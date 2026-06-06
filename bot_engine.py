"""
TrendBot — Trend Following + Swing por niveles
Estrategias: Weinstein Stage 2/4, HH/HL estructura, EMA pullback
Sin sesgo: opera LONG en alcistas y SHORT en bajistas
"""
from __future__ import annotations

import copy
import base64
import fcntl
import hashlib
import hmac
import json
import math
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from pattern_memory import PatternMemory, detect_patterns
from typing import Optional
from urllib.parse import urlencode

import requests
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Indicadores
# ─────────────────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def _vol_ratio(df: pd.DataFrame, n: int = 20) -> pd.Series:
    avg = df["volume"].rolling(n).mean()
    return df["volume"] / avg.replace(0, np.nan)

def _rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=n - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=n - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    up   = h.diff()
    down = -l.diff()
    plus_dm  = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr14    = _atr(df, n)
    plus_di  = 100 * plus_dm.ewm(span=n, adjust=False).mean()  / atr14.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(span=n, adjust=False).mean() / atr14.replace(0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    return dx.ewm(span=n, adjust=False).mean()

def _swing_highs(df: pd.DataFrame, n: int = 3) -> list[float]:
    highs = []
    for i in range(n, len(df) - n):
        val = df["high"].iloc[i]
        if all(val > df["high"].iloc[i - j] for j in range(1, n + 1)) and \
           all(val > df["high"].iloc[i + j] for j in range(1, n + 1)):
            highs.append(float(val))
    return highs[-6:]

def _swing_lows(df: pd.DataFrame, n: int = 3) -> list[float]:
    lows = []
    for i in range(n, len(df) - n):
        val = df["low"].iloc[i]
        if all(val < df["low"].iloc[i - j] for j in range(1, n + 1)) and \
           all(val < df["low"].iloc[i + j] for j in range(1, n + 1)):
            lows.append(float(val))
    return lows[-6:]

def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema20"]     = _ema(df["close"], 20)
    df["ema50"]     = _ema(df["close"], 50)
    df["ema150"]    = _ema(df["close"], 150)
    df["ema200"]    = _ema(df["close"], 200)
    df["atr"]       = _atr(df, 14)
    df["vol_ratio"] = _vol_ratio(df, 20)
    df["rsi"]       = _rsi(df["close"], 14)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Motor principal
# ─────────────────────────────────────────────────────────────────────────────

class TrendBot:

    def __init__(self, cfg: dict, restore_state: Optional[dict] = None):
        self.cfg        = cfg
        self._lock      = threading.RLock()
        self._running   = False
        self._thread: Optional[threading.Thread] = None
        self._estados: dict[str, dict] = {}
        self._estados_mom: dict[str, dict] = {}
        self._logs: deque  = deque(maxlen=200)
        self._balance      = float(cfg.get("balance_inicial", 1000.0))
        self._balance_ini  = self._balance
        self._ganancia     = 0.0
        self._wins         = 0
        self._losses       = 0
        self._operations   = 0
        self._historial: deque  = deque(maxlen=500)
        self._historial_counter = 0
        self._df_cache: dict[str, pd.DataFrame] = {}
        self._exchange_info: dict[str, dict] = {}
        self._pattern_memory = PatternMemory(
            str(Path(__file__).with_name("pattern_memory.json"))
        )
        self._public_connected = False
        self._private_connected = False
        self._connection_error: Optional[str] = None
        self._last_account_sync: Optional[str] = None
        self._real_balance: Optional[float] = None
        self._real_available_balance: Optional[float] = None
        self._exchange_positions: dict[str, dict] = {}
        self._live_prices: dict[str, float] = {}
        self._position_mode = "net_mode"
        self._alerts: deque = deque(maxlen=50)
        self._last_status: dict = {
            "running": False,
            "balance": round(self._balance, 2),
            "balance_ini": round(self._balance_ini, 2),
            "pnl": 0.0,
            "open_pnl": 0.0,
            "total_pnl": 0.0,
            "wins": 0,
            "losses": 0,
            "operations": 0,
            "win_rate": 0,
            "symbols": {},
            "configured_symbols": self.cfg.get("symbols", []),
            "execution_mode": self.cfg.get("execution_mode", "SIMULADO"),
            "last_update": datetime.now(timezone.utc).isoformat(),
        }
        self._last_logs_cache: list[dict] = []
        self._run_lock_handle = None
        if restore_state:
            self._restore_runtime_state(restore_state)

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, msg: str, level: str = "info"):
        now = datetime.now(timezone.utc)
        entry = {"ts": now.isoformat(), "level": level, "msg": msg}
        with self._lock:
            self._logs.appendleft(entry)
            self._last_logs_cache = list(self._logs)
        print(f"[{now.strftime('%H:%M:%S')}] {msg}", flush=True)

    def _send_telegram(self, msg: str):
        token = self.cfg.get("telegram_token", "").strip()
        chat_id = self.cfg.get("telegram_chat_id", "").strip()
        if not token or not chat_id:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"},
                timeout=5,
            )
        except Exception:
            pass

    def _push_alert(self, msg: str, kind: str = "info", sym: str = ""):
        """kind: info | success | warning | error"""
        with self._lock:
            self._alerts.append({
                "id":  int(datetime.now(timezone.utc).timestamp() * 1000),
                "ts":  datetime.now(timezone.utc).strftime("%H:%M:%S"),
                "msg": msg,
                "kind": kind,
                "sym": sym,
            })
        if kind == "success":
            self._send_telegram(f"🤖 <b>Axiom Bot</b>\n{msg}")

    # ── Control ───────────────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        if not self._acquire_run_lock():
            self._log("No se inició: ya hay otra instancia de TrendBot operando", "error")
            self._push_alert("Otra instancia de TrendBot ya está operando. Detén la duplicada antes de iniciar.", "error")
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._log("TrendBot iniciado")

    def stop(self):
        self._running = False
        self._log("TrendBot detenido")
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._release_run_lock()

    def _acquire_run_lock(self) -> bool:
        if self._run_lock_handle:
            return True
        lock_path = Path("/tmp/trendbot.run.lock")
        handle = lock_path.open("w")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return False
        handle.write(str(threading.get_ident()))
        handle.flush()
        self._run_lock_handle = handle
        return True

    def _release_run_lock(self):
        handle = self._run_lock_handle
        if not handle:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._run_lock_handle = None

    def update_config(self, new_cfg: dict):
        with self._lock:
            self.cfg = new_cfg
            for sym in self.cfg.get("symbols", []):
                if sym not in self._estados:
                    self._estados[sym] = self._estado_vacio()
                elif not self._estados[sym].get("posicion_abierta"):
                    self._estados[sym]["apalancamiento"] = self._symbol_leverage(sym)
                    self._estados[sym]["capital"] = self._symbol_capital(sym)
                if sym in self._estados_mom and not self._estados_mom[sym].get("posicion_abierta"):
                    self._estados_mom[sym]["apalancamiento"] = self._symbol_leverage(sym)
                    self._estados_mom[sym]["capital"] = self._symbol_capital(sym)
            self._estados = {sym: self._estados[sym] for sym in self.cfg.get("symbols", []) if sym in self._estados}
            self._df_cache = {sym: df for sym, df in self._df_cache.items() if sym in self._estados}
        self._log("Configuración actualizada")

    def _restore_runtime_state(self, state: dict):
        self._balance = float(state.get("balance", self._balance))
        self._balance_ini = float(state.get("balance_ini", self._balance_ini))
        self._ganancia = float(state.get("ganancia", self._ganancia))
        repaired_sim_balance = self._repair_simulated_balance_if_needed()
        self._wins = int(state.get("wins", self._wins))
        self._losses = int(state.get("losses", self._losses))
        self._operations = int(state.get("operations", self._operations))
        if repaired_sim_balance:
            self._wins = 0
            self._losses = 0
            self._operations = 0
        self._last_account_sync = state.get("last_account_sync")
        hist = state.get("historial", [])
        if isinstance(hist, list):
            self._historial = deque(hist[-500:], maxlen=500)
            self._historial_counter = int(state.get("historial_counter", len(self._historial)))
        restored_mom = state.get("estados_mom", {})
        if isinstance(restored_mom, dict):
            self._estados_mom = restored_mom
            for est in self._estados_mom.values():
                est["signal_confirmado"] = 0
                est["signal_short_confirmado"] = 0
        restored_states = state.get("estados", {})
        if isinstance(restored_states, dict):
            self._estados = restored_states
            # Resetear contadores de confirmación para evitar entrada inmediata al reiniciar
            for est in self._estados.values():
                est["signal_confirmado"] = 0
                est["signal_short_confirmado"] = 0
        restored_logs = state.get("logs", [])
        if isinstance(restored_logs, list):
            self._logs = deque(restored_logs[:200], maxlen=200)
            self._last_logs_cache = list(self._logs)
        if self._is_live_mode() and not self._has_private_keys():
            self._balance_ini = self._balance
            self._ganancia = 0.0

    def _repair_simulated_balance_if_needed(self):
        if self._is_live_mode():
            return False
        configured_initial = float(self.cfg.get("balance_inicial", 1000.0))
        if configured_initial <= 0:
            return False
        restored_initial_mismatch = abs(self._balance_ini - configured_initial) > 0.01
        impossible_restored_balance = self._balance < max(10.0, configured_initial * 0.05)
        if restored_initial_mismatch or impossible_restored_balance:
            self._balance = configured_initial
            self._balance_ini = configured_initial
            self._ganancia = 0.0
            return True
        return False

    def export_runtime_state(self) -> dict:
        with self._lock:
            cfg = copy.deepcopy(self.cfg)
            cfg["api_key"] = ""
            cfg["api_secret"] = ""
            cfg["okx_passphrase"] = ""
            balance_ini = self._balance_ini
            ganancia = self._ganancia
            if self._is_live_mode() and not self._has_private_keys():
                balance_ini = self._balance
                ganancia = 0.0
            return {
                "cfg": cfg,
                "running": self._running,
                "balance": self._balance,
                "balance_ini": balance_ini,
                "ganancia": ganancia,
                "wins": self._wins,
                "losses": self._losses,
                "operations": self._operations,
                "estados": copy.deepcopy(self._estados),
                "logs": list(self._logs),
                "last_account_sync": self._last_account_sync,
                "historial": list(self._historial),
                "historial_counter": self._historial_counter,
                "estados_mom": copy.deepcopy(self._estados_mom),
            }

    # ── Conexión / API ────────────────────────────────────────────────────────

    def _execution_mode(self) -> str:
        return str(self.cfg.get("execution_mode", "SIMULADO")).upper()

    def _is_live_mode(self) -> bool:
        return self._execution_mode() in {"REAL", "TESTNET"}

    def _trade_symbols(self) -> set[str]:
        return set(self.cfg.get("trade_symbols", []) or self.cfg.get("symbols", []))

    def _symbol_leverage(self, sym: str) -> int:
        mapping = self.cfg.get("symbol_leverage", {}) or {}
        if sym in mapping:
            try:
                return int(mapping[sym])
            except Exception:
                pass
        return int(self.cfg.get("apalancamiento", 3))

    def _symbol_capital(self, sym: str) -> float:
        mapping = self.cfg.get("symbol_capital", {}) or {}
        if sym in mapping:
            try:
                value = float(mapping[sym])
                if value > 0:
                    return value
            except Exception:
                pass
        return float(self.cfg.get("capital_usd", 25.0))

    def _base_url(self) -> str:
        return "https://www.okx.com"

    def _api_symbol(self, sym: str) -> str:
        base, quote = sym.split("/") if "/" in sym else (sym.replace("USDT", ""), "USDT")
        return f"{base}-{quote}-SWAP"

    def _ui_symbol(self, inst_id: str) -> str:
        parts = str(inst_id or "").split("-")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return str(inst_id or "").replace("USDT", "/USDT")

    def _has_private_keys(self) -> bool:
        return bool(self.cfg.get("api_key")) and bool(self.cfg.get("api_secret")) and bool(self.cfg.get("okx_passphrase"))

    def _request_public(self, path: str, params: Optional[dict] = None, timeout: tuple[int, int] = (5, 10)):
        r = requests.get(f"{self._base_url()}{path}", params=params or {}, timeout=timeout)
        r.raise_for_status()
        payload = r.json()
        if isinstance(payload, dict) and payload.get("code") not in (None, "0"):
            raise RuntimeError(payload.get("msg") or str(payload))
        return payload

    def _okx_headers(self, method: str, request_path: str, body: str = "") -> dict:
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        prehash = f"{ts}{method.upper()}{request_path}{body}"
        signature = base64.b64encode(
            hmac.new(
                str(self.cfg.get("api_secret", "")).encode(),
                prehash.encode(),
                hashlib.sha256,
            ).digest()
        ).decode()
        headers = {
            "OK-ACCESS-KEY": str(self.cfg.get("api_key", "")),
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": str(self.cfg.get("okx_passphrase", "")),
        }
        # Solo TESTNET activa el paper trading de OKX.
        # binance_testnet=True no debe interferir cuando execution_mode=REAL.
        if self._execution_mode() == "TESTNET":
            headers["x-simulated-trading"] = "1"
        return headers

    def _request_signed(self, method: str, path: str, params: Optional[dict] = None):
        if not self._has_private_keys():
            raise RuntimeError("Faltan API key, API secret y passphrase de OKX")
        method = method.upper()
        query = urlencode(params or {}, doseq=True)
        request_path = f"{path}?{query}" if method == "GET" and query else path
        body = "" if method == "GET" else json.dumps(params or {}, separators=(",", ":"))
        headers = self._okx_headers(method, request_path, body)
        headers["Content-Type"] = "application/json"
        url = f"{self._base_url()}{path}"
        if method == "GET":
            r = requests.get(url, params=params or {}, headers=headers, timeout=(5, 15))
        else:
            r = requests.post(url, data=body, headers=headers, timeout=(5, 15))
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            detail = (r.text or "").strip()
            if detail:
                raise requests.HTTPError(f"{e} | OKX: {detail}", response=r) from e
            raise
        payload = r.json()
        if isinstance(payload, dict) and payload.get("code") not in (None, "0"):
            raise RuntimeError(payload.get("msg") or str(payload))
        return payload

    def _load_exchange_info(self):
        payload = self._request_public("/api/v5/public/instruments", {"instType": "SWAP"})
        exchange_info: dict[str, dict] = {}
        for item in payload.get("data", []):
            if item.get("state") != "live" or item.get("settleCcy") != "USDT":
                continue
            inst_id = item.get("instId", "")
            base = item.get("uly", "").split("-")[0] or item.get("baseCcy", "")
            if not inst_id or not base:
                continue
            exchange_info[f"{base}/USDT"] = {
                "symbol": inst_id,
                "qty_step": float(item.get("lotSz", 1.0) or 1.0),
                "qty_min": float(item.get("minSz", 0.0) or 0.0),
                "price_tick": float(item.get("tickSz", 0.01) or 0.01),
                "ct_val": float(item.get("ctVal", 1.0) or 1.0),
                "ct_val_ccy": item.get("ctValCcy", base),
            }
        self._exchange_info = exchange_info

    def _round_step(self, value: float, step: float) -> float:
        if step <= 0:
            return value
        return math.floor(value / step) * step

    def _format_okx_number(self, value: float) -> str:
        return f"{value:.12f}".rstrip("0").rstrip(".")

    def _contracts_to_base_qty(self, sym: str, contracts: float) -> float:
        info = self._exchange_info.get(sym, {})
        return float(contracts) * float(info.get("ct_val") or 1.0)

    def _base_qty_to_contracts(self, sym: str, base_qty: float) -> float:
        info = self._exchange_info.get(sym, {})
        ct_val = float(info.get("ct_val") or 1.0)
        return float(base_qty) / ct_val if ct_val > 0 else float(base_qty)

    def _position_side(self, direction: str) -> str:
        return "long" if direction == "LONG" else "short"

    def _order_side(self, direction: str, closing: bool = False) -> str:
        if direction == "LONG":
            return "sell" if closing else "buy"
        return "buy" if closing else "sell"

    def _trade_mode(self) -> str:
        return "isolated" if str(self.cfg.get("margin_type", "ISOLATED")).upper() == "ISOLATED" else "cross"

    def _order_payload_base(self, sym: str, direction: str) -> dict:
        payload = {"instId": self._api_symbol(sym), "tdMode": self._trade_mode()}
        if self._position_mode == "long_short_mode":
            payload["posSide"] = self._position_side(direction)
        return payload

    def _sync_account_config(self):
        if not self._has_private_keys():
            return
        try:
            payload = self._request_signed("GET", "/api/v5/account/config")
            data = (payload.get("data") or [{}])[0]
            self._position_mode = str(data.get("posMode") or "net_mode")
        except Exception as e:
            self._log(f"No pude leer modo de posiciones OKX: {e}", "warning")

    def _sync_account(self):
        if not self._private_connected or not self._has_private_keys():
            self._exchange_positions = {}
            return
        account = self._request_signed("GET", "/api/v5/account/balance", {"ccy": "USDT"})
        data = (account.get("data") or [{}])[0]
        usdt = next((a for a in data.get("details", []) if a.get("ccy") == "USDT"), None)
        if not usdt:
            return
        wallet_balance = float(usdt.get("cashBal") or usdt.get("eq") or 0.0)
        available_balance = float(usdt.get("availBal") or usdt.get("availEq") or wallet_balance or 0.0)
        first_real_sync = self._real_balance is None
        # Siempre actualiza el balance real (visible en UI)
        self._real_balance = wallet_balance
        self._real_available_balance = available_balance
        self._last_account_sync = datetime.now(timezone.utc).isoformat()
        self._connection_error = None
        # Solo en modo real/testnet afecta el balance del bot
        if self._is_live_mode():
            if first_real_sync:
                self._balance_ini = wallet_balance
            self._balance = wallet_balance
            self._ganancia = self._balance - self._balance_ini
        self._sync_exchange_positions()

    def _effective_order_capital(self, requested_capital: float) -> float:
        if not self._is_live_mode():
            return requested_capital
        # Forzar sincronización si no tenemos balance reciente
        if self._real_available_balance is None:
            try:
                self._sync_account()
            except Exception as e:
                raise RuntimeError(f"No se puede verificar balance en OKX antes de operar: {e}") from e
        available = self._real_available_balance
        if available is None:
            available = self._real_balance
        if available is None:
            raise RuntimeError("Balance OKX no disponible — verifica tu conexión y credenciales")
        if available <= 0:
            raise RuntimeError(f"Balance disponible en OKX es ${available:.2f} USDT — sin fondos suficientes para abrir posición")
        safe_capital = available * 0.95
        if requested_capital > safe_capital:
            self._log(
                f"Capital solicitado ${requested_capital:.2f} supera disponible OKX ${available:.2f} → ajustado a ${safe_capital:.2f}",
                "warning"
            )
            self._push_alert(
                f"⚠️ Capital ajustado ${safe_capital:.2f} (OKX disponible: ${available:.2f} USDT)",
                "warning"
            )
        return min(requested_capital, safe_capital)

    def _sync_exchange_positions(self):
        if not self._private_connected or not self._has_private_keys():
            self._exchange_positions = {}
            return
        try:
            payload = self._request_signed("GET", "/api/v5/account/positions", {"instType": "SWAP"})
        except Exception as e:
            self._log(f"No se pudo sincronizar PnL real de posiciones OKX: {e}", "warning")
            return

        live_positions = {}
        for pos in payload.get("data", []):
            try:
                amt = float(pos.get("pos", 0.0) or 0.0)
            except Exception:
                continue
            if amt == 0.0:
                continue
            raw_sym = pos.get("instId", "")
            if not raw_sym or "-USDT-" not in raw_sym:
                continue

            sym = self._ui_symbol(raw_sym)
            pos_side = str(pos.get("posSide") or "net").lower()
            if pos_side == "short":
                raw_qty = -abs(amt)
            elif pos_side == "long":
                raw_qty = abs(amt)
            else:
                raw_qty = amt
            try:
                lev = int(float(pos.get("lever", 1) or 1))
            except Exception:
                lev = 1
            base_qty = self._contracts_to_base_qty(sym, abs(raw_qty))
            live_positions[sym] = {
                "qty": base_qty,
                "raw_qty": -base_qty if raw_qty < 0 else base_qty,
                "okx_contracts": abs(raw_qty),
                "entry": float(pos.get("avgPx", 0.0) or 0.0),
                "mark_price": float(pos.get("markPx", 0.0) or 0.0),
                "unrealized_pnl": float(pos.get("upl", 0.0) or 0.0),
                "leverage": lev,
            }
        self._exchange_positions = live_positions
        if self._is_live_mode():
            self._clear_missing_exchange_positions(live_positions)

    def _clear_missing_exchange_positions(self, live_positions: dict[str, dict]):
        """Limpia posiciones locales que OKX ya no reporta como abiertas."""
        with self._lock:
            for sym, est in list(self._estados.items()):
                if not est.get("posicion_abierta") or sym in live_positions:
                    continue
                old_dir = est.get("direccion_pos") or "?"
                new_st = self._estado_vacio()
                new_st["tendencia"] = est.get("tendencia", "NEUTRAL")
                new_st["fase"] = est.get("fase", "")
                new_st["cooldown_restante"] = int(self.cfg.get("cooldown_ciclos", 3))
                new_st["ultimo_cierre"] = datetime.now(timezone.utc).isoformat()
                new_st["ultimo_resultado"] = "SYNC"
                self._estados[sym] = new_st
                self._log(f"[{sym}] Posición {old_dir} limpiada: OKX ya no la reporta abierta", "warning")

    def _reconcile_positions(self):
        """
        Consulta posiciones abiertas en OKX y sincroniza el estado interno.
        Evita operar ciego si el bot se reinició con posiciones ya abiertas.
        """
        if not self._is_live_mode() or not self._private_connected:
            return
        try:
            payload = self._request_signed("GET", "/api/v5/account/positions", {"instType": "SWAP"})
        except Exception as e:
            self._log(f"No se pudo reconciliar posiciones: {e}", "warning")
            return

        reconciled = 0
        for pos in payload.get("data", []):
            amt = float(pos.get("pos", 0.0) or 0.0)
            if amt == 0.0:
                continue

            raw_sym = pos.get("instId", "")
            if not raw_sym or "-USDT-" not in raw_sym:
                continue
            sym = self._ui_symbol(raw_sym)

            if sym not in self.cfg.get("symbols", []):
                continue

            pos_side = str(pos.get("posSide") or "net").lower()
            signed_amt = -abs(amt) if pos_side == "short" else abs(amt) if pos_side == "long" else amt
            direction = "LONG" if signed_amt > 0 else "SHORT"
            entry_price = float(pos.get("avgPx", 0.0) or 0.0)
            leverage    = int(float(pos.get("lever", 1) or 1))
            okx_contracts = abs(amt)
            qty           = self._contracts_to_base_qty(sym, okx_contracts)
            capital       = round((qty * entry_price) / leverage, 2)
            self._exchange_positions[sym] = {
                "qty": qty,
                "raw_qty": -qty if signed_amt < 0 else qty,
                "okx_contracts": okx_contracts,
                "entry": entry_price,
                "mark_price": float(pos.get("markPx", 0.0) or 0.0),
                "unrealized_pnl": float(pos.get("upl", 0.0) or 0.0),
                "leverage": leverage,
            }

            with self._lock:
                est = self._estados.setdefault(sym, self._estado_vacio())
                if est.get("posicion_abierta"):
                    continue  # ya la conocemos

                # Calcular SL/TP provisionales con ATR del cache
                atr_val = 0.0
                df_cached = self._df_cache.get(sym)
                if df_cached is not None and len(df_cached) > 0:
                    atr_val = float(df_cached.iloc[-1].get("atr", 0.0))

                atr_sl   = float(self.cfg.get("atr_sl_mult", 1.5))
                atr_tp   = float(self.cfg.get("atr_tp_mult", 4.0))
                atr_tr   = float(self.cfg.get("atr_trail_mult", 2.0))

                if atr_val > 0:
                    if direction == "LONG":
                        sl_p = entry_price - atr_val * atr_sl
                        tp_p = entry_price + atr_val * atr_tp
                    else:
                        sl_p = entry_price + atr_val * atr_sl
                        tp_p = entry_price - atr_val * atr_tp
                else:
                    pct = 0.02
                    sl_p = entry_price * (1 - pct) if direction == "LONG" else entry_price * (1 + pct)
                    tp_p = entry_price * (1 + pct * 3) if direction == "LONG" else entry_price * (1 - pct * 3)

                est.update({
                    "posicion_abierta":     True,
                    "direccion_pos":        direction,
                    "precio_entrada":       entry_price,
                    "sl_inicial":           sl_p,
                    "sl_actual":            sl_p,
                    "tp":                   tp_p,
                    "precio_ext":           entry_price,
                    "exchange_qty":         qty,
                    "capital":              capital,
                    "apalancamiento":       leverage,
                    "breakeven_activado":   False,
                    "salida_parcial_hecha": False,
                    "pnl_pct":              0.0,
                    "ts_apertura":          datetime.now(timezone.utc).isoformat(),
                })
                reconciled += 1
                self._log(
                    f"[{sym}] Posición {direction} reconciliada desde OKX — "
                    f"entrada ${entry_price:,.4f} | qty {qty} | lev {leverage}x | "
                    f"SL ${sl_p:,.4f} TP ${tp_p:,.4f}",
                    "warning"
                )

        if reconciled:
            self._log(f"Reconciliación completa: {reconciled} posición(es) importada(s) de OKX", "success")
        else:
            self._log("Reconciliación completa: no hay posiciones abiertas en OKX", "info")

    def _set_leverage(self, sym: str, leverage: int):
        if not self._is_live_mode():
            return
        payload = {
            "instId": self._api_symbol(sym),
            "lever": str(leverage),
            "mgnMode": self._trade_mode(),
        }
        if self._position_mode == "long_short_mode":
            for pos_side in ("long", "short"):
                self._request_signed("POST", "/api/v5/account/set-leverage", {**payload, "posSide": pos_side})
            return
        self._request_signed("POST", "/api/v5/account/set-leverage", payload)

    def _set_margin_type(self, sym: str):
        return

    def _conectar(self):
        self._connection_error = None
        self._request_public("/api/v5/public/time", timeout=(5, 10))
        self._public_connected = True
        self._load_exchange_info()
        if self._is_live_mode():
            if not self._has_private_keys():
                raise RuntimeError("Modo real/demo requiere API key, API secret y passphrase de OKX")
            self._sync_account_config()
            self._request_signed("GET", "/api/v5/account/balance", {"ccy": "USDT"})
            self._private_connected = True
            self._sync_account()
        elif self._has_private_keys():
            # Simulado con claves — conectar en modo lectura para ver balance real
            try:
                self._sync_account_config()
                self._request_signed("GET", "/api/v5/account/balance", {"ccy": "USDT"})
                self._private_connected = True
                self._sync_account()
                self._log("API privada conectada (solo lectura — modo simulado)", "info")
            except Exception as e:
                self._private_connected = False
                self._log(f"API privada no disponible: {e}", "warning")
        else:
            self._private_connected = False

    def test_connection(self) -> dict:
        try:
            self._conectar()
            return {
                "ok": True, "public_api": self._public_connected,
                "private_api": self._private_connected,
                "mode": self._execution_mode(),
                "balance": round(self._balance, 2),
                "error": None,
            }
        except Exception as e:
            self._connection_error = str(e)
            self._public_connected = False
            self._private_connected = False
            return {"ok": False, "error": str(e)}

    def _round_price(self, sym: str, price: float) -> float:
        info = self._exchange_info.get(sym, {})
        tick = float(info.get("price_tick", 0.01))
        if tick <= 0:
            return round(price, 4)
        return round(round(price / tick) * tick, 8)

    def _place_sl_order(self, sym: str, direction: str, sl_price: float) -> Optional[int]:
        """SL exchange pendiente de migración a OKX."""
        if not self._is_live_mode():
            return None
        self._log(f"[{sym}] SL exchange pendiente de migración OKX; gestión local activa", "warning")
        return None

    def _cancel_sl_order(self, sym: str, order_id: Optional[int]) -> bool:
        """Cancela la orden SL existente cuando la ejecución OKX esté activa."""
        if not self._is_live_mode() or not order_id:
            return True
        self._log(f"[{sym}] Cancelación SL OKX pendiente de migración #{order_id}", "warning")
        return True

    def _update_sl_order(self, sym: str, direction: str, old_order_id: Optional[int], new_sl_price: float) -> Optional[int]:
        """Cancela el SL viejo y coloca uno nuevo. Retorna el nuevo order_id."""
        self._cancel_sl_order(sym, old_order_id)
        return self._place_sl_order(sym, direction, new_sl_price)

    def _place_live_order(self, sym: str, direction: str, capital: float, leverage: int, price: float) -> tuple[float, dict]:
        info = self._exchange_info.get(sym)
        if not info:
            raise RuntimeError(f"No hay reglas de mercado OKX para {sym}")
        # _effective_order_capital fuerza sync, valida balance y ajusta capital
        capital = self._effective_order_capital(capital)
        if capital <= 0:
            raise RuntimeError("Balance disponible insuficiente para abrir orden")
        # Verificar que el margen requerido (capital) no supere el balance disponible
        available = self._real_available_balance or self._real_balance
        if available is not None and capital > available:
            raise RuntimeError(
                f"Margen requerido ${capital:.2f} supera balance disponible ${available:.2f} USDT en OKX"
            )

        self._set_margin_type(sym)
        self._set_leverage(sym, leverage)

        ct_val = float(info.get("ct_val") or 1.0)
        base_qty = (capital * leverage) / price if price > 0 else 0.0
        contracts = self._round_step(base_qty / ct_val if ct_val > 0 else base_qty, float(info["qty_step"]))
        if contracts < float(info["qty_min"]):
            raise RuntimeError(f"Cantidad {contracts:g} menor al mínimo {info['qty_min']} contrato(s) para {sym}")

        payload = {
            **self._order_payload_base(sym, direction),
            "side": self._order_side(direction),
            "ordType": "market",
            "sz": self._format_okx_number(contracts),
        }
        order = self._request_signed("POST", "/api/v5/trade/order", payload)
        data = (order.get("data") or [{}])[0]
        ord_id = data.get("ordId") or data.get("clOrdId") or "—"
        base_qty_filled = self._contracts_to_base_qty(sym, contracts)
        normalized = {
            **data,
            "orderId": ord_id,
            "ordId": ord_id,
            "executedQty": base_qty_filled,
            "origQty": base_qty_filled,
            "okxContracts": contracts,
            "okx_payload": payload,
        }
        return base_qty_filled, normalized

    def _close_live_order(self, sym: str, direction: str, qty: float) -> dict:
        if qty <= 0:
            raise RuntimeError(f"Cantidad inválida para cerrar {sym}")
        info = self._exchange_info.get(sym)
        if not info:
            raise RuntimeError(f"No hay reglas de mercado OKX para {sym}")
        contracts = self._round_step(self._base_qty_to_contracts(sym, float(qty)), float(info["qty_step"]))
        if contracts < float(info["qty_min"]):
            raise RuntimeError(f"Cantidad {contracts:g} menor al mínimo {info['qty_min']} contrato(s) para cerrar {sym}")

        payload = {
            **self._order_payload_base(sym, direction),
            "side": self._order_side(direction, closing=True),
            "ordType": "market",
            "sz": self._format_okx_number(contracts),
        }
        if self._position_mode != "long_short_mode":
            payload["reduceOnly"] = "true"
        order = self._request_signed("POST", "/api/v5/trade/order", payload)
        data = (order.get("data") or [{}])[0]
        ord_id = data.get("ordId") or data.get("clOrdId") or "—"
        return {
            **data,
            "orderId": ord_id,
            "ordId": ord_id,
            "executedQty": self._contracts_to_base_qty(sym, contracts),
            "origQty": self._contracts_to_base_qty(sym, contracts),
            "okxContracts": contracts,
            "okx_payload": payload,
        }

    def _fetch_df(self, sym: str, tf: str, limit: int = 300) -> pd.DataFrame:
        bar = {"1h": "1H"}.get(tf, tf)
        payload = self._request_public(
            "/api/v5/market/candles",
            params={"instId": self._api_symbol(sym), "bar": bar, "limit": limit},
            timeout=(5, 10),
        )
        raw = list(reversed(payload.get("data", [])))
        cols = ["ts","open","high","low","close","volume","vol_ccy","vol_quote","confirm"]
        df = pd.DataFrame(raw, columns=cols)[["ts","open","high","low","close","volume"]]
        df = df.astype({"ts":"int64","open":"float64","high":"float64",
                        "low":"float64","close":"float64","volume":"float64"})
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return _prepare(df)

    def _fetch_ticker_price(self, sym: str) -> float:
        payload = self._request_public(
            "/api/v5/market/ticker",
            params={"instId": self._api_symbol(sym)},
            timeout=(5, 10),
        )
        data = (payload.get("data") or [{}])[0]
        return float(data.get("last") or data.get("askPx") or data.get("bidPx") or 0.0)

    # ── Status público ────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        if not self._lock.acquire(timeout=0.5):
            return dict(self._last_status)
        try:
            syms_data = {}
            open_pnl_total = 0.0
            trade_syms = self._trade_symbols()
            for sym, est in self._estados.items():
                df = self._df_cache.get(sym)
                if df is None or len(df) < 2:
                    continue
                u    = df.iloc[-1]
                prev = df.iloc[-2]
                live_price = float(self._live_prices.get(sym) or u["close"])
                position = {
                    "open":       est.get("posicion_abierta", False),
                    "direction":  est.get("direccion_pos"),
                    "entry":      est.get("precio_entrada"),
                    "sl":         est.get("sl_actual"),
                    "tp":         est.get("tp"),
                    "capital":    est.get("capital", self.cfg.get("capital_usd", 25.0)) if est.get("posicion_abierta") else self.cfg.get("capital_usd", 25.0),
                    "apalancamiento": est.get("apalancamiento", self._symbol_leverage(sym)) if est.get("posicion_abierta") else self._symbol_leverage(sym),
                    "breakeven":  est.get("breakeven_activado", False),
                    "partial":    est.get("salida_parcial_hecha", False),
                    "pnl_pct":    est.get("pnl_pct", 0.0),
                    "pnl_usd":    0.0,
                }
                if position["open"] and position["entry"]:
                    exchange_pos = self._exchange_positions.get(sym) if self._private_connected else None
                    if exchange_pos:
                        mark_price = float(exchange_pos.get("mark_price") or 0.0)
                        position["pnl_usd"] = round(float(exchange_pos.get("unrealized_pnl") or 0.0), 2)
                        position["mark_price"] = round(mark_price, 8)
                        position["exchange_qty"] = float(exchange_pos.get("qty") or 0.0)
                        position["pnl_source"] = "OKX"
                        if mark_price > 0:
                            if position["direction"] == "LONG":
                                position["pnl_pct"] = round((mark_price - float(position["entry"])) / float(position["entry"]) * 100, 2)
                            else:
                                position["pnl_pct"] = round((float(position["entry"]) - mark_price) / float(position["entry"]) * 100, 2)
                    else:
                        qty = float(est.get("exchange_qty") or 0.0)
                        if qty <= 0:
                            qty = (float(position["capital"]) * int(position["apalancamiento"])) / float(position["entry"])
                        if position["direction"] == "LONG":
                            position["pnl_usd"] = round((live_price - float(position["entry"])) * qty, 2)
                        else:
                            position["pnl_usd"] = round((float(position["entry"]) - live_price) * qty, 2)
                        position["pnl_source"] = "LOCAL"
                    open_pnl_total += position["pnl_usd"]

                est_m = self._estados_mom.get(sym, {})
                mom_pos = {
                    "open":       est_m.get("posicion_abierta", False),
                    "direction":  est_m.get("direccion_pos"),
                    "entry":      est_m.get("precio_entrada"),
                    "sl":         est_m.get("sl_actual"),
                    "tp":         est_m.get("tp"),
                    "capital":    est_m.get("capital", self.cfg.get("capital_usd", 25.0)) if est_m.get("posicion_abierta") else self.cfg.get("capital_usd", 25.0),
                    "apalancamiento": est_m.get("apalancamiento", self._symbol_leverage(sym)) if est_m.get("posicion_abierta") else self._symbol_leverage(sym),
                    "pnl_pct":    est_m.get("pnl_pct", 0.0),
                    "pnl_usd":    0.0,
                    "score_l":    est_m.get("score_mom_l", 0),
                    "score_s":    est_m.get("score_mom_s", 0),
                    "checklist":  est_m.get("checklist_mom", []),
                }
                if mom_pos["open"] and mom_pos["entry"]:
                    exchange_pos = self._exchange_positions.get(sym) if self._private_connected else None
                    if exchange_pos:
                        mark_price = float(exchange_pos.get("mark_price") or 0.0)
                        mom_pos["pnl_usd"] = round(float(exchange_pos.get("unrealized_pnl") or 0.0), 2)
                        mom_pos["mark_price"] = round(mark_price, 8)
                        mom_pos["exchange_qty"] = float(exchange_pos.get("qty") or 0.0)
                        mom_pos["pnl_source"] = "OKX"
                        if mark_price > 0:
                            if mom_pos["direction"] == "LONG":
                                mom_pos["pnl_pct"] = round((mark_price - float(mom_pos["entry"])) / float(mom_pos["entry"]) * 100, 2)
                            else:
                                mom_pos["pnl_pct"] = round((float(mom_pos["entry"]) - mark_price) / float(mom_pos["entry"]) * 100, 2)
                    else:
                        qty_m = float(est_m.get("exchange_qty") or 0.0)
                        if qty_m <= 0:
                            qty_m = (float(mom_pos["capital"]) * int(mom_pos["apalancamiento"])) / float(mom_pos["entry"])
                        if mom_pos["direction"] == "LONG":
                            mom_pos["pnl_usd"] = round((live_price - float(mom_pos["entry"])) * qty_m, 2)
                        else:
                            mom_pos["pnl_usd"] = round((float(mom_pos["entry"]) - live_price) * qty_m, 2)
                        mom_pos["pnl_source"] = "LOCAL"
                    open_pnl_total += mom_pos["pnl_usd"]

                try:
                    df_sig = df.iloc[:-1] if len(df) > 20 else df
                    hh_hl, lh_ll, _, _ = self._estructura_hh_hl(df_sig)
                    if hh_hl:
                        estructura = "HH/HL"
                    elif lh_ll:
                        estructura = "LH/LL"
                    else:
                        estructura = "NEUTRAL"
                except Exception:
                    estructura = "NEUTRAL"

                syms_data[sym] = {
                    "price":        round(live_price, 6),
                    "price_source": "OKX ticker" if sym in self._live_prices else "OKX candle",
                    "change_pct":   round((live_price - float(prev["close"])) / float(prev["close"]) * 100, 2),
                    "ema20":        round(float(u["ema20"]), 4),
                    "ema50":        round(float(u["ema50"]), 4),
                    "ema200":       round(float(u["ema200"]), 4),
                    "rsi":          round(float(u["rsi"]), 1),
                    "vol_ratio":    round(float(prev["vol_ratio"]), 2),
                    "atr":          round(float(u["atr"]), 4),
                    "estructura":   estructura,
                    "tendencia":    est.get("tendencia", "NEUTRAL"),
                    "fase":         est.get("fase", "—"),
                    "signal_long":  est.get("signal_long", False),
                    "signal_short": est.get("signal_short", False),
                    "score_long":   est.get("score_long", 0.0),
                    "score_short":  est.get("score_short", 0.0),
                    "checklist":    est.get("checklist", []),
                    "watch_only":   sym not in trade_syms,
                    "position":     position,
                    "mom_position": mom_pos,
                }
            recommendations = self._build_recommendations(syms_data, trade_syms)
            status = {
                "running":        self._running,
                "balance":        round(self._balance, 2),
                "balance_ini":    round(self._balance_ini, 2),
                "real_balance":   round(self._real_balance, 2) if self._real_balance is not None else None,
                "pnl":            round(self._ganancia, 2),
                "open_pnl":       round(open_pnl_total, 2),
                "total_pnl":      round(self._ganancia + open_pnl_total, 2),
                "wins":           self._wins,
                "losses":         self._losses,
                "operations":     self._operations,
                "win_rate":     round(self._wins / (self._wins + self._losses) * 100) if (self._wins + self._losses) > 0 else 0,
                "symbols":      syms_data,
                "recommendations": recommendations,
                "configured_symbols": self.cfg.get("symbols", []),
                "watch_symbols": self.cfg.get("watch_symbols", self.cfg.get("symbols", [])),
                "trade_symbols": self.cfg.get("trade_symbols", self.cfg.get("symbols", [])),
                "symbol_leverage": self.cfg.get("symbol_leverage", {}),
                "symbol_capital": self.cfg.get("symbol_capital", {}),
                "execution_mode": self._execution_mode(),
                "connection": {
                    "public_api":  self._public_connected,
                    "private_api": self._private_connected,
                    "mode":        self._execution_mode(),
                    "last_account_sync": self._last_account_sync,
                    "error":       self._connection_error,
                },
                "last_update": datetime.now(timezone.utc).isoformat(),
                "alerts":     list(self._alerts),
            }
            self._last_status = status
            return status
        finally:
            self._lock.release()

    def _build_recommendations(self, syms_data: dict[str, dict], trade_syms: set[str]) -> list[dict]:
        recs = []
        score_min = float(self.cfg.get("score_minimo", 6.5))
        for sym, data in syms_data.items():
            pos = data.get("position") or {}
            mom_pos = data.get("mom_position") or {}
            if pos.get("open") or mom_pos.get("open"):
                continue

            long_score = float(data.get("score_long") or 0.0)
            short_score = float(data.get("score_short") or 0.0)
            side = "LONG" if long_score >= short_score else "SHORT"
            mnv_score = max(long_score, short_score)
            mom_l = float(mom_pos.get("score_l") or 0.0)
            mom_s = float(mom_pos.get("score_s") or 0.0)
            mom_side = "LONG" if mom_l >= mom_s else "SHORT"
            mom_score = max(mom_l, mom_s)

            source = "MNV"
            score = mnv_score
            side_used = side
            if mom_score >= 5 and (mom_score / 6) > (mnv_score / 8):
                source = "MOM"
                score = mom_score
                side_used = mom_side

            trend = data.get("tendencia", "NEUTRAL")
            structure = data.get("estructura", "NEUTRAL")
            rsi_val = float(data.get("rsi") or 0.0)
            vol_ratio = float(data.get("vol_ratio") or 0.0)

            priority = score
            if trend == "ALCISTA" and side_used == "LONG":
                priority += 1.0
            elif trend == "BAJISTA" and side_used == "SHORT":
                priority += 1.0
            elif trend == "NEUTRAL":
                priority -= 0.75

            if structure == "HH/HL" and side_used == "LONG":
                priority += 0.5
            elif structure == "LH/LL" and side_used == "SHORT":
                priority += 0.5

            if source == "MOM":
                priority += 0.4
            if side_used == "LONG" and rsi_val > 68:
                priority -= 0.8
            if side_used == "SHORT" and rsi_val < 32:
                priority -= 0.8
            if vol_ratio >= 3:
                priority -= 0.5

            if source == "MNV":
                checklist = data.get("checklist") or []
                missing = [item.get("name", "") for item in checklist if not item.get("ok")]
            else:
                checklist = mom_pos.get("checklist") or []
                missing = [item.get("name", "") for item in checklist if not item.get("ok")]

            ready = (
                (source == "MNV" and score >= score_min and trend != "NEUTRAL") or
                (source == "MOM" and score >= 5)
            )
            critical_missing = {
                "Reset bajista",
                "No extendida",
                "RSI",
                "RSI 32–52",
                "RSI 48–68",
                "Rebote + Rechazo",
                "Pullback + Rebote",
                "Espacio a soporte",
                "Espacio a resistencia",
                "Volumen sano",
                "Impulso 3 velas",
                "Caída 3 velas",
                "LH/LL",
                "HH/HL",
            }
            if any(name in critical_missing for name in missing):
                ready = False
            if source == "MNV" and side_used == "LONG" and bool(data.get("signal_long")):
                ready = True
            if source == "MNV" and side_used == "SHORT" and bool(data.get("signal_short")):
                ready = True

            recs.append({
                "symbol": sym,
                "side": side_used,
                "source": source,
                "score": round(score, 2),
                "priority": round(priority, 2),
                "ready": bool(ready),
                "operable": sym in trade_syms,
                "watch_only": sym not in trade_syms,
                "price": data.get("price"),
                "trend": trend,
                "structure": structure,
                "rsi": data.get("rsi"),
                "vol_ratio": data.get("vol_ratio"),
                "reason": self._recommendation_reason(source, side_used, trend, structure, score, missing),
                "missing": missing[:3],
            })

        recs.sort(key=lambda item: (item["ready"], item["priority"]), reverse=True)
        return recs[:6]

    def _recommendation_reason(self, source: str, side: str, trend: str, structure: str, score: float, missing: list[str]) -> str:
        parts = [f"{source} {side}", f"score {score:g}"]
        if trend != "NEUTRAL":
            parts.append(trend.lower())
        if structure in {"HH/HL", "LH/LL"}:
            parts.append(structure)
        if missing:
            parts.append("falta " + ", ".join(missing[:2]))
        else:
            parts.append("setup completo")
        return " · ".join(parts)

    def get_logs(self, n: int = 50) -> list:
        if not self._lock.acquire(timeout=0.2):
            return self._last_logs_cache[:n]
        try:
            return list(self._logs)[:n]
        finally:
            self._lock.release()

    def get_historial(self) -> list:
        if self._lock.acquire(timeout=2):
            try:
                return list(reversed(list(self._historial)))
            finally:
                self._lock.release()
        return list(reversed(list(self._historial)))

    def get_pattern_stats(self) -> dict:
        return self._pattern_memory.get_stats()

    # ── Estrategia Momentum ───────────────────────────────────────────────────

    # Sólo estos pares tienen permitido abrir posiciones MOM
    _MOM_TRADE_SYMS = {"BTC/USDT", "ETH/USDT"}

    def _verificar_momentum(self, df: pd.DataFrame, sym: str):
        """
        Estrategia Momentum conservadora: solo BTC/ETH, ADX>20, RSI obligatorio,
        SL 2×ATR, TP 1.5R — movimientos pequeños pero certeros.
        """
        if len(df) < 20:
            empty = [(f"ind{i}", False, "sin datos") for i in range(6)]
            return False, False, 0.0, 0.0, 0.0, 0.0, empty, empty, 0, 0

        u    = df.iloc[-1]
        p    = float(u["close"])
        e20  = float(u["ema20"])
        e50  = float(u["ema50"])
        e200 = float(u["ema200"])
        rsi  = float(u["rsi"])
        atr  = float(u["atr"])
        adx  = float(_adx(df).iloc[-1])

        e20_prev = float(df.iloc[-6]["ema20"]) if len(df) >= 6 else e20
        p3       = float(df.iloc[-4]["close"])  if len(df) >= 4 else p

        def _c(name, cond_l, cond_s, detail_ok, detail_no):
            nonlocal score_l, score_s
            if cond_l:
                score_l += 1
                conds_l.append((name, True,  detail_ok))
            else:
                conds_l.append((name, False, detail_no))
            if cond_s:
                score_s += 1
                conds_s.append((name, True,  detail_ok))
            else:
                conds_s.append((name, False, detail_no))

        score_l, score_s = 0, 0
        conds_l: list = []
        conds_s: list = []

        # 1. Precio vs EMA50
        conds_l.append(("Precio > EMA50", p > e50,
                         f"${p:,.4f} > ${e50:,.4f}" if p > e50 else f"${p:,.4f} ≤ ${e50:,.4f}",))
        if p > e50: score_l += 1
        conds_s.append(("Precio < EMA50", p < e50,
                         f"${p:,.4f} < ${e50:,.4f}" if p < e50 else f"${p:,.4f} ≥ ${e50:,.4f}",))
        if p < e50: score_s += 1

        # 2. Precio vs EMA200
        conds_l.append(("Precio > EMA200", p > e200,
                         f"${p:,.4f} > ${e200:,.4f}" if p > e200 else f"${p:,.4f} ≤ ${e200:,.4f}",))
        if p > e200: score_l += 1
        conds_s.append(("Precio < EMA200", p < e200,
                         f"${p:,.4f} < ${e200:,.4f}" if p < e200 else f"${p:,.4f} ≥ ${e200:,.4f}",))
        if p < e200: score_s += 1

        # 3. EMA50 vs EMA200 alineadas
        conds_l.append(("EMA50 > EMA200", e50 > e200,
                         "EMAs alcistas alineadas" if e50 > e200 else "EMAs no alineadas para LONG",))
        if e50 > e200: score_l += 1
        conds_s.append(("EMA50 < EMA200", e50 < e200,
                         "EMAs bajistas alineadas" if e50 < e200 else "EMAs no alineadas para SHORT",))
        if e50 < e200: score_s += 1

        # 4. RSI en zona correcta (rango más estricto, condición obligatoria)
        rsi_ok_l = 48.0 <= rsi <= 68.0
        rsi_ok_s = 32.0 <= rsi <= 52.0
        conds_l.append(("RSI 48–68", rsi_ok_l,
                         f"RSI {rsi:.0f} — momentum confirmado" if rsi_ok_l else f"RSI {rsi:.0f} — fuera de zona (48–68)",))
        if rsi_ok_l: score_l += 1
        conds_s.append(("RSI 32–52", rsi_ok_s,
                         f"RSI {rsi:.0f} — momentum confirmado" if rsi_ok_s else f"RSI {rsi:.0f} — fuera de zona (32–52)",))\

        if rsi_ok_s: score_s += 1

        # 5. Impulso de precio: vs 3 velas atrás (umbral reducido en modo lento)
        imp_pct  = float(self.cfg.get("slow_impulso_pct", 0.05)) / 100 if self.cfg.get("slow_trend") else 0.001
        mom_up   = p > p3 * (1 + imp_pct)
        mom_down = p < p3 * (1 - imp_pct)
        pct3 = (p / p3 - 1) * 100
        conds_l.append(("Impulso 3 velas", mom_up,
                         f"+{pct3:.2f}% vs hace 3 velas" if mom_up else f"{pct3:.2f}% — sin impulso alcista",))
        if mom_up: score_l += 1
        conds_s.append(("Caída 3 velas", mom_down,
                         f"{pct3:.2f}% vs hace 3 velas" if mom_down else f"{pct3:.2f}% — sin caída bajista",))
        if mom_down: score_s += 1

        # 6. EMA20 con pendiente favorable
        slope_up   = e20 > e20_prev * 1.0005
        slope_down = e20 < e20_prev * 0.9995
        conds_l.append(("EMA20 subiendo", slope_up,
                         f"EMA20 {e20:.4f} > {e20_prev:.4f}" if slope_up else "EMA20 plana o bajando",))
        if slope_up: score_l += 1
        conds_s.append(("EMA20 bajando", slope_down,
                         f"EMA20 {e20:.4f} < {e20_prev:.4f}" if slope_down else "EMA20 plana o subiendo",))
        if slope_down: score_s += 1

        # 7. Si la caída ya está extendida, solo buscar SHORT después de respiro y rechazo.
        reset_short_ok, reset_short_detail = self._short_extension_reset(
            df, p, atr, rsi_ok_s, mom_down and slope_down
        )
        conds_s.append((
            "Reset bajista",
            reset_short_ok,
            reset_short_detail,
        ))
        if reset_short_ok:
            score_s += 1

        nearest_res = self._nearest_resistance(df, p)
        res_room_pct = (nearest_res - p) / p * 100 if nearest_res else 999.0
        min_room_pct = max(1.2, (atr / p * 100) * 1.2) if p > 0 else 1.2
        room_to_resistance = nearest_res is None or res_room_pct >= min_room_pct
        conds_l.append((
            "Espacio a resistencia",
            room_to_resistance,
            "sin resistencia cercana" if nearest_res is None else (
                f"resistencia ${nearest_res:,.4f} a {res_room_pct:.1f}%"
                if room_to_resistance else
                f"resistencia ${nearest_res:,.4f} demasiado cerca ({res_room_pct:.1f}%)"
            ),
        ))
        if room_to_resistance:
            score_l += 1

        nearest_sup = self._nearest_support(df, p)
        support_room_pct = (p - nearest_sup) / p * 100 if nearest_sup else 999.0
        min_support_room_pct = max(1.2, (atr / p * 100) * 1.2) if p > 0 else 1.2
        room_to_support = nearest_sup is None or support_room_pct >= min_support_room_pct
        conds_s.append((
            "Espacio a soporte",
            room_to_support,
            "sin soporte cercano" if nearest_sup is None else (
                f"soporte ${nearest_sup:,.4f} a {support_room_pct:.1f}%"
                if room_to_support else
                f"soporte ${nearest_sup:,.4f} demasiado cerca ({support_room_pct:.1f}%)"
            ),
        ))
        if room_to_support:
            score_s += 1

        capitulation_long, cap_detail_long = self._capitulation_risk(df, "LONG")
        conds_l.append(("Volumen sano", not capitulation_long, cap_detail_long if not capitulation_long else f"{cap_detail_long} — esperar enfriamiento"))
        if not capitulation_long:
            score_l += 1

        capitulation_short, cap_detail_short = self._capitulation_risk(df, "SHORT")
        conds_s.append(("Volumen sano", not capitulation_short, cap_detail_short if not capitulation_short else f"{cap_detail_short} — esperar rebote"))
        if not capitulation_short:
            score_s += 1

        # Requiere ventaja clara sobre la dirección contraria (≥2 pts)
        # Si ambas suman ≥4 simultáneamente → mercado lateral, no operar
        conflicto = score_l >= 4 and score_s >= 4
        # EMA200 debe estar moviéndose — filtro de mercado lateral
        e200_prev = float(df.iloc[-10]["ema200"]) if len(df) >= 10 else e200
        ema200_trending_up   = e200 > e200_prev * 1.001
        ema200_trending_down = e200 < e200_prev * 0.999
        # ADX mínimo: 20 normal, configurable en modo lento
        adx_min = float(self.cfg.get("slow_adx_min", 15.0)) if self.cfg.get("slow_trend") else 20.0
        adx_ok = adx >= adx_min
        # RSI obligatorio: si falla RSI no hay entrada
        senal_l = score_l >= 6 and rsi_ok_l and mom_up and score_l >= score_s + 2 and not conflicto and ema200_trending_up and adx_ok and room_to_resistance and not capitulation_long
        senal_s = score_s >= 6 and rsi_ok_s and mom_down and score_s >= score_l + 2 and not conflicto and ema200_trending_down and adx_ok and reset_short_ok and room_to_support and not capitulation_short

        # SL más ancho (2×ATR) para no ser golpeado por ruido
        min_dist_l = max(atr * 2.0, p * 0.02)
        sl_l = min(e20 - atr * 0.5, p - min_dist_l)
        riesgo_l = max(p - sl_l, atr * 0.8)
        tp_l = p + riesgo_l * 1.5   # TP 1.5R — más alcanzable

        min_dist_s = max(atr * 2.0, p * 0.02)
        sl_s = max(e20 + atr * 0.5, p + min_dist_s)
        riesgo_s = max(sl_s - p, atr * 0.8)
        tp_s = p - riesgo_s * 1.5   # TP 1.5R — más alcanzable

        return senal_l, senal_s, sl_l, tp_l, sl_s, tp_s, conds_l, conds_s, score_l, score_s

    # ── Análisis de tendencia ─────────────────────────────────────────────────

    def _ema_sep_threshold(self) -> float:
        """Separación mínima entre EMAs. Modo lento usa umbral reducido."""
        if self.cfg.get("slow_trend"):
            return 1.0 + float(self.cfg.get("slow_ema_sep_pct", 0.1)) / 100
        return 1.003

    def _analizar_tendencia(self, df: pd.DataFrame) -> tuple[str, str]:
        u = df.iloc[-1]
        p = float(u["close"])
        e20, e50, e150, e200 = float(u["ema20"]), float(u["ema50"]), float(u["ema150"]), float(u["ema200"])

        sep = self._ema_sep_threshold()
        # EMA200 tendencia establecida: debe llevar 20 velas moviéndose en la misma dirección
        ema200_20 = float(df.iloc[-20]["ema200"]) if len(df) >= 20 else float(df.iloc[0]["ema200"])
        # EMAs deben estar SEPARADAS — umbral normal 0.3%, modo lento configurable
        emas_sep_up   = e50 > e150 * sep   and e150 > e200 * sep
        emas_sep_down = e50 < e150 / sep   and e150 < e200 / sep

        stage2 = (p > e200 and p > e150 and p > e50 and
                  emas_sep_up and
                  float(u["ema200"]) > ema200_20)

        stage4 = (p < e200 and p < e150 and p < e50 and
                  emas_sep_down and
                  float(u["ema200"]) < ema200_20)

        if stage2:
            dist_ema50 = (p - e50) / e50 * 100
            if dist_ema50 < 1.5:
                fase = "Pullback a EMA50"
            elif dist_ema50 < 4.0:
                fase = "Tendencia limpia"
            else:
                fase = "Extendido — esperar retroceso"
            return "ALCISTA", fase

        if stage4:
            dist_ema50 = (e50 - p) / e50 * 100
            if dist_ema50 < 1.5:
                fase = "Rebote a EMA50"
            elif dist_ema50 < 4.0:
                fase = "Tendencia bajista limpia"
            else:
                fase = "Muy extendido — esperar rebote"
            return "BAJISTA", fase

        return ("NEUTRAL", "Construyendo Stage 2") if p > e200 else ("NEUTRAL", "Construyendo Stage 4")

    # ── Estructura HH/HL ──────────────────────────────────────────────────────

    def _estructura_hh_hl(self, df: pd.DataFrame) -> tuple[bool, bool, float, float]:
        highs = _swing_highs(df)
        lows  = _swing_lows(df)

        hh_hl = False
        lh_ll = False
        ultimo_hl = 0.0
        ultimo_lh = float("inf")

        if len(highs) >= 3 and len(lows) >= 3:
            hh_hl = highs[-1] > highs[-3] and lows[-1] > lows[-3]
            lh_ll = highs[-1] < highs[-3] and lows[-1] < lows[-3]
        elif len(highs) >= 2 and len(lows) >= 2:
            hh_hl = highs[-1] > highs[-2] and lows[-1] > lows[-2]
            lh_ll = highs[-1] < highs[-2] and lows[-1] < lows[-2]

        if lows:
            ultimo_hl = lows[-1]
        if highs:
            ultimo_lh = highs[-1]

        return hh_hl, lh_ll, ultimo_hl, ultimo_lh

    # ── Helpers de calidad de entrada ─────────────────────────────────────────

    def _pullback_tocado_ema(self, df: pd.DataFrame, ema_col: str, tolerancia: float = 1.01) -> bool:
        """Verifica que en las últimas 5 velas cerradas el precio tocó el EMA."""
        ventana = df.iloc[-6:-1] if len(df) >= 7 else df.iloc[:-1]
        ema_vals = ventana[ema_col].values
        low_vals = ventana["low"].values
        return bool(any(low_vals[i] <= ema_vals[i] * tolerancia for i in range(len(low_vals))))

    def _rebote_confirmado(self, df: pd.DataFrame, ema_col: str) -> bool:
        """La última vela cerrada cerró POR ENCIMA del EMA (rebote)."""
        u = df.iloc[-1]
        return float(u["close"]) > float(u[ema_col]) and float(u["close"]) > float(u["open"])

    def _resistencia_tocada_ema(self, df: pd.DataFrame, ema_col: str, tolerancia: float = 1.01) -> bool:
        """Verifica que en las últimas 5 velas el precio tocó la resistencia EMA desde abajo."""
        ventana = df.iloc[-6:-1] if len(df) >= 7 else df.iloc[:-1]
        ema_vals = ventana[ema_col].values
        high_vals = ventana["high"].values
        return bool(any(high_vals[i] >= ema_vals[i] / tolerancia for i in range(len(high_vals))))

    def _rechazo_confirmado(self, df: pd.DataFrame, ema_col: str) -> bool:
        """La última vela cerrada cerró POR DEBAJO del EMA (rechazo de resistencia)."""
        u = df.iloc[-1]
        return float(u["close"]) < float(u[ema_col]) and float(u["close"]) < float(u["open"])

    def _nearest_resistance(self, df: pd.DataFrame, price: float) -> Optional[float]:
        candidates = [h for h in _swing_highs(df) if h > price * 1.001]
        return min(candidates) if candidates else None

    def _nearest_support(self, df: pd.DataFrame, price: float) -> Optional[float]:
        candidates = [l for l in _swing_lows(df) if l < price * 0.999]
        return max(candidates) if candidates else None

    def _capitulation_risk(self, df: pd.DataFrame, side: str) -> tuple[bool, str]:
        """Detecta velas de agotamiento por volumen antes de perseguir entrada."""
        if len(df) < 5:
            return False, "sin datos suficientes"
        u = df.iloc[-1]
        prev = df.iloc[-2]
        close = float(u["close"])
        open_ = float(u["open"])
        prev_close = float(prev["close"])
        vol_ratio = float(u.get("vol_ratio", 0.0) or 0.0)
        move_pct = (close - prev_close) / prev_close * 100 if prev_close > 0 else 0.0
        if side == "SHORT":
            risky = close < open_ and move_pct < -1.2 and vol_ratio >= 1.6
            return risky, f"venta fuerte {move_pct:.1f}% con volumen {vol_ratio:.2f}x"
        risky = close > open_ and move_pct > 1.2 and vol_ratio >= 1.6
        return risky, f"compra fuerte {move_pct:.1f}% con volumen {vol_ratio:.2f}x"

    def _short_extension_reset(self, df: pd.DataFrame, price: float, atr: float, rsi_ok: bool, rejection_ok: bool) -> tuple[bool, str]:
        lookback_7d = min(168, max(24, len(df) - 1))
        recent_high_7d = float(df.iloc[-lookback_7d:]["high"].max()) if lookback_7d > 0 else price
        drop_7d = (recent_high_7d - price) / recent_high_7d * 100 if recent_high_7d > 0 else 0.0
        max_drop_7d = float(self.cfg.get("max_7d_drop_short_pct", 4.0))

        lookback_3d = min(72, max(12, len(df) - 1))
        recent_high_3d = float(df.iloc[-lookback_3d:]["high"].max()) if lookback_3d > 0 else price
        recent_low_3d = float(df.iloc[-lookback_3d:]["low"].min()) if lookback_3d > 0 else price
        drop_3d = (recent_high_3d - price) / recent_high_3d * 100 if recent_high_3d > 0 else 0.0
        max_drop_3d = float(self.cfg.get("max_3d_drop_short_pct", 2.5))

        lookback_24h = min(24, max(6, len(df) - 1))
        recent_high_24h = float(df.iloc[-lookback_24h:]["high"].max()) if lookback_24h > 0 else price
        drop_24h = (recent_high_24h - price) / recent_high_24h * 100 if recent_high_24h > 0 else 0.0
        max_drop_24h = float(self.cfg.get("max_24h_drop_short_pct", 2.0))

        extended = drop_7d > max_drop_7d or drop_3d > max_drop_3d or drop_24h > max_drop_24h
        rebound_pct = (price - recent_low_3d) / recent_low_3d * 100 if recent_low_3d > 0 else 0.0
        min_rebound_pct = max(1.2, (atr / price * 100) * 1.2) if price > 0 else 1.2
        reset_ok = rebound_pct >= min_rebound_pct and rsi_ok and rejection_ok

        if not extended:
            return True, f"Caída 7d {drop_7d:.1f}% / 3d {drop_3d:.1f}% / 24h {drop_24h:.1f}% — entrada no tardía"
        if reset_ok:
            return True, f"Caída extendida, pero hubo rebote {rebound_pct:.1f}% y rechazo — SHORT permitido"
        return False, f"Caída 7d {drop_7d:.1f}% / 3d {drop_3d:.1f}% / 24h {drop_24h:.1f}% — esperar rebote/rechazo antes de otro SHORT"

    # ── Señal LONG ────────────────────────────────────────────────────────────

    def _verificar_long(self, df: pd.DataFrame, sym: str) -> tuple[bool, list, float, float, float]:
        c      = self.cfg
        u      = df.iloc[-1]
        prev   = df.iloc[-2]
        p      = float(u["close"])
        atr    = float(u["atr"])
        vr     = float(prev["vol_ratio"])
        rsi    = float(u["rsi"])
        e20    = float(u["ema20"])
        e50    = float(u["ema50"])
        e200   = float(u["ema200"])
        rr_min = float(c.get("rr_minimo", 2.0))
        score  = 0.0
        conds  = []

        # 1. Stage 2 — filtro duro (EMAs separadas, EMA200 subiendo 20 velas)
        e150    = float(u["ema150"])
        ema200_20 = float(df.iloc[-20]["ema200"]) if len(df) >= 20 else float(df.iloc[0]["ema200"])
        sep = self._ema_sep_threshold()
        emas_sep_up = float(u["ema50"]) > e150 * sep and e150 > e200 * sep
        stage2 = (p > e200 and emas_sep_up and float(u["ema200"]) > ema200_20)
        if stage2:
            score += 2.0
            conds.append(("Stage 2", True, f"Tendencia alcista establecida — ${p:,.2f} > EMA200 ${e200:,.2f}"))
        else:
            conds.append(("Stage 2", False, f"Sin Stage 2 — EMAs no alineadas/separadas o EMA200 no sube 20v"))

        # 2. Estructura HH/HL — filtro duro
        hh_hl, _, ultimo_hl, _ = self._estructura_hh_hl(df)
        if hh_hl:
            score += 1.5
            conds.append(("HH/HL", True, f"Higher Highs + Higher Lows — HL ${ultimo_hl:,.2f}"))
        else:
            conds.append(("HH/HL", False, "Sin estructura HH/HL confirmada"))

        # 3. RSI: zona óptima 40-65 (hay momentum sin sobrecompra) — filtro duro
        rsi_ok = 40.0 <= rsi <= 65.0
        if rsi_ok:
            score += 1.0
            conds.append(("RSI", True, f"RSI {rsi:.0f} — momentum sin sobrecompra"))
        else:
            motivo = "sobrecomprado (>65)" if rsi > 65 else "sin momentum (<40)"
            conds.append(("RSI", False, f"RSI {rsi:.0f} — {motivo}"))

        # 4. Pullback real: precio tocó EMA50 en últimas 5 velas Y última vela cierra arriba (rebote)
        tocado_ema50 = self._pullback_tocado_ema(df, "ema50", tolerancia=1.008)
        rebote_ema50 = self._rebote_confirmado(df, "ema50")
        tocado_ema20 = self._pullback_tocado_ema(df, "ema20", tolerancia=1.005)
        rebote_ema20 = self._rebote_confirmado(df, "ema20")

        pullback_real = (tocado_ema50 and rebote_ema50) or (tocado_ema20 and rebote_ema20)

        dist_e50 = abs(p - e50) / e50 * 100
        dist_hl  = abs(p - ultimo_hl) / ultimo_hl * 100 if ultimo_hl > 0 else 999

        if pullback_real:
            score += 1.5
            zona = f"EMA50 ${e50:,.2f}" if tocado_ema50 else f"EMA20 ${e20:,.2f}"
            conds.append(("Pullback + Rebote", True, f"Toque y rebote confirmado — {zona} (dist {dist_e50:.1f}%)"))
        else:
            conds.append(("Pullback + Rebote", False,
                          f"Sin toque de EMA50 ({dist_e50:.1f}%) o sin vela rebote — esperar retroceso"))

        # 5. Evitar entradas tardías después de una subida prolongada.
        lookback = min(168, max(24, len(df) - 1))
        recent_low_7d = float(df.iloc[-lookback:]["low"].min()) if lookback > 0 else p
        rise_7d = (p - recent_low_7d) / recent_low_7d * 100 if recent_low_7d > 0 else 0.0
        max_rise_7d = float(c.get("max_7d_rise_long_pct", 6.0))
        mature_ok = rise_7d <= max_rise_7d
        if mature_ok:
            score += 1.0
            conds.append(("No extendida", True, f"Subida 7d {rise_7d:.1f}% — entrada no tardía"))
        else:
            conds.append(("No extendida", False, f"Subida 7d {rise_7d:.1f}% > {max_rise_7d:.1f}% — esperar corrección"))

        # 6. Volumen bajo en el pullback (retroceso tranquilo)
        vol_bajo = vr < float(c.get("vol_pullback_max", 0.85))
        if vol_bajo:
            score += 1.0
            conds.append(("Vol bajo en retroceso", True, f"Vol {vr:.2f}x — retroceso tranquilo"))
        else:
            conds.append(("Vol bajo en retroceso", False, f"Vol {vr:.2f}x — demasiada presión vendedora"))

        # 7. Espacio real hasta resistencia: no comprar debajo de techo cercano.
        nearest_res = self._nearest_resistance(df, p)
        res_room_pct = (nearest_res - p) / p * 100 if nearest_res else 999.0
        min_room_pct = max(1.2, (atr / p * 100) * 1.2) if p > 0 else 1.2
        room_to_resistance = nearest_res is None or res_room_pct >= min_room_pct
        if room_to_resistance:
            score += 0.8
            detail = "sin resistencia cercana" if nearest_res is None else f"resistencia ${nearest_res:,.2f} a {res_room_pct:.1f}%"
            conds.append(("Espacio a resistencia", True, detail))
        else:
            conds.append(("Espacio a resistencia", False, f"resistencia ${nearest_res:,.2f} demasiado cerca ({res_room_pct:.1f}%)"))

        # 8. Evitar velas de euforia con volumen alto.
        capitulation_long, cap_detail_long = self._capitulation_risk(df, "LONG")
        if not capitulation_long:
            score += 0.5
            conds.append(("Volumen sano", True, cap_detail_long))
        else:
            conds.append(("Volumen sano", False, f"{cap_detail_long} — esperar enfriamiento"))

        # 9. SL: debajo del mínimo reciente Y debajo del EMA50, tomar el MÁS BAJO (más margen)
        n_sl = min(5, len(df) - 2)
        recent_low = float(df.iloc[-(n_sl + 1):-1]["low"].min())
        sl_structural = recent_low - atr * 0.5
        sl_ema        = e50 - atr * float(c.get("atr_sl_mult", 1.5))
        sl = min(sl_structural, sl_ema)   # más bajo = más margen para el trade

        # Mínimo: al menos 1% del precio o 1 ATR (lo que sea mayor), para evitar SL pegado
        min_sl_dist = max(atr * 1.0, p * 0.010)
        sl = min(sl, p - min_sl_dist)
        # Máximo: SL no puede estar más de 3% del precio (evita pérdidas enormes en coins volátiles)
        sl = max(sl, p * 0.97)

        # 10. TP: primera resistencia real, máximo 6R (evita TPs irreales por swing lejano)
        highs = _swing_highs(df)
        highs_arriba = [h for h in highs if h > p * 1.005]
        riesgo = p - sl
        rr_base = p + riesgo * rr_min  # mínimo TP para cumplir R:R
        tp = min(highs_arriba) if highs_arriba and min(highs_arriba) > rr_base else rr_base
        tp = min(tp, p + riesgo * 6.0)  # nunca más de 6R (evita TPs irreales)

        reward = tp - p
        rr = reward / riesgo if riesgo > 0 else 0
        rr_ok = rr >= rr_min

        if rr_ok:
            score += 1.5
            conds.append(("R:R", True, f"R:R {rr:.1f} — SL ${sl:,.2f} / TP ${tp:,.2f}"))
        else:
            conds.append(("R:R", False, f"R:R {rr:.1f} < {rr_min} — SL ${sl:,.2f} / TP ${tp:,.2f}"))

        senal = (
            stage2
            and hh_hl
            and rsi_ok
            and pullback_real
            and mature_ok
            and vol_bajo
            and room_to_resistance
            and not capitulation_long
            and rr_ok
            and score >= float(c.get("score_minimo", 6.5))
        )
        return senal, conds, score, sl, tp

    # ── Señal SHORT ───────────────────────────────────────────────────────────

    def _verificar_short(self, df: pd.DataFrame, sym: str) -> tuple[bool, list, float, float, float]:
        c      = self.cfg
        u      = df.iloc[-1]
        prev   = df.iloc[-2]
        p      = float(u["close"])
        atr    = float(u["atr"])
        vr     = float(prev["vol_ratio"])
        rsi    = float(u["rsi"])
        e20    = float(u["ema20"])
        e50    = float(u["ema50"])
        e200   = float(u["ema200"])
        rr_min = float(c.get("rr_minimo", 2.0))
        score  = 0.0
        conds  = []

        # 1. Stage 4 — filtro duro (EMAs separadas, EMA200 bajando 20 velas)
        e150    = float(u["ema150"])
        ema200_20 = float(df.iloc[-20]["ema200"]) if len(df) >= 20 else float(df.iloc[0]["ema200"])
        sep = self._ema_sep_threshold()
        emas_sep_down = float(u["ema50"]) < e150 / sep and e150 < e200 / sep
        stage4 = (p < e200 and emas_sep_down and float(u["ema200"]) < ema200_20)
        if stage4:
            score += 2.0
            conds.append(("Stage 4", True, f"Tendencia bajista establecida — ${p:,.2f} < EMA200 ${e200:,.2f}"))
        else:
            conds.append(("Stage 4", False, f"Sin Stage 4 — EMAs no alineadas/separadas o EMA200 no baja 20v"))

        # 2. Estructura LH/LL — filtro duro
        _, lh_ll, _, ultimo_lh = self._estructura_hh_hl(df)
        if lh_ll:
            score += 1.5
            conds.append(("LH/LL", True, f"Lower Highs + Lower Lows — LH ${ultimo_lh:,.2f}"))
        else:
            conds.append(("LH/LL", False, "Sin estructura LH/LL confirmada"))

        # 3. RSI: 45-65 para SHORT (el rebote debe tener momentum real, no rebote débil desde sobreventa)
        rsi_ok = 45.0 <= rsi <= 65.0
        if rsi_ok:
            score += 1.0
            conds.append(("RSI", True, f"RSI {rsi:.0f} — momentum bajista sin sobreventa"))
        else:
            motivo = "sin rebote suficiente (<45)" if rsi < 45 else "demasiado alto (>65)"
            conds.append(("RSI", False, f"RSI {rsi:.0f} — {motivo}"))

        # 4. Rebote real a resistencia: precio tocó EMA50/20 desde abajo Y última vela rechazada
        tocado_ema50 = self._resistencia_tocada_ema(df, "ema50", tolerancia=1.008)
        rechazo_ema50 = self._rechazo_confirmado(df, "ema50")
        tocado_ema20 = self._resistencia_tocada_ema(df, "ema20", tolerancia=1.005)
        rechazo_ema20 = self._rechazo_confirmado(df, "ema20")

        rebote_resistencia = (tocado_ema50 and rechazo_ema50) or (tocado_ema20 and rechazo_ema20)

        dist_e50 = abs(p - e50) / e50 * 100

        if rebote_resistencia:
            score += 1.5
            zona = f"EMA50 ${e50:,.2f}" if tocado_ema50 else f"EMA20 ${e20:,.2f}"
            conds.append(("Rebote + Rechazo", True, f"Toque y rechazo confirmado — {zona} (dist {dist_e50:.1f}%)"))
        else:
            conds.append(("Rebote + Rechazo", False,
                          f"Sin toque de EMA50 ({dist_e50:.1f}%) o sin vela rechazo — esperar rebote"))

        # 5. Evitar perseguir caídas fuertes: si ya cayó demasiado, exigir rebote y rechazo.
        reset_short_ok, reset_short_detail = self._short_extension_reset(
            df, p, atr, rsi_ok, rebote_resistencia
        )
        if reset_short_ok:
            score += 1.0
            conds.append(("Reset bajista", True, reset_short_detail))
        else:
            conds.append(("Reset bajista", False, reset_short_detail))

        # 6. Volumen bajo en el rebote
        vol_bajo = vr < float(c.get("vol_pullback_max", 0.85))
        if vol_bajo:
            score += 1.0
            conds.append(("Vol bajo en rebote", True, f"Vol {vr:.2f}x — rebote sin compradores fuertes"))
        else:
            conds.append(("Vol bajo en rebote", False, f"Vol {vr:.2f}x — demasiada presión compradora"))

        # 7. Espacio real hasta soporte: no vender encima de soporte cercano.
        nearest_sup = self._nearest_support(df, p)
        support_room_pct = (p - nearest_sup) / p * 100 if nearest_sup else 999.0
        min_support_room_pct = max(1.2, (atr / p * 100) * 1.2) if p > 0 else 1.2
        room_to_support = nearest_sup is None or support_room_pct >= min_support_room_pct
        if room_to_support:
            score += 0.8
            detail = "sin soporte cercano" if nearest_sup is None else f"soporte ${nearest_sup:,.2f} a {support_room_pct:.1f}%"
            conds.append(("Espacio a soporte", True, detail))
        else:
            conds.append(("Espacio a soporte", False, f"soporte ${nearest_sup:,.2f} demasiado cerca ({support_room_pct:.1f}%)"))

        # 8. Evitar vender en capitulación: alto volumen tras vela roja grande suele rebotar.
        capitulation_short, cap_detail_short = self._capitulation_risk(df, "SHORT")
        if not capitulation_short:
            score += 0.5
            conds.append(("Volumen sano", True, cap_detail_short))
        else:
            conds.append(("Volumen sano", False, f"{cap_detail_short} — esperar rebote"))

        # 9. SL: encima del máximo reciente Y encima del EMA50, tomar el MÁS ALTO (más margen)
        n_sl = min(5, len(df) - 2)
        recent_high = float(df.iloc[-(n_sl + 1):-1]["high"].max())
        sl_structural = recent_high + atr * 0.5
        sl_ema        = e50 + atr * float(c.get("atr_sl_mult", 1.5))
        sl = max(sl_structural, sl_ema)   # más alto = más margen para el trade

        # Mínimo: al menos 1% del precio o 1 ATR (lo que sea mayor)
        min_sl_dist = max(atr * 1.0, p * 0.010)
        sl = max(sl, p + min_sl_dist)
        # Máximo: SL no puede estar más de 3% del precio
        sl = min(sl, p * 1.03)

        # 10. TP: primer soporte real, máximo 6R (evita TPs irreales por swing lejano)
        lows = _swing_lows(df)
        lows_abajo = [l for l in lows if l < p * 0.995]
        riesgo = sl - p
        rr_base = p - riesgo * rr_min  # mínimo TP para cumplir R:R
        tp = max(lows_abajo) if lows_abajo and max(lows_abajo) < rr_base else rr_base
        tp = max(tp, p - riesgo * 6.0)  # nunca más de 6R

        reward = p - tp
        rr = reward / riesgo if riesgo > 0 else 0
        rr_ok = rr >= rr_min

        if rr_ok:
            score += 1.5
            conds.append(("R:R", True, f"R:R {rr:.1f} — SL ${sl:,.2f} / TP ${tp:,.2f}"))
        else:
            conds.append(("R:R", False, f"R:R {rr:.1f} < {rr_min} — SL ${sl:,.2f} / TP ${tp:,.2f}"))

        senal = (
            stage4
            and lh_ll
            and rsi_ok
            and rebote_resistencia
            and reset_short_ok
            and vol_bajo
            and room_to_support
            and not capitulation_short
            and rr_ok
            and score >= float(c.get("score_minimo", 6.5))
        )
        return senal, conds, score, sl, tp

    # ── Gestión de salida ─────────────────────────────────────────────────────

    def _verificar_salida(self, df: pd.DataFrame, est: dict) -> tuple[bool, bool, float, float, list]:
        u       = df.iloc[-1]
        p       = float(u["close"])
        atr     = float(u["atr"])
        entrada = est["precio_entrada"]
        sl      = est["sl_actual"]
        tp      = est["tp"]
        be      = est["breakeven_activado"]
        dir_    = est["direccion_pos"]
        eventos = []
        cerrar  = False
        parcial = False
        exit_price = p   # precio de salida real (SL o TP, no precio live arbitrario)
        c       = self.cfg

        rr_be      = float(c.get("breakeven_rr", 1.0))
        rr_parcial = float(c.get("parcial_rr", 1.5))
        trail_mult = float(c.get("atr_trail_mult", 2.0))

        if dir_ == "LONG":
            riesgo = entrada - sl
            profit = p - entrada

            if not be and riesgo > 0 and profit >= riesgo * rr_be:
                sl = max(sl, entrada + atr * 0.2)
                be = True
                eventos.append(f"Break-even activado → SL ${sl:,.4f}")

            if be:
                pmax = est.get("precio_ext", p)
                sl_trail = pmax - atr * trail_mult
                if sl_trail > sl:
                    sl = sl_trail
                    eventos.append(f"Trailing SL → ${sl:,.4f}")

            if p <= sl:
                cerrar = True
                exit_price = sl  # salir exactamente en el SL (no peor)
                eventos.append(f"SL tocado ({(sl - entrada) / entrada * 100:+.2f}%)")
            elif tp and p >= tp:
                cerrar = True
                exit_price = tp
                eventos.append(f"TP alcanzado ({(tp - entrada) / entrada * 100:+.2f}%) ✅")
            elif not est["salida_parcial_hecha"] and riesgo > 0 and profit >= riesgo * rr_parcial:
                parcial = True
                eventos.append(f"Salida parcial 50% ${p:,.4f} (+{profit / entrada * 100:.2f}%)")

        else:  # SHORT
            riesgo = sl - entrada
            profit = entrada - p

            if not be and riesgo > 0 and profit >= riesgo * rr_be:
                sl = min(sl, entrada - atr * 0.2)
                be = True
                eventos.append(f"Break-even activado → SL ${sl:,.4f}")

            if be:
                pmin = est.get("precio_ext", p)
                sl_trail = pmin + atr * trail_mult
                if sl_trail < sl:
                    sl = sl_trail
                    eventos.append(f"Trailing SL → ${sl:,.4f}")

            if p >= sl:
                cerrar = True
                exit_price = sl
                eventos.append(f"SL tocado ({(entrada - sl) / entrada * 100:+.2f}%)")
            elif tp and p <= tp:
                cerrar = True
                exit_price = tp
                eventos.append(f"TP alcanzado ({(entrada - tp) / entrada * 100:+.2f}%) ✅")
            elif not est["salida_parcial_hecha"] and riesgo > 0 and profit >= riesgo * rr_parcial:
                parcial = True
                eventos.append(f"Salida parcial 50% ${p:,.4f} (+{profit / entrada * 100:.2f}%)")

        est["breakeven_activado"] = be
        return cerrar, parcial, sl, exit_price, eventos

    # ── Estado vacío ──────────────────────────────────────────────────────────

    def _estado_vacio(self) -> dict:
        return {
            "posicion_abierta":    False,
            "direccion_pos":       None,
            "precio_entrada":      None,
            "sl_inicial":          None,
            "sl_actual":           None,
            "tp":                  None,
            "precio_ext":          None,
            "exchange_qty":        0.0,
            "sl_order_id":         None,
            "breakeven_activado":  False,
            "salida_parcial_hecha":False,
            "capital":             self.cfg.get("capital_usd", 25.0),
            "apalancamiento":      self.cfg.get("apalancamiento", 3),
            "tendencia":           "NEUTRAL",
            "fase":                "—",
            "signal_long":         False,
            "signal_short":        False,
            "score_long":          0.0,
            "score_short":         0.0,
            "checklist":           [],
            "pnl_pct":             0.0,
            "cooldown_restante":   0,
            "ultimo_cierre":       None,
            "ultimo_resultado":    None,
            "signal_confirmado":         0,
            "signal_short_confirmado":   0,
        }

    def _current_open_positions(self) -> int:
        mnv = sum(1 for est in self._estados.values() if est.get("posicion_abierta"))
        mom = sum(1 for est in self._estados_mom.values() if est.get("posicion_abierta"))
        return mnv + mom

    def _can_open_new_position(self, sym: str, tendencia: str) -> tuple[bool, str]:
        est = self._estados.get(sym, {})
        cooldown = int(est.get("cooldown_restante", 0) or 0)
        if cooldown > 0:
            return False, f"Cooldown {cooldown} ciclo(s)"
        # Una posición por moneda — si ya tiene una abierta (MNV o MOM) no abre otra
        if est.get("posicion_abierta") or self._estados_mom.get(sym, {}).get("posicion_abierta"):
            return False, f"Ya hay una posición abierta en {sym}"
        if tendencia == "NEUTRAL":
            return False, "Sin contexto tendencial"
        return True, "OK"

    # ── Loop principal ────────────────────────────────────────────────────────

    def _run(self):
        self._log("Conectando a OKX…")
        try:
            self._conectar()
            mode_str = self._execution_mode()
            if self._is_live_mode():
                self._log(f"Conectado ✓ — API lista ({mode_str})", "success")
            else:
                self._log("Conectado ✓ — mercados listos (modo simulado)", "success")
        except Exception as e:
            self._connection_error = str(e)
            self._public_connected = False
            self._private_connected = False
            self._log(f"Error de conexión: {e}", "error")
            self._running = False
            self._release_run_lock()
            return

        symbols = self.cfg.get("symbols", ["BTC/USDT"])
        with self._lock:
            for sym in symbols:
                if sym not in self._estados:
                    self._estados[sym] = self._estado_vacio()

        if self._is_live_mode():
            self._log("Reconciliando posiciones con OKX…")
            self._reconcile_positions()

        self._log(f"Monitoreando {len(symbols)} par(es)…")

        while self._running:
            try:
                cfg = self.cfg
                if self._is_live_mode() and self._private_connected:
                    try:
                        self._sync_account()
                    except Exception as e:
                        self._connection_error = str(e)
                        self._log(f"Error sincronizando cuenta: {e}", "error")

                for sym in cfg.get("symbols", []):
                    if not self._running:
                        break
                    try:
                        df_1h = self._fetch_df(sym, "1h", 300)

                        with self._lock:
                            self._df_cache[sym] = df_1h
                            est = self._estados.setdefault(sym, self._estado_vacio())
                            if int(est.get("cooldown_restante", 0) or 0) > 0 and not est.get("posicion_abierta"):
                                est["cooldown_restante"] = max(int(est.get("cooldown_restante", 0)) - 1, 0)

                        # Usar solo velas cerradas para señales (excluir la vela live actual)
                        df_signal = df_1h.iloc[:-1].copy() if len(df_1h) > 20 else df_1h
                        if len(df_signal) < 50:
                            continue

                        try:
                            live_ticker_price = self._fetch_ticker_price(sym)
                        except Exception as price_error:
                            live_ticker_price = float(df_1h.iloc[-1]["close"])
                            self._log(f"[{sym}] Precio ticker OKX no disponible, usando vela: {price_error}", "warning")
                        with self._lock:
                            self._live_prices[sym] = live_ticker_price

                        # ── Memoria de patrones ───────────────────────────
                        try:
                            last_candle_ts = float(df_signal.iloc[-1]["ts"].timestamp())
                            current_price_pm = live_ticker_price
                            current_ts_pm = float(df_1h.iloc[-1]["ts"].timestamp())
                            if sym == "BTC/USDT":
                                detected = detect_patterns(df_signal)
                                for pat_name, pat_dir in detected:
                                    self._pattern_memory.record(sym, pat_name, pat_dir, current_price_pm, last_candle_ts)
                                self._pattern_memory.update(sym, current_price_pm, current_ts_pm)
                        except Exception:
                            pass

                        tendencia, fase = self._analizar_tendencia(df_signal)
                        senal_l, conds_l, score_l, sl_l, tp_l = self._verificar_long(df_signal, sym)
                        senal_s, conds_s, score_s, sl_s, tp_s = self._verificar_short(df_signal, sym)

                        if tendencia == "ALCISTA":
                            checklist = [{"name": n, "ok": ok, "detail": d} for n, ok, d in conds_l]
                        elif tendencia == "BAJISTA":
                            checklist = [{"name": n, "ok": ok, "detail": d} for n, ok, d in conds_s]
                        else:
                            checklist = [{"name": n, "ok": ok, "detail": d} for n, ok, d in conds_l]

                        # Precio live de OKX para entry, P&L y salidas.
                        precio = live_ticker_price
                        u_live = df_1h.iloc[-1]
                        rsi_live = float(u_live["rsi"])
                        df_live = df_1h.copy()
                        df_live.loc[df_live.index[-1], "close"] = precio

                        # Chequeo spread: si el precio live se alejó >1.5% del cierre de la última vela
                        # cerrada, el setup ya no es válido para entrar
                        signal_close = float(df_signal.iloc[-1]["close"])
                        spread_pct = abs(precio - signal_close) / signal_close * 100

                        pending_logs: list[tuple[str, str]] = []

                        with self._lock:
                            est["tendencia"]    = tendencia
                            est["fase"]         = fase
                            est["signal_long"]  = senal_l
                            est["signal_short"] = senal_s
                            est["score_long"]   = score_l
                            est["score_short"]  = score_s
                            est["checklist"]    = checklist

                            # ── Gestión de posición abierta ───────────────
                            if est["posicion_abierta"]:
                                dir_ = est["direccion_pos"]
                                if dir_ == "LONG" and precio > (est["precio_ext"] or 0):
                                    est["precio_ext"] = precio
                                elif dir_ == "SHORT" and precio < (est["precio_ext"] or float("inf")):
                                    est["precio_ext"] = precio

                                cerrar, parcial, nuevo_sl, exit_price, eventos = self._verificar_salida(df_live, est)

                                sl_movio = False
                                if dir_ == "LONG" and nuevo_sl > est["sl_actual"]:
                                    est["sl_actual"] = nuevo_sl
                                    sl_movio = True
                                elif dir_ == "SHORT" and nuevo_sl < est["sl_actual"]:
                                    est["sl_actual"] = nuevo_sl
                                    sl_movio = True

                                if sl_movio and self._is_live_mode() and not cerrar:
                                    try:
                                        new_id = self._update_sl_order(sym, dir_, est.get("sl_order_id"), nuevo_sl)
                                        est["sl_order_id"] = new_id
                                    except Exception as _sle:
                                        pending_logs.append((f"[{sym}] Error actualizando SL exchange: {_sle}", "warning"))

                                if dir_ == "LONG":
                                    est["pnl_pct"] = round((precio - est["precio_entrada"]) / est["precio_entrada"] * 100, 2)
                                else:
                                    est["pnl_pct"] = round((est["precio_entrada"] - precio) / est["precio_entrada"] * 100, 2)

                                for ev in eventos:
                                    pending_logs.append((f"[{sym}] {ev}", "info"))

                                if parcial and not est["salida_parcial_hecha"]:
                                    qty_total = float(est.get("exchange_qty") or 0.0)
                                    if qty_total <= 0:
                                        cap = float(est.get("capital") or 0.0)
                                        apal = int(est.get("apalancamiento") or 1)
                                        ent = float(est.get("precio_entrada") or precio)
                                        qty_total = (cap * apal) / ent if ent > 0 else 0.0
                                    qty_close = qty_total / 2

                                    if qty_close > 0 and self._is_live_mode():
                                        try:
                                            self._close_live_order(sym, dir_, qty_close)
                                        except Exception as pe:
                                            pending_logs.append((f"[{sym}] Error parcial real: {pe}", "error"))
                                            qty_close = 0.0

                                    if qty_close > 0:
                                        ent = float(est["precio_entrada"])
                                        gan_p = ((precio - ent) * qty_close) if dir_ == "LONG" else ((ent - precio) * qty_close)
                                        self._balance  += gan_p
                                        self._ganancia += gan_p
                                        est["exchange_qty"] = max(qty_total - qty_close, 0.0)
                                        est["capital"] = round(float(est.get("capital", 0.0)) / 2, 2)
                                        est["salida_parcial_hecha"] = True
                                        pending_logs.append((f"[{sym}] Parcial +${gan_p:.2f} | Restante {est['exchange_qty']:.6f}", "success"))
                                        self._push_alert(f"Salida parcial {sym} +${gan_p:.2f}", "success", sym)

                                if cerrar:
                                    cap  = est["capital"]
                                    apal = est["apalancamiento"]
                                    ent  = est["precio_entrada"]
                                    cant = float(est.get("exchange_qty") or ((cap * apal) / ent))

                                    if self._is_live_mode():
                                        try:
                                            self._close_live_order(sym, dir_, cant)
                                            self._cancel_sl_order(sym, est.get("sl_order_id"))
                                            est["sl_order_id"] = None
                                        except Exception as le:
                                            pending_logs.append((f"[{sym}] Error cerrando real: {le}", "error"))
                                            cerrar = False

                                    if cerrar:
                                        # Usar exit_price (SL o TP exacto), no precio live arbitrario
                                        gan = ((exit_price - ent) * cant) if dir_ == "LONG" else ((ent - exit_price) * cant)
                                        self._balance  += gan
                                        self._ganancia += gan
                                        self._operations += 1
                                        if gan >= 0:
                                            self._wins += 1
                                        else:
                                            self._losses += 1

                                        # ── Registrar en historial ────────────
                                        _ev0 = eventos[0] if eventos else ""
                                        _motivo = ("TP" if "TP" in _ev0 else
                                                   "Trailing" if "Trailing" in _ev0 else
                                                   "SL" if "SL" in _ev0 else "Manual")
                                        _sl_ini   = float(est.get("sl_inicial", est.get("sl_actual", 0)))
                                        _riesgo   = abs(ent - _sl_ini) if _sl_ini else 0
                                        _rr_real  = round(((exit_price - ent) / _riesgo if dir_ == "LONG" else (ent - exit_price) / _riesgo), 2) if _riesgo > 0 else 0.0
                                        _cap      = float(est.get("capital", 0))
                                        _pct_cap  = round(gan / _cap * 100, 2) if _cap > 0 else 0.0
                                        _ts_open  = est.get("ts_apertura", "")
                                        _ts_close = datetime.now(timezone.utc).isoformat()
                                        _duracion = 0
                                        if _ts_open:
                                            try:
                                                _d = datetime.fromisoformat(_ts_close.replace("Z","")) - datetime.fromisoformat(_ts_open.replace("Z",""))
                                                _duracion = max(0, int(_d.total_seconds() / 60))
                                            except Exception:
                                                pass
                                        self._historial_counter += 1
                                        self._historial.append({
                                            "id":       self._historial_counter,
                                            "sym":      sym,
                                            "dir":      dir_,
                                            "entrada":  round(ent, 6),
                                            "sl_ini":   round(_sl_ini, 6),
                                            "tp":       round(float(est.get("tp", 0)), 6),
                                            "salida":   round(exit_price, 6),
                                            "motivo":   _motivo,
                                            "pnl":      round(gan, 2),
                                            "pct":      _pct_cap,
                                            "rr":       _rr_real,
                                            "capital":  _cap,
                                            "apal":     int(est.get("apalancamiento", 3)),
                                            "ts_open":  _ts_open,
                                            "ts_close": _ts_close,
                                            "min":      _duracion,
                                            "estrategia": "MNV",
                                        })

                                        pending_logs.append((
                                            f"[{sym}] Cerrado {'✅' if gan >= 0 else '❌'} ${gan:+.2f} | Balance ${self._balance:.2f}",
                                            "success" if gan >= 0 else "error"
                                        ))
                                        _pct_close = round(gan / est.get("capital", 1) * 100, 2) if est.get("capital") else 0
                                        self._push_alert(
                                            f"{'✅ WIN' if gan >= 0 else '❌ LOSS'} {sym} {dir_} ${gan:+.2f} vía {_motivo}",
                                            "success" if gan >= 0 else "error", sym
                                        )
                                        self._send_telegram(
                                            f"{'✅ GANANCIA' if gan >= 0 else '❌ PÉRDIDA'} <b>{sym}</b>\n"
                                            f"💰 P&L: <b>${gan:+.2f}</b> ({_pct_close:+.2f}%)\n"
                                            f"📍 Entrada: ${ent:,.4f} → Salida: ${exit_price:,.4f}\n"
                                            f"📋 Motivo: {_motivo}\n"
                                            f"💼 Balance: ${self._balance:,.2f}"
                                        )
                                        if self._is_live_mode():
                                            try:
                                                self._sync_account()
                                            except Exception as se:
                                                pending_logs.append((f"[{sym}] Error sync balance: {se}", "error"))

                                        new_st = self._estado_vacio()
                                        new_st["tendencia"] = tendencia
                                        new_st["fase"] = fase
                                        loss_cd = int(cfg.get("cooldown_loss_ciclos", cfg.get("cooldown_ciclos", 8)))
                                        base_cd = int(cfg.get("cooldown_ciclos", 3))
                                        short_cd = int(cfg.get("cooldown_short_after_drop_ciclos", 12))
                                        new_st["cooldown_restante"] = loss_cd if gan < 0 else max(base_cd, short_cd) if dir_ == "SHORT" else base_cd
                                        new_st["ultimo_cierre"] = datetime.now(timezone.utc).isoformat()
                                        new_st["ultimo_resultado"] = "WIN" if gan >= 0 else "LOSS"
                                        self._estados[sym] = new_st

                            # ── Nueva entrada ──────────────────────────────
                            elif cfg.get("modo_operador", "AUTOMATICO") == "AUTOMATICO" and sym in self._trade_symbols():
                                can_open, open_reason = self._can_open_new_position(sym, tendencia)
                                cap  = self._symbol_capital(sym)
                                apal = self._symbol_leverage(sym)
                                dir_nueva = None

                                if not can_open:
                                    pending_logs.append((f"[{sym}] Sin entrada: {open_reason}", "info"))
                                    est["signal_confirmado"] = 0
                                elif spread_pct > 1.5:
                                    pending_logs.append((f"[{sym}] Sin entrada: precio se alejó {spread_pct:.1f}% del setup — esperar siguiente vela", "info"))
                                    est["signal_confirmado"] = 0
                                elif senal_l and tendencia == "ALCISTA":
                                    # Confirmación de 2 ciclos: el setup debe verse 2 veces seguidas
                                    est["signal_confirmado"] = int(est.get("signal_confirmado", 0)) + 1
                                    est["signal_short_confirmado"] = 0
                                    if est["signal_confirmado"] < 2:
                                        pending_logs.append((f"[{sym}] Setup detectado — esperando confirmación ciclo 2/2", "info"))
                                        self._push_alert(f"⏳ {sym} LONG — setup detectado, esperando ciclo 2/2", "warning", sym)
                                    else:
                                        # Verificar R:R real al precio live — tolerancia 15% sobre el mínimo
                                        live_risk_l = precio - sl_l
                                        live_rr_l = (tp_l - precio) / live_risk_l if live_risk_l > 0 else 0
                                        rr_min_live = float(cfg.get("rr_minimo", 2.0)) * 0.85
                                        if live_risk_l <= 0 or live_rr_l < rr_min_live:
                                            pending_logs.append((f"[{sym}] Sin entrada LONG: R:R real {live_rr_l:.1f} al precio live ${precio:,.4f}", "info"))
                                        else:
                                            dir_nueva = "LONG"
                                            sl_usar, tp_usar = sl_l, tp_l
                                            est["signal_confirmado"] = 0
                                elif senal_s and tendencia == "BAJISTA":
                                    est["signal_short_confirmado"] = int(est.get("signal_short_confirmado", 0)) + 1
                                    est["signal_confirmado"] = 0
                                    if est["signal_short_confirmado"] < 2:
                                        pending_logs.append((f"[{sym}] Setup SHORT detectado — esperando confirmación ciclo 2/2", "info"))
                                        self._push_alert(f"⏳ {sym} SHORT — setup detectado, esperando ciclo 2/2", "warning", sym)
                                    else:
                                        live_risk_s = sl_s - precio
                                        live_rr_s = (precio - tp_s) / live_risk_s if live_risk_s > 0 else 0
                                        rr_min_live = float(cfg.get("rr_minimo", 2.0)) * 0.85
                                        if live_risk_s <= 0 or live_rr_s < rr_min_live:
                                            pending_logs.append((f"[{sym}] Sin entrada SHORT: R:R real {live_rr_s:.1f} al precio live ${precio:,.4f}", "info"))
                                        else:
                                            dir_nueva = "SHORT"
                                            sl_usar, tp_usar = sl_s, tp_s
                                            est["signal_short_confirmado"] = 0
                                else:
                                    est["signal_confirmado"] = 0
                                    est["signal_short_confirmado"] = 0

                                if dir_nueva:
                                    exchange_qty = 0.0
                                    _sl_order_id = None
                                    if self._is_live_mode():
                                        try:
                                            cap = self._effective_order_capital(cap)
                                            exchange_qty, order = self._place_live_order(sym, dir_nueva, cap, apal, precio)
                                            pending_logs.append((
                                                f"[{sym}] Orden {dir_nueva} enviada #{order.get('orderId', '—')}",
                                                "success",
                                            ))
                                            _sl_order_id = self._place_sl_order(sym, dir_nueva, sl_usar)
                                            self._sync_account()
                                        except Exception as le:
                                            pending_logs.append((f"[{sym}] Error orden real: {le}", "error"))
                                            dir_nueva = None

                                if dir_nueva:
                                    est.update({
                                        "posicion_abierta":    True,
                                        "direccion_pos":       dir_nueva,
                                        "precio_entrada":      precio,
                                        "sl_inicial":          sl_usar,
                                        "sl_actual":           sl_usar,
                                        "tp":                  tp_usar,
                                        "precio_ext":          precio,
                                        "exchange_qty":        exchange_qty or ((cap * apal) / precio if precio > 0 else 0.0),
                                        "sl_order_id":         _sl_order_id,
                                        "breakeven_activado":  False,
                                        "salida_parcial_hecha":False,
                                        "capital":             cap,
                                        "apalancamiento":      apal,
                                        "pnl_pct":             0.0,
                                        "ts_apertura":         datetime.now(timezone.utc).isoformat(),
                                    })
                                    pending_logs.append((
                                        f"[{sym}] ENTRADA {dir_nueva} ${precio:,.2f} "
                                        f"SL ${sl_usar:,.2f} TP ${tp_usar:,.2f} | "
                                        f"RSI {rsi_live:.0f} | {tendencia} | {fase}",
                                        "success"
                                    ))
                                    self._push_alert(
                                        f"🚀 ENTRADA {dir_nueva} {sym} @ ${precio:,.4f} | SL ${sl_usar:,.4f} | TP ${tp_usar:,.4f}",
                                        "success", sym
                                    )

                        for msg, level in pending_logs:
                            self._log(msg, level)

                        # ── Estrategia Momentum ───────────────────────────────
                        senal_ml, senal_ms, sl_ml, tp_ml, sl_ms, tp_ms, conds_ml, conds_ms, score_ml, score_ms = \
                            self._verificar_momentum(df_signal, sym)
                        pending_logs_m: list[tuple[str, str]] = []

                        with self._lock:
                            est_m = self._estados_mom.setdefault(sym, self._estado_vacio())
                            est_m["score_mom_l"]   = score_ml
                            est_m["score_mom_s"]   = score_ms
                            mom_dir_for_checklist = est_m.get("direccion_pos") or ("SHORT" if score_ms > score_ml else "LONG")
                            active_mom_conds = conds_ms if mom_dir_for_checklist == "SHORT" else conds_ml
                            est_m["checklist_mom"] = [{"name": n, "ok": ok, "detail": d} for n, ok, d in active_mom_conds]
                            if int(est_m.get("cooldown_restante", 0) or 0) > 0 and not est_m.get("posicion_abierta"):
                                est_m["cooldown_restante"] = max(int(est_m.get("cooldown_restante", 0)) - 1, 0)

                            if est_m.get("posicion_abierta"):
                                dir_m = est_m["direccion_pos"]
                                if dir_m == "LONG" and precio > (est_m.get("precio_ext") or 0):
                                    est_m["precio_ext"] = precio
                                elif dir_m == "SHORT" and precio < (est_m.get("precio_ext") or float("inf")):
                                    est_m["precio_ext"] = precio

                                cerrar_m, parcial_m, nuevo_sl_m, exit_price_m, eventos_m = self._verificar_salida(df_live, est_m)

                                if dir_m == "LONG" and nuevo_sl_m > est_m["sl_actual"]:
                                    est_m["sl_actual"] = nuevo_sl_m
                                elif dir_m == "SHORT" and nuevo_sl_m < est_m["sl_actual"]:
                                    est_m["sl_actual"] = nuevo_sl_m

                                est_m["pnl_pct"] = round(
                                    ((precio - est_m["precio_entrada"]) / est_m["precio_entrada"] * 100) if dir_m == "LONG"
                                    else ((est_m["precio_entrada"] - precio) / est_m["precio_entrada"] * 100), 2
                                )
                                for ev in eventos_m:
                                    pending_logs_m.append((f"[{sym}][MOM] {ev}", "info"))

                                if parcial_m and not est_m["salida_parcial_hecha"]:
                                    _qty_tm = float(est_m.get("exchange_qty") or 0.0)
                                    if _qty_tm <= 0:
                                        _cm = float(est_m.get("capital") or 0.0)
                                        _am = int(est_m.get("apalancamiento") or 1)
                                        _em = float(est_m.get("precio_entrada") or precio)
                                        _qty_tm = (_cm * _am) / _em if _em > 0 else 0.0
                                    _qty_pm = _qty_tm / 2
                                    if _qty_pm > 0 and self._is_live_mode():
                                        try:
                                            self._close_live_order(sym, dir_m, _qty_pm)
                                        except Exception as _pe:
                                            pending_logs_m.append((f"[{sym}][MOM] Error parcial: {_pe}", "error"))
                                            _qty_pm = 0.0
                                    if _qty_pm > 0:
                                        _ent_pm = float(est_m["precio_entrada"])
                                        _gan_pm = ((precio - _ent_pm) * _qty_pm) if dir_m == "LONG" else ((_ent_pm - precio) * _qty_pm)
                                        self._balance  += _gan_pm
                                        self._ganancia += _gan_pm
                                        est_m["exchange_qty"] = max(_qty_tm - _qty_pm, 0.0)
                                        est_m["capital"] = round(float(est_m.get("capital", 0.0)) / 2, 2)
                                        est_m["salida_parcial_hecha"] = True
                                        pending_logs_m.append((f"[{sym}][MOM] Parcial +${_gan_pm:.2f}", "success"))

                                if cerrar_m:
                                    _cap_m  = float(est_m["capital"])
                                    _apal_m = int(est_m["apalancamiento"])
                                    _ent_m  = float(est_m["precio_entrada"])
                                    _cant_m = float(est_m.get("exchange_qty") or ((_cap_m * _apal_m) / _ent_m if _ent_m > 0 else 0.0))
                                    if self._is_live_mode():
                                        try:
                                            self._close_live_order(sym, dir_m, _cant_m)
                                            self._cancel_sl_order(sym, est_m.get("sl_order_id"))
                                            est_m["sl_order_id"] = None
                                        except Exception as _le:
                                            pending_logs_m.append((f"[{sym}][MOM] Error cerrando: {_le}", "error"))
                                            cerrar_m = False
                                    if cerrar_m:
                                        _gan_m = ((exit_price_m - _ent_m) * _cant_m) if dir_m == "LONG" else ((_ent_m - exit_price_m) * _cant_m)
                                        self._balance  += _gan_m
                                        self._ganancia += _gan_m
                                        self._operations += 1
                                        if _gan_m >= 0: self._wins   += 1
                                        else:           self._losses += 1
                                        _ev0_m  = eventos_m[0] if eventos_m else ""
                                        _mot_m  = ("TP" if "TP" in _ev0_m else "Trailing" if "Trailing" in _ev0_m else "SL" if "SL" in _ev0_m else "Manual")
                                        _sl_im  = float(est_m.get("sl_inicial", est_m.get("sl_actual", 0)))
                                        _rsgm   = abs(_ent_m - _sl_im) if _sl_im else 0
                                        _rr_m   = round(((exit_price_m - _ent_m) / _rsgm if dir_m == "LONG" else (_ent_m - exit_price_m) / _rsgm), 2) if _rsgm > 0 else 0.0
                                        _ts_om  = est_m.get("ts_apertura", "")
                                        _ts_cm  = datetime.now(timezone.utc).isoformat()
                                        _dur_m  = 0
                                        if _ts_om:
                                            try:
                                                _dur_m = max(0, int((datetime.fromisoformat(_ts_cm.replace("Z","")) - datetime.fromisoformat(_ts_om.replace("Z",""))).total_seconds() / 60))
                                            except Exception:
                                                pass
                                        self._historial_counter += 1
                                        self._historial.append({
                                            "id": self._historial_counter, "sym": sym, "dir": dir_m,
                                            "entrada": round(_ent_m, 6), "sl_ini": round(_sl_im, 6),
                                            "tp": round(float(est_m.get("tp", 0)), 6), "salida": round(exit_price_m, 6),
                                            "motivo": _mot_m, "pnl": round(_gan_m, 2),
                                            "pct": round(_gan_m / _cap_m * 100, 2) if _cap_m > 0 else 0.0,
                                            "rr": _rr_m, "capital": _cap_m, "apal": int(est_m.get("apalancamiento", 3)),
                                            "ts_open": _ts_om, "ts_close": _ts_cm, "min": _dur_m, "estrategia": "MOM",
                                        })
                                        pending_logs_m.append((
                                            f"[{sym}][MOM] Cerrado {'✅' if _gan_m >= 0 else '❌'} ${_gan_m:+.2f} | Balance ${self._balance:.2f}",
                                            "success" if _gan_m >= 0 else "error"
                                        ))
                                        _pct_close_m = round(_gan_m / _cap_m * 100, 2) if _cap_m > 0 else 0
                                        self._push_alert(
                                            f"{'✅ WIN' if _gan_m >= 0 else '❌ LOSS'} [MOM] {sym} {dir_m} ${_gan_m:+.2f} vía {_mot_m}",
                                            "success" if _gan_m >= 0 else "error", sym
                                        )
                                        self._send_telegram(
                                            f"{'✅ GANANCIA' if _gan_m >= 0 else '❌ PÉRDIDA'} <b>{sym}</b> [MOM]\n"
                                            f"💰 P&L: <b>${_gan_m:+.2f}</b> ({_pct_close_m:+.2f}%)\n"
                                            f"📍 Entrada: ${_ent_m:,.4f} → Salida: ${exit_price_m:,.4f}\n"
                                            f"📋 Motivo: {_mot_m}\n"
                                            f"💼 Balance: ${self._balance:,.2f}"
                                        )
                                        if self._is_live_mode():
                                            try:
                                                self._sync_account()
                                            except Exception as _se:
                                                pending_logs_m.append((f"[{sym}][MOM] Error sync: {_se}", "error"))
                                        _nst_m = self._estado_vacio()
                                        _nst_m["tendencia"] = tendencia
                                        _nst_m["fase"]      = fase
                                        _short_cd_m = int(cfg.get("cooldown_short_after_drop_ciclos", 12))
                                        _nst_m["cooldown_restante"] = 20 if _gan_m < 0 else max(3, _short_cd_m) if dir_m == "SHORT" else 3
                                        _nst_m["ultimo_cierre"]     = datetime.now(timezone.utc).isoformat()
                                        _nst_m["ultimo_resultado"]  = "WIN" if _gan_m >= 0 else "LOSS"
                                        self._estados_mom[sym] = _nst_m

                            elif cfg.get("modo_operador", "AUTOMATICO") == "AUTOMATICO" and sym in self._trade_symbols() and sym in self._MOM_TRADE_SYMS and not int(est_m.get("cooldown_restante", 0) or 0) and not est_m.get("posicion_abierta") and not self._estados.get(sym, {}).get("posicion_abierta"):
                                _dir_mn = None
                                _sl_mn = _tp_mn = 0.0
                                # Filtro de tendencia: MOM no puede ir en contra de la tendencia principal
                                # Confirmación de 2 ciclos para evitar entradas prematuras
                                if senal_ml and tendencia != "BAJISTA":
                                    est_m["signal_confirmado"] = int(est_m.get("signal_confirmado", 0)) + 1
                                    est_m["signal_short_confirmado"] = 0
                                    if est_m["signal_confirmado"] < 2:
                                        pending_logs_m.append((f"[{sym}][MOM] Setup LONG — esperando confirmación ciclo 2/2", "info"))
                                        self._push_alert(f"⏳ {sym} [MOM] LONG — setup detectado, esperando ciclo 2/2", "warning", sym)
                                    else:
                                        _lrisk_ml = precio - sl_ml
                                        _lrr_ml   = (tp_ml - precio) / _lrisk_ml if _lrisk_ml > 0 else 0
                                        if _lrisk_ml > 0 and _lrr_ml >= 1.5:
                                            _dir_mn, _sl_mn, _tp_mn = "LONG", sl_ml, tp_ml
                                            est_m["signal_confirmado"] = 0
                                        else:
                                            pending_logs_m.append((f"[{sym}][MOM] Sin LONG: R:R {_lrr_ml:.1f}", "info"))
                                elif senal_ms and tendencia != "ALCISTA":
                                    est_m["signal_short_confirmado"] = int(est_m.get("signal_short_confirmado", 0)) + 1
                                    est_m["signal_confirmado"] = 0
                                    if est_m["signal_short_confirmado"] < 2:
                                        pending_logs_m.append((f"[{sym}][MOM] Setup SHORT — esperando confirmación ciclo 2/2", "info"))
                                        self._push_alert(f"⏳ {sym} [MOM] SHORT — setup detectado, esperando ciclo 2/2", "warning", sym)
                                    else:
                                        _lrisk_ms = sl_ms - precio
                                        _lrr_ms   = (precio - tp_ms) / _lrisk_ms if _lrisk_ms > 0 else 0
                                        if _lrisk_ms > 0 and _lrr_ms >= 1.5:
                                            _dir_mn, _sl_mn, _tp_mn = "SHORT", sl_ms, tp_ms
                                            est_m["signal_short_confirmado"] = 0
                                        else:
                                            pending_logs_m.append((f"[{sym}][MOM] Sin SHORT: R:R {_lrr_ms:.1f}", "info"))
                                else:
                                    est_m["signal_confirmado"] = 0
                                    est_m["signal_short_confirmado"] = 0
                                    if senal_ml and tendencia == "BAJISTA":
                                        pending_logs_m.append((f"[{sym}][MOM] LONG bloqueado — tendencia BAJISTA", "info"))
                                    elif senal_ms and tendencia == "ALCISTA":
                                        pending_logs_m.append((f"[{sym}][MOM] SHORT bloqueado — tendencia ALCISTA", "info"))
                                if _dir_mn:
                                    _cap_mn  = self._symbol_capital(sym)
                                    _apal_mn = self._symbol_leverage(sym)
                                    _qty_mn  = 0.0
                                    _sl_oid_mn = None
                                    if self._is_live_mode():
                                        try:
                                            _cap_mn = self._effective_order_capital(_cap_mn)
                                            _qty_mn, _ord_mn = self._place_live_order(sym, _dir_mn, _cap_mn, _apal_mn, precio)
                                            _qty_mn = float(_ord_mn.get("executedQty") or _ord_mn.get("origQty") or 0)
                                            precio  = float(_ord_mn.get("avgPx") or _ord_mn.get("price") or precio)
                                            pending_logs_m.append((f"[{sym}][MOM] Orden {_dir_mn} #{_ord_mn.get('orderId','—')}", "success"))
                                            _sl_oid_mn = self._place_sl_order(sym, _dir_mn, _sl_mn)
                                            self._sync_account()
                                        except Exception as _le2:
                                            pending_logs_m.append((f"[{sym}][MOM] Error orden: {_le2}", "error"))
                                            _dir_mn = None
                                    if _dir_mn:
                                        est_m.update({
                                            "posicion_abierta": True, "direccion_pos": _dir_mn,
                                            "precio_entrada": precio, "sl_inicial": _sl_mn, "sl_actual": _sl_mn,
                                            "tp": _tp_mn, "precio_ext": precio,
                                            "exchange_qty": _qty_mn or ((_cap_mn * _apal_mn) / precio if precio > 0 else 0.0),
                                            "sl_order_id": _sl_oid_mn,
                                            "breakeven_activado": False, "salida_parcial_hecha": False,
                                            "capital": _cap_mn, "apalancamiento": _apal_mn, "pnl_pct": 0.0,
                                            "ts_apertura": datetime.now(timezone.utc).isoformat(),
                                        })
                                        pending_logs_m.append((
                                            f"[{sym}][MOM] ENTRADA {_dir_mn} ${precio:,.4f} "
                                            f"SL ${_sl_mn:,.4f} TP ${_tp_mn:,.4f} | {score_ml if _dir_mn=='LONG' else score_ms}/6",
                                            "success"
                                        ))
                                        self._push_alert(
                                            f"🚀 [MOM] ENTRADA {_dir_mn} {sym} @ ${precio:,.4f} | SL ${_sl_mn:,.4f} | TP ${_tp_mn:,.4f}",
                                            "success", sym
                                        )

                        for msg, level in pending_logs_m:
                            self._log(msg, level)

                        # Log resumen del símbolo
                        score_max = max(score_l, score_s)
                        score_min = float(cfg.get("score_minimo", 6.5))
                        nivel = "warning" if score_max >= score_min * 0.85 else "info"
                        if tendencia == "ALCISTA":
                            faltan = [n for n, ok, _ in conds_l if not ok]
                            accion = "LISTO ✓" if senal_l else ("FALTA: " + ", ".join(faltan[:2]) if faltan else "ESPERAR")
                        elif tendencia == "BAJISTA":
                            faltan = [n for n, ok, _ in conds_s if not ok]
                            accion = "LISTO ✓" if senal_s else ("FALTA: " + ", ".join(faltan[:2]) if faltan else "ESPERAR")
                        else:
                            accion = "SIN TENDENCIA"
                        if sym not in self._trade_symbols():
                            accion = f"OBS | {accion}"
                        self._log(
                            f"[{sym}] ${precio:,.2f} | RSI {rsi_live:.0f} | {tendencia} | {fase} | "
                            f"L:{score_l:.1f} S:{score_s:.1f} | {accion}",
                            nivel
                        )

                    except Exception as e:
                        self._log(f"[{sym}] Error: {e}", "error")

                intervalo = int(self.cfg.get("intervalo_segundos", 60))
                self._log(f"─── Ciclo completo. Próx análisis en {intervalo}s ───")
                self._pattern_memory.save()
                time.sleep(intervalo)

            except Exception as e:
                self._log(f"Error en loop: {e}", "error")
                time.sleep(30)
