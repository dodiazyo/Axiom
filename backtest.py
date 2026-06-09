#!/usr/bin/env python3
"""
Axiom Backtester
================
Corre la lógica REAL de las estrategias MNV y MOM de bot_engine.py sobre
velas históricas de OKX, sin reimplementarla: instancia TrendBot y llama a
sus propios métodos (_analizar_tendencia, _verificar_long, _verificar_short,
_verificar_momentum, _verificar_salida, _apply_btc_market_filter,
_can_open_new_position). Si cambias la estrategia en bot_engine.py, el
backtest la prueba automáticamente.

Incluye costes reales: comisión taker, funding aproximado y slippage.

Uso:
    # Descargar datos de OKX y backtestear 365 días (requiere internet):
    python3 backtest.py --symbols BTC/USDT ETH/USDT --days 365

    # Modo rápido con menos historia:
    python3 backtest.py --symbols BTC/USDT --days 120

    # Validación con datos sintéticos (sin internet):
    python3 backtest.py --synthetic

    # Probar sizing por riesgo (1% del balance por trade):
    python3 backtest.py --symbols BTC/USDT ETH/USDT --days 365 --sizing risk --risk-pct 1.0

Salidas:
    backtest_data/           caché de velas descargadas (CSV)
    backtest_trades.csv      todos los trades con detalle
    backtest_equity.csv      curva de capital vela a vela
    backtest_report.txt      resumen de métricas
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from bot_engine import TrendBot, _prepare

OKX_URL  = "https://www.okx.com"
DATA_DIR = Path(__file__).with_name("backtest_data")

# Velas mínimas de calentamiento antes de empezar a operar (EMA200 estable)
WARMUP = 250
# Ventana de velas que ve la estrategia en cada paso (igual que el bot en vivo)
WINDOW = 300


# ─────────────────────────────────────────────────────────────────────────────
# Datos
# ─────────────────────────────────────────────────────────────────────────────

def download_okx_candles(sym: str, days: int, bar: str = "1H") -> pd.DataFrame:
    """Descarga velas históricas del swap perpetuo en OKX, con caché local."""
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / f"{sym.replace('/', '')}_{bar}_{days}d.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        print(f"  [{sym}] {len(df)} velas desde caché ({cache.name})")
        return df

    inst     = sym.replace("/", "-") + "-SWAP"
    end_ms   = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 3600 * 1000
    rows: list = []
    after = end_ms  # OKX pagina hacia atrás con 'after' (devuelve velas < after)

    print(f"  [{sym}] descargando ~{days * 24} velas de OKX…", end="", flush=True)
    while True:
        r = requests.get(
            f"{OKX_URL}/api/v5/market/history-candles",
            params={"instId": inst, "bar": bar, "limit": 100, "after": after},
            timeout=(5, 15),
        )
        r.raise_for_status()
        payload = r.json()
        if payload.get("code") not in (None, "0"):
            raise RuntimeError(f"OKX: {payload.get('msg')}")
        data = payload.get("data", [])
        if not data:
            break
        rows.extend(data)
        oldest = int(data[-1][0])
        after = oldest
        if oldest <= start_ms:
            break
        if len(rows) % 2000 < 100:
            print(".", end="", flush=True)
        time.sleep(0.12)  # respeta el rate limit público de OKX (20 req / 2 s)
    print(f" {len(rows)} velas")

    cols = ["ts", "open", "high", "low", "close", "volume", "vol_ccy", "vol_quote", "confirm"]
    df = pd.DataFrame(rows, columns=cols)[["ts", "open", "high", "low", "close", "volume"]]
    df = df.astype({"ts": "int64", "open": "float64", "high": "float64",
                    "low": "float64", "close": "float64", "volume": "float64"})
    df = df[df["ts"] >= start_ms].drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.to_csv(cache, index=False)
    return df


def synthetic_candles(n: int = 4000, seed: int = 7, p0: float = 50000.0) -> pd.DataFrame:
    """Velas sintéticas con regímenes de tendencia/rango para validar el motor."""
    rng = np.random.default_rng(seed)
    drift = np.zeros(n)
    i = 0
    while i < n:
        seg = int(rng.integers(150, 500))
        mu = rng.choice([0.0006, -0.0005, 0.0, 0.0003, -0.0003])
        drift[i:i + seg] = mu
        i += seg
    rets   = drift + rng.normal(0, 0.008, n)
    close  = p0 * np.exp(np.cumsum(rets))
    open_  = np.roll(close, 1); open_[0] = p0
    spread = np.abs(rng.normal(0, 0.004, n))
    high   = np.maximum(open_, close) * (1 + spread)
    low    = np.minimum(open_, close) * (1 - spread)
    vol    = rng.lognormal(10, 0.5, n) * (1 + 4 * np.abs(rets) / 0.008)
    ts     = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"ts": ts, "open": open_, "high": high, "low": low,
                         "close": close, "volume": vol})


# ─────────────────────────────────────────────────────────────────────────────
# Backtester
# ─────────────────────────────────────────────────────────────────────────────

class AxiomBacktester:
    def __init__(self, symbols: list[str], dfs: dict[str, pd.DataFrame],
                 cfg_overrides: dict | None = None,
                 fee_rate: float = 0.0005,        # taker OKX por lado
                 funding_8h: float = 0.0001,      # 0.01% / 8h aprox. (coste)
                 slippage_bps: float = 2.0,       # 0.02% por ejecución a mercado
                 confirm_candles: int = 1,        # velas consecutivas con señal
                 cooldown_win: int = 3,           # velas de espera tras ganar
                 cooldown_loss: int = 8,          # velas de espera tras perder
                 sizing: str = "fixed",           # "fixed" (como el bot) o "risk"
                 risk_pct: float = 1.0,           # % del balance arriesgado/trade
                 solo: str | None = None,         # "MNV" o "MOM": aislar una estrategia
                 mom_rr: float | None = None):    # override del TP de MOM en múltiplos de R
        cfg = {
            "symbols": symbols, "watch_symbols": symbols, "trade_symbols": symbols,
            "momentum_symbols": [s for s in symbols if s in ("BTC/USDT", "ETH/USDT")],
            "momentum_enabled": True,   # el backtester la deja disponible para experimentar
            "capital_usd": 25.0, "apalancamiento": 3, "balance_inicial": 1000.0,
            "symbol_leverage": {}, "symbol_capital": {},
            "rr_minimo": 2.0, "score_minimo": 6.5, "vol_pullback_max": 0.85,
            "atr_sl_mult": 1.5, "atr_tp_mult": 4.0, "atr_trail_mult": 2.0,
            "breakeven_rr": 1.0, "parcial_rr": 1.5,
            "max_posiciones": 2, "modo_operador": "AUTOMATICO",
            "execution_mode": "SIMULADO", "margin_type": "ISOLATED",
            "max_entry_chase_pct": 0.6,
        }
        cfg.update(cfg_overrides or {})
        self.cfg = cfg
        self.bot = TrendBot(cfg)            # nunca se llama start(): solo usamos su lógica
        self.symbols = symbols
        self.fee_rate, self.funding_1h = fee_rate, funding_8h / 8.0
        self.slip = slippage_bps / 10_000.0
        self.confirm_candles = max(1, confirm_candles)
        self.cooldown_win, self.cooldown_loss = cooldown_win, cooldown_loss
        self.sizing, self.risk_pct = sizing, risk_pct
        self.solo, self.mom_rr = solo, mom_rr

        # Alinear todos los símbolos por timestamp (intersección)
        common = None
        for s in symbols:
            tset = set(dfs[s]["ts"])
            common = tset if common is None else common & tset
        common = sorted(common)
        self.index = pd.DatetimeIndex(common)
        self.dfs = {s: dfs[s].set_index("ts").loc[self.index].reset_index() for s in symbols}
        # Indicadores precalculados UNA vez sobre toda la historia (EMA de
        # historia completa ≈ EMA de ventana 300 tras el warmup; mucho más rápido)
        print("  Precalculando indicadores…")
        self.prepared = {s: _prepare(self.dfs[s]) for s in symbols}
        self.n = len(self.index)

        self.balance = float(cfg["balance_inicial"])
        self.balance_ini = self.balance
        self.fees_total = 0.0
        self.funding_total = 0.0
        self.trades: list[dict] = []
        self.equity: list[dict] = []
        # confirmaciones de señal por símbolo y estrategia
        self.conf = {s: {"mnv_l": 0, "mnv_s": 0, "mom_l": 0, "mom_s": 0} for s in symbols}
        # señales pendientes de ejecutar al open de la vela siguiente
        self.pending: dict[str, dict] = {}

    # ── tamaño de posición ────────────────────────────────────────────────────
    def _position_size(self, sym: str, entry: float, sl: float) -> tuple[float, float, int]:
        """Retorna (qty, capital, apalancamiento)."""
        apal = self.bot._symbol_leverage(sym)
        if self.sizing == "risk":
            risk_usd = self.balance * self.risk_pct / 100.0
            dist = abs(entry - sl)
            qty = risk_usd / dist if dist > 0 else 0.0
            capital = qty * entry / apal
            cap_max = self.balance * 0.5           # nunca más del 50% del balance en margen
            if capital > cap_max and capital > 0:
                qty *= cap_max / capital
                capital = cap_max
            return qty, round(capital, 2), apal
        capital = self.bot._symbol_capital(sym)
        qty = (capital * apal) / entry if entry > 0 else 0.0
        return qty, capital, apal

    # ── apertura / cierre ─────────────────────────────────────────────────────
    def _open(self, sym: str, estrategia: str, direction: str, ts, open_px: float,
              sl: float, tp: float, score: float):
        entry = open_px * (1 + self.slip) if direction == "LONG" else open_px * (1 - self.slip)
        qty, capital, apal = self._position_size(sym, entry, sl)
        if qty <= 0:
            return
        fee = qty * entry * self.fee_rate
        self.fees_total += fee
        self.balance -= fee
        est = self.bot._estados[sym] if estrategia == "MNV" else self.bot._estados_mom.setdefault(sym, self.bot._estado_vacio())
        est.update({
            "posicion_abierta": True, "direccion_pos": direction,
            "precio_entrada": entry, "sl_inicial": sl, "sl_actual": sl, "tp": tp,
            "precio_ext": entry, "exchange_qty": qty, "capital": capital,
            "apalancamiento": apal, "breakeven_activado": False,
            "salida_parcial_hecha": False, "pnl_pct": 0.0,
            "ts_apertura": str(ts), "score_open": score,
            "entrada_temprana": est.get("entrada_temprana", False),
        })

    def _close(self, sym: str, estrategia: str, est: dict, ts, exit_px: float,
               motivo: str, qty_frac: float = 1.0):
        dir_ = est["direccion_pos"]
        ent  = float(est["precio_entrada"])
        qty  = float(est["exchange_qty"]) * qty_frac
        px   = exit_px * (1 - self.slip) if dir_ == "LONG" else exit_px * (1 + self.slip)
        gan  = (px - ent) * qty if dir_ == "LONG" else (ent - px) * qty
        fee  = qty * px * self.fee_rate
        self.fees_total += fee
        self.balance += gan - fee

        if qty_frac < 1.0:                       # salida parcial
            est["exchange_qty"] = float(est["exchange_qty"]) - qty
            est["capital"] = round(float(est["capital"]) / 2, 2)
            est["salida_parcial_hecha"] = True
            self.trades.append({
                "ts_open": est["ts_apertura"], "ts_close": str(ts), "sym": sym,
                "estrategia": estrategia, "dir": dir_, "entrada": ent, "salida": px,
                "qty": qty, "pnl": round(gan - fee, 4), "motivo": "Parcial",
                "sl_ini": est.get("sl_inicial"), "tp": est.get("tp"),
            })
            return

        sl_ini = float(est.get("sl_inicial") or 0)
        riesgo = abs(ent - sl_ini) if sl_ini else 0
        rr = round(((px - ent) / riesgo if dir_ == "LONG" else (ent - px) / riesgo), 2) if riesgo > 0 else 0.0
        self.trades.append({
            "ts_open": est["ts_apertura"], "ts_close": str(ts), "sym": sym,
            "estrategia": estrategia, "dir": dir_, "entrada": ent, "salida": px,
            "qty": qty, "pnl": round(gan - fee, 4), "motivo": motivo, "rr": rr,
            "sl_ini": est.get("sl_inicial"), "tp": est.get("tp"),
            "score": est.get("score_open", 0.0),
        })
        nuevo = self.bot._estado_vacio()
        nuevo["cooldown_restante"] = self.cooldown_loss if gan < 0 else self.cooldown_win
        if estrategia == "MNV":
            self.bot._estados[sym] = nuevo
        else:
            self.bot._estados_mom[sym] = nuevo

    # ── gestión de una posición abierta durante la vela i ────────────────────
    def _manage(self, sym: str, estrategia: str, est: dict, candle, window: pd.DataFrame, ts):
        dir_ = est["direccion_pos"]
        qty  = float(est["exchange_qty"])
        ent  = float(est["precio_entrada"])

        # funding por vela mantenida
        funding = qty * float(candle["close"]) * self.funding_1h
        self.funding_total += funding
        self.balance -= funding

        # 1) toque intra-vela de SL / TP con high-low (caso peor: SL primero)
        sl, tp = float(est["sl_actual"]), float(est.get("tp") or 0)
        if dir_ == "LONG":
            if float(candle["low"]) <= sl:
                self._close(sym, estrategia, est, ts, sl, "SL"); return
            if tp and float(candle["high"]) >= tp:
                self._close(sym, estrategia, est, ts, tp, "TP"); return
            est["precio_ext"] = max(float(est["precio_ext"] or ent), float(candle["high"]))
        else:
            if float(candle["high"]) >= sl:
                self._close(sym, estrategia, est, ts, sl, "SL"); return
            if tp and float(candle["low"]) <= tp:
                self._close(sym, estrategia, est, ts, tp, "TP"); return
            est["precio_ext"] = min(float(est["precio_ext"] or ent), float(candle["low"]))

        # 2) al cierre de la vela: breakeven / trailing / invalidación / parcial
        #    usando la MISMA función que el bot en vivo
        cerrar, parcial, nuevo_sl, exit_price, eventos = self.bot._verificar_salida(window, est)
        if dir_ == "LONG" and nuevo_sl > est["sl_actual"]:
            est["sl_actual"] = nuevo_sl
        elif dir_ == "SHORT" and nuevo_sl < est["sl_actual"]:
            est["sl_actual"] = nuevo_sl
        if cerrar:
            ev0 = eventos[0] if eventos else ""
            motivo = ("TP" if "TP" in ev0 else "Trailing" if "Trailing" in ev0
                      else "SL" if "SL" in ev0 else "Invalidación")
            self._close(sym, estrategia, est, ts, exit_price, motivo)
        elif parcial and not est["salida_parcial_hecha"]:
            self._close(sym, estrategia, est, ts, float(candle["close"]), "Parcial", qty_frac=0.5)

    # ── loop principal ────────────────────────────────────────────────────────
    def run(self):
        bot = self.bot
        for s in self.symbols:
            bot._estados[s] = bot._estado_vacio()
            bot._estados_mom[s] = bot._estado_vacio()

        t0 = time.time()
        for i in range(WARMUP, self.n):
            ts = self.index[i]

            # contexto maestro de BTC con la ventana hasta la vela anterior cerrada
            btc_ctx = {}
            if "BTC/USDT" in self.prepared:
                btc_win = self.prepared["BTC/USDT"].iloc[max(0, i - WINDOW):i]
                try:
                    btc_ctx = bot._btc_market_context(btc_win)
                except Exception:
                    btc_ctx = {}

            open_unreal = 0.0
            for sym in self.symbols:
                prep = self.prepared[sym]
                candle = prep.iloc[i]                                  # vela "actual"
                win_closed = prep.iloc[max(0, i - WINDOW):i]           # velas cerradas
                win_with_now = prep.iloc[max(0, i - WINDOW):i + 1]
                if len(win_closed) < 50:
                    continue

                est   = bot._estados[sym]
                est_m = bot._estados_mom[sym]

                # ejecutar señal pendiente de la vela anterior al OPEN de esta
                pend = self.pending.pop(sym, None)
                if pend is not None:
                    open_px = float(candle["open"])
                    sl, tp, dir_, strat = pend["sl"], pend["tp"], pend["dir"], pend["strat"]
                    if strat == "MOM" and self.mom_rr:
                        _r = (open_px - sl) if dir_ == "LONG" else (sl - open_px)
                        if _r > 0:
                            tp = open_px + _r * self.mom_rr if dir_ == "LONG" else open_px - _r * self.mom_rr
                    # re-chequear R:R al precio de ejecución (como hace el bot en vivo)
                    riesgo = (open_px - sl) if dir_ == "LONG" else (sl - open_px)
                    reward = (tp - open_px) if dir_ == "LONG" else (open_px - tp)
                    rr_min = (float(self.cfg["rr_minimo"]) * 0.85) if strat == "MNV" else 1.5
                    dest = est if strat == "MNV" else est_m
                    if riesgo > 0 and reward / riesgo >= rr_min and not dest.get("posicion_abierta"):
                        if strat == "MNV":
                            can, _why = bot._can_open_new_position(sym, "ALCISTA" if dir_ == "LONG" else "BAJISTA")
                        else:
                            can = (not est.get("posicion_abierta")
                                   and not int(est_m.get("cooldown_restante", 0) or 0)
                                   and bot._current_open_positions() < int(self.cfg["max_posiciones"]))
                        if can:
                            self._open(sym, strat, dir_, ts, open_px, sl, tp, pend["score"])

                # cooldowns en VELAS (corrige el desajuste ciclos↔velas del bot en vivo)
                for d in (est, est_m):
                    if int(d.get("cooldown_restante", 0) or 0) > 0 and not d.get("posicion_abierta"):
                        d["cooldown_restante"] = int(d["cooldown_restante"]) - 1

                # gestionar posiciones abiertas durante esta vela
                if est.get("posicion_abierta"):
                    self._manage(sym, "MNV", est, candle, win_with_now, ts)
                if est_m.get("posicion_abierta"):
                    self._manage(sym, "MOM", est_m, candle, win_with_now, ts)

                # ── señales sobre velas cerradas (sin lookahead) ──────────────
                tendencia, fase = bot._analizar_tendencia(win_closed)
                est["tendencia"], est["fase"] = tendencia, fase

                senal_l, conds_l, score_l, sl_l, tp_l = bot._verificar_long(win_closed, sym)
                senal_s, conds_s, score_s, sl_s, tp_s = bot._verificar_short(win_closed, sym)
                if btc_ctx:
                    senal_l, senal_s, conds_l, conds_s, _blk = bot._apply_btc_market_filter(
                        sym, senal_l, senal_s, conds_l, conds_s, btc_ctx)

                early  = any(n == "Stage 2 temprano" and ok for n, ok, _ in conds_l)
                accum  = any(n == "Acumulacion" and ok for n, ok, _ in conds_l)
                ctx_ok = tendencia == "ALCISTA" or early or accum
                cf = self.conf[sym]

                if (self.solo != "MOM" and not est.get("posicion_abierta")
                        and not int(est.get("cooldown_restante", 0) or 0)):
                    cf["mnv_l"] = cf["mnv_l"] + 1 if (senal_l and ctx_ok) else 0
                    cf["mnv_s"] = cf["mnv_s"] + 1 if (senal_s and tendencia == "BAJISTA") else 0
                    if cf["mnv_l"] >= self.confirm_candles:
                        can, _w = bot._can_open_new_position(sym, "ALCISTA")
                        if can:
                            self.pending[sym] = {"strat": "MNV", "dir": "LONG", "sl": sl_l,
                                                 "tp": tp_l, "score": score_l}
                            est["entrada_temprana"] = bool((early or accum) and tendencia != "ALCISTA")
                            cf["mnv_l"] = 0
                    elif cf["mnv_s"] >= self.confirm_candles:
                        can, _w = bot._can_open_new_position(sym, "BAJISTA")
                        if can:
                            self.pending[sym] = {"strat": "MNV", "dir": "SHORT", "sl": sl_s,
                                                 "tp": tp_s, "score": score_s}
                            cf["mnv_s"] = 0

                # MOM (solo símbolos momentum; una posición por símbolo)
                if (self.solo != "MNV" and sym in bot._momentum_trade_symbols()
                        and not est_m.get("posicion_abierta") and not est.get("posicion_abierta")
                        and not int(est_m.get("cooldown_restante", 0) or 0)
                        and sym not in self.pending):
                    (s_ml, s_ms, sl_ml, tp_ml, sl_ms, tp_ms,
                     _cl, _cs, sc_ml, sc_ms) = bot._verificar_momentum(win_closed, sym)
                    if btc_ctx:
                        s_ml, s_ms, _cl, _cs, _b = bot._apply_btc_market_filter(
                            sym, s_ml, s_ms, _cl, _cs, btc_ctx)
                    cf["mom_l"] = cf["mom_l"] + 1 if (s_ml and tendencia != "BAJISTA") else 0
                    cf["mom_s"] = cf["mom_s"] + 1 if (s_ms and tendencia != "ALCISTA") else 0
                    if cf["mom_l"] >= self.confirm_candles:
                        self.pending[sym] = {"strat": "MOM", "dir": "LONG", "sl": sl_ml,
                                             "tp": tp_ml, "score": sc_ml}
                        cf["mom_l"] = 0
                    elif cf["mom_s"] >= self.confirm_candles:
                        self.pending[sym] = {"strat": "MOM", "dir": "SHORT", "sl": sl_ms,
                                             "tp": tp_ms, "score": sc_ms}
                        cf["mom_s"] = 0

                # PnL no realizado al cierre de la vela
                for d in (est, est_m):
                    if d.get("posicion_abierta"):
                        e = float(d["precio_entrada"]); q = float(d["exchange_qty"])
                        c = float(candle["close"])
                        open_unreal += (c - e) * q if d["direccion_pos"] == "LONG" else (e - c) * q

            self.equity.append({"ts": str(ts), "balance": round(self.balance, 2),
                                "equity": round(self.balance + open_unreal, 2)})
            if (i - WARMUP) % 500 == 0:
                pct = (i - WARMUP) / (self.n - WARMUP) * 100
                print(f"  … {pct:5.1f}%  ({self.index[i].date()})  equity ${self.balance + open_unreal:,.2f}")

        # cerrar posiciones que quedaron abiertas al final, a precio de cierre
        ts_end = self.index[-1]
        for sym in self.symbols:
            last_close = float(self.prepared[sym].iloc[-1]["close"])
            for strat, d in (("MNV", self.bot._estados[sym]), ("MOM", self.bot._estados_mom[sym])):
                if d.get("posicion_abierta"):
                    self._close(sym, strat, d, ts_end, last_close, "Fin de datos")

        print(f"\n  Backtest completado en {time.time() - t0:,.1f}s")
        return self.report()

    # ── métricas ──────────────────────────────────────────────────────────────
    def report(self) -> str:
        tr = pd.DataFrame(self.trades)
        eq = pd.DataFrame(self.equity)
        lines = []
        w = lines.append

        w("=" * 64)
        w("AXIOM BACKTEST — RESUMEN")
        w("=" * 64)
        w(f"Periodo:        {self.index[WARMUP].date()} → {self.index[-1].date()} "
          f"({(self.n - WARMUP) / 24:.0f} días, velas 1h)")
        w(f"Símbolos:       {', '.join(self.symbols)}")
        variantes = []
        if self.solo:    variantes.append(f"solo={self.solo}")
        if self.mom_rr:  variantes.append(f"mom_rr={self.mom_rr}")
        if self.cfg.get("slow_trend"): variantes.append(
            f"slow_trend (adx≥{self.cfg.get('slow_adx_min', 15)}, imp {self.cfg.get('slow_impulso_pct', 0.05)}%)")
        if variantes:
            w("Variante:       " + " · ".join(variantes))
        w(f"Sizing:         {self.sizing}"
          + (f" ({self.risk_pct}%/trade)" if self.sizing == "risk" else
             f" (${self.cfg['capital_usd']} × {self.cfg['apalancamiento']}x)"))
        w(f"Costes:         fee {self.fee_rate*100:.3f}%/lado · funding {self.funding_1h*8*100:.3f}%/8h · "
          f"slippage {self.slip*10000:.0f} bps")
        w("-" * 64)
        ret = (self.balance / self.balance_ini - 1) * 100
        w(f"Balance:        ${self.balance_ini:,.2f} → ${self.balance:,.2f}  ({ret:+.2f}%)")
        w(f"Comisiones:     ${self.fees_total:,.2f}   |   Funding: ${self.funding_total:,.2f}")

        if eq.empty or tr.empty:
            w("Sin trades ejecutados — revisa parámetros o periodo.")
            text = "\n".join(lines)
            print("\n" + text)
            return text

        eq["peak"] = eq["equity"].cummax()
        eq["dd"]   = (eq["equity"] / eq["peak"] - 1) * 100
        max_dd = eq["dd"].min()
        w(f"Máx. drawdown:  {max_dd:.2f}%")

        full = tr[tr["motivo"] != "Parcial"]
        wins   = full[full["pnl"] > 0]
        losses = full[full["pnl"] < 0]
        pf = wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) and losses["pnl"].sum() != 0 else float("inf")
        w("-" * 64)
        w(f"Trades:         {len(full)} cierres completos  (+{len(tr) - len(full)} parciales)")
        w(f"Win rate:       {len(wins) / len(full) * 100:.1f}%   |   Profit factor: {pf:.2f}")
        w(f"Esperanza:      ${full['pnl'].mean():+.3f}/trade   |   "
          f"Mejor ${full['pnl'].max():+.2f} / Peor ${full['pnl'].min():+.2f}")
        if "rr" in full:
            w(f"R múltiple:     promedio {full['rr'].mean():+.2f}R")

        w("-" * 64)
        w("Por estrategia:")
        for strat, g in full.groupby("estrategia"):
            gw = g[g["pnl"] > 0]
            w(f"  {strat}: {len(g):4d} trades | WR {len(gw)/len(g)*100:5.1f}% | "
              f"PnL ${g['pnl'].sum():+10.2f} | media ${g['pnl'].mean():+.3f}")
        w("Por símbolo:")
        for sym, g in full.groupby("sym"):
            gw = g[g["pnl"] > 0]
            w(f"  {sym}: {len(g):4d} trades | WR {len(gw)/len(g)*100:5.1f}% | PnL ${g['pnl'].sum():+10.2f}")
        w("Por motivo de cierre:")
        for mot, g in full.groupby("motivo"):
            w(f"  {mot:14s}: {len(g):4d} trades | PnL ${g['pnl'].sum():+10.2f}")
        w("=" * 64)

        tr.to_csv(Path(__file__).with_name("backtest_trades.csv"), index=False)
        eq.to_csv(Path(__file__).with_name("backtest_equity.csv"), index=False)
        text = "\n".join(lines)
        Path(__file__).with_name("backtest_report.txt").write_text(text)
        print("\n" + text)
        print("\nArchivos: backtest_trades.csv · backtest_equity.csv · backtest_report.txt")
        return text


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Backtester de Axiom (usa la lógica real del bot)")
    ap.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT"])
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--synthetic", action="store_true", help="validar con datos sintéticos (sin internet)")
    ap.add_argument("--sizing", choices=["fixed", "risk"], default="fixed")
    ap.add_argument("--risk-pct", type=float, default=1.0)
    ap.add_argument("--confirm-candles", type=int, default=1)
    ap.add_argument("--cooldown-win", type=int, default=3)
    ap.add_argument("--cooldown-loss", type=int, default=8)
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--score-minimo", type=float, default=None)
    ap.add_argument("--solo", choices=["MNV", "MOM"], default=None,
                    help="aislar una sola estrategia")
    ap.add_argument("--slow-trend", action="store_true",
                    help="activar el modo de tendencias lentas del bot (slow_trend=True)")
    ap.add_argument("--slow-adx-min", type=float, default=15.0)
    ap.add_argument("--slow-impulso-pct", type=float, default=0.05)
    ap.add_argument("--mom-rr", type=float, default=None,
                    help="override del TP de MOM en múltiplos de R (ej. 2.5)")
    args = ap.parse_args()

    print("Axiom Backtester")
    print("─" * 40)
    if args.synthetic:
        print("Modo sintético (validación del motor):")
        dfs = {s: synthetic_candles(4000, seed=11 + k) for k, s in enumerate(args.symbols)}
    else:
        print(f"Descargando {args.days} días de OKX:")
        dfs = {s: download_okx_candles(s, args.days) for s in args.symbols}
        # BTC siempre necesario para el filtro maestro
        if "BTC/USDT" not in dfs:
            dfs["BTC/USDT"] = download_okx_candles("BTC/USDT", args.days)

    overrides = {}
    if args.score_minimo is not None:
        overrides["score_minimo"] = args.score_minimo
    if args.slow_trend:
        overrides["slow_trend"] = True
        overrides["slow_adx_min"] = args.slow_adx_min
        overrides["slow_impulso_pct"] = args.slow_impulso_pct

    bt = AxiomBacktester(
        args.symbols, dfs, cfg_overrides=overrides, fee_rate=args.fee,
        confirm_candles=args.confirm_candles,
        cooldown_win=args.cooldown_win, cooldown_loss=args.cooldown_loss,
        sizing=args.sizing, risk_pct=args.risk_pct,
        solo=args.solo, mom_rr=args.mom_rr,
    )
    print(f"\nCorriendo {bt.n - WARMUP:,} velas…")
    bt.run()


if __name__ == "__main__":
    sys.exit(main())
