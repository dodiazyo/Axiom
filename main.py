from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from bot_engine import TrendBot

app = FastAPI(title="Axiom")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")

# ── Config por defecto ────────────────────────────────────────────────────────
DEFAULT_CFG = {
    "symbols":          [],
    "watch_symbols":    [],
    "trade_symbols":    [],
    "capital_usd":      25.0,
    "apalancamiento":   3,
    "symbol_leverage":  {},
    "symbol_capital":   {},
    "balance_inicial":  1000.0,
    "rr_minimo":        2.0,
    "score_minimo":     6.5,
    "vol_pullback_max": 0.85,
    "atr_sl_mult":      1.5,
    "atr_tp_mult":      4.0,
    "atr_trail_mult":   2.0,
    "breakeven_rr":     1.0,
    "parcial_rr":       1.5,
    "intervalo_segundos": 60,
    "cooldown_ciclos":  3,
    "cooldown_loss_ciclos": 8,
    "max_posiciones":   2,
    "modo_operador":    "AUTOMATICO",
    "execution_mode":   "SIMULADO",
    "margin_type":      "ISOLATED",
    "api_key":          "",
    "api_secret":       "",
    "binance_testnet":  True,
}

bot: Optional[TrendBot] = None
ws_clients: list[WebSocket] = []
MARKETS_CACHE: dict = {"ts": 0.0, "data": []}
STATE_FILE = Path(__file__).with_name("runtime_state.json")


def _normalize_symbols(config: dict) -> dict:
    watch = [s.strip().upper() for s in config.get("watch_symbols", []) if s and s.strip()]
    trade = [s.strip().upper() for s in config.get("trade_symbols", []) if s and s.strip()]

    if not watch and config.get("symbols"):
        watch = [s.strip().upper() for s in config.get("symbols", []) if s and s.strip()]
    if not trade:
        trade = watch[:]

    all_symbols = []
    seen: set[str] = set()
    for sym in watch + trade:
        if sym not in seen:
            all_symbols.append(sym)
            seen.add(sym)

    config["watch_symbols"] = watch
    config["trade_symbols"] = trade
    config["symbols"] = all_symbols
    symbol_leverage = {}
    for sym, lev in (config.get("symbol_leverage", {}) or {}).items():
        try:
            symbol_leverage[str(sym).strip().upper()] = int(lev)
        except Exception:
            continue
    config["symbol_leverage"] = symbol_leverage
    symbol_capital = {}
    for sym, amount in (config.get("symbol_capital", {}) or {}).items():
        try:
            value = float(amount)
            if value > 0:
                symbol_capital[str(sym).strip().upper()] = value
        except Exception:
            continue
    config["symbol_capital"] = symbol_capital
    config["execution_mode"] = str(config.get("execution_mode", "SIMULADO")).upper()
    config["margin_type"] = "ISOLATED"
    config["modo_operador"] = str(config.get("modo_operador", "AUTOMATICO")).upper()
    return config


def _recover_empty_symbol_config(state: dict) -> dict:
    cfg = dict(state.get("cfg") or {})
    if cfg.get("symbols") or cfg.get("watch_symbols") or cfg.get("trade_symbols"):
        return _normalize_symbols(cfg)

    recovered = []
    seen: set[str] = set()
    for bucket in ("estados", "estados_mom"):
        values = state.get(bucket, {})
        if not isinstance(values, dict):
            continue
        for sym in values.keys():
            clean = str(sym).strip().upper()
            if clean and clean not in seen:
                recovered.append(clean)
                seen.add(clean)

    if recovered:
        cfg["symbols"] = recovered
        cfg["watch_symbols"] = recovered
        cfg["trade_symbols"] = recovered

    return _normalize_symbols(cfg)


def _preserve_runtime_secrets(config: dict) -> dict:
    """Keep credentials in memory when the UI posts blank masked fields."""
    if not bot:
        return config
    current_cfg = getattr(bot, "cfg", {}) or {}
    for key in ("api_key", "api_secret"):
        if not config.get(key) and current_cfg.get(key):
            config[key] = current_cfg[key]
    return config


def _preserve_runtime_symbols(config: dict) -> dict:
    """Ignore accidental empty symbol posts from the UI after reconnects/reloads."""
    if config.get("symbols") or config.get("watch_symbols") or config.get("trade_symbols"):
        return config

    candidates: list[dict] = []
    if bot:
        candidates.append(getattr(bot, "cfg", {}) or {})
        candidates.append({
            "symbols": list(getattr(bot, "_estados", {}).keys()),
            "watch_symbols": list(getattr(bot, "_estados", {}).keys()),
            "trade_symbols": list((getattr(bot, "cfg", {}) or {}).get("trade_symbols", [])),
        })

    saved = _load_runtime_state()
    if saved and saved.get("cfg"):
        candidates.append(saved["cfg"])

    for candidate in candidates:
        symbols = candidate.get("symbols") or candidate.get("watch_symbols") or candidate.get("trade_symbols")
        if symbols:
            config["symbols"] = list(symbols)
            config["watch_symbols"] = list(candidate.get("watch_symbols") or symbols)
            config["trade_symbols"] = list(candidate.get("trade_symbols") or symbols)
            return config

    return config


def _fetch_markets() -> list[str]:
    now = time.time()
    if MARKETS_CACHE["data"] and now - MARKETS_CACHE["ts"] < 300:
        return MARKETS_CACHE["data"]

    r = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=10)
    r.raise_for_status()
    payload = r.json()
    markets = []
    for item in payload.get("symbols", []):
        if item.get("status") != "TRADING":
            continue
        if item.get("quoteAsset") != "USDT":
            continue
        if item.get("contractType") != "PERPETUAL":
            continue
        symbol = item.get("symbol", "")
        if symbol.endswith("USDT"):
            markets.append(symbol.replace("USDT", "/USDT"))

    markets = sorted(set(markets))
    MARKETS_CACHE["ts"] = now
    MARKETS_CACHE["data"] = markets
    return markets


def _save_runtime_state():
    if not bot:
        return
    payload = bot.export_runtime_state()
    STATE_FILE.write_text(json.dumps(payload, ensure_ascii=True, indent=2))


def _load_runtime_state() -> Optional[dict]:
    if not STATE_FILE.exists():
        return None
    try:
        state = json.loads(STATE_FILE.read_text())
        if isinstance(state, dict) and state.get("cfg"):
            state["cfg"] = _recover_empty_symbol_config(state)
        return state
    except Exception:
        return None


async def broadcast(msg: dict):
    data = json.dumps(msg)
    for ws in list(ws_clients):
        try:
            await ws.send_text(data)
        except Exception:
            if ws in ws_clients:
                ws_clients.remove(ws)


async def broadcast_loop():
    while True:
        if bot:
            status = bot.get_status()
            _save_runtime_state()
            await broadcast({"type": "status", "data": status})
        await asyncio.sleep(1)


@app.on_event("startup")
async def startup():
    global bot
    saved = _load_runtime_state()
    if saved and saved.get("cfg"):
        bot = TrendBot(saved["cfg"], restore_state=saved)
        if saved.get("running"):
            bot.start()
    asyncio.create_task(broadcast_loop())


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    try:
        if bot:
            await ws.send_text(json.dumps({"type": "status", "data": bot.get_status()}))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if ws in ws_clients:
            ws_clients.remove(ws)


# ── Modelos ───────────────────────────────────────────────────────────────────

class ConfigIn(BaseModel):
    symbols:            list[str] = Field(default_factory=list)
    watch_symbols:      list[str] = Field(default_factory=list)
    trade_symbols:      list[str] = Field(default_factory=list)
    capital_usd:        float = Field(25.0,  ge=1.0)
    apalancamiento:     int   = Field(3,     ge=1, le=10)
    symbol_leverage:    dict[str, int] = Field(default_factory=dict)
    symbol_capital:     dict[str, float] = Field(default_factory=dict)
    balance_inicial:    float = Field(1000.0,ge=10.0)
    rr_minimo:          float = Field(2.0,   ge=0.5)
    score_minimo:       float = Field(6.5,   ge=1.0)
    vol_pullback_max:   float = Field(0.85,  ge=0.1, le=2.0)
    atr_sl_mult:        float = Field(1.5,   ge=0.5)
    atr_tp_mult:        float = Field(4.0,   ge=1.0)
    atr_trail_mult:     float = Field(2.0,   ge=0.5)
    breakeven_rr:       float = Field(1.0,   ge=0.5)
    parcial_rr:         float = Field(1.5,   ge=0.5)
    intervalo_segundos: int   = Field(60,    ge=15)
    cooldown_ciclos:    int   = Field(3,     ge=0, le=20)
    cooldown_loss_ciclos: int = Field(8,     ge=0, le=50)
    max_posiciones:     int   = Field(2,     ge=1, le=20)
    modo_operador:      str   = Field("AUTOMATICO")
    execution_mode:     str   = Field("SIMULADO")
    margin_type:        str   = Field("ISOLATED")
    api_key:            str   = Field("")
    api_secret:         str   = Field("")
    binance_testnet:    bool  = Field(True)


class BinanceCredsIn(BaseModel):
    execution_mode:     str   = Field("SIMULADO")
    margin_type:        str   = Field("ISOLATED")
    api_key:            str   = Field("")
    api_secret:         str   = Field("")
    binance_testnet:    bool  = Field(True)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    if not bot:
        return {"running": False, "symbols": {}, "configured_symbols": []}
    return bot.get_status()


@app.get("/api/logs")
def get_logs():
    if not bot:
        return []
    return bot.get_logs(100)


@app.get("/api/alerts")
def get_alerts():
    if not bot:
        return []
    with bot._lock:
        return list(bot._alerts)


@app.get("/api/historial")
def get_historial():
    if not bot:
        saved = _load_runtime_state()
        if saved and saved.get("historial"):
            return list(reversed(saved["historial"][-500:]))
        return []
    return bot.get_historial()


@app.get("/api/pattern-memory")
def get_pattern_memory():
    if not bot:
        return {"patterns": {}, "pending_count": 0, "total_observations": 0, "recent": []}
    return bot.get_pattern_stats()


@app.post("/api/start")
def start_bot(cfg: ConfigIn):
    global bot
    config = _preserve_runtime_secrets(_normalize_symbols(_preserve_runtime_symbols(cfg.model_dump())))
    restore_state = None
    if bot:
        current = bot.export_runtime_state()
        restore_state = {
            **current,
            "cfg": config,
        }
        bot.stop()
        time.sleep(0.5)
    else:
        saved = _load_runtime_state()
        if saved:
            restore_state = {
                **saved,
                "cfg": config,
            }
    bot = TrendBot(config, restore_state=restore_state)
    bot.start()
    _save_runtime_state()
    return {"ok": True, "msg": "Axiom iniciado"}


@app.post("/api/stop")
def stop_bot():
    global bot
    if bot:
        bot.stop()
        _save_runtime_state()
    return {"ok": True, "msg": "Axiom detenido"}


@app.get("/api/config")
def get_config():
    cfg = None
    if bot:
        cfg = bot.cfg
    else:
        saved = _load_runtime_state()
        if saved and saved.get("cfg"):
            cfg = saved["cfg"]
    if cfg is None:
        cfg = DEFAULT_CFG
    payload = dict(cfg)
    api_key = str(cfg.get("api_key", "") or "")
    api_secret = str(cfg.get("api_secret", "") or "")
    payload["api_credentials_saved"] = bool(api_key and api_secret)
    payload["api_key_hint"] = f"****{api_key[-4:]}" if api_key else ""
    payload["api_key"] = ""
    payload["api_secret"] = ""
    return payload


@app.post("/api/config")
def update_config(cfg: ConfigIn):
    global bot
    config = _preserve_runtime_secrets(_normalize_symbols(_preserve_runtime_symbols(cfg.model_dump())))
    if bot:
        bot.update_config(config)
    else:
        saved = _load_runtime_state() or {}
        saved["cfg"] = config
        STATE_FILE.write_text(json.dumps(saved, ensure_ascii=True, indent=2))
        return {"ok": True, "msg": "Configuracion guardada"}
    _save_runtime_state()
    return {"ok": True, "msg": "Configuracion actualizada"}


@app.get("/api/markets")
def get_markets():
    return {"markets": _fetch_markets()}


@app.post("/api/test-credentials")
def test_credentials(creds: BinanceCredsIn):
    config = _normalize_symbols({
        **DEFAULT_CFG,
        **creds.model_dump(),
        "symbols": [],
        "watch_symbols": [],
        "trade_symbols": [],
    })
    probe = TrendBot(config)
    return probe.test_connection()
