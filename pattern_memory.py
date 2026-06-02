"""
Pattern Memory — detecta patrones de velas y registra si la predicción fue correcta.
Después de RESOLVE_AFTER velas de 1h, verifica si el precio se movió en la dirección esperada.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Detección de patrones
# ─────────────────────────────────────────────────────────────────────────────

def detect_patterns(df: pd.DataFrame) -> list[tuple[str, str]]:
    """
    Detecta patrones en las últimas velas cerradas.
    Retorna lista de (nombre_patron, dirección) donde dirección es BULLISH/BEARISH/NEUTRAL.
    """
    patterns: list[tuple[str, str]] = []
    if len(df) < 3:
        return patterns

    c  = df.iloc[-1]
    p  = df.iloc[-2]
    pp = df.iloc[-3]

    def _vals(row):
        return float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])

    o,  h,  l,  cl  = _vals(c)
    po, ph, pl, pcl = _vals(p)
    ppo,pph,ppl,ppcl= _vals(pp)

    body    = abs(cl - o)
    rng     = h - l
    if rng < 1e-12:
        return patterns

    upper_wick = h - max(o, cl)
    lower_wick = min(o, cl) - l
    body_pct   = body / rng
    is_bull    = cl > o
    is_bear    = cl < o

    p_body  = abs(pcl - po)
    p_rng   = ph - pl if ph - pl > 1e-12 else 1e-12
    p_bull  = pcl > po
    p_bear  = pcl < po

    pp_bull = ppcl > ppo
    pp_bear = ppcl < ppo

    # ── 1 vela ──────────────────────────────────────────────────────────────

    # Doji: cuerpo muy pequeño
    if body_pct < 0.08:
        patterns.append(("Doji", "NEUTRAL"))

    # Hammer: mecha inferior ≥2× cuerpo, mecha superior pequeña, tras vela bajista
    if body > 0 and lower_wick >= 2.0 * body and upper_wick <= body * 0.5 and p_bear:
        patterns.append(("Hammer", "BULLISH"))

    # Inverted Hammer: mecha superior ≥2× cuerpo, tras vela bajista
    if body > 0 and upper_wick >= 2.0 * body and lower_wick <= body * 0.5 and p_bear:
        patterns.append(("Inverted Hammer", "BULLISH"))

    # Shooting Star: mecha superior ≥2× cuerpo, tras vela alcista
    if body > 0 and upper_wick >= 2.0 * body and lower_wick <= body * 0.5 and p_bull:
        patterns.append(("Shooting Star", "BEARISH"))

    # Hanging Man: mecha inferior ≥2× cuerpo, tras vela alcista
    if body > 0 and lower_wick >= 2.0 * body and upper_wick <= body * 0.5 and p_bull:
        patterns.append(("Hanging Man", "BEARISH"))

    # Bullish Marubozu: cuerpo >80%, casi sin mechas, vela alcista
    if is_bull and body_pct > 0.80 and upper_wick < body * 0.1 and lower_wick < body * 0.1:
        patterns.append(("Bullish Marubozu", "BULLISH"))

    # Bearish Marubozu
    if is_bear and body_pct > 0.80 and upper_wick < body * 0.1 and lower_wick < body * 0.1:
        patterns.append(("Bearish Marubozu", "BEARISH"))

    # Pin Bar alcista: mecha inferior muy larga (≥3× cuerpo), cuerpo en tercio superior del rango
    if body > 0 and lower_wick >= 3.0 * body and (min(o, cl) - l) / rng > 0.5:
        patterns.append(("Pin Bar Alcista", "BULLISH"))

    # Pin Bar bajista: mecha superior muy larga
    if body > 0 and upper_wick >= 3.0 * body and (h - max(o, cl)) / rng > 0.5:
        patterns.append(("Pin Bar Bajista", "BEARISH"))

    # ── 2 velas ──────────────────────────────────────────────────────────────

    # Bullish Engulfing: vela anterior bajista, actual alcista y envuelve
    if p_bear and is_bull and o <= pcl and cl >= po:
        patterns.append(("Bullish Engulfing", "BULLISH"))

    # Bearish Engulfing
    if p_bull and is_bear and o >= pcl and cl <= po:
        patterns.append(("Bearish Engulfing", "BEARISH"))

    # Tweezer Bottom: mismo mínimo ±0.05% del rango, primera bajista segunda alcista
    if p_bear and is_bull and abs(l - pl) / max(rng, p_rng) < 0.05:
        patterns.append(("Tweezer Bottom", "BULLISH"))

    # Tweezer Top
    if p_bull and is_bear and abs(h - ph) / max(rng, p_rng) < 0.05:
        patterns.append(("Tweezer Top", "BEARISH"))

    # Piercing Pattern: vela bajista fuerte, luego alcista que cierra >50% del cuerpo anterior
    if p_bear and is_bull and o < pcl and cl > (po + pcl) / 2 and p_body / p_rng > 0.5:
        patterns.append(("Piercing Pattern", "BULLISH"))

    # Dark Cloud Cover: vela alcista fuerte, luego bajista que cierra <50%
    if p_bull and is_bear and o > pcl and cl < (po + pcl) / 2 and p_body / p_rng > 0.5:
        patterns.append(("Dark Cloud Cover", "BEARISH"))

    # ── 3 velas ──────────────────────────────────────────────────────────────

    # Three White Soldiers: 3 alcistas consecutivas, cada una cierra más alto
    if pp_bull and p_bull and is_bull and ppcl < pcl < cl and o > po and po > ppo:
        patterns.append(("Three White Soldiers", "BULLISH"))

    # Three Black Crows
    if pp_bear and p_bear and is_bear and ppcl > pcl > cl and o < po and po < ppo:
        patterns.append(("Three Black Crows", "BEARISH"))

    # Morning Star: bajista fuerte + pequeño cuerpo + alcista
    pp_body_pct = abs(ppcl - ppo) / max(pph - ppl, 1e-12)
    if (pp_bear and pp_body_pct > 0.5
            and p_body / p_rng < 0.30
            and is_bull and cl > (ppo + ppcl) / 2):
        patterns.append(("Morning Star", "BULLISH"))

    # Evening Star
    if (pp_bull and pp_body_pct > 0.5
            and p_body / p_rng < 0.30
            and is_bear and cl < (ppo + ppcl) / 2):
        patterns.append(("Evening Star", "BEARISH"))

    return patterns


# ─────────────────────────────────────────────────────────────────────────────
# Memoria de patrones
# ─────────────────────────────────────────────────────────────────────────────

class PatternMemory:
    RESOLVE_AFTER_H = 5       # horas hasta resolver (velas de 1h)
    MIN_MOVE_PCT    = 0.30    # movimiento mínimo para contar como "correcto"
    MAX_OBS         = 5_000
    MAX_PENDING     = 500

    def __init__(self, path: str = "pattern_memory.json"):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text())
                raw.setdefault("patterns", {})
                raw.setdefault("symbols", {})
                raw.setdefault("pending", [])
                raw.setdefault("observations", [])
                return raw
            except Exception:
                pass
        return {"patterns": {}, "symbols": {}, "pending": [], "observations": []}

    def save(self):
        try:
            with self._lock:
                snapshot = json.dumps(self._data, ensure_ascii=False, indent=2)
            self._path.write_text(snapshot)
        except Exception:
            pass

    def record(self, symbol: str, pattern: str, direction: str, price: float, candle_ts_sec: float):
        """Registra un patrón detectado en la vela cuyo timestamp Unix es candle_ts_sec."""
        with self._lock:
            for item in self._data["pending"]:
                if (item["symbol"] == symbol
                        and item["pattern"] == pattern
                        and abs(item["candle_ts"] - candle_ts_sec) < 60):
                    return  # ya registrado
            entry = {
                "symbol":      symbol,
                "pattern":     pattern,
                "direction":   direction,
                "entry_price": round(price, 10),
                "candle_ts":   candle_ts_sec,
                "resolve_ts":  candle_ts_sec + self.RESOLVE_AFTER_H * 3600,
            }
            self._data["pending"].append(entry)
            if len(self._data["pending"]) > self.MAX_PENDING:
                self._data["pending"] = self._data["pending"][-self.MAX_PENDING:]

    def update(self, symbol: str, current_price: float, current_ts_sec: float):
        """Verifica patrones pendientes de este símbolo y resuelve los que ya cumplieron el tiempo."""
        resolved: list[int] = []
        with self._lock:
            for i, item in enumerate(self._data["pending"]):
                if item["symbol"] != symbol:
                    continue
                if current_ts_sec < item["resolve_ts"]:
                    continue

                entry     = item["entry_price"]
                change    = (current_price - entry) / entry * 100
                direction = item["direction"]

                if direction == "BULLISH":
                    correct = change >= self.MIN_MOVE_PCT
                elif direction == "BEARISH":
                    correct = change <= -self.MIN_MOVE_PCT
                else:
                    correct = abs(change) < self.MIN_MOVE_PCT

                obs = {
                    "symbol":      symbol,
                    "pattern":     item["pattern"],
                    "direction":   direction,
                    "entry_price": round(entry, 10),
                    "exit_price":  round(current_price, 10),
                    "change_pct":  round(change, 3),
                    "correct":     correct,
                    "ts":          datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
                }
                self._data["observations"].append(obs)
                if len(self._data["observations"]) > self.MAX_OBS:
                    self._data["observations"] = self._data["observations"][-self.MAX_OBS:]

                # Actualizar stats por patrón
                pk = item["pattern"]
                if pk not in self._data["patterns"]:
                    self._data["patterns"][pk] = {
                        "total": 0, "correct": 0,
                        "bullish": 0, "bearish": 0, "neutral": 0
                    }
                self._data["patterns"][pk]["total"]   += 1
                self._data["patterns"][pk]["correct"]  += int(correct)
                dk = direction.lower()
                self._data["patterns"][pk][dk] = self._data["patterns"][pk].get(dk, 0) + 1

                # Actualizar stats por símbolo
                if symbol not in self._data["symbols"]:
                    self._data["symbols"][symbol] = {}
                if pk not in self._data["symbols"][symbol]:
                    self._data["symbols"][symbol][pk] = {"total": 0, "correct": 0}
                self._data["symbols"][symbol][pk]["total"]   += 1
                self._data["symbols"][symbol][pk]["correct"] += int(correct)

                resolved.append(i)

            for i in sorted(resolved, reverse=True):
                self._data["pending"].pop(i)

    def get_pattern_confidence(self, symbol: str, direction: str, min_samples: int = 8) -> tuple[float | None, int]:
        """
        Retorna (win_rate 0-100, n_muestras) para el símbolo+dirección dados.
        Si hay menos de min_samples observaciones, retorna (None, n) — sin opinión.
        Combina stats del símbolo específico (peso 2) con stats globales (peso 1).
        """
        dir_key = direction.lower()
        with self._lock:
            # Stats por símbolo
            sym_stats = self._data["symbols"].get(symbol, {})
            sym_total = sym_correct = 0
            for pat_name, ps in sym_stats.items():
                # Solo contar patrones de esta dirección
                global_pat = self._data["patterns"].get(pat_name, {})
                pat_dir = "bullish" if direction == "BULLISH" else "bearish" if direction == "BEARISH" else "neutral"
                if global_pat.get(pat_dir, 0) > 0:
                    sym_total   += ps["total"]
                    sym_correct += ps["correct"]

            # Stats globales por dirección
            g_total = g_correct = 0
            for pat_name, gs in self._data["patterns"].items():
                pat_dir_count = gs.get(dir_key, 0)
                if pat_dir_count == 0:
                    continue
                ratio = pat_dir_count / max(gs["total"], 1)
                g_total   += gs["total"]
                g_correct += gs["correct"]

            total = sym_total * 2 + g_total
            correct = sym_correct * 2 + g_correct

            if total < min_samples:
                return None, total

            return round(correct / total * 100, 1), total

    def get_stats(self) -> dict:
        with self._lock:
            patterns_out = {}
            for name, s in self._data["patterns"].items():
                total   = s["total"]
                correct = s["correct"]
                patterns_out[name] = {
                    "total":    total,
                    "correct":  correct,
                    "win_rate": round(correct / total * 100) if total else 0,
                    "bullish":  s.get("bullish", 0),
                    "bearish":  s.get("bearish", 0),
                    "neutral":  s.get("neutral", 0),
                }
            recent = list(reversed(self._data["observations"][-100:]))
            return {
                "patterns":           patterns_out,
                "pending_count":      len(self._data["pending"]),
                "total_observations": len(self._data["observations"]),
                "recent":             recent,
            }
