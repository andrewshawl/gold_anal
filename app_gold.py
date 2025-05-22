#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyzer_xauusd_m1.py · versión 25-may-2025 (patch-bugs 1-4)
===========================================================

✦ Cambios clave
────────────────
1. DD intermedia ahora se exporta en columnas dd_at_<umbral>              (Bug #1)
2. Off-by-one cuando no hay step_plan corregido (niveles extra = max-1)   (Bug #2)
3. count_rollovers() evita tz_convert redundantes                         (Bug #3)
4. Validación STOP-loss: se fuerza a negativo y se avisa al usuario       (Bug #4)

El resto reproduce el script original al 100 %.
"""

from __future__ import annotations
import io, zipfile, tempfile, calendar, time, math, warnings, logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytz
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

# ─────────────────────────── logging básico ────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ──────────────────────── constantes generales ─────────────────────────────
TZ_CDMX       = pytz.timezone("America/Mexico_City")
CONTRACT_SIZE = 100      # onzas por lote

# ════════════════ utilidades de gaps / sesiones ═════════════════════════════
def classify_gap(prev_ts: pd.Timestamp, curr_ts: pd.Timestamp,
                 daily_min: int = 45, weekend_min: int = 60) -> str:
    delta = (curr_ts - prev_ts).total_seconds() / 60
    if delta < daily_min:
        return "no_gap"
    if prev_ts.weekday() == 4 and curr_ts.weekday() in {6, 0} and delta >= weekend_min:
        return "weekend_gap"
    if 15 <= prev_ts.hour <= 18:
        return "daily_break"
    return "gap"

def gap_priority(gt: str) -> int:
    return {"no_gap": 0, "gap": 1, "daily_break": 2, "weekend_gap": 3}[gt]

def scan_gap_ext(df: pd.DataFrame, a: int, b: int) -> tuple[str, float, float]:
    """Escanea el tramo [a,b] para hallar el peor gap."""
    ts = df.datetime_cdmx
    worst, worst_delta, worst_pts = "no_gap", 0.0, 0.0
    for i in range(a + 1, b + 1):
        gtype = classify_gap(ts.iat[i - 1], ts.iat[i])
        if gtype == "no_gap":
            continue
        delta = (ts.iat[i] - ts.iat[i - 1]).total_seconds() / 60
        gap_pts = abs(df.Open.iat[i] - df.Close.iat[i - 1])
        if (gap_priority(gtype) > gap_priority(worst)) or (
            gtype == worst and gap_pts > worst_pts
        ):
            worst, worst_delta, worst_pts = gtype, delta, gap_pts
    return worst, worst_delta, worst_pts

def session_label(hr: int) -> str:
    if 0 <= hr < 2:
        return "Londres"
    if 2 <= hr < 7:
        return "NY-temp"
    if 7 <= hr < 12:
        return "NY-tarde"
    if 12 <= hr < 17:
        return "Asia-mat"
    return "Asia-noche"

def count_rollovers(start: pd.Timestamp, end: pd.Timestamp, rollover_hr: int) -> int:
    """
    Nº de roll-overs (hora fija CDMX) entre dos timestamps YA en TZ_CDMX.
    """
    if end <= start:
        return 0
    first = start.normalize() + pd.Timedelta(hours=rollover_hr)
    if start >= first:
        first += pd.Timedelta(days=1)
    if end < first:
        return 0
    return (end - first).days + 1

# ═══════════════ helpers plan de pasos (“segmento”) ════════════════════════
def total_levels(plan: List[Tuple[int, float]]) -> int:
    return sum(n for n, _ in plan)

def compute_total_lots(
    lot0: float,
    q0: float,
    step_plan: List[Tuple[int, float]],
    max_steps: int,
) -> float:
    add_levels = total_levels(step_plan) if step_plan else max_steps - 1
    if add_levels <= 0:
        return lot0

    if step_plan:
        seg_idx = 0
        remaining, q = step_plan[0]
    else:
        seg_idx, remaining, q = None, add_levels, q0

    total = lot = lot0
    for _ in range(add_levels):
        lot *= q
        total += lot
        remaining -= 1
        if remaining == 0 and step_plan and seg_idx + 1 < len(step_plan):
            seg_idx += 1
            remaining, q = step_plan[seg_idx]
    return total

# ═════════════ estilo Streamlit y cabecera ═════════════════════════════════
st.set_page_config("Analyzer XAUUSD M1", layout="wide")
st.markdown(
    """
    <style>
      .title{font-size:3.5rem;font-weight:bold;text-align:center;color:#1abc9c;margin:0.2rem;}
      .subtle{text-align:center;color:#666;margin:-1rem 0 2rem;}
      th{text-align:center;}
    </style>""",
    unsafe_allow_html=True,
)
st.markdown('<div class="title">Analyzer XAUUSD M1</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtle">Análisis técnico · Monte Carlo</div>', unsafe_allow_html=True
)

# ═════════════ carga y preparación del archivo ════════════════════════════
@st.cache_data(show_spinner=False)
def load_and_prepare(file) -> pd.DataFrame:
    head = file.read(256)
    file.seek(0)
    is_csv = head.split(b"\n", 1)[0].lstrip().startswith(b"<DATE>")

    if is_csv:
        df = pd.read_csv(
            file,
            sep="\t",
            usecols=[
                "<DATE>", "<TIME>", "<OPEN>", "<HIGH>",
                "<LOW>", "<CLOSE>", "<TICKVOL>",
            ],
        ).rename(
            columns={
                "<DATE>": "Date",  "<TIME>": "Time",
                "<OPEN>": "Open",  "<HIGH>": "High",
                "<LOW>":  "Low",   "<CLOSE>": "Close",
                "<TICKVOL>": "Volume",
            }
        )
        dt_utc = pd.to_datetime(
            df["Date"] + " " + df["Time"], format="%Y.%m.%d %H:%M:%S", utc=True
        )
    else:
        cols = ["Symbol", "Date", "Time", "Open", "High",
                "Low", "Close", "Volume"]
        df = pd.read_csv(
            file, names=cols, header=None,
            delim_whitespace=True, skipinitialspace=True
        )
        dt_utc = pd.to_datetime(
            df["Date"] + df["Time"], format="%Y%m%d%H%M%S", utc=True
        )

    dt_cdmx = dt_utc.dt.tz_convert(TZ_CDMX)
    df = (
        df.assign(
            datetime_utc=dt_utc,
            datetime_cdmx=dt_cdmx,
            hour_cdmx=dt_cdmx.dt.hour,
            dow=dt_cdmx.dt.dayofweek,
            range_pts=df["High"] - df["Low"],
        )
        .dropna(subset=["datetime_utc"])
        .sort_values("datetime_utc")[[
            "datetime_utc", "datetime_cdmx", "hour_cdmx", "dow",
            "Open", "High", "Low", "Close", "Volume", "range_pts",
        ]]
    )
    return df

# ═════════════ plan parser ════════════════════════════════════════════════
def parse_plan(text: str) -> List[Tuple[int, float]]:
    out = []
    for tk in text.split(","):
        if not tk.strip():
            continue
        n, fac = tk.split(":", 1)
        out.append((int(float(n)), float(fac)))
    return out

# ═════════════ funciones cacheadas de análisis básico ═════════════════════
@st.cache_data(show_spinner=False)
def count_candles(df, thr):
    tot = len(df)
    return pd.DataFrame(
        [{"threshold": t, "count": int((df.range_pts >= t).sum())} for t in thr]
    ).assign(pct_total=lambda d: d["count"] / tot * 100)

@st.cache_data(show_spinner=False)
def count_by_hour(df, thr):
    base = df.groupby("hour_cdmx").size()
    rec = []
    for t in thr:
        sel = df[df.range_pts >= t].groupby("hour_cdmx").size()
        for h in range(24):
            n, tot = int(sel.get(h, 0)), int(base.get(h, 0))
            rec.append(
                dict(
                    threshold=t,
                    hour_cdmx=h,
                    count=n,
                    pct_in_hour=round(n / tot * 100, 3) if tot else np.nan,
                )
            )
    return pd.DataFrame(rec)

@st.cache_data(show_spinner=False)
def list_big_candles(df, thr):
    base = df[df.range_pts >= min(thr)][
        ["datetime_utc", "datetime_cdmx", "range_pts"]
    ]
    base["exceeded_thresholds"] = base["range_pts"].apply(
        lambda x: ";".join(str(t) for t in thr if x >= t)
    )
    return base

@st.cache_data(show_spinner=False)
def find_streaks(df, thr):
    rec, arr = [], df.range_pts.to_numpy()
    for t in thr:
        mask = arr >= t
        grp = np.cumsum(np.concatenate(([0], np.diff(mask) != 0)))
        for g in np.unique(grp[mask]):
            idx = np.where(grp == g)[0]
            rec.append(
                dict(
                    threshold=t,
                    start_cdmx=df.datetime_cdmx.iat[idx[0]],
                    end_cdmx=df.datetime_cdmx.iat[idx[-1]],
                    length=len(idx),
                )
            )
    return pd.DataFrame(rec)

@st.cache_data(show_spinner=False)
def detect_gaps(df, mins=45):
    ts = df.datetime_cdmx
    out = []
    for i in range(1, len(ts)):
        g = classify_gap(ts.iat[i - 1], ts.iat[i], mins)
        if g == "no_gap":
            continue
        delta = (ts.iat[i] - ts.iat[i - 1]).total_seconds() / 60
        out.append(
            dict(
                prev_cdmx=ts.iat[i - 1],
                next_cdmx=ts.iat[i],
                delta_min=delta,
                abs_gap=abs(df.Open.iat[i] - df.Close.iat[i - 1]),
                gap_type=g,
            )
        )
    return pd.DataFrame(out)

# ═════════════ figuras matplotlib (compactas) ═════════════════════════════
def fig_counts(d):
    fig, ax = plt.subplots()
    ax.bar(d.threshold, d["count"])
    ax.set(xlabel="Umbral (pts)", ylabel="# velas ≥ umbral", title="Conteo de velas")
    fig.tight_layout()
    return fig

def fig_hour_heat(d):
    piv = (
        d.pivot(index="threshold", columns="hour_cdmx", values="pct_in_hour")
        .sort_index(ascending=False)
        .reindex(columns=range(24), fill_value=np.nan)
    )
    fig, ax = plt.subplots()
    im = ax.imshow(piv, aspect="auto")
    ax.set_xticks(range(24)); ax.set_xticklabels(range(24))
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.set(xlabel="Hora CDMX", ylabel="Umbral (pts)", title="% velas ≥ umbral por hora")
    fig.colorbar(im, ax=ax, label="%")
    fig.tight_layout()
    return fig

def fig_hist(df):
    bins = np.arange(0, df.range_pts.max() + 0.25, 0.25)
    fig, ax = plt.subplots()
    ax.hist(df.range_pts, bins=bins, log=True)
    ax.set(xlabel="Rango (pts)", ylabel="Frecuencia (log)", title="Histograma de rangos")
    fig.tight_layout()
    return fig

def fig_hist_tail(df, thr):
    data = df[df.range_pts >= thr].range_pts
    fig, ax = plt.subplots()
    if data.empty:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
        ax.set_xlim(0, 1)
    else:
        ax.hist(data, bins=np.arange(thr, data.max() + 0.25, 0.25))
    ax.set(xlabel="Rango (pts)", ylabel="Frecuencia", title=f"Histograma ≥ {thr} pts")
    fig.tight_layout()
    return fig

def fig_session_box(df):
    df2 = df.copy()
    df2["session"] = df2.hour_cdmx.apply(session_label)
    fig, ax = plt.subplots()
    df2.boxplot(column="range_pts", by="session", ax=ax, grid=False)
    plt.suptitle("")
    ax.set_ylabel("Rango (pts)")
    ax.set_title("Rangos por sesión")
    fig.tight_layout()
    return fig

def fig_dow_hour_heat(df):
    piv = (
        df.groupby(["dow", "hour_cdmx"]).range_pts.mean()
        .unstack(fill_value=np.nan)
        .reindex(index=range(7))
    )
    fig, ax = plt.subplots()
    im = ax.imshow(piv, aspect="auto")
    ax.set_xticks(range(24)); ax.set_xticklabels(range(24))
    ax.set_yticks(range(7)); ax.set_yticklabels([calendar.day_abbr[d] for d in range(7)])
    ax.set(xlabel="Hora CDMX", ylabel="Día semana", title="Heat-map rango medio")
    fig.colorbar(im, ax=ax, label="pts")
    fig.tight_layout()
    return fig

def fig_gaps_scatter(dg):
    fig, ax = plt.subplots()
    ax.scatter(dg.delta_min, dg.abs_gap, c="tab:blue")
    ax.set(xlabel="Duración gap (min)", ylabel="Tamaño gap (pts)",
           title="Gaps: tamaño vs duración")
    fig.tight_layout()
    return fig

def fig_gaps_top(dg):
    g = dg.nlargest(10, "abs_gap").sort_values("abs_gap")
    labels = [d.strftime("%Y-%m-%d\n%H:%M") for d in g.prev_cdmx]
    fig, ax = plt.subplots()
    ax.barh(labels, g.abs_gap, color="tab:blue")
    ax.set(xlabel="Tamaño gap (pts)", title="Top-10 gaps más grandes")
    fig.tight_layout()
    return fig

# ═══════════════ MONTE CARLO (plan segmental) ══════════════════════════════
def sample_start(df, max_lv, n, lookahead=10) -> np.ndarray:
    hist = len(df)
    need = max_lv * lookahead
    if hist - need <= 0:
        lookahead = max(hist // max_lv - 1, 1)
        warnings.warn("Histórico corto: lookahead reducido.")
    lim = hist - max_lv * lookahead
    return np.random.choice(max(lim, 1), size=n, replace=n > lim)

def calc_eq(price, entry, lots, side):
    return (
        sum(l * (price - e if side == "BUY" else e - price) for l, e in zip(lots, entry))
        * CONTRACT_SIZE
    )

def add_lv(low, high, side, last, d):
    if side == "BUY" and low <= last - d:
        return math.floor((last - low) / d)
    if side == "SELL" and high >= last + d:
        return math.floor((high - last) / d)
    return 0

def simulate(
    df,
    idx0,
    *,
    side,
    distance,
    lot0,
    q0,
    tp_offset,
    stop_loss,
    max_lv,
    step_plan,
    dd_inter,
    swap_long,
    swap_short,
    rollover_hr,
):
    # ── segmentos (Bug #2) ────────────────────────────────────────────────
    if step_plan:
        segments = step_plan
    else:
        segments = [(max_lv - 1, q0)]  # niveles extra

    seg_idx = 0
    remaining_seg, current_q = segments[0] if segments else (0, q0)

    closes, lows, highs = df.Close.values, df.Low.values, df.High.values
    entry = [df.Open.iat[idx0]]
    lots  = [lot0]
    pmp   = entry[0]
    last  = entry[0]
    steps = 1
    idx   = idx0

    eq   = calc_eq(entry[0], entry, lots, side)
    peak = eq
    max_dd = 0.0
    rec  = {thr: None for thr in dd_inter}
    adding = True

    hard = total_levels(step_plan) if step_plan else max_lv - 1

    while idx < len(df) - 1:
        idx += 1
        price, low, high = closes[idx], lows[idx], highs[idx]

        # ── draw-down ─────────────────────────────────────────────────────
        eq = calc_eq(price, entry, lots, side)
        peak = max(peak, eq)
        max_dd = min(max_dd, eq - peak)
        dd_usd = -max_dd
        for thr in dd_inter:
            if rec[thr] is None and dd_usd >= thr:
                rec[thr] = dd_usd

        # ── STOP global ──────────────────────────────────────────────────
        if max_dd <= stop_loss:
            break_ts = df.datetime_cdmx.iat[idx]
            nights = count_rollovers(df.datetime_cdmx.iat[idx0], break_ts, rollover_hr)
            swap_rate = swap_long if side == "BUY" else swap_short
            swap_usd = nights * sum(lots) * swap_rate
            gtype, gmin, gap_pts = scan_gap_ext(df, idx0, idx)
            result = dict(
                broke=True,
                dd_pico=max_dd,
                steps_used=steps,
                dur_min=idx - idx0,
                side=side,
                start_ts=df.datetime_cdmx.iat[idx0],
                break_ts=break_ts,
                stuck_session=session_label(break_ts.hour),
                exit_session=None,
                gap_type=gtype,
                gap_min=gmin,
                gap_abs_pts=gap_pts,
                swap_usd=swap_usd,
                exit_pnl_usd=eq + swap_usd,
            )
            # exporta DD intermedia (Bug #1)
            for thr in dd_inter:
                result[f"dd_at_{thr}"] = rec[thr]
            return result

        # ── take-profit ───────────────────────────────────────────────────
        if (side == "BUY" and price >= pmp + tp_offset) or (
            side == "SELL" and price <= pmp - tp_offset
        ):
            break

        # ── añadir niveles ───────────────────────────────────────────────
        if adding and hard > 0:
            n_new = add_lv(low, high, side, last, distance)
            n_new = max(0, min(n_new, hard - (steps - 1)))

            for _ in range(n_new):
                if remaining_seg == 0 and seg_idx + 1 < len(segments):
                    seg_idx += 1
                    remaining_seg, current_q = segments[seg_idx]

                last = last - distance if side == "BUY" else last + distance
                entry.append(last)
                lots.append(lots[-1] * current_q)
                steps += 1
                remaining_seg = max(remaining_seg - 1, 0)
                pmp = sum(l * e for l, e in zip(lots, entry)) / sum(lots)

                if steps - 1 >= hard:
                    adding = False

    # ── cierre normal ─────────────────────────────────────────────────────
    end_ts = df.datetime_cdmx.iat[idx]
    nights = count_rollovers(df.datetime_cdmx.iat[idx0], end_ts, rollover_hr)
    swap_rate = swap_long if side == "BUY" else swap_short
    swap_usd = nights * sum(lots) * swap_rate
    gtype, gmin, gap_pts = scan_gap_ext(df, idx0, idx)
    result = dict(
        broke=False,
        dd_pico=max_dd,
        steps_used=steps,
        dur_min=idx - idx0,
        side=side,
        start_ts=df.datetime_cdmx.iat[idx0],
        exit_ts=end_ts,
        stuck_session=None,
        exit_session=session_label(end_ts.hour),
        gap_type=gtype,
        gap_min=gmin,
        gap_abs_pts=gap_pts,
        swap_usd=swap_usd,
        exit_pnl_usd=calc_eq(closes[idx], entry, lots, side) + swap_usd,
    )
    for thr in dd_inter:
        result[f"dd_at_{thr}"] = rec[thr]
    return result

def run_mc(
    df,
    *,
    distance,
    lot0,
    q0,
    tp_offset,
    stop_loss,
    max_steps,
    n_samples,
    step_plan,
    dd_inter,
    swap_long,
    swap_short,
    rollover_hr,
):
    levels_extra = total_levels(step_plan) if step_plan else max_steps - 1
    idxs = sample_start(df, levels_extra, n_samples)

    out, prog = [], st.progress(0.0)
    next_up = max(n_samples // 20, 1)
    for i, base in enumerate(idxs, 1):
        for side in ("BUY", "SELL"):
            out.append(
                simulate(
                    df,
                    base,
                    side=side,
                    distance=distance,
                    lot0=lot0,
                    q0=q0,
                    tp_offset=tp_offset,
                    stop_loss=stop_loss,
                    max_lv=max_steps,
                    step_plan=step_plan,
                    dd_inter=dd_inter,
                    swap_long=swap_long,
                    swap_short=swap_short,
                    rollover_hr=rollover_hr,
                )
            )
        if i % next_up == 0 or i == len(idxs):
            prog.progress(i / len(idxs))
    prog.empty()
    return pd.DataFrame(out)

# ═══════════════ SIDEBAR INPUTS ═════════════════════════════════════════════
uploaded = st.file_uploader("Archivo MT5 / CSV (<DATE>)", ["txt", "csv"])

c_thr_in = st.sidebar.text_input("Umbrales conteo", "5,10,15,20,25,30")
s_thr_in = st.sidebar.text_input("Umbrales rachas", "10,15,20,25,30,35,40,45,50,55,60")
tail_thr = st.sidebar.number_input("Umbral cola hist", 1, 100, 5)
gap_min  = st.sidebar.number_input("Min gap (min)", 1, 240, 45)
want_parq = st.sidebar.checkbox("Parquet", False)

st.sidebar.markdown("---"); st.sidebar.subheader("Monte Carlo")

lot0      = st.sidebar.number_input("LOT0 (lot)", 0.01, 5.0, 0.10, 0.01, format="%.2f")
q_factor  = st.sidebar.number_input("Factor inicial q", 1.01, 2.0, 1.10, 0.01, format="%.2f")
distance  = st.sidebar.slider("distance (USD)", 0.05, 3.0, 0.50, 0.05)
tp_offset = st.sidebar.number_input("tp_offset (USD)", 0.01, 5.0, 0.06, 0.01, format="%.2f")
stop_loss = st.sidebar.number_input("STOP-loss (USD < 0)", -500000.0, -1.0, -200000.0, 1000.0, format="%.0f")
max_steps = st.sidebar.number_input("max_steps", 1, 10000, 30, 1)
n_samples = st.sidebar.number_input("n_samples", 100, 100000, 1000, 100)

step_plan_in = st.sidebar.text_input("Plan de pasos (n:factor)", "3:1.10,5:2")
dd_int_in    = st.sidebar.text_input("Umbrales DD intermedios (USD)", "100000,150000")

swap_long   = st.sidebar.number_input("Swap BUY (USD/lot/noche)", -20.0, 20.0, -4.0, 0.1)
swap_short  = st.sidebar.number_input("Swap SELL (USD/lot/noche)", -20.0, 20.0,  1.0, 0.1)
rollover_hr = st.sidebar.number_input("Hora rollover CDMX", 0, 23, 16, 1)

try:
    step_plan   = parse_plan(step_plan_in)
    dd_int_thr  = [int(x) for x in dd_int_in.split(",") if x.strip()]
except ValueError as e:
    st.error(f"Error en planes / umbrales: {e}")
    st.stop()

# ── Bug #4 – STOP-loss debe ser negativo ───────────────────────────────────
if stop_loss >= 0:
    st.sidebar.warning("STOP-loss debe ser negativo. Se usará su negativo.")
    stop_loss = -abs(stop_loss)

levels_eff = total_levels(step_plan) if step_plan else max_steps
lot_tot    = compute_total_lots(lot0, q_factor, step_plan, max_steps)
st.sidebar.caption(f"Tamaño total teórico: **{lot_tot:,.2f} lots**")

run_basic = st.button("Ejecutar análisis")
run_mc_bt = st.button("⏱ Monte Carlo")

# ─────────── BLOQUE 1 – ANÁLISIS BÁSICO ────────────────────────────────────
if run_basic:
    try:
        cnt_thr = [int(x) for x in c_thr_in.split(",") if x.strip()]
        stk_thr = [int(x) for x in s_thr_in.split(",") if x.strip()]
        assert distance > 0 and tp_offset > 0
    except (ValueError, AssertionError):
        st.error("Umbrales / parámetros inválidos.")
        st.stop()

    if not uploaded:
        st.warning("Sube el archivo primero.")
        st.stop()

    with st.spinner("Procesando archivo…"):
        df = load_and_prepare(uploaded)
        st.session_state["df"] = df

        max_r, avg_r = df.range_pts.max(), df.range_pts.mean()
        d_gaps = detect_gaps(df, gap_min)

        c1, c2, c3 = st.columns(3)
        c1.metric("Máx rango", f"{max_r:.2f} pts")
        c2.metric("Rango medio", f"{avg_r:.2f} pts")
        c3.metric("Gaps", len(d_gaps))

        d_counts  = count_candles(df, cnt_thr)
        d_hour    = count_by_hour(df, cnt_thr)
        d_big     = list_big_candles(df, cnt_thr)
        d_streaks = find_streaks(df, stk_thr)

        with st.expander("Tablas detalladas", False):
            st.subheader("Conteo global");      st.dataframe(d_counts)
            st.subheader("Distribución hora");  st.dataframe(d_hour)
            st.subheader("Top-10 rachas");      st.dataframe(
                d_streaks.sort_values("length", ascending=False).head(10)
            )
            st.subheader("Top-10 gaps");        st.dataframe(
                d_gaps.sort_values("abs_gap", ascending=False).head(10)
            )

        tabs = st.tabs(
            [
                "Conteo", "Heat-map hora", "Histograma", "Hist cola",
                "Sesión box", "DOW×hora", "Gaps scat", "Gaps Top-10",
            ]
        )
        with tabs[0]:
            st.pyplot(fig_counts(d_counts), use_container_width=True)
        with tabs[1]:
            st.pyplot(fig_hour_heat(d_hour), use_container_width=True)
        with tabs[2]:
            st.plotly_chart(
                px.histogram(
                    df, x="range_pts", nbins=60, marginal="box",
                    title="Histograma rangos",
                    labels={"range_pts": "Rango (pts)"},
                ).update_layout(template="plotly_white"),
                use_container_width=True,
            )
        with tabs[3]:
            st.pyplot(fig_hist_tail(df, tail_thr), use_container_width=True)
        with tabs[4]:
            st.pyplot(fig_session_box(df), use_container_width=True)
        with tabs[5]:
            st.pyplot(fig_dow_hour_heat(df), use_container_width=True)
        with tabs[6]:
            st.pyplot(
                fig_gaps_scatter(d_gaps) if not d_gaps.empty else plt.figure(),
                use_container_width=True,
            )
        with tabs[7]:
            st.pyplot(
                fig_gaps_top(d_gaps) if not d_gaps.empty else plt.figure(),
                use_container_width=True,
            )

        tmp = tempfile.TemporaryDirectory(prefix="basic_")
        def _csv(name, _df):
            Path(tmp.name, name).write_text(_df.to_csv(index=False), encoding="utf-8")

        _csv("counts.csv",            d_counts)
        _csv("counts_by_hour.csv",    d_hour)
        _csv("big_candles.csv",       d_big)
        _csv("streaks.csv",           d_streaks)
        _csv("gaps.csv",              d_gaps)
        if want_parq:
            d_counts.to_parquet(Path(tmp.name, "counts.parquet"), index=False)
            d_hour.to_parquet(Path(tmp.name, "counts_by_hour.parquet"), index=False)

        # guardar figuras
        fig_counts(d_counts).savefig(Path(tmp.name, "counts.png"), dpi=150)
        fig_hour_heat(d_hour).savefig(Path(tmp.name, "hour_heat.png"), dpi=150)
        fig_hist(df).savefig(Path(tmp.name, "hist.png"), dpi=150)
        fig_hist_tail(df, tail_thr).savefig(Path(tmp.name, "hist_tail.png"), dpi=150)
        fig_session_box(df).savefig(Path(tmp.name, "session_box.png"), dpi=150)
        fig_dow_hour_heat(df).savefig(Path(tmp.name, "dow_hour_heat.png"), dpi=150)
        if not d_gaps.empty:
            fig_gaps_scatter(d_gaps).savefig(Path(tmp.name, "gaps_scatter.png"), dpi=150)
            fig_gaps_top(d_gaps).savefig(Path(tmp.name, "gaps_top.png"), dpi=150)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in Path(tmp.name).iterdir():
                zf.write(p, arcname=p.name)
        buf.seek(0)
        st.session_state["zip_basic"] = buf
        with st.sidebar.expander("📦 Descargar ZIP Análisis"):
            st.download_button(
                "ZIP análisis", data=buf,
                file_name="xauusd_analysis.zip", mime="application/zip",
            )
        st.success("¡Análisis estándar listo!")

# ─────────── BLOQUE 2 – MONTE CARLO ────────────────────────────────────────
if run_mc_bt:
    if "df" not in st.session_state:
        st.warning("Ejecuta primero el análisis estándar.")
        st.stop()
    df = st.session_state["df"]

    with st.spinner("Monte Carlo…"):
        t0 = time.time()
        df_mc = run_mc(
            df,
            distance=distance,
            lot0=lot0,
            q0=q_factor,
            tp_offset=tp_offset,
            stop_loss=stop_loss,
            max_steps=max_steps,
            n_samples=n_samples,
            step_plan=step_plan,
            dd_inter=dd_int_thr,
            swap_long=swap_long,
            swap_short=swap_short,
            rollover_hr=rollover_hr,
        )
        exec_t = time.time() - t0
    logging.info("Monte Carlo completado en %.2f s", exec_t)

    # mantén dd_at_*  (ya no los eliminamos)
    df_mc_c = df_mc.assign(
        session=lambda d: np.where(d["broke"], d["stuck_session"], d["exit_session"])
    )

    broke_buy  = df_mc_c[df_mc_c.side == "BUY"].broke.mean() * 100
    broke_sell = df_mc_c[df_mc_c.side == "SELL"].broke.mean() * 100

    st.header("Monte Carlo – riesgo de ruina")
    c1, c2, c3 = st.columns(3)
    c1.metric("% quiebras BUY",  f"{broke_buy:.2f}%")
    c2.metric("% quiebras SELL", f"{broke_sell:.2f}%")
    c3.metric("% quiebras global", f"{df_mc_c.broke.mean() * 100:.2f}%")
    c4, c5 = st.columns(2)
    c4.metric("DD pico media", f"{df_mc_c.dd_pico.mean():,.2f} USD")
    c5.metric("Duración media", f"{df_mc_c.dur_min.mean():.1f} min")

    with st.expander("Debug Monte Carlo"):
        st.write("Escalones promedio:", df_mc_c.steps_used.mean().round(2))
        st.write("Máx escalones:", df_mc_c.steps_used.max())
        st.write(f"Tiempo ejecución: {exec_t:.2f} s")

    st.plotly_chart(
        px.histogram(
            df_mc_c, x="dd_pico", nbins=60,
            title="Distribución dd_pico",
        ).update_layout(template="plotly_white"),
        use_container_width=True,
    )
    st.plotly_chart(
        px.histogram(
            df_mc_c, x="steps_used", nbins=levels_eff,
            title="Distribución steps_used",
        ).update_layout(template="plotly_white"),
        use_container_width=True,
    )

    broke_df = df_mc_c[df_mc_c.broke]
    if broke_df.empty:
        st.info("No hubo quiebras en esta corrida. 🎉")
    else:
        cols = [
            "start_ts", "break_ts", "session", "dd_pico", "steps_used",
            "dur_min", "side", "gap_type", "gap_min", "gap_abs_pts",
            "swap_usd", "exit_pnl_usd",
        ]
        st.subheader("Top-50 quiebras (CDMX)")
        st.dataframe(broke_df.sort_values("dd_pico").head(50)[cols])

    tmp = tempfile.TemporaryDirectory(prefix="mc_")
    df_mc_c.to_csv(Path(tmp.name, "montecarlo_results.csv"), index=False)
    if "zip_basic" in st.session_state:
        Path(tmp.name, "analysis_basic.zip").write_bytes(
            st.session_state["zip_basic"].getvalue()
        )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in Path(tmp.name).iterdir():
            zf.write(p, arcname=p.name)
    buf.seek(0)
    with st.sidebar.expander("📦 Descargar ZIP Monte Carlo"):
        st.download_button(
            "ZIP completo (Monte Carlo)",
            data=buf,
            file_name="xauusd_analysis_mc.zip",
            mime="application/zip",
        )
    st.success("¡Monte Carlo listo!")
