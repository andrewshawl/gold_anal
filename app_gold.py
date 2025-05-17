#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyzer_xauusd_m1.py
=====================

Streamlit GUI para:

1.  Análisis estándar de velas XAUUSD M1
    – 8 gráficas, KPIs, tablas, exportación ZIP (CSV + PNG + Parquet opc.).

2.  Módulo Monte Carlo Martingala
    – distance, LOT0, Q, tp_offset (fijo), STOP, max_steps, n_samples.
    – BUY + SELL por timestamp ⇒ 2 × n_samples partidas.
    – Métricas de quiebra, DD, duración, histogramas, Top-50 quiebras.
    – CSV Monte Carlo añadido al mismo ZIP.

Todo está contenido en un solo archivo.
"""
from __future__ import annotations
# ───────── std libs ─────────
import io, zipfile, tempfile, calendar, random, time
from pathlib import Path
from typing import List
# ───────── 3rd-party ────────
import numpy as np
import pandas as pd
import pytz
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

###############################################################################
# CONFIG STREAMLIT
###############################################################################
st.set_page_config(page_title="Analyzer XAUUSD M1", layout="wide")
st.markdown("""
<style>
  .title {font-size:3.5rem;font-weight:bold;text-align:center;color:#1abc9c;margin:0.2rem;}
  .subtle{text-align:center;color:#666;margin:-1rem 0 2rem;}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="title"> Analyzer XAUUSD M1 </div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Análisis técnico + Monte Carlo</div>', unsafe_allow_html=True)

###############################################################################
# CONSTANTES
###############################################################################
TZ_UTC, TZ_CDMX = pytz.UTC, pytz.timezone("America/Mexico_City")
CONTRACT_SIZE   = 100              # oz por lote XAUUSD

###############################################################################
# CARGA Y PREPARACIÓN
###############################################################################
@st.cache_data(show_spinner=False)
def load_and_prepare(file) -> pd.DataFrame:
    cols = ["Symbol","Date","Time","Open","High","Low","Close","Volume"]
    df = pd.read_csv(
        file, names=cols, header=None,
        dtype=dict(Date=str, Time=str,
                   Open=np.float64, High=np.float64,
                   Low=np.float64, Close=np.float64, Volume=np.int64),
        skipinitialspace=True, na_filter=False
    )
    dt = pd.to_datetime(df["Date"] + df["Time"], format="%Y%m%d%H%M%S",
                        errors="coerce", utc=True)
    df = df.assign(
        datetime_utc = dt,
        datetime_cdmx= dt.dt.tz_convert(TZ_CDMX),
        hour_cdmx    = dt.dt.tz_convert(TZ_CDMX).dt.hour,
        dow          = dt.dt.tz_convert(TZ_CDMX).dt.dayofweek,
        range_pts    = df["High"] - df["Low"]
    ).dropna(subset=["datetime_utc"])
    return df.sort_values("datetime_utc", ignore_index=True)

###############################################################################
# FUNCIONES DE CÁLCULO (cacheadas)
###############################################################################
@st.cache_data(show_spinner=False)
def count_candles(df:pd.DataFrame, thr:List[int])->pd.DataFrame:
    total=len(df)
    rows=[{"threshold":t,"count":int((df["range_pts"]>=t).sum())} for t in thr]
    return pd.DataFrame(rows).assign(pct_total=lambda d: d["count"]/total*100)

@st.cache_data(show_spinner=False)
def count_by_hour(df:pd.DataFrame, thr:List[int])->pd.DataFrame:
    total_h=df.groupby("hour_cdmx").size()
    rec=[]
    for t in thr:
        c=df[df["range_pts"]>=t].groupby("hour_cdmx").size()
        for h in range(24):
            n=int(c.get(h,0)); tot=int(total_h.get(h,0))
            rec.append(dict(threshold=t,hour_cdmx=h,count=n,
                            pct_in_hour=round(n/tot*100,3) if tot else np.nan))
    return pd.DataFrame(rec)

@st.cache_data(show_spinner=False)
def list_big_candles(df:pd.DataFrame,thr:List[int])->pd.DataFrame:
    t0=min(thr)
    big=df[df["range_pts"]>=t0][["datetime_utc","datetime_cdmx","range_pts"]].copy()
    big["exceeded_thresholds"]=big["range_pts"].apply(
        lambda x:";".join(str(t) for t in sorted(thr) if x>=t))
    return big

@st.cache_data(show_spinner=False)
def find_streaks(df:pd.DataFrame,thr:List[int])->pd.DataFrame:
    rec,arr=[],df["range_pts"].to_numpy()
    for t in thr:
        mask=arr>=t
        grp=np.cumsum(np.concatenate(([0],np.diff(mask).astype(int)!=0)))
        for g in np.unique(grp[mask]):
            idx=np.where(grp==g)[0]
            rec.append(dict(threshold=t,
                            start_cdmx=df.datetime_cdmx.iat[idx[0]],
                            end_cdmx  =df.datetime_cdmx.iat[idx[-1]],
                            length    =len(idx)))
    return pd.DataFrame(rec)

@st.cache_data(show_spinner=False)
def detect_gaps(df:pd.DataFrame,min_minutes:int=90)->pd.DataFrame:
    delta=df["datetime_utc"].diff().dt.total_seconds().div(60)
    gaps=np.where(delta>min_minutes)[0]
    out=[]
    for n in gaps:
        prev,next=df.iloc[n-1],df.iloc[n]
        out.append(dict(prev_cdmx=prev.datetime_cdmx,
                        next_cdmx=next.datetime_cdmx,
                        delta_min=float(delta.iat[n]),
                        abs_gap=float(abs(next.Open-prev.Close))))
    return pd.DataFrame(out)

###############################################################################
# FUNCIONES DE PLOT (Matplotlib)
###############################################################################
def fig_counts(df):
    fig,ax=plt.subplots(); ax.bar(df.threshold,df["count"])
    ax.set(xlabel="Umbral (pts)",ylabel="# velas ≥ umbral",
           title="Conteo de velas por rango"); fig.tight_layout(); return fig

def fig_hour_heat(df_hour):
    pivot=df_hour.pivot(index="threshold",columns="hour_cdmx",
                        values="pct_in_hour").sort_index(ascending=False)
    fig,ax=plt.subplots(); im=ax.imshow(pivot,aspect="auto")
    ax.set_xticks(range(24)); ax.set_xticklabels(range(24))
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    ax.set(xlabel="Hora CDMX",ylabel="Umbral (pts)",
           title="% velas ≥ umbral por hora")
    fig.colorbar(im,ax=ax,label="%"); fig.tight_layout(); return fig

def fig_hist(df):
    bins=np.arange(0,df.range_pts.max()+0.25,0.25)
    fig,ax=plt.subplots(); ax.hist(df.range_pts,bins=bins,log=True)
    ax.set(xlabel="Rango (pts)",ylabel="Frecuencia (log)",
           title="Histograma de rangos"); fig.tight_layout(); return fig

def fig_hist_tail(df,thr):
    data=df[df.range_pts>=thr].range_pts
    fig,ax=plt.subplots()
    if data.empty:
        ax.text(0.5,0.5,"Sin datos",ha="center",va="center")
    else:
        bins=np.arange(thr,data.max()+0.25,0.25); ax.hist(data,bins=bins)
    ax.set(xlabel="Rango (pts)",ylabel="Frecuencia",
           title=f"Histograma rangos ≥ {thr} pts"); fig.tight_layout(); return fig

def fig_session_box(df):
    bins=[0,2,7,12,17,24]
    labels=["Londres","NY-temp","NY-tarde","Asia-mat","Asia-noche"]
    df_s=df.copy()
    df_s["session"]=pd.cut(df_s.hour_cdmx,bins=bins,labels=labels,
                           right=False,include_lowest=True)
    fig,ax=plt.subplots()
    df_s.boxplot(column="range_pts",by="session",ax=ax,grid=False)
    plt.suptitle(""); ax.set_ylabel("Rango (pts)"); ax.set_title("Rangos por sesión")
    fig.tight_layout(); return fig

def fig_dow_hour_heat(df):
    pivot=(df.groupby(["dow","hour_cdmx"]).range_pts.mean()
             .unstack(fill_value=np.nan).reindex(index=range(7)))
    fig,ax=plt.subplots(); im=ax.imshow(pivot,aspect="auto")
    ax.set_xticks(range(24)); ax.set_xticklabels(range(24))
    ax.set_yticks(range(7)); ax.set_yticklabels([calendar.day_abbr[d] for d in range(7)])
    ax.set(xlabel="Hora CDMX",ylabel="Día semana",title="Heat-map rango medio")
    fig.colorbar(im,ax=ax,label="pts"); fig.tight_layout(); return fig

def fig_gaps_scatter(df_gaps):
    fig,ax=plt.subplots()
    ax.scatter(df_gaps.delta_min,df_gaps.abs_gap)
    ax.set(xlabel="Duración gap (min)",ylabel="Tamaño gap (pts)",
           title="Gaps: tamaño vs duración"); fig.tight_layout(); return fig

def fig_gaps_top(df_gaps):
    g=df_gaps.nlargest(10,"abs_gap").sort_values("abs_gap")
    labels=[d.strftime("%Y-%m-%d\n%H:%M") for d in g.prev_cdmx]
    fig,ax=plt.subplots(); ax.barh(labels,g.abs_gap)
    ax.set(xlabel="Tamaño gap (pts)",title="Top-10 gaps más grandes")
    fig.tight_layout(); return fig

###############################################################################
# MONTE CARLO – MARTINGALA GLOBAL (TP fijo)
###############################################################################
def sample_start_indices(df,max_steps,n,lookahead_per_step=10):
    limit=len(df)-max_steps*lookahead_per_step
    if limit<=0: raise ValueError("Histórico muy corto.")
    return np.random.choice(limit,size=n,replace=False)

def pnl_per_oz(side,price,entry): return price-entry if side=="BUY" else entry-price

def simulate_grid(df, start_idx, *, side, distance, lot0, q, tp_offset,
                  stop_loss, max_steps):
    closes = df.Close.values
    lows   = df.Low.values
    highs  = df.High.values

    # entradas y lotes
    entry = [df.Open.iat[start_idx]]
    lots  = [lot0]
    total_lots = lot0

    # equity / drawdown tracking
    equity = 0.0
    peak_equity = 0.0
    max_drawdown = 0.0  # siempre ≤ 0

    # precio medio ponderado (para TP)
    pmp = entry[0]
    last = entry[0]
    steps = 1
    idx = start_idx

    while idx < len(df) - 1:
        idx += 1
        price = closes[idx]

        # 1) calcular equity actual en USD
        pnl_per_ozs = [lots[i] * (price - entry[i]) for i in range(len(entry))]
        equity = sum(pnl_per_ozs) * CONTRACT_SIZE

        # 2) actualizar peak
        if equity > peak_equity:
            peak_equity = equity

        # 3) drawdown actual
        drawdown = equity - peak_equity  # negativo o cero
        if drawdown < max_drawdown:
            max_drawdown = drawdown

        # 4) check stop_loss sobre drawdown
        if max_drawdown <= stop_loss:
            return dict(
                broke=True,
                dd_pico=max_drawdown,
                steps_used=steps,
                dur_min=idx - start_idx,
                side=side,
                start_ts=df.datetime_cdmx.iat[start_idx]
            )

        # 5) check TP
        if (side == "BUY" and price >= pmp + tp_offset) or \
           (side == "SELL" and price <= pmp - tp_offset):
            break

        # 6) añadir nivel Martingala
        if steps < max_steps:
            triggered = (side == "BUY" and lows[idx] <= last - distance) or \
                        (side == "SELL" and highs[idx] >= last + distance)
            if triggered:
                new_entry = lows[idx] if side == "BUY" else highs[idx]
                entry.append(new_entry)
                lots.append(lots[-1] * q)
                total_lots += lots[-1]
                last = new_entry
                steps += 1
                # recalcular precio medio ponderado
                pmp = sum(l * e for l, e in zip(lots, entry)) / total_lots

    return dict(
        broke=False,
        dd_pico=max_drawdown,
        steps_used=steps,
        dur_min=idx - start_idx,
        side=side,
        start_ts=df.datetime_cdmx.iat[start_idx]
    )

def run_monte_carlo(df,*,distance,lot0,q,tp_offset,stop_loss,
                    max_steps,n_samples,seed=42):
    np.random.seed(seed); random.seed(seed)
    idxs=sample_start_indices(df,max_steps,n_samples)
    res=[]; prog=st.progress(0.0,"Monte Carlo…")
    for i,base in enumerate(idxs,1):
        for side in ("BUY","SELL"):
            res.append(simulate_grid(df,base,side=side,distance=distance,lot0=lot0,
                                     q=q,tp_offset=tp_offset,stop_loss=stop_loss,
                                     max_steps=max_steps))
        if i%50==0 or i==len(idxs): prog.progress(i/len(idxs))
    prog.empty(); return pd.DataFrame(res)

###############################################################################
# UI  – PARÁMETROS
###############################################################################

uploaded=st.file_uploader("Archivo MT5 (.txt/.csv)",["txt","csv"])

#  parámetros básicos
c_thr_in=st.sidebar.text_input("Umbrales conteo","5,10,15,20,25,30")
s_thr_in=st.sidebar.text_input("Umbrales rachas","10,15,20,25,30,35,40,45,50,55,60")
tail_thr=st.sidebar.number_input("Umbral cola hist",1,value=5)
gap_min =st.sidebar.number_input("Min gap (min)",1,value=90)
want_parq=st.sidebar.checkbox("Parquet",False)

#  parámetros Monte Carlo
st.sidebar.markdown("---"); st.sidebar.subheader("Monte Carlo")
lot0      =st.sidebar.number_input("LOT0 (lot)",0.01,5.0,0.10,0.01)
q_factor  =st.sidebar.number_input("Factor",1.01,2.0,1.10,0.01)
distance  =st.sidebar.slider("distance (USD)",0.05,3.0,0.50,0.05)
tp_offset =st.sidebar.number_input("tp_offset (USD)",0.01,5.0,0.06,0.01)
stop_loss =st.sidebar.number_input("STOP-loss (USD)",-1e5,-1000.0,-10000.0,500.0,format="%.0f")

# Límites ampliados:
max_steps = st.sidebar.number_input("max_steps", 1, 10000, 30, 1)
n_samples = st.sidebar.number_input("n_samples", 100, 100000, 1000, 100)

lot_tot=lot0*(q_factor**max_steps-1)/(q_factor-1)

run_basic=st.button("Ejecutar análisis")
run_mc   =st.button("⏱ Monte Carlo")

###############################################################################
# ANÁLISIS BÁSICO COMPLETO
###############################################################################
if run_basic:
    try:
        cnt_thr=[int(x) for x in c_thr_in.split(",") if x.strip()]
        stk_thr=[int(x) for x in s_thr_in.split(",") if x.strip()]
    except ValueError:
        st.error("Umbrales inválidos."); st.stop()
    if not uploaded:
        st.warning("Sube el archivo primero."); st.stop()

    with st.spinner("Procesando archivo…"):
        df=load_and_prepare(uploaded); st.session_state["df"]=df

        max_range,avg_range=df.range_pts.max(),df.range_pts.mean()
        d_gaps=detect_gaps(df,gap_min); total_gaps=len(d_gaps)

        c1,c2,c3=st.columns(3)
        c1.metric("Máx rango",f"{max_range:.2f} pts")
        c2.metric("Rango medio",f"{avg_range:.2f} pts")
        c3.metric("Gaps",str(total_gaps))

        d_counts =count_candles(df,cnt_thr)
        d_hour   =count_by_hour(df,cnt_thr)
        d_big    =list_big_candles(df,cnt_thr)
        d_streaks=find_streaks(df,stk_thr)

        with st.expander("Tablas detalladas"):
            st.subheader("Conteo global");         st.dataframe(d_counts)
            st.subheader("Distribución hora");     st.dataframe(d_hour)
            st.subheader("Top-10 rachas");         st.dataframe(d_streaks.sort_values("length",ascending=False).head(10))
            st.subheader("Top-10 gaps");           st.dataframe(d_gaps.sort_values("abs_gap",ascending=False).head(10))

        tabs=st.tabs(["Conteo","Heat-map hora","Histograma","Hist cola",
                      "Sesión box","DOW×hora","Gaps scat","Gaps Top-10"])
        with tabs[0]: st.pyplot(fig_counts(d_counts),use_container_width=True)
        with tabs[1]: st.pyplot(fig_hour_heat(d_hour),use_container_width=True)
        with tabs[2]:
            st.plotly_chart(
                px.histogram(df,x="range_pts",nbins=60,
                             title="Histograma rangos",
                             labels={"range_pts":"Rango (pts)"},
                             marginal="box").update_layout(template="plotly_white"),
                use_container_width=True)
        with tabs[3]: st.pyplot(fig_hist_tail(df,tail_thr),use_container_width=True)
        with tabs[4]: st.pyplot(fig_session_box(df),use_container_width=True)
        with tabs[5]: st.pyplot(fig_dow_hour_heat(df),use_container_width=True)
        with tabs[6]:
            st.pyplot(fig_gaps_scatter(d_gaps) if not d_gaps.empty else plt.figure(),
                      use_container_width=True)
        with tabs[7]:
            st.pyplot(fig_gaps_top(d_gaps) if not d_gaps.empty else plt.figure(),
                      use_container_width=True)

        # Export ZIP
        tmp=tempfile.TemporaryDirectory()
        def save_csv(name,_df): Path(tmp.name,name).write_text(_df.to_csv(index=False),encoding="utf-8")
        save_csv("counts.csv",d_counts); save_csv("counts_by_hour.csv",d_hour)
        save_csv("big_candles.csv",d_big); save_csv("streaks.csv",d_streaks); save_csv("gaps.csv",d_gaps)
        if want_parq:
            d_counts.to_parquet(Path(tmp.name,"counts.parquet"),index=False)
            d_hour.to_parquet(Path(tmp.name,"counts_by_hour.parquet"),index=False)
        # PNGs
        fig_counts(d_counts).savefig(Path(tmp.name,"counts.png"),dpi=150)
        fig_hour_heat(d_hour).savefig(Path(tmp.name,"hour_heat.png"),dpi=150)
        fig_hist(df).savefig(Path(tmp.name,"hist.png"),dpi=150)
        fig_hist_tail(df,tail_thr).savefig(Path(tmp.name,"hist_tail.png"),dpi=150)
        fig_session_box(df).savefig(Path(tmp.name,"session_box.png"),dpi=150)
        fig_dow_hour_heat(df).savefig(Path(tmp.name,"dow_hour_heat.png"),dpi=150)
        if not d_gaps.empty:
            fig_gaps_scatter(d_gaps).savefig(Path(tmp.name,"gaps_scatter.png"),dpi=150)
            fig_gaps_top(d_gaps).savefig(Path(tmp.name,"gaps_top.png"),dpi=150)
        # ZIP
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as zf:
            for p in Path(tmp.name).iterdir(): zf.write(p,arcname=p.name)
        buf.seek(0); st.session_state["zip_tmp"]=tmp
        with st.sidebar.expander("📦 Descargar ZIP"):
            st.download_button("ZIP análisis",data=buf,
                               file_name="xauusd_analysis.zip",mime="application/zip")
        st.success("¡Análisis estándar listo!")

###############################################################################
# MONTE CARLO
###############################################################################
if run_mc:
    if "df" not in st.session_state:
        st.warning("Ejecuta primero el análisis estándar."); st.stop()
    df=st.session_state["df"]
    with st.spinner("Monte Carlo…"):
        t0=time.time()
        df_mc=run_monte_carlo(df,distance=distance,lot0=lot0,q=q_factor,
                              tp_offset=tp_offset,stop_loss=stop_loss,
                              max_steps=max_steps,n_samples=n_samples)
        t_exec=time.time()-t0

    broke_buy =df_mc[df_mc.side=="BUY"].broke.mean()*100
    broke_sell=df_mc[df_mc.side=="SELL"].broke.mean()*100
    st.header("Monte Carlo – riesgo de ruina")
    c1,c2,c3=st.columns(3)
    c1.metric("% quiebras BUY",f"{broke_buy:.2f}%")
    c2.metric("% quiebras SELL",f"{broke_sell:.2f}%")
    c3.metric("% quiebras global",f"{df_mc.broke.mean()*100:.2f}%")
    c4,c5=st.columns(2)
    c4.metric("DD pico medio",f"{df_mc.dd_pico.mean():,.2f} USD")
    c5.metric("Duración media",f"{df_mc.dur_min.mean():.1f} min")
    with st.expander("Debug"):
        st.write("Escalones promedio:",df_mc.steps_used.mean().round(2))
        st.write("Máx escalones:",df_mc.steps_used.max())

    st.plotly_chart(px.histogram(df_mc,x="dd_pico",nbins=60,
                     title="Distribución dd_pico").update_layout(template="plotly_white"),
                     use_container_width=True)
    st.plotly_chart(px.histogram(df_mc,x="steps_used",nbins=max_steps,
                     title="Distribución steps_used").update_layout(template="plotly_white"),
                     use_container_width=True)
    st.subheader("Top-50 quiebras")
    st.dataframe(df_mc[df_mc.broke].sort_values("dd_pico").head(50))

    tmp=st.session_state.get("zip_tmp") or tempfile.TemporaryDirectory()
    Path(tmp.name,"montecarlo_results.csv").write_text(df_mc.to_csv(index=False),encoding="utf-8")
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as zf:
        for p in Path(tmp.name).iterdir(): zf.write(p,arcname=p.name)
    buf.seek(0)
    with st.sidebar.expander("📦 Descargar ZIP"):
        st.download_button("ZIP completo (Monte Carlo)",
                           data=buf,file_name="xauusd_analysis_mc.zip",
                           mime="application/zip")
    st.success("¡Monte Carlo listo!")
