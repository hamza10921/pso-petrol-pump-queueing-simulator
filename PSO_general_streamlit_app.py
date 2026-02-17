"""
PSO Petrol Pump Queueing Simulator
- 9 models: M/M/1, M/M/2, M/M/c, M/G/1, M/G/2, M/G/c, G/G/1, G/G/2, G/G/c
- FIFO only
- Chi-square GOF (Exponential) for inter-arrival and service durations (from CSV) to match report requirement
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from engine import SimulationConfig, simulate
from disciplines import FCFS
from processes import (
    PoissonArrival,
    EmpiricalArrivalTimes,
    EmpiricalInterarrival,
    ExponentialService,
    EmpiricalServiceTimes,
)
from metrics import to_dataframes, chi_square_exponential_gof

st.set_page_config(page_title="PSO Petrol Pump Queueing Simulator", layout="wide")
st.title("PSO Petrol Pump Queueing Simulator")
st.caption("FIFO queueing simulator + Chi-square validation (report requirement)")

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("Queueing Model (9)")
    model = st.selectbox(
        "Select model",
        [
            "M/M/1", "M/M/2", "M/M/c",
            "M/G/1", "M/G/2", "M/G/c",
            "G/G/1", "G/G/2", "G/G/c",
        ],
        index=1,
    )

    st.header("Mode")
    mode = st.radio("Choose mode", ["Rate-based (λ, μ)", "Data-driven (CSV)"], index=0)

    st.header("Servers (c)")
    if model.endswith("/1"):
        c = 1
        st.info("Servers fixed at 1 by model.")
    elif model.endswith("/2"):
        c = 2
        st.info("Servers fixed at 2 by model.")
    else:
        c = st.number_input("Servers (dispensers) c", 1, 20, 2, 1)

    st.header("Simulation size")
    N = st.number_input("Customers to simulate (N)", 1, 5000, 200, 10)
    seed = st.number_input("Random seed", 0, 10_000_000, 42, 1)

    st.header("Parameters")
    needs_lambda = model.startswith("M/")
    needs_mu = "/M/" in model
    lam = st.number_input("Arrival rate λ (per minute)", min_value=1e-9, value=3.0, step=0.1, format="%.4f") if needs_lambda else None
    mu = st.number_input("Service rate μ (per minute per server)", min_value=1e-9, value=4.0, step=0.1, format="%.4f") if needs_mu else None

    st.header("Charts")
    bins = st.slider("Histogram bins", 5, 50, 10)
    alpha = st.select_slider("Chi-square α", options=[0.10, 0.05, 0.01], value=0.05)
    chi_bins = st.slider("Chi-square bins (k)", 5, 20, 10)

# -------------------------
# CSV upload
# -------------------------
st.subheader("Inputs")
st.markdown("### Customer Data (CSV)")
st.caption("Expected columns (PSO): Arrival_min, ServiceDuration_min")

template = "Arrival_min,ServiceDuration_min\n912.0000,0.0833\n912.4939,0.0232\n913.0000,0.8669\n"
st.download_button("Download CSV Template", data=template, file_name="pso_template.csv", mime="text/csv")

uploaded = st.file_uploader("Import CSV", type=["csv"])
data_df = None

if uploaded is not None:
    raw = pd.read_csv(uploaded)
    raw.columns = [c.strip() for c in raw.columns]
    col_map = {c.lower(): c for c in raw.columns}

    def pick(*names):
        for n in names:
            if n.lower() in col_map:
                return col_map[n.lower()]
        return None

    arr_col = pick("Arrival_min", "arrival_min", "arrival_time")
    svc_col = pick("ServiceDuration_min", "serviceduration_min", "service_time")

    if arr_col is None:
        st.error("CSV must include Arrival_min.")
    else:
        df = raw.copy()
        df["Arrival_min"] = pd.to_numeric(df[arr_col], errors="coerce")

        if svc_col is not None:
            df["ServiceDuration_min"] = pd.to_numeric(df[svc_col], errors="coerce")
        else:
            df["ServiceDuration_min"] = np.nan

        data_df = df.dropna(subset=["Arrival_min"]).sort_values("Arrival_min").reset_index(drop=True)
        st.success(f"Loaded {len(data_df)} rows.")
        st.dataframe(data_df.head(200), use_container_width=True, height=260)

# -------------------------
# Chi-square section (report requirement)
# -------------------------
st.markdown("## Statistical Validation (Chi-square)")
if data_df is None:
    st.info("Upload CSV to run Chi-square GOF on inter-arrival & service duration data (report style).")
else:
    arr = data_df["Arrival_min"].dropna().astype(float).values
    if len(arr) >= 2:
        inter = np.diff(np.sort(arr))
        res_arr = chi_square_exponential_gof(inter, bins=int(chi_bins), alpha=float(alpha))
        st.markdown("### Inter-arrival times (test Exponential ⇒ Poisson arrivals)")
        if res_arr.get("ok"):
            st.write(f"λ̂ (from data) = {res_arr['lambda_hat']:.6f},  χ² = {res_arr['chi_square']:.4f}, df = {res_arr['df']}, p = {res_arr['p_value']:.4f}")
            st.write(f"Decision @ α={res_arr['alpha']}: **{res_arr['decision']}**")
            st.dataframe(res_arr["table"], use_container_width=True)
        else:
            st.warning(res_arr.get("error"))

    svc = data_df.get("ServiceDuration_min", pd.Series([], dtype=float)).dropna().astype(float).values
    st.markdown("### Service times (test Exponential for M-service assumption)")
    if len(svc) >= 2:
        res_svc = chi_square_exponential_gof(svc, bins=int(chi_bins), alpha=float(alpha))
        if res_svc.get("ok"):
            st.write(f"μ̂ (from data) = {res_svc['lambda_hat']:.6f},  χ² = {res_svc['chi_square']:.4f}, df = {res_svc['df']}, p = {res_svc['p_value']:.4f}")
            st.write(f"Decision @ α={res_svc['alpha']}: **{res_svc['decision']}**")
            st.dataframe(res_svc["table"], use_container_width=True)
        else:
            st.warning(res_svc.get("error"))
    else:
        st.info("ServiceDuration_min column missing or too few valid values for Chi-square.")

# -------------------------
# Run buttons
# -------------------------
run_col1, run_col2 = st.columns([1, 1])
run_btn = run_col1.button("▶ Run Simulation", type="primary", use_container_width=True)
reset_btn = run_col2.button("↩ Reset", use_container_width=True)

if reset_btn:
    st.session_state.pop("last_sim", None)
    st.rerun()

# -------------------------
# Build processes by model
# -------------------------
def build_arrival_process():
    # M arrivals
    if model.startswith("M/"):
        if lam is None:
            raise ValueError("λ required for M-arrivals.")
        return PoissonArrival(lam=float(lam))

    # G arrivals from CSV
    if data_df is None:
        raise ValueError("CSV required for G-arrivals.")
    arr = data_df["Arrival_min"].dropna().astype(float).values
    arr = np.sort(arr)
    if len(arr) < 2:
        raise ValueError("Need at least 2 Arrival_min rows for G-arrivals.")
    inter = np.diff(arr)
    return EmpiricalInterarrival(interarrivals=inter)


def build_service_process():
    # M service
    if "/M/" in model:
        if mu is None:
            raise ValueError("μ required for M-service.")
        return ExponentialService(mu=float(mu))

    # G service from CSV
    if data_df is None or "ServiceDuration_min" not in data_df.columns:
        raise ValueError("CSV with ServiceDuration_min required for G-service.")
    svc = data_df["ServiceDuration_min"].dropna().astype(float).values
    if len(svc) < 1:
        raise ValueError("No valid ServiceDuration_min values.")
    return EmpiricalServiceTimes(service_times=svc)

# -------------------------
# Run simulation
# -------------------------
if run_btn:
    try:
        cfg = SimulationConfig(n_servers=int(c), n_customers=int(N), seed=int(seed))
        arr_proc = build_arrival_process()
        svc_proc = build_service_process()
        discipline = FCFS()
        sim = simulate(cfg, arr_proc, svc_proc, discipline)
        st.session_state["last_sim"] = sim
        st.success("Simulation completed.")
    except Exception as e:
        st.error(str(e))

# -------------------------
# Results
# -------------------------
sim = st.session_state.get("last_sim", None)
if sim:
    event_df, timeline_df, kpi_df, ts_df = to_dataframes(sim)

    st.subheader("Simulation Results")

    if not kpi_df.empty:
        k = kpi_df.iloc[0].to_dict()
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Avg Waiting Time", f'{k.get("avg_wait", 0):.4f}')
        c2.metric("Avg Turnaround", f'{k.get("avg_turnaround", 0):.4f}')
        c3.metric("Avg Service", f'{k.get("avg_service", 0):.4f}')
        c4.metric("Avg Queue Length (Lq)", f'{k.get("Lq_time_avg", 0):.4f}')
        c5.metric("Avg System Length (L)", f'{k.get("L_time_avg", 0):.4f}')
        c6.metric("Throughput", f'{k.get("throughput_per_time", 0):.4f}')

    util = (sim.get("kpis", {}) or {}).get("utilization", {})
    if util:
        util_df = pd.DataFrame({"Server": list(util.keys()), "Utilization": list(util.values())})
        st.markdown("### Server Utilization")
        st.plotly_chart(px.bar(util_df, x="Server", y="Utilization"), use_container_width=True)

    st.markdown("### Customer Results")
    st.dataframe(event_df, use_container_width=True, height=280)

    st.markdown("### Distributions")
    d1, d2, d3, d4 = st.columns(4)

    with d1:
        if "WaitTime" in event_df.columns:
            st.plotly_chart(px.histogram(event_df, x="WaitTime", nbins=bins, title="Waiting Time"), use_container_width=True)

    with d2:
        if "ServiceTime" in event_df.columns:
            st.plotly_chart(px.histogram(event_df, x="ServiceTime", nbins=bins, title="Service Time"), use_container_width=True)

    with d3:
        if "Arrival" in event_df.columns and len(event_df) > 1:
            ia = np.diff(np.sort(event_df["Arrival"].astype(float).values))
            st.plotly_chart(px.histogram(pd.DataFrame({"InterArrival": ia}), x="InterArrival", nbins=bins, title="Inter-arrival Time"), use_container_width=True)

    with d4:
        if not ts_df.empty and "queue_length" in ts_df.columns:
            st.plotly_chart(px.line(ts_df, x="time", y="queue_length", title="Queue Length Over Time"), use_container_width=True)

    st.markdown("### Service Timeline (Gantt)")
    if not timeline_df.empty:
        fig = go.Figure()
        for _, r in timeline_df.iterrows():
            fig.add_trace(go.Bar(
                x=[r["BusyDuration"]],
                y=[f"Server {int(r['Server_ID'])}"],
                base=r["BusyStart"],
                orientation="h",
                hovertemplate=(
                    f"Server {int(r['Server_ID'])}<br>"
                    f"Customer {int(r['Customer_ID'])}<br>"
                    "Start %{base:.3f}<br>"
                    "Duration %{x:.3f}<extra></extra>"
                ),
            ))
        fig.update_layout(barmode="stack", xaxis_title="Time (min)", yaxis_title="Server", showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Export")
    out = io.StringIO()
    event_df.to_csv(out, index=False)
    st.download_button("Export Customer Results CSV", data=out.getvalue(), file_name="pso_sim_results.csv", mime="text/csv")
