# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ems_core import (
    SCENARIOS,
    simulate_day,
    audit_constraints,
    audit_all_scenarios,
    results_timeseries_df,
)

st.set_page_config(page_title="EMS SmartGrid — EV + PV + BESS", layout="wide")
st.title("EMS SmartGrid — EV + PV + BESS (SOC BESS 20%–80%)")

# -----------------------------
# Sidebar: paramètres
# -----------------------------
st.sidebar.header("Paramètres")

mode = st.sidebar.selectbox("Mode", ["1 scénario", "Comparaison multi-scénarios"])

default_scenarios = list(SCENARIOS.keys())
if mode == "1 scénario":
    scenario_selected = st.sidebar.selectbox("Scénario", default_scenarios)
    scenario_list = [scenario_selected]
else:
    scenario_list = st.sidebar.multiselect("Scénarios", default_scenarios, default=default_scenarios[:3])
    if not scenario_list:
        st.warning("Sélectionne au moins un scénario.")
        st.stop()

st.sidebar.subheader("Pas & bornes")
dt_min = st.sidebar.slider("dt (min)", 1, 30, 5, step=1)
nb_bornes = st.sidebar.slider("Nombre de bornes", 1, 20, 6, step=1)
P_borne_uni_max = st.sidebar.slider("P max par VE (kW)", 3.0, 50.0, 22.0, step=0.5)
P_borne_tot_max = st.sidebar.slider("P totale VE (kW)", 10.0, 500.0, 132.0, step=1.0)
P_grid_max = st.sidebar.slider("P max réseau (kW)", 50.0, 800.0, 300.0, step=10.0)

st.sidebar.subheader("Tarifs")
tariff_hp = st.sidebar.number_input("Tarif HP (€/kWh)", value=0.24, step=0.01, format="%.2f")
tariff_hc = st.sidebar.number_input("Tarif HC (€/kWh)", value=0.17, step=0.01, format="%.2f")

st.sidebar.subheader("BESS SOC (bornes)")
bess_soc_min = st.sidebar.slider("SOC min (%)", 0, 50, 20, step=1)
bess_soc_max = st.sidebar.slider("SOC max (%)", 50, 100, 80, step=1)
if bess_soc_max <= bess_soc_min:
    st.sidebar.error("SOC max doit être > SOC min")
    st.stop()

run = st.sidebar.button("▶️ Lancer simulation", type="primary")

# -----------------------------
# Simulation (cache)
# -----------------------------
@st.cache_data(show_spinner=False)
def _run_one(scn: str, dt_min: int, nb_bornes: int, P_borne_uni_max: float, P_borne_tot_max: float, P_grid_max: float,
             tariff_hp: float, tariff_hc: float, bess_soc_min: float, bess_soc_max: float):
    return simulate_day(
        scn,
        dt_min=dt_min,
        nb_bornes=nb_bornes,
        P_borne_uni_max=P_borne_uni_max,
        P_borne_tot_max=P_borne_tot_max,
        P_grid_max=P_grid_max,
        tariff_hp=tariff_hp,
        tariff_hc=tariff_hc,
        bess_soc_min_pct=bess_soc_min,
        bess_soc_max_pct=bess_soc_max,
    )

def shade_hc(ax):
    ax.axvspan(22, 24, alpha=0.12)
    ax.axvspan(0, 6, alpha=0.12)

def format_time_axis(ax):
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 2))
    ax.grid(True, alpha=0.25)

def plot_flux(res: dict):
    t_h = res["t_min"] / 60.0
    pv = res["pv_kW"]
    load = res["load_kW"]
    ev = res["ev_kW"]
    imp = res["grid_import_kW"]
    exp = res["grid_export_kW"]
    net_load = load + ev - pv

    fig = plt.figure(figsize=(12, 5))
    ax = plt.gca()
    ax.plot(t_h, pv, label="PV (kW)", linewidth=2)
    ax.plot(t_h, load, label="Bâtiment (kW)", linewidth=2)
    ax.plot(t_h, ev, label="VE (kW)", linewidth=2)
    ax.plot(t_h, imp, label="Import réseau (kW)", linewidth=2)
    ax.plot(t_h, exp, label="Export réseau (kW)", linewidth=2)
    ax.fill_between(t_h, 0, net_load, alpha=0.08, label="Net load (Load+EV-PV)")
    shade_hc(ax)
    format_time_axis(ax)
    ax.set_xlabel("Heure (h)")
    ax.set_ylabel("Puissance (kW)")
    ax.set_title(f"Flux de puissance — {res['scenario']}")
    ax.legend(ncol=2)
    plt.tight_layout()
    return fig

def plot_bess_soc(res: dict):
    t_h = res["t_min"] / 60.0
    soc = res["bess_soc_pct"]
    bl = res["bess_limits"]

    fig = plt.figure(figsize=(12, 4))
    ax = plt.gca()
    ax.plot(t_h, soc, linewidth=2)
    ax.axhline(bl["soc_min_pct"], linestyle="--", alpha=0.7)
    ax.axhline(bl["soc_max_pct"], linestyle="--", alpha=0.7)
    shade_hc(ax)
    format_time_axis(ax)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Heure (h)")
    ax.set_ylabel("SOC BESS (%)")
    ax.set_title(f"SOC BESS — {res['scenario']} (bornes {bl['soc_min_pct']:.0f}%–{bl['soc_max_pct']:.0f}%)")
    plt.tight_layout()
    return fig

def plot_sci(res: dict):
    t_h = res["t_min"] / 60.0
    fig = plt.figure(figsize=(12, 4))
    ax = plt.gca()
    ax.plot(t_h, res["SCI_inst_pct"], linewidth=2)
    shade_hc(ax)
    format_time_axis(ax)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Heure (h)")
    ax.set_ylabel("SCI instantané (%)")
    ax.set_title(f"SCI — {res['scenario']} | SCI global = {res['SCI_global_pct']:.1f}%")
    plt.tight_layout()
    return fig

def plot_cost(res: dict):
    t_h = res["t_min"] / 60.0
    total = res["cost_hp_cum"] + res["cost_hc_cum"]

    fig = plt.figure(figsize=(12, 4))
    ax = plt.gca()
    ax.plot(t_h, res["cost_hp_cum"], label="Coût cumulé HP (€)", linewidth=2)
    ax.plot(t_h, res["cost_hc_cum"], label="Coût cumulé HC (€)", linewidth=2)
    ax.plot(t_h, total, label="Total (€)", linewidth=2)
    shade_hc(ax)
    format_time_axis(ax)
    ax.set_xlabel("Heure (h)")
    ax.set_ylabel("Coût cumulé (€)")
    ax.set_title(f"Coût cumulé — {res['scenario']} | Total = {res['cost_total']:.2f} €")
    ax.legend()
    plt.tight_layout()
    return fig

# -----------------------------
# Run
# -----------------------------
if not run:
    st.info("Régle les paramètres à gauche, puis clique **Lancer simulation**.")
    st.stop()

with st.spinner("Simulation en cours..."):
    results = [
        _run_one(
            scn,
            dt_min, nb_bornes, P_borne_uni_max, P_borne_tot_max, P_grid_max,
            tariff_hp, tariff_hc, float(bess_soc_min), float(bess_soc_max)
        )
        for scn in scenario_list
    ]

# -----------------------------
# Résultats
# -----------------------------
st.subheader("KPI")

kpi_rows = []
for r in results:
    kpi_rows.append({
        "scenario": r["scenario"],
        "SCI_global_%": round(r["SCI_global_pct"], 2),
        "IEJ": round(r["IEJ"], 4),
        "peak_grid_import_kW": round(r["peak_grid_import_kW"], 2),
        "E_import_total_kWh": round(r["E_import_total_kWh"], 2),
        "cost_total_€": round(r["cost_total"], 2),
        "SOC_BESS_min_%": round(float(np.min(r["bess_soc_pct"])), 2),
        "SOC_BESS_max_%": round(float(np.max(r["bess_soc_pct"])), 2),
    })
kpi_df = pd.DataFrame(kpi_rows)
st.dataframe(kpi_df, use_container_width=True)

st.subheader("Audit contraintes")
audit_big = audit_all_scenarios(results)
st.dataframe(audit_big, use_container_width=True)

# Download audit CSV
st.download_button(
    "⬇️ Télécharger Audit (CSV)",
    data=audit_big.to_csv(index=False).encode("utf-8"),
    file_name="audit_constraints.csv",
    mime="text/csv",
)

# -----------------------------
# Graphiques
# -----------------------------
st.subheader("Graphiques")

if mode == "1 scénario":
    r = results[0]
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_flux(r), clear_figure=True)
        st.pyplot(plot_sci(r), clear_figure=True)
    with c2:
        st.pyplot(plot_bess_soc(r), clear_figure=True)
        st.pyplot(plot_cost(r), clear_figure=True)

    ts = results_timeseries_df(r)
    st.download_button(
        "⬇️ Télécharger séries temporelles (CSV)",
        data=ts.to_csv(index=False).encode("utf-8"),
        file_name=f"timeseries_{r['scenario']}.csv",
        mime="text/csv",
    )
else:
    # Comparaison: import réseau
    fig = plt.figure(figsize=(12, 4))
    ax = plt.gca()
    for r in results:
        t_h = r["t_min"] / 60.0
        ax.plot(t_h, r["grid_import_kW"], linewidth=2, label=r["scenario"])
    shade_hc(ax); format_time_axis(ax)
    ax.set_xlabel("Heure (h)"); ax.set_ylabel("Import réseau (kW)")
    ax.set_title("Comparaison — Import réseau")
    ax.legend(ncol=2)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # Comparaison: SOC BESS
    fig = plt.figure(figsize=(12, 4))
    ax = plt.gca()
    for r in results:
        t_h = r["t_min"] / 60.0
        ax.plot(t_h, r["bess_soc_pct"], linewidth=2, label=r["scenario"])
    shade_hc(ax); format_time_axis(ax)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Heure (h)"); ax.set_ylabel("SOC BESS (%)")
    ax.set_title("Comparaison — SOC BESS")
    ax.legend(ncol=2)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # Comparaison: SCI instant
    fig = plt.figure(figsize=(12, 4))
    ax = plt.gca()
    for r in results:
        t_h = r["t_min"] / 60.0
        ax.plot(t_h, r["SCI_inst_pct"], linewidth=2, label=r["scenario"])
    shade_hc(ax); format_time_axis(ax)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Heure (h)"); ax.set_ylabel("SCI instantané (%)")
    ax.set_title("Comparaison — SCI instantané")
    ax.legend(ncol=2)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)