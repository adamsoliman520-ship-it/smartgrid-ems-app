# ems_core.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ============================================================
# 0) Score de priorisation (simulation)
# ============================================================
def vehicule_prioritisation(user_profile: str, t_now_min: int, departure_min: int, charge_level: float) -> float:
    if user_profile == "Urgences":
        Pu = 7
    elif user_profile == "Administration":
        Pu = 2
    elif user_profile == "Autres":
        Pu = 1
    else:
        raise ValueError("Invalid user profile")

    dt = departure_min - t_now_min
    if dt < 0:
        dt += 24 * 60

    if dt < 60:
        H = 3
    elif dt < 120:
        H = 2
    else:
        H = 1

    return float(Pu + H + 2 * (1 - float(charge_level)))

# ============================================================
# 1) Modèles EV / BESS
# ============================================================
@dataclass
class EV:
    ev_id: str
    profile: str
    arrival_min: int
    departure_min: int
    E_max_kWh: float
    P_max_kW: float
    soc_ini_pct: float
    soc_target_pct: float

    soc_pct: float = None
    energy_received_kWh: float = 0.0

    def __post_init__(self):
        self.soc_pct = float(self.soc_ini_pct)

    def active(self, t_min: int) -> bool:
        return (self.arrival_min <= t_min < self.departure_min) and (self.soc_pct < self.soc_target_pct - 1e-9)

    def remaining_need_kWh(self) -> float:
        need_pct = max(self.soc_target_pct - self.soc_pct, 0.0)
        return (need_pct / 100.0) * self.E_max_kWh

@dataclass
class BESS:
    # capacité physique
    E_max_kWh: float = 600.0
    P_ch_max_kW: float = 120.0
    P_dis_max_kW: float = 240.0
    eff: float = 0.87

    # bornes d'exploitation SOC
    soc_min_pct: float = 20.0
    soc_max_pct: float = 80.0

    # état
    E_kWh: float = 360.0

    def E_min_oper_kWh(self) -> float:
        return (self.soc_min_pct / 100.0) * self.E_max_kWh

    def E_max_oper_kWh(self) -> float:
        return (self.soc_max_pct / 100.0) * self.E_max_kWh

    def soc_pct(self) -> float:
        return 100.0 * self.E_kWh / self.E_max_kWh

    def clamp_operating_range(self):
        self.E_kWh = float(np.clip(self.E_kWh, self.E_min_oper_kWh(), self.E_max_oper_kWh()))

# ============================================================
# 2) Scénarios
# ============================================================
SCENARIOS: Dict[str, Dict] = {
    "pluvieux": {
        "pv_peak_kW": 25.0, "pv_width_min": 240,
        "load_base_kW": 60.0, "load_multiplier": 1.00,
        "n_evs": 20, "seed_evs": 42,
        "bess_soc0": 0.60,
    },
    "nuageux": {
        "pv_peak_kW": 120.0, "pv_width_min": 260,
        "load_base_kW": 60.0, "load_multiplier": 1.00,
        "n_evs": 20, "seed_evs": 42,
        "bess_soc0": 0.60,
    },
    "ensoleille": {
        "pv_peak_kW": 200.0, "pv_width_min": 300,
        "load_base_kW": 60.0, "load_multiplier": 1.00,
        "n_evs": 20, "seed_evs": 42,
        "bess_soc0": 0.60,
    },
    "ete_canicul": {
        "pv_peak_kW": 230.0, "pv_width_min": 320,
        "load_base_kW": 70.0, "load_multiplier": 1.15,
        "n_evs": 22, "seed_evs": 7,
        "bess_soc0": 0.50,
    },
    "hiver_froid": {
        "pv_peak_kW": 80.0, "pv_width_min": 220,
        "load_base_kW": 75.0, "load_multiplier": 1.20,
        "n_evs": 18, "seed_evs": 101,
        "bess_soc0": 0.70,
    },
    "pic_vehicules": {
        "pv_peak_kW": 120.0, "pv_width_min": 260,
        "load_base_kW": 60.0, "load_multiplier": 1.00,
        "n_evs": 35, "seed_evs": 123,
        "bess_soc0": 0.60,
    },
}

# ============================================================
# 3) Profils PV / Load
# ============================================================
def pv_profile_kW(t_min: int, pv_peak_kW: float, pv_width_min: float) -> float:
    x = (t_min - 720) / (pv_width_min / 2.0)
    return float(pv_peak_kW * np.exp(-0.5 * x * x))

def load_profile_kW(t_min: int, base_kW: float, mult: float) -> float:
    hour = t_min / 60.0
    morning = 40.0 * np.exp(-0.5 * ((hour - 9.0) / 1.5) ** 2)
    noon = 55.0 * np.exp(-0.5 * ((hour - 13.0) / 2.0) ** 2)
    afternoon = 45.0 * np.exp(-0.5 * ((hour - 17.5) / 1.8) ** 2)
    return float(mult * (base_kW + morning + noon + afternoon))

# ============================================================
# 4) Génération parc EV
# ============================================================
def generate_evs(seed: int, n_evs: int) -> List[EV]:
    rng = np.random.default_rng(seed)
    profiles = rng.choice(["Urgences", "Administration", "Autres"], size=n_evs, p=[0.2, 0.3, 0.5])

    evs: List[EV] = []
    for i in range(n_evs):
        arrival = int(rng.integers(7 * 60, 18 * 60))
        stay = int(rng.integers(60, 8 * 60))
        departure = min(arrival + stay, 23 * 60 + 55)

        E_max = float(rng.choice([50.0, 52.0, 60.0]))
        P_max = float(rng.choice([6.6, 11.0, 22.0]))
        soc_ini = float(rng.integers(10, 70))
        soc_target = float(rng.choice([60.0, 70.0, 80.0]))
        soc_target = max(soc_target, soc_ini + 5)
        soc_target = min(soc_target, 100.0)

        evs.append(
            EV(
                ev_id=f"EV{i+1}",
                profile=str(profiles[i]),
                arrival_min=arrival,
                departure_min=departure,
                E_max_kWh=E_max,
                P_max_kW=P_max,
                soc_ini_pct=soc_ini,
                soc_target_pct=soc_target,
            )
        )
    return evs

# ============================================================
# 5) Allocation puissance EV
# ============================================================
def allocate_ev_power(
    evs: List[EV],
    t_min: int,
    dt_h: float,
    nb_bornes: int,
    P_borne_uni_max: float,
    P_borne_tot_max: float,
) -> Dict[str, float]:
    active = [ev for ev in evs if ev.active(t_min)]
    if not active:
        return {}

    scored: List[Tuple[EV, float]] = []
    for ev in active:
        score = vehicule_prioritisation(
            user_profile=ev.profile,
            t_now_min=t_min,
            departure_min=ev.departure_min,
            charge_level=ev.soc_pct / 100.0,
        )
        scored.append((ev, float(score)))

    scored.sort(key=lambda x: x[1], reverse=True)
    selected = scored[:nb_bornes]

    alloc: Dict[str, float] = {}
    remaining_total = float(P_borne_tot_max)

    for ev, _s in selected:
        if remaining_total <= 1e-9:
            break

        P_cap = min(ev.P_max_kW, P_borne_uni_max)
        need_kWh = ev.remaining_need_kWh()
        P_need = need_kWh / dt_h if dt_h > 0 else 0.0
        P_set = min(P_cap, P_need, remaining_total)

        if P_set > 1e-9:
            alloc[ev.ev_id] = float(P_set)
            remaining_total -= P_set

    return alloc

# ============================================================
# 6) KPI
# ============================================================
def jain_index(x: np.ndarray) -> float:
    x = np.array(x, dtype=float)
    if np.all(x <= 1e-12):
        return 1.0
    return float((x.sum() ** 2) / (len(x) * (x**2).sum() + 1e-12))

# ============================================================
# 7) Simulation journée (EMS) — BESS SOC bornée [20%, 80%]
# ============================================================
def simulate_day(
    scenario_name: str,
    dt_min: int = 5,
    nb_bornes: int = 6,
    P_borne_uni_max: float = 22.0,
    P_borne_tot_max: float = 132.0,
    P_grid_max: float = 300.0,
    tariff_hp: float = 0.24,
    tariff_hc: float = 0.17,
    bess_soc_min_pct: float = 20.0,
    bess_soc_max_pct: float = 80.0,
) -> Dict:
    sc = SCENARIOS[scenario_name]

    bess = BESS(
        E_max_kWh=600.0,
        P_ch_max_kW=120.0,
        P_dis_max_kW=240.0,
        eff=0.87,
        soc_min_pct=float(bess_soc_min_pct),
        soc_max_pct=float(bess_soc_max_pct),
        E_kWh=float(sc["bess_soc0"] * 600.0),
    )
    bess.clamp_operating_range()

    evs = generate_evs(seed=int(sc["seed_evs"]), n_evs=int(sc["n_evs"]))
    ev_ids = [ev.ev_id for ev in evs]

    n_steps = int(24 * 60 / dt_min)
    t_min_arr = np.arange(n_steps) * dt_min
    dt_h = dt_min / 60.0

    pv_hist = np.zeros(n_steps)
    load_hist = np.zeros(n_steps)
    ev_hist = np.zeros(n_steps)
    grid_imp_hist = np.zeros(n_steps)
    grid_exp_hist = np.zeros(n_steps)
    bess_p_hist = np.zeros(n_steps)
    bess_soc_hist = np.zeros(n_steps)

    sci_inst_hist = np.full(n_steps, np.nan)
    cost_hp_cum = np.zeros(n_steps)
    cost_hc_cum = np.zeros(n_steps)

    soc_ev_hist = np.zeros((n_steps, len(evs)))

    nb_bornes_used_hist = np.zeros(n_steps)
    ev_pmax_single_hist = np.zeros(n_steps)
    ev_ptotal_hist = np.zeros(n_steps)
    bess_dis_hist = np.zeros(n_steps)
    bess_ch_hist = np.zeros(n_steps)

    pv_energy_kWh = 0.0
    export_energy_kWh = 0.0
    cum_hp = 0.0
    cum_hc = 0.0

    E_min_oper = bess.E_min_oper_kWh()
    E_max_oper = bess.E_max_oper_kWh()

    for k in range(n_steps):
        t_min = int(t_min_arr[k])
        hour = t_min / 60.0
        is_hc = (hour >= 22.0) or (hour < 6.0)

        pv = pv_profile_kW(t_min, sc["pv_peak_kW"], sc["pv_width_min"])
        load = load_profile_kW(t_min, sc["load_base_kW"], sc["load_multiplier"])

        pv_hist[k] = pv
        load_hist[k] = load

        pv_to_build = min(pv, load)
        remaining_pv = pv - pv_to_build
        deficit_build = load - pv_to_build
        grid_import = deficit_build

        alloc = allocate_ev_power(
            evs=evs,
            t_min=t_min,
            dt_h=dt_h,
            nb_bornes=nb_bornes,
            P_borne_uni_max=P_borne_uni_max,
            P_borne_tot_max=P_borne_tot_max,
        )
        P_ev_target = float(sum(alloc.values()))

        pv_to_ev = min(remaining_pv, P_ev_target)
        remaining_pv -= pv_to_ev
        ev_need_after_pv = P_ev_target - pv_to_ev

        # BESS -> EV (limité par SOC_min)
        P_bess_dis_internal = 0.0
        P_bess_to_ev = 0.0
        if ev_need_after_pv > 1e-9 and bess.E_kWh > E_min_oper + 1e-9:
            P_req_internal = ev_need_after_pv / bess.eff
            max_by_power = bess.P_dis_max_kW
            max_by_energy = (bess.E_kWh - E_min_oper) / dt_h if dt_h > 0 else 0.0
            P_bess_dis_internal = min(P_req_internal, max_by_power, max_by_energy)
            P_bess_to_ev = P_bess_dis_internal * bess.eff
            bess.E_kWh -= P_bess_dis_internal * dt_h

        ev_need_after_pv_bess = ev_need_after_pv - P_bess_to_ev

        grid_import += max(ev_need_after_pv_bess, 0.0)

        # Grid limit: reduce EV first
        if grid_import > P_grid_max + 1e-9:
            excess = grid_import - P_grid_max
            to_reduce = min(excess, P_ev_target)
            P_ev_target -= to_reduce
            grid_import -= to_reduce

            for ev_id in list(alloc.keys())[::-1]:
                if to_reduce <= 1e-9:
                    break
                cut = min(alloc[ev_id], to_reduce)
                alloc[ev_id] -= cut
                to_reduce -= cut
                if alloc[ev_id] <= 1e-9:
                    del alloc[ev_id]

        P_ev_final = float(sum(alloc.values()))
        ev_hist[k] = P_ev_final

        # Charge BESS depuis grid en HC (limité par SOC_max)
        P_bess_ch_from_grid = 0.0
        if is_hc and bess.E_kWh < E_max_oper - 1e-9:
            margin = max(P_grid_max - grid_import, 0.0)
            if margin > 1e-9:
                max_by_power = bess.P_ch_max_kW
                max_by_energy = (E_max_oper - bess.E_kWh) / (dt_h * bess.eff) if dt_h > 0 else 0.0
                P_bess_ch_from_grid = min(margin, max_by_power, max_by_energy)
                grid_import += P_bess_ch_from_grid
                bess.E_kWh += P_bess_ch_from_grid * dt_h * bess.eff

        # PV restant -> charge BESS puis export (limité par SOC_max)
        P_bess_ch_from_pv = 0.0
        grid_export = 0.0
        if remaining_pv > 1e-9:
            if bess.E_kWh < E_max_oper - 1e-9:
                max_by_power = bess.P_ch_max_kW
                max_by_energy = (E_max_oper - bess.E_kWh) / (dt_h * bess.eff) if dt_h > 0 else 0.0
                P_bess_ch_from_pv = min(remaining_pv, max_by_power, max_by_energy)
                bess.E_kWh += P_bess_ch_from_pv * dt_h * bess.eff
                remaining_pv -= P_bess_ch_from_pv
            grid_export = max(remaining_pv, 0.0)

        bess.clamp_operating_range()

        # Update EV SOC
        for i_ev, ev in enumerate(evs):
            if ev.ev_id in alloc:
                P_i = alloc[ev.ev_id]
                e_i = P_i * dt_h
                ev.energy_received_kWh += e_i
                ev.soc_pct = min(100.0, ev.soc_pct + 100.0 * e_i / ev.E_max_kWh)
            soc_ev_hist[k, i_ev] = ev.soc_pct

        grid_imp_hist[k] = grid_import
        grid_exp_hist[k] = grid_export
        bess_p_hist[k] = (P_bess_dis_internal) - (P_bess_ch_from_grid + P_bess_ch_from_pv)
        bess_soc_hist[k] = bess.soc_pct()

        nb_bornes_used_hist[k] = len(alloc)
        ev_ptotal_hist[k] = P_ev_final
        ev_pmax_single_hist[k] = max(alloc.values()) if len(alloc) > 0 else 0.0
        bess_dis_hist[k] = P_bess_dis_internal
        bess_ch_hist[k] = (P_bess_ch_from_grid + P_bess_ch_from_pv)

        # SCI
        pv_e = pv * dt_h
        exp_e = grid_export * dt_h
        pv_energy_kWh += pv_e
        export_energy_kWh += exp_e
        if pv_e > 1e-12:
            sci_inst_hist[k] = 100.0 * (pv_e - exp_e) / pv_e

        # Cost (import only)
        price = tariff_hc if is_hc else tariff_hp
        cost_step = grid_import * dt_h * price
        if is_hc:
            cum_hc += cost_step
        else:
            cum_hp += cost_step
        cost_hp_cum[k] = cum_hp
        cost_hc_cum[k] = cum_hc

    sci_global = 100.0 * (pv_energy_kWh - export_energy_kWh) / (pv_energy_kWh + 1e-12)
    energies = np.array([ev.energy_received_kWh for ev in evs], dtype=float)
    iej = jain_index(energies)
    peak_imp = float(np.max(grid_imp_hist))
    E_import_total = float(np.sum(grid_imp_hist) * dt_h)
    cost_total = float(cost_hp_cum[-1] + cost_hc_cum[-1])

    return {
        "scenario": scenario_name,
        "t_min": t_min_arr,
        "pv_kW": pv_hist,
        "load_kW": load_hist,
        "ev_kW": ev_hist,
        "grid_import_kW": grid_imp_hist,
        "grid_export_kW": grid_exp_hist,
        "bess_power_kW": bess_p_hist,
        "bess_soc_pct": bess_soc_hist,
        "soc_ev_pct": soc_ev_hist,
        "ev_ids": ev_ids,
        "SCI_inst_pct": sci_inst_hist,
        "SCI_global_pct": float(sci_global),
        "IEJ": float(iej),
        "peak_grid_import_kW": peak_imp,
        "E_import_total_kWh": E_import_total,
        "cost_hp_cum": cost_hp_cum,
        "cost_hc_cum": cost_hc_cum,
        "cost_total": cost_total,
        "nb_bornes_used": nb_bornes_used_hist,
        "ev_pmax_single_kW": ev_pmax_single_hist,
        "ev_ptotal_kW": ev_ptotal_hist,
        "bess_dis_internal_kW": bess_dis_hist,
        "bess_ch_internal_kW": bess_ch_hist,
        "bess_limits": {
            "E_max_kWh": bess.E_max_kWh,
            "soc_min_pct": bess.soc_min_pct,
            "soc_max_pct": bess.soc_max_pct,
            "E_min_oper_kWh": bess.E_min_oper_kWh(),
            "E_max_oper_kWh": bess.E_max_oper_kWh(),
            "P_ch_max_kW": bess.P_ch_max_kW,
            "P_dis_max_kW": bess.P_dis_max_kW,
        },
        "sim_params": {
            "dt_min": dt_min,
            "nb_bornes": nb_bornes,
            "P_borne_uni_max": P_borne_uni_max,
            "P_borne_tot_max": P_borne_tot_max,
            "P_grid_max": P_grid_max,
        }
    }

# ============================================================
# 8) Audit contraintes
# ============================================================
def audit_constraints(res: Dict) -> pd.DataFrame:
    sp = res["sim_params"]
    bl = res["bess_limits"]

    P_grid_max = float(sp["P_grid_max"])
    nb_bornes = int(sp["nb_bornes"])
    P_borne_uni_max = float(sp["P_borne_uni_max"])
    P_borne_tot_max = float(sp["P_borne_tot_max"])

    P_dis_max = float(bl["P_dis_max_kW"])
    P_ch_max = float(bl["P_ch_max_kW"])
    soc_floor = float(bl["soc_min_pct"])
    soc_ceil = float(bl["soc_max_pct"])

    def status(ok: bool) -> str:
        return "OK" if ok else "ERROR"

    peak_grid = float(np.max(res["grid_import_kW"]))
    max_bornes_used = int(np.max(res["nb_bornes_used"]))
    max_ev_single = float(np.max(res["ev_pmax_single_kW"]))
    max_ev_total = float(np.max(res["ev_ptotal_kW"]))
    max_bess_dis = float(np.max(res["bess_dis_internal_kW"]))
    max_bess_ch = float(np.max(res["bess_ch_internal_kW"]))
    soc_min = float(np.min(res["bess_soc_pct"]))
    soc_max = float(np.max(res["bess_soc_pct"]))
    iej = float(res["IEJ"])
    sci = float(res["SCI_global_pct"])

    rows = [
        {"Contrainte": f"Pic import réseau ≤ {P_grid_max:.0f} kW", "Résultat": f"{peak_grid:.2f} kW", "Statut": status(peak_grid <= P_grid_max + 1e-9)},
        {"Contrainte": f"Nb bornes utilisées ≤ {nb_bornes}", "Résultat": f"{max_bornes_used:d}", "Statut": status(max_bornes_used <= nb_bornes)},
        {"Contrainte": f"Puissance max par VE ≤ {P_borne_uni_max:.1f} kW", "Résultat": f"{max_ev_single:.2f} kW", "Statut": status(max_ev_single <= P_borne_uni_max + 1e-9)},
        {"Contrainte": f"Puissance totale VE ≤ {P_borne_tot_max:.1f} kW", "Résultat": f"{max_ev_total:.2f} kW", "Statut": status(max_ev_total <= P_borne_tot_max + 1e-9)},
        {"Contrainte": f"Décharge BESS (interne) ≤ {P_dis_max:.1f} kW", "Résultat": f"{max_bess_dis:.2f} kW", "Statut": status(max_bess_dis <= P_dis_max + 1e-9)},
        {"Contrainte": f"Charge BESS (interne) ≤ {P_ch_max:.1f} kW", "Résultat": f"{max_bess_ch:.2f} kW", "Statut": status(max_bess_ch <= P_ch_max + 1e-9)},
        {"Contrainte": f"SOC BESS min ≥ {soc_floor:.1f} %", "Résultat": f"{soc_min:.2f} %", "Statut": status(soc_min + 1e-9 >= soc_floor)},
        {"Contrainte": f"SOC BESS max ≤ {soc_ceil:.1f} %", "Résultat": f"{soc_max:.2f} %", "Statut": status(soc_max <= soc_ceil + 1e-9)},
        {"Contrainte": "Indice de Jain ∈ [0, 1]", "Résultat": f"{iej:.4f}", "Statut": status(0.0 - 1e-9 <= iej <= 1.0 + 1e-9)},
        {"Contrainte": "SCI global ∈ [0, 100] %", "Résultat": f"{sci:.2f} %", "Statut": status(0.0 - 1e-9 <= sci <= 100.0 + 1e-9)},
    ]

    return pd.DataFrame(rows, columns=["Contrainte", "Résultat", "Statut"])

def audit_all_scenarios(results: List[Dict]) -> pd.DataFrame:
    frames = []
    for res in results:
        df = audit_constraints(res).copy()
        df.insert(0, "Scénario", res["scenario"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def results_timeseries_df(res: Dict) -> pd.DataFrame:
    t_h = res["t_min"] / 60.0
    df = pd.DataFrame({
        "hour": t_h,
        "pv_kW": res["pv_kW"],
        "load_kW": res["load_kW"],
        "ev_kW": res["ev_kW"],
        "grid_import_kW": res["grid_import_kW"],
        "grid_export_kW": res["grid_export_kW"],
        "bess_power_kW": res["bess_power_kW"],
        "bess_soc_pct": res["bess_soc_pct"],
        "SCI_inst_pct": res["SCI_inst_pct"],
        "cost_hp_cum": res["cost_hp_cum"],
        "cost_hc_cum": res["cost_hc_cum"],
    })
    return df