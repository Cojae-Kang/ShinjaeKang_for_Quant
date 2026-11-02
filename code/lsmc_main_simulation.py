# ==== PACKET HEADER ====
# TARGET : Table 4 (main) / Table S7 (27-scenario grid) / tau paths used by Fig 6
# INPUT  : none on disk: all parameters hard-coded (Table 1-3 of the manuscript); seed 42, N=100k
# OUTPUT : 04_Data/A_Deposit_v1_data/derived/lsmc_27_scenarios_results.csv (+ tau_*.npy, lsmc_summary.txt)
# NOTE   : deposit package v1 file name kept (README/LICENSE refer to it); R8 reproduction gate: byte-identical CSV
# ORIGINAL: lsmc_main_simulation.py  (packet copy; the original file is unchanged)
# ============================================
"""
LSMC Real Options Valuation — Biochar CDR Project
===================================================
S. Kang and J. Kim (KAIST) --- companion code for the article

Computes ENPV and Option Premium for 27 scenarios:
  3 carbon price models  (GBM / G-OU / MS-DR)
  × 3 biochar scenarios  (Low $250 / Base $981 / High $1,712 /t)
  × 3 feedstock scenarios (Low $54.9 / Base $82.4 / High $109.9 /dry-t)

Algorithm  : Longstaff–Schwartz Monte Carlo — annual-step Bermudan deferral option
             Investor may invest at any year t = 1 … 9 (or abandon).
             If invest at t = τ, pays TCI at τ, receives ATCFs for years τ+1 … T.
Discount   : WACC = 8.0 % for ATCF present-value  |  r_f = 4.1 % for continuation value
Basis      : 14-feature multivariate polynomial (4 linear + 4 quadratic + 6 cross-terms)
Paths      : N_TRAIN = N_EVAL = 100,000  (set N_QUICK = 10,000 for test runs)
Units      : all cash-flow matrices are in USD millions ($M) throughout.


Outputs (in results/ sub-folder):
  lsmc_27_scenarios_0705.csv  — full 27-row result table
  lsmc_summary_0705.txt       — human-readable summary
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import os, time

np.random.seed(42)
# Working-capital recovery adopted (option (a), see the accompanying article):
#   WC recovered in full, non-taxable, at Year 10 (as stated in tex 3.2).
#   Terms added to: terminal V, iv_tgt (train), static NPV, iv_actual (eval), iv_T.


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIRMED PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

FCI  = 3_960_062.0          # Fixed Capital Investment ($)
WC   = 594_009.0            # Working Capital ($)
TCI  = FCI + WC             # Total Capital Investment = $4,554,071

# MACRS 7-yr, 200% DB, half-year convention  (years 1–10)
MACRS_RATES = np.array([0.1429, 0.2449, 0.1749, 0.1249,
                         0.0893, 0.0892, 0.0893, 0.0446,
                         0.0,    0.0])
DEP = FCI * MACRS_RATES     # $/ yr

WACC   = 0.080
RF     = 0.041
TAX    = 0.250
T_YRS  = 10                 # project evaluation horizon

Q_BC   = 3_000.0            # biochar production  t / yr
Q_NET  = 6_605.79           # net CDR credits    tCO2e / yr
Q_FEED = 10_000.0           # feedstock input    dry-t / yr
Q_ELEC = 720_000.0          # electricity        kWh / yr

OPEX_FIXED     = 863_243.0  # labor+maint+insur+audit+rent+overhead  ($/yr)
OPEX_OTHER_VAR = 104_935.0  # propane $4,935 + packaging $60,000 + consumables $40,000

# TCI in $M (used throughout in cash-flow matrices)
TCI_M = TCI / 1e6           # 4.554071
WC_M  = WC / 1e6           # 0.594009 $M, recovered Year 10

# ── Carbon price model parameters ────────────────────────────────────────────
PC0 = 150.75                # current carbon price  $/tCO2e  (2024 VCM CA)

MU_GBM  = 0.0618;   SIG_GBM  = 0.3858           # GBM

THETA_GOU = 149.56;  KAPPA_GOU = 11.8073;  SIG_GOU = 0.4885  # G-OU

# MS-DR  state 0 = Spike, state 1 = Base
MU_SPIKE = 0.6699;   SIG_SPIKE = 0.0679
MU_BASE  = -0.1284;  SIG_BASE  = 0.4155
P01_MONTH = 1.0;     P10_MONTH = 0.2915   # monthly transition
_Pm  = np.array([[1-P01_MONTH, P01_MONTH], [P10_MONTH, 1-P10_MONTH]])
_Py  = np.linalg.matrix_power(_Pm, 12)    # annual = P_month^12
P01_YR = _Py[0, 1];  P10_YR = _Py[1, 0]

# ── Electricity G-OU parameters ───────────────────────────────────────────────
THETA_ELEC = 0.173269;  KAPPA_ELEC = 1.3918;  SIG_ELEC = 0.2736
# Jensen-corrected level-space mean: E[P_e∞] = theta * exp(-sigma^2 / (4*kappa))
PE0 = THETA_ELEC * np.exp(-(SIG_ELEC**2) / (4 * KAPPA_ELEC))   # ≈ 0.17100 $/kWh

# ── Scenario grids ────────────────────────────────────────────────────────────
PB_GRID = {'Low': 250.0, 'Base': 981.0, 'High': 1_712.0}   # $/t
CF_GRID = {'Low': 54.9,  'Base': 82.4,  'High': 109.9}     # $/dry-t

TRI_PB = (250.0, 981.0, 1_712.0)   # training triangular distribution (min, mode, max)
TRI_CF = (54.9,  82.4,  109.9)

# ── Path counts — change N_QUICK to 10_000 for fast test run ─────────────────
N_TRAIN = 100_000
N_EVAL  = 100_000

# ══════════════════════════════════════════════════════════════════════════════
# 2.  PATH GENERATION  (all paths shape: N × (T_YRS+1))
# ══════════════════════════════════════════════════════════════════════════════

def gen_gbm(p0: float, n: int) -> np.ndarray:
    paths = np.empty((n, T_YRS + 1))
    paths[:, 0] = p0
    drift = MU_GBM - 0.5 * SIG_GBM**2
    for t in range(1, T_YRS + 1):
        paths[:, t] = paths[:, t-1] * np.exp(drift + SIG_GBM * np.random.standard_normal(n))
    return paths


def gen_gou(p0: float, n: int,
            theta: float = THETA_GOU,
            kappa: float = KAPPA_GOU,
            sigma: float = SIG_GOU) -> np.ndarray:
    mu_log  = np.log(theta) - sigma**2 / (2 * kappa)
    exp_k   = np.exp(-kappa)                                  # DT = 1 yr
    std_log = sigma * np.sqrt((1 - exp_k**2) / (2 * kappa))
    paths   = np.empty((n, T_YRS + 1))
    paths[:, 0] = p0
    log_x = np.full(n, np.log(p0))
    for t in range(1, T_YRS + 1):
        log_x = log_x * exp_k + mu_log * (1 - exp_k) + std_log * np.random.standard_normal(n)
        paths[:, t] = np.exp(log_x)
    return paths


def gen_msdr(p0: float, n: int) -> np.ndarray:
    paths  = np.empty((n, T_YRS + 1))
    paths[:, 0] = p0
    states = np.ones(n, dtype=int)        # start in Base (1)
    for t in range(1, T_YRS + 1):
        u = np.random.uniform(size=n)
        # --- BUG FIX 2026-06-22: preserve old state before transition ---
        # OLD (buggy, in-place): the second line re-read `states` already
        # modified by the first line using the SAME u, causing a 1->0->1
        # bounce whenever u < P10_YR < P01_YR. The spike state (0) was thus
        # never reachable from base (1); with all paths starting in base,
        # MS-DR degenerated to a pure base-regime (mu=-12.84%/yr) lognormal,
        # and the spike regime (+66.99%/yr) was entirely absent.
        #     states[((states == 1) & (u < P10_YR))] = 0
        #     states[((states == 0) & (u < P01_YR))] = 1
        old = states.copy()
        states[(old == 1) & (u < P10_YR)] = 0
        states[(old == 0) & (u < P01_YR)] = 1
        mu  = np.where(states == 0, MU_SPIKE,  MU_BASE)
        sig = np.where(states == 0, SIG_SPIKE, SIG_BASE)
        paths[:, t] = paths[:, t-1] * np.exp(
            (mu - 0.5 * sig**2) + sig * np.random.standard_normal(n))
    return paths


def gen_elec(n: int) -> np.ndarray:
    """Electricity paths starting at Jensen-corrected PE0."""
    return gen_gou(PE0, n, theta=THETA_ELEC, kappa=KAPPA_ELEC, sigma=SIG_ELEC)


def gen_carbon(model: str, p0: float, n: int) -> np.ndarray:
    return {'GBM': gen_gbm, 'G-OU': gen_gou, 'MS-DR': gen_msdr}[model](p0, n)

# ══════════════════════════════════════════════════════════════════════════════
# 3.  AFTER-TAX CASH FLOW  (returns $M matrix, shape N × (T_YRS+1))
# ══════════════════════════════════════════════════════════════════════════════

def compute_atcf_M(pc: np.ndarray, pe: np.ndarray,
                   cf: np.ndarray, pb: np.ndarray) -> np.ndarray:
    """
    After-tax cash flow in $M for each path and year.
    atcf[:, 0] = 0 (t=0 placeholder).
    atcf[:, t] = after-tax operating CF at year t  ($M), t = 1 … T_YRS.

    Tax rule:  EBIT = EBITDA – D_t
      EBIT > 0  →  tax = EBIT × TAX   →  ATCF = EBITDA – tax  = EBITDA(1-TAX) + D_t×TAX
      EBIT ≤ 0  →  tax = 0            →  ATCF = EBITDA          (D_t cancels)
    """
    n    = pc.shape[0]
    atcf = np.zeros((n, T_YRS + 1))
    for t in range(1, T_YRS + 1):
        rev    = pb * Q_BC + pc[:, t] * Q_NET
        opex   = OPEX_FIXED + cf * Q_FEED + pe[:, t] * Q_ELEC + OPEX_OTHER_VAR
        ebitda = rev - opex
        ebit   = ebitda - DEP[t - 1]
        tax    = np.where(ebit > 0, ebit * TAX, 0.0)
        atcf[:, t] = (ebitda - tax) / 1e6   # convert to $M
    return atcf

# ══════════════════════════════════════════════════════════════════════════════
# 4.  REGRESSION BASIS  (14 features; intercept added by LinearRegression)
# ══════════════════════════════════════════════════════════════════════════════

def basis(pc: np.ndarray, pe: np.ndarray,
          cf: np.ndarray, pb: np.ndarray) -> np.ndarray:
    return np.column_stack([
        pc,    pe,    cf,    pb,
        pc**2, pe**2, cf**2, pb**2,
        pc*pe, pc*cf, pc*pb,
        pe*cf, pe*pb,
        cf*pb
    ])

# ══════════════════════════════════════════════════════════════════════════════
# 5.  LSMC TRAINING — backward induction on broad triangular distribution
# ══════════════════════════════════════════════════════════════════════════════

def train_lsmc(model: str) -> tuple[list, list, StandardScaler]:
    """
    Learn optimal exercise policy from N_TRAIN broad-distribution paths.

    At each t, fit two OLS models on scaled basis functions:
      reg_cont[t]: E[V*(t+1)/(1+rf) | S_t]  — continuation value
      reg_iv[t]  : E[IV_t | S_t]             — expected intrinsic value

    IV_t (in $M, from t's perspective) = sum_{k=t+1}^{T} ATCF_k/(1+WACC)^{k-t} – TCI_M

    V*(t) is updated to max(IV_t, continuation_t) at each step.
    V*(T) = max(ATCF_T – TCI_M, 0)  — terminal boundary (≈ 0 for all practical paths).

    Returns reg_cont, reg_iv (lists indexed by t=1..T-1), and fitted StandardScaler.
    """
    print(f"  [TRAIN] {model}  N={N_TRAIN:,} ...", end=" ", flush=True)
    t0 = time.time()

    pb_tr = np.random.triangular(*TRI_PB, size=N_TRAIN)
    cf_tr = np.random.triangular(*TRI_CF, size=N_TRAIN)
    pc_tr = gen_carbon(model, PC0, N_TRAIN)
    pc_tr[:, 0] = np.random.uniform(50, 600, N_TRAIN)  # broad initial spread
    pe_tr = gen_elec(N_TRAIN)

    atcf = compute_atcf_M(pc_tr, pe_tr, cf_tr, pb_tr)

    # Terminal: max(ATCF_T - TCI_M, 0)  — in $M at time T
    V = np.maximum(atcf[:, T_YRS] - TCI_M + WC_M, 0.0)  # [WC] invest at T -> immediate recovery

    # Fit scaler on t=1 features for numerical stability
    X1     = basis(pc_tr[:, 1], pe_tr[:, 1], cf_tr, pb_tr)
    scaler = StandardScaler().fit(X1)

    reg_cont = [None] * (T_YRS + 1)
    reg_iv   = [None] * (T_YRS + 1)

    for t in range(T_YRS - 1, 0, -1):
        # Continuation value: E[V*(t+1)/(1+rf) | S_t]
        cont_tgt = V / (1 + RF)

        # Intrinsic value at t: NPV if investing NOW (in $M, from t's view)
        disc = np.array([(1 + WACC) ** (-(k - t)) for k in range(t + 1, T_YRS + 1)])
        iv_tgt = np.dot(atcf[:, t+1:], disc) - TCI_M + WC_M * (1 + WACC) ** (-(T_YRS - t))  # [WC]

        Xt = scaler.transform(basis(pc_tr[:, t], pe_tr[:, t], cf_tr, pb_tr))
        rc = LinearRegression().fit(Xt, cont_tgt)
        ri = LinearRegression().fit(Xt, iv_tgt)
        reg_cont[t] = rc
        reg_iv[t]   = ri

        iv_hat  = ri.predict(Xt)
        phi_hat = rc.predict(Xt)
        # Update V: exercise (take IV) if predicted IV > continuation
        V = np.where(iv_hat > phi_hat, iv_tgt, cont_tgt)

    print(f"done in {time.time()-t0:.1f}s")
    return reg_cont, reg_iv, scaler

# ══════════════════════════════════════════════════════════════════════════════
# 6.  LSMC EVALUATION — forward pass for one (P_b, C_f) scenario
# ══════════════════════════════════════════════════════════════════════════════

def eval_scenario(model: str, pb_val: float, cf_val: float,
                  reg_cont: list, reg_iv: list,
                  scaler: StandardScaler) -> dict:
    """
    For a fixed (model, P_b, C_f) scenario:
      1. Generate N_EVAL fresh carbon + electricity paths from PC0.
      2. Compute Static NPV = E[NPV | invest at t=0] using WACC.
      3. Apply learned exercise policy to find optimal τ* per path.
      4. Compute deferral option value = E[IV(τ*) * (1+rf)^{-τ*}].
      5. ENPV = max(Static NPV, deferral value, 0).
      6. Option Premium = max(ENPV – max(Static NPV, 0), 0).
    """
    pc = gen_carbon(model, PC0, N_EVAL)
    pe = gen_elec(N_EVAL)
    pb = np.full(N_EVAL, pb_val)
    cf = np.full(N_EVAL, cf_val)

    atcf = compute_atcf_M(pc, pe, cf, pb)

    # ── Static NPV: invest at t=0, discount ATCFs at WACC ────────────────────
    disc_wacc = np.array([(1 + WACC) ** (-t) for t in range(1, T_YRS + 1)])
    static_npv = float(np.mean(np.dot(atcf[:, 1:], disc_wacc)) - TCI_M
                       + WC_M * (1 + WACC) ** (-T_YRS))  # [WC]

    # ── Deferral option: forward pass using learned policy ────────────────────
    ex_year = np.full(N_EVAL, -1, dtype=int)   # -1 = not yet exercised
    payoffs = np.zeros(N_EVAL)

    for t in range(1, T_YRS):   # t = 1 … T-1 = 9
        if reg_cont[t] is None or reg_iv[t] is None:
            continue
        alive = ex_year == -1
        if not alive.any():
            break

        Xt = scaler.transform(basis(pc[alive, t], pe[alive, t],
                                    cf[alive], pb[alive]))
        phi_hat = reg_cont[t].predict(Xt)
        iv_hat  = reg_iv[t].predict(Xt)

        # Actual intrinsic value (not predicted) for paths being exercised
        disc = np.array([(1 + WACC) ** (-(k - t)) for k in range(t + 1, T_YRS + 1)])
        iv_actual = (np.dot(atcf[alive, t+1:], disc) - TCI_M
                     + WC_M * (1 + WACC) ** (-(T_YRS - t)))  # [WC]

        do_ex     = (iv_hat > phi_hat) & (iv_hat > 0)
        idx_alive = np.where(alive)[0]
        idx_ex    = idx_alive[do_ex]

        ex_year[idx_ex] = t
        payoffs[idx_ex] = iv_actual[do_ex] * (1 + RF) ** (-t)

    # ── Terminal: last chance at t = T (pay TCI, get ATCF_T only) ────────────
    never = ex_year == -1
    if never.any():
        # IV at T (from T's perspective): ATCF_T – TCI_M
        iv_T  = atcf[never, T_YRS] - TCI_M + WC_M  # [WC]
        do_T  = iv_T > 0
        idx_n = np.where(never)[0]
        idx_T = idx_n[do_T]
        ex_year[idx_T] = T_YRS
        payoffs[idx_T] = iv_T[do_T] * (1 + RF) ** (-T_YRS)

    deferral_val = float(np.mean(payoffs))

    # ── ENPV and Option Premium ───────────────────────────────────────────────
    enpv           = max(static_npv, deferral_val, 0.0)
    option_premium = max(enpv - max(static_npv, 0.0), 0.0)

    # ── Exercise timing statistics ────────────────────────────────────────────
    exercised = ex_year[ex_year > 0]
    avg_tau   = float(np.mean(exercised)) if exercised.size > 0 else float('nan')
    abandon_p = float(np.sum(ex_year == -1)) / N_EVAL * 100.0

    return {
        'Static NPV ($M)':     round(static_npv,      4),
        'Deferral Value ($M)': round(deferral_val,     4),
        'ENPV ($M)':           round(enpv,             4),
        'Option Premium ($M)': round(option_premium,   4),
        'Avg tau* (yr)':       round(avg_tau, 2) if not np.isnan(avg_tau) else float('nan'),
        'Abandon (%)':         round(abandon_p, 2),
        'ex_year':             ex_year.copy(),   # per-path exercise year (-1 = abandon)
    }

# ══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    records = []

    for model in ['GBM', 'G-OU', 'MS-DR']:
        print(f"\n{'='*58}")
        print(f"  Carbon model: {model}")
        print(f"{'='*58}")

        reg_cont, reg_iv, scaler = train_lsmc(model)

        for pb_name, pb_val in PB_GRID.items():
            for cf_name, cf_val in CF_GRID.items():
                tag = f"P_b={pb_name}(${pb_val:,.0f})  C_f={cf_name}(${cf_val})"
                print(f"  Eval {tag}", end=" … ", flush=True)
                t0  = time.time()
                res = eval_scenario(model, pb_val, cf_val,
                                    reg_cont, reg_iv, scaler)
                print(f"ENPV={res['ENPV ($M)']:+.3f}M  "
                      f"Prem={res['Option Premium ($M)']:.3f}M  "
                      f"({time.time()-t0:.0f}s)")
                records.append({
                    'Carbon Model':         model,
                    'Biochar Scenario':     pb_name,
                    'Biochar Price ($/t)':  pb_val,
                    'Feedstock Scenario':   cf_name,
                    'Feedstock Cost ($/t)': cf_val,
                    **res,
                })

    # ── Save τ* per-path arrays (for Fig 4.3) ────────────────────────────────
    # 2026-06-22: write results INSIDE Code_0705 (one level up from this engine),
    # so the bug-fixed run stays fully contained in the working copy.
    out_dir  = os.path.normpath(
        os.path.join(os.path.dirname(__file__), '..', 'results'))
    os.makedirs(out_dir, exist_ok=True)
    tau_dir = os.path.join(out_dir, 'tau')
    os.makedirs(tau_dir, exist_ok=True)
    for r in records:
        fname = f"tau_{r['Carbon Model']}_{r['Biochar Scenario']}_{r['Feedstock Scenario']}_0705.npy"
        np.save(os.path.join(tau_dir, fname), r['ex_year'])

    # ── Save CSV (drop ex_year column — not CSV-serialisable) ─────────────────
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'ex_year'}
                        for r in records])
    csv_path = os.path.join(out_dir, 'lsmc_27_scenarios_0705.csv')
    df.to_csv(csv_path, index=False)

    # ── Console summary ───────────────────────────────────────────────────────
    sep = '─' * 100
    print(f"\n{sep}")
    print(f"  LSMC RESULTS  |  N_TRAIN={N_TRAIN:,}  N_EVAL={N_EVAL:,}  seed=42")
    print(f"  WACC={WACC*100:.1f}%  r_f={RF*100:.1f}%  TAX={TAX*100:.0f}%  T={T_YRS}yr  TCI=${TCI_M:.3f}M")
    print(f"  P_e0={PE0*100:.4f} c/kWh (Jensen)  PC0=${PC0}/tCO2e")
    print(sep)
    hdr = (f"{'Model':<8} {'Pb':>6} {'Cf':>6}  "
           f"{'StNPV':>8} {'Defer':>8} {'ENPV':>8} {'OptPrem':>9}  "
           f"{'tau*':>6} {'Aban%':>6}")
    print(hdr)
    print('─' * len(hdr))
    for r in records:
        tau_s = f"{r['Avg tau* (yr)']:.1f}" if not np.isnan(r['Avg tau* (yr)']) else '  N/A'
        print(f"{r['Carbon Model']:<8} "
              f"{r['Biochar Price ($/t)']:>6,.0f} "
              f"{r['Feedstock Cost ($/t)']:>6.1f}  "
              f"{r['Static NPV ($M)']:>+8.3f} "
              f"{r['Deferral Value ($M)']:>8.3f} "
              f"{r['ENPV ($M)']:>+8.3f} "
              f"{r['Option Premium ($M)']:>9.3f}  "
              f"{tau_s:>6} "
              f"{r['Abandon (%)']:>5.1f}%")
    print(sep)
    print(f"\nCSV saved → {csv_path}")

    txt_path = os.path.join(out_dir, 'lsmc_summary_0705.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(df.to_string(index=False))
    print(f"TXT saved → {txt_path}")


if __name__ == '__main__':
    main()
