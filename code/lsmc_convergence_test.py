# ==== PACKET HEADER ====
# TARGET : Table S5 / Fig S4 data (path-count convergence, 5 seeds x 4 N levels)
# INPUT  : none on disk: parameters identical to lsmc_main_simulation.py
# OUTPUT : 04_Data/A_Deposit_v1_data/derived/lsmc_convergence_raw.csv, lsmc_convergence_summary.csv
# NOTE   : deposit package v1 file name kept
# ORIGINAL: lsmc_convergence_test.py  (packet copy; the original file is unchanged)
# ============================================
"""
make_lsmc_convergence_test.py  —  LSMC Path Count Convergence Test
S. Kang and J. Kim (KAIST) --- companion code for the article

Scenario : Base (P_b=$981/t, C_f=$82.4/dry-t) for all 3 models.
N levels : 10,000 / 25,000 / 50,000 / 100,000
Reps     : 5 independent seeds per N level
           Seeds [101, 202, 303, 404, 505] — identical across N for comparability.
Training : N_TRAIN=100,000, seed=42 (same policy as lsmc_main_simulation.py).

Output:
  results/lsmc_convergence_raw_0705.csv     — per-replication ENPV
  results/lsmc_convergence_summary_0705.csv — mean/std/CI/change% per (model, N)
  Console: LaTeX table rows for tab:lsmc_conv_gou and tab:lsmc_conv_gbm_msdr
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import os, time

try:
    from scipy import stats as sc_stats
    T_CRIT_DF4 = float(sc_stats.t.ppf(0.975, df=4))
except ImportError:
    T_CRIT_DF4 = 2.7764   # t_{0.025, df=4}, hardcoded fallback

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS — identical to lsmc_main_simulation.py
# ══════════════════════════════════════════════════════════════════════════════
FCI  = 3_960_062.0
WC   = 594_009.0
TCI  = FCI + WC
MACRS_RATES = np.array([0.1429, 0.2449, 0.1749, 0.1249,
                         0.0893, 0.0892, 0.0893, 0.0446, 0.0, 0.0])
DEP    = FCI * MACRS_RATES
WACC   = 0.080
RF     = 0.041
TAX    = 0.250
T_YRS  = 10
Q_BC   = 3_000.0
Q_NET  = 6_605.79
Q_FEED = 10_000.0
Q_ELEC = 720_000.0
OPEX_FIXED     = 863_243.0
OPEX_OTHER_VAR = 104_935.0
TCI_M = TCI / 1e6
WC_M  = WC / 1e6           # 0.594009 $M, recovered Year 10

PC0 = 150.75
MU_GBM  = 0.0618;   SIG_GBM  = 0.3858
THETA_GOU = 149.56; KAPPA_GOU = 11.8073; SIG_GOU = 0.4885
MU_SPIKE = 0.6699;  SIG_SPIKE = 0.0679
MU_BASE  = -0.1284; SIG_BASE  = 0.4155
P01_MONTH = 1.0;    P10_MONTH = 0.2915
_Pm  = np.array([[1 - P01_MONTH, P01_MONTH],
                  [P10_MONTH,     1 - P10_MONTH]])
_Py  = np.linalg.matrix_power(_Pm, 12)
P01_YR = _Py[0, 1];  P10_YR = _Py[1, 0]
THETA_ELEC = 0.173269; KAPPA_ELEC = 1.3918; SIG_ELEC = 0.2736
PE0 = THETA_ELEC * np.exp(-(SIG_ELEC**2) / (4 * KAPPA_ELEC))
TRI_PB = (250.0, 981.0, 1_712.0)
TRI_CF = (54.9,  82.4,  109.9)
N_TRAIN = 100_000

# ── Convergence test settings ──────────────────────────────────────────────────
CONV_N_LIST = [10_000, 25_000, 50_000, 100_000]
CONV_N_REPS = 5
EVAL_SEEDS  = [101, 202, 303, 404, 505]
CONV_PB     = 981.0
CONV_CF     = 82.4

# ══════════════════════════════════════════════════════════════════════════════
# PATH GENERATORS
# ══════════════════════════════════════════════════════════════════════════════
def gen_gbm(p0, n):
    paths = np.empty((n, T_YRS + 1))
    paths[:, 0] = p0
    drift = MU_GBM - 0.5 * SIG_GBM**2
    for t in range(1, T_YRS + 1):
        paths[:, t] = paths[:, t-1] * np.exp(
            drift + SIG_GBM * np.random.standard_normal(n))
    return paths

def gen_gou(p0, n, theta=THETA_GOU, kappa=KAPPA_GOU, sigma=SIG_GOU):
    mu_log  = np.log(theta) - sigma**2 / (2 * kappa)
    exp_k   = np.exp(-kappa)
    std_log = sigma * np.sqrt((1 - exp_k**2) / (2 * kappa))
    paths   = np.empty((n, T_YRS + 1))
    paths[:, 0] = p0
    log_x = np.full(n, np.log(p0))
    for t in range(1, T_YRS + 1):
        log_x = (log_x * exp_k
                 + mu_log * (1 - exp_k)
                 + std_log * np.random.standard_normal(n))
        paths[:, t] = np.exp(log_x)
    return paths

def gen_msdr(p0, n):
    paths  = np.empty((n, T_YRS + 1))
    paths[:, 0] = p0
    states = np.ones(n, dtype=int)
    for t in range(1, T_YRS + 1):
        u = np.random.uniform(size=n)
        # --- BUG FIX 2026-06-22: preserve old state before transition ---
        # OLD (buggy, in-place) suppressed the spike regime entirely; see
        # lsmc_main_simulation.py for the full explanation.
        #     states[((states == 1) & (u < P10_YR))] = 0
        #     states[((states == 0) & (u < P01_YR))] = 1
        old = states.copy()
        states[(old == 1) & (u < P10_YR)] = 0
        states[(old == 0) & (u < P01_YR)] = 1
        mu  = np.where(states == 0, MU_SPIKE, MU_BASE)
        sig = np.where(states == 0, SIG_SPIKE, SIG_BASE)
        paths[:, t] = paths[:, t-1] * np.exp(
            (mu - 0.5 * sig**2) + sig * np.random.standard_normal(n))
    return paths

def gen_elec(n):
    return gen_gou(PE0, n, theta=THETA_ELEC, kappa=KAPPA_ELEC, sigma=SIG_ELEC)

def gen_carbon(model, p0, n):
    return {'GBM': gen_gbm, 'G-OU': gen_gou, 'MS-DR': gen_msdr}[model](p0, n)

# ══════════════════════════════════════════════════════════════════════════════
# CASH FLOW
# ══════════════════════════════════════════════════════════════════════════════
def compute_atcf_M(pc, pe, cf, pb):
    n = pc.shape[0]
    atcf = np.zeros((n, T_YRS + 1))
    for t in range(1, T_YRS + 1):
        rev    = pb * Q_BC + pc[:, t] * Q_NET
        opex   = OPEX_FIXED + cf * Q_FEED + pe[:, t] * Q_ELEC + OPEX_OTHER_VAR
        ebitda = rev - opex
        ebit   = ebitda - DEP[t - 1]
        tax    = np.where(ebit > 0, ebit * TAX, 0.0)
        atcf[:, t] = (ebitda - tax) / 1e6
    return atcf

# ══════════════════════════════════════════════════════════════════════════════
# BASIS
# ══════════════════════════════════════════════════════════════════════════════
def basis(pc, pe, cf, pb):
    return np.column_stack([
        pc, pe, cf, pb,
        pc**2, pe**2, cf**2, pb**2,
        pc*pe, pc*cf, pc*pb,
        pe*cf, pe*pb,
        cf*pb
    ])

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING  (N_TRAIN=100,000, seed already set by caller)
# ══════════════════════════════════════════════════════════════════════════════
def train_lsmc(model):
    print(f"  [TRAIN] {model}  N={N_TRAIN:,} ...", end=" ", flush=True)
    t0 = time.time()
    pb_tr = np.random.triangular(*TRI_PB, size=N_TRAIN)
    cf_tr = np.random.triangular(*TRI_CF, size=N_TRAIN)
    pc_tr = gen_carbon(model, PC0, N_TRAIN)
    pc_tr[:, 0] = np.random.uniform(50, 600, N_TRAIN)
    pe_tr = gen_elec(N_TRAIN)
    atcf  = compute_atcf_M(pc_tr, pe_tr, cf_tr, pb_tr)
    V     = np.maximum(atcf[:, T_YRS] - TCI_M + WC_M, 0.0)  # [WC]
    X1    = basis(pc_tr[:, 1], pe_tr[:, 1], cf_tr, pb_tr)
    scaler = StandardScaler().fit(X1)
    reg_cont = [None] * (T_YRS + 1)
    reg_iv   = [None] * (T_YRS + 1)
    for t in range(T_YRS - 1, 0, -1):
        cont_tgt = V / (1 + RF)
        disc = np.array([(1 + WACC) ** (-(k - t))
                         for k in range(t + 1, T_YRS + 1)])
        iv_tgt = np.dot(atcf[:, t+1:], disc) - TCI_M + WC_M * (1 + WACC) ** (-(T_YRS - t))  # [WC]
        Xt = scaler.transform(basis(pc_tr[:, t], pe_tr[:, t], cf_tr, pb_tr))
        rc = LinearRegression().fit(Xt, cont_tgt)
        ri = LinearRegression().fit(Xt, iv_tgt)
        reg_cont[t] = rc
        reg_iv[t]   = ri
        iv_hat  = ri.predict(Xt)
        phi_hat = rc.predict(Xt)
        V = np.where(iv_hat > phi_hat, iv_tgt, cont_tgt)
    print(f"done in {time.time()-t0:.1f}s")
    return reg_cont, reg_iv, scaler

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION  (explicit n_eval — seed must be set by caller beforehand)
# ══════════════════════════════════════════════════════════════════════════════
def eval_enpv(model, pb_val, cf_val, reg_cont, reg_iv, scaler, n_eval):
    """Returns ENPV ($M) for given evaluation path count."""
    pc = gen_carbon(model, PC0, n_eval)
    pe = gen_elec(n_eval)
    pb = np.full(n_eval, pb_val)
    cf = np.full(n_eval, cf_val)
    atcf = compute_atcf_M(pc, pe, cf, pb)

    disc_wacc = np.array([(1 + WACC) ** (-t) for t in range(1, T_YRS + 1)])
    static_npv = float(np.mean(np.dot(atcf[:, 1:], disc_wacc)) - TCI_M
                       + WC_M * (1 + WACC) ** (-T_YRS))  # [WC]

    ex_year = np.full(n_eval, -1, dtype=int)
    payoffs = np.zeros(n_eval)

    for t in range(1, T_YRS):
        if reg_cont[t] is None or reg_iv[t] is None:
            continue
        alive = ex_year == -1
        if not alive.any():
            break
        Xt = scaler.transform(
            basis(pc[alive, t], pe[alive, t], cf[alive], pb[alive]))
        phi_hat  = reg_cont[t].predict(Xt)
        iv_hat   = reg_iv[t].predict(Xt)
        disc     = np.array([(1 + WACC) ** (-(k - t))
                              for k in range(t + 1, T_YRS + 1)])
        iv_actual = (np.dot(atcf[alive, t+1:], disc) - TCI_M
                     + WC_M * (1 + WACC) ** (-(T_YRS - t)))  # [WC]
        do_ex     = (iv_hat > phi_hat) & (iv_hat > 0)
        idx_alive = np.where(alive)[0]
        idx_ex    = idx_alive[do_ex]
        ex_year[idx_ex] = t
        payoffs[idx_ex] = iv_actual[do_ex] * (1 + RF) ** (-t)

    never = ex_year == -1
    if never.any():
        iv_T = atcf[never, T_YRS] - TCI_M + WC_M  # [WC]
        do_T = iv_T > 0
        idx_n = np.where(never)[0]
        idx_T = idx_n[do_T]
        ex_year[idx_T] = T_YRS
        payoffs[idx_T] = iv_T[do_T] * (1 + RF) ** (-T_YRS)

    deferral_val = float(np.mean(payoffs))
    return max(static_npv, deferral_val, 0.0)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    t_total = time.time()
    raw_records     = []
    summary_records = []

    for model in ['GBM', 'G-OU', 'MS-DR']:
        print(f"\n{'='*62}")
        print(f"  Model: {model}")
        print(f"{'='*62}")

        np.random.seed(42)
# Working-capital recovery adopted (option (a), see the accompanying article):
#   WC recovered in full, non-taxable, at Year 10 (as stated in tex 3.2).
#   Terms added to: terminal V, iv_tgt (train), static NPV, iv_actual (eval), iv_T.
   # reproducible training (same as lsmc_main_simulation.py)
        reg_cont, reg_iv, scaler = train_lsmc(model)

        prev_mean = None
        for n_eval in CONV_N_LIST:
            enpv_reps = []
            t_n = time.time()
            for rep_idx, seed in enumerate(EVAL_SEEDS):
                np.random.seed(seed)
                enpv = eval_enpv(model, CONV_PB, CONV_CF,
                                 reg_cont, reg_iv, scaler, n_eval)
                enpv_reps.append(enpv)
                raw_records.append({
                    'Model': model, 'N': n_eval,
                    'Rep': rep_idx + 1, 'Seed': seed,
                    'ENPV_M': round(enpv, 5),
                })
            print(f"  N={n_eval:>7,}  ENPVs={[round(v,3) for v in enpv_reps]}  "
                  f"({time.time()-t_n:.1f}s)")

            enpv_arr  = np.array(enpv_reps)
            mean_enpv = float(np.mean(enpv_arr))
            std_enpv  = float(np.std(enpv_arr, ddof=1))
            ci_hw     = T_CRIT_DF4 * std_enpv / np.sqrt(CONV_N_REPS)
            change_pct = (abs(mean_enpv - prev_mean) / abs(prev_mean) * 100
                          if prev_mean is not None and abs(prev_mean) > 1e-9
                          else None)
            summary_records.append({
                'Model':        model,
                'N':            n_eval,
                'Mean_ENPV_M':  round(mean_enpv, 4),
                'Std_M':        round(std_enpv, 4),
                'CI_HW_M':      round(ci_hw, 4),
                'Change_Pct':   round(change_pct, 2) if change_pct is not None else None,
            })
            prev_mean = mean_enpv

    # ── Save CSV ───────────────────────────────────────────────────────────────
    # 2026-06-22: write inside Code_0705 (one level up), to stay contained.
    out_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '..', 'results'))
    os.makedirs(out_dir, exist_ok=True)

    df_raw = pd.DataFrame(raw_records)
    df_sum = pd.DataFrame(summary_records)
    df_raw.to_csv(os.path.join(out_dir, 'lsmc_convergence_raw_0705.csv'), index=False)
    df_sum.to_csv(os.path.join(out_dir, 'lsmc_convergence_summary_0705.csv'), index=False)
    print(f"\nCSVs saved → {out_dir}")

    # ── LaTeX rows ─────────────────────────────────────────────────────────────
    def fmt_m(v):
        return f"\\${v:.3f}\\,M"

    print(f"\n{'='*70}")
    print("  tab:lsmc_conv_gou  (G-OU model, Mid scenario)")
    print(f"{'='*70}")
    for row in [r for r in summary_records if r['Model'] == 'G-OU']:
        n_s   = f"{row['N']:,}"
        chg   = "---" if row['Change_Pct'] is None else f"{row['Change_Pct']:.2f}\\%"
        print(f"{n_s}  & {fmt_m(row['Mean_ENPV_M'])} & {fmt_m(row['Std_M'])} "
              f"& {fmt_m(row['CI_HW_M'])} & {chg} \\\\")

    print(f"\n{'='*70}")
    print("  tab:lsmc_conv_gbm_msdr  (GBM vs MS-DR, Mid scenario)")
    print(f"{'='*70}")
    gbm_rows  = [r for r in summary_records if r['Model'] == 'GBM']
    msdr_rows = [r for r in summary_records if r['Model'] == 'MS-DR']
    for g, m in zip(gbm_rows, msdr_rows):
        n_s    = f"{g['N']:,}"
        if g['Change_Pct'] is None:
            chg = "---"
        else:
            chg = f"{g['Change_Pct']:.2f}\\% / {m['Change_Pct']:.2f}\\%"
        print(f"{n_s}  & {fmt_m(g['Mean_ENPV_M'])} & {fmt_m(m['Mean_ENPV_M'])} & {chg} \\\\")

    print(f"\nTotal elapsed: {(time.time()-t_total)/60:.1f} min")


if __name__ == '__main__':
    main()
