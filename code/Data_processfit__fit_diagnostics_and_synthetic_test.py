# -*- coding: utf-8 -*-
# ==== PACKET HEADER ====
# TARGET : 04_Data/E_ProcessFit_0918/fit_statistics_real_47m.csv, synthetic_refit_{replicates,summary}.csv,
#          delta_aic_reference.csv
#          (evidence for the Section 5.3 statement on the origin of the GBM vs MS-DR divergence; 2026-09-18 meeting)
# INPUT  : 04_Data/D_MSCI_licensed_local_only/... weekly MSCI series -> 47 monthly means (licensed; the real-sample
#          part cannot run from the public deposit). Calibrated parameters of Table 3 are hard-coded for the generators.
# OUTPUT : (A) log-likelihood, AIC and BIC of GBM, G-OU (AR(1) on log price) and MS-DR (Hamilton filter MLE) on the
#          observed 47-month sample, with each fitted model's 10-year projection (mean, median, 80% band at t = 10);
#          (B) the same three fits re-estimated on 200 synthetic 47-month samples generated from the calibrated GBM,
#          and (C) from the calibrated G-OU; the cross-model divergence of the 10-year projections is summarized
#          per generator. Question answered: is the divergence a property of the process families or of fitting
#          them to a sample they do not describe?
# NOTE   : seed 42 throughout; MS-DR MLE uses three starts (paper estimates, symmetric, wide); values are reported
#          as computed, no rounding to targets.
# ============================================
import os, io, sys, time
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
HERE = os.path.dirname(os.path.abspath(__file__))
PKT = os.path.abspath(os.path.join(HERE, ".."))
# Licensed MSCI weekly series: not distributed with this repository (see README, Data policy)
MSCI_TXT = os.path.join(PKT, "data_licensed", "MSCI_Biochar_CarbonCreditPrice_Weekly_2022-2025_0727.txt")

OUTDIR = os.path.join(PKT, "results", "process_fit"); os.makedirs(OUTDIR, exist_ok=True)
SEED = 42
P0 = 150.75
DT = 1.0 / 12
N_MONTHS = 47
H_STEPS = 120            # 10-year monthly projection
N_PATHS_REAL = 20_000
N_PATHS_SYN = 3_000
R_REP = 200

# calibrated parameters (Table 3 / Table S4), annual
GBM_MU, GBM_SIG = 0.0618, 0.3858
GOU_KAPPA, GOU_THETA, GOU_SIG = 11.807, 149.56, 0.4885
MSDR_PAPER = dict(mu0=0.6699, sig0=0.0679, mu1=-0.1284, sig1=0.4155, p00=0.0000, p11=0.7085)

# ---------------------------------------------------------------- fitting
def fit_gbm(x):
    r = np.diff(x); m = r.mean(); s = r.std(ddof=0)
    ll = norm.logpdf(r, m, s).sum()
    return dict(ll=ll, k=2, m=m, s=s, mu_ann=m * 12 + 0.5 * (s ** 2) * 12, sig_ann=s * np.sqrt(12))

def fit_ar1(x):
    y, z = x[1:], x[:-1]
    b, a = np.polyfit(z, y, 1)
    e = y - (a + b * z); s = e.std(ddof=0)
    ll = norm.logpdf(e, 0, s).sum()
    kappa = -12 * np.log(b) if 0 < b < 1 else np.nan
    theta_log = a / (1 - b) if b != 1 else np.nan
    return dict(ll=ll, k=3, a=a, b=b, s=s, kappa=kappa, theta_log=theta_log)

def msdr_negll(p, r):
    mu0, sig0, mu1, sig1, p00, p11 = p
    mc = np.array([(mu0 - 0.5 * sig0 ** 2) * DT, (mu1 - 0.5 * sig1 ** 2) * DT])
    sc = np.array([sig0 * DT ** 0.5, sig1 * DT ** 0.5])
    T = np.array([[p00, 1 - p00], [1 - p11, p11]])
    p01, p10 = 1 - p00, 1 - p11
    pi0 = p10 / (p10 + p01) if (p10 + p01) > 0 else 0.5
    prev = np.array([pi0, 1 - pi0]); ll = 0.0
    for rt in r:
        pred = T.T @ prev
        eta = norm.pdf(rt, mc, sc)
        joint = eta * pred; lik = joint.sum()
        if not np.isfinite(lik) or lik <= 0:
            return 1e6
        ll += np.log(lik); prev = joint / lik
    return -ll

def fit_msdr(x):
    r = np.diff(x)
    sd = r.std(ddof=0) * np.sqrt(12); m = r.mean() * 12
    starts = [
        [MSDR_PAPER["mu0"], MSDR_PAPER["sig0"], MSDR_PAPER["mu1"], MSDR_PAPER["sig1"], 0.05, 0.70],
        [m + sd, 0.5 * sd, m - 0.3 * sd, 1.2 * sd, 0.3, 0.8],
        [m + 2 * sd, 0.2 * sd, m, sd, 0.1, 0.9],
    ]
    bounds = [(-3, 3), (0.01, 3), (-3, 3), (0.01, 3), (0.0, 1.0), (0.0, 1.0)]
    best = None
    for s0 in starts:
        res = minimize(msdr_negll, s0, args=(r,), method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    mu0, sig0, mu1, sig1, p00, p11 = best.x
    # label the regime with the higher drift as the shock regime (0)
    if mu1 > mu0:
        mu0, sig0, mu1, sig1, p00, p11 = mu1, sig1, mu0, sig0, p11, p00
    return dict(ll=-best.fun, k=6, mu0=mu0, sig0=sig0, mu1=mu1, sig1=sig1, p00=p00, p11=p11)

# ---------------------------------------------------------------- projection (monthly, 10 years)
def project(model, f, n, rng):
    if model == "GBM":
        z = rng.standard_normal((n, H_STEPS))
        logp = np.log(P0) + np.cumsum(f["m"] + f["s"] * z, axis=1)
        return np.exp(logp[:, -1])
    if model == "G-OU":
        x = np.full(n, np.log(P0)); z = rng.standard_normal((n, H_STEPS))
        for t in range(H_STEPS):
            x = f["a"] + f["b"] * x + f["s"] * z[:, t]
        return np.exp(x)
    if model == "MS-DR":
        mc = np.array([(f["mu0"] - 0.5 * f["sig0"] ** 2) * DT, (f["mu1"] - 0.5 * f["sig1"] ** 2) * DT])
        sc = np.array([f["sig0"] * DT ** 0.5, f["sig1"] * DT ** 0.5])
        st = np.ones(n, dtype=int); x = np.full(n, np.log(P0))
        for t in range(H_STEPS):
            u = rng.uniform(size=n); old = st.copy()
            st[(old == 1) & (u < 1 - f["p11"])] = 0
            st[(old == 0) & (u < 1 - f["p00"])] = 1
            x = x + mc[st] + sc[st] * rng.standard_normal(n)
        return np.exp(x)
    raise ValueError(model)

def summarize(p10):
    q10, q50, q90 = np.percentile(p10, [10, 50, 90])
    return dict(mean=p10.mean(), median=q50, q10=q10, q90=q90, band=q90 - q10)

# ---------------------------------------------------------------- generators for the synthetic test
def gen_gbm(rng):
    d = (GBM_MU - 0.5 * GBM_SIG ** 2) * DT; v = GBM_SIG * DT ** 0.5
    return np.log(P0) + np.concatenate([[0.0], np.cumsum(d + v * rng.standard_normal(N_MONTHS - 1))])

def gen_gou(rng):
    th = np.log(GOU_THETA) - GOU_SIG ** 2 / (2 * GOU_KAPPA)
    e = np.exp(-GOU_KAPPA * DT); sd = GOU_SIG * np.sqrt((1 - e ** 2) / (2 * GOU_KAPPA))
    x = np.empty(N_MONTHS); x[0] = np.log(P0)
    for t in range(1, N_MONTHS):
        x[t] = e * x[t - 1] + (1 - e) * th + sd * rng.standard_normal()
    return x

FITS = {"GBM": fit_gbm, "G-OU": fit_ar1, "MS-DR": fit_msdr}

def fit_all(x):
    return {k: fn(x) for k, fn in FITS.items()}

def divergence(sums):
    means = np.array([sums[k]["mean"] for k in FITS]); bands = np.array([sums[k]["band"] for k in FITS])
    return dict(mean_ratio_max_min=means.max() / means.min(), band_ratio_max_min=bands.max() / bands.min(),
                mean_GBM_over_GOU=sums["GBM"]["mean"] / sums["G-OU"]["mean"],
                mean_MSDR_over_GOU=sums["MS-DR"]["mean"] / sums["G-OU"]["mean"],
                band_GBM_over_GOU=sums["GBM"]["band"] / sums["G-OU"]["band"],
                band_MSDR_over_GOU=sums["MS-DR"]["band"] / sums["G-OU"]["band"])

def part_d(x_obs, n_rep=2000):
    """(D) reference distribution of the AIC gap AIC(GBM) - AIC(AR1) (positive favours G-OU) at n = 47,
    under each generator; no MS-DR fit, so it runs in seconds. Answers: how often does a sample of this length
    drawn from the calibrated GBM produce a gap at least as large as the observed one?"""
    obs_gap = (2 * 2 - 2 * fit_gbm(x_obs)["ll"]) - (2 * 3 - 2 * fit_ar1(x_obs)["ll"])
    rows = []
    for gen_name, gen in (("GBM", gen_gbm), ("G-OU", gen_gou)):
        rng_g = np.random.default_rng(SEED + 7)
        gaps = np.empty(n_rep); kap = np.empty(n_rep)
        for i in range(n_rep):
            x = gen(rng_g)
            fg, fa = fit_gbm(x), fit_ar1(x)
            gaps[i] = (4 - 2 * fg["ll"]) - (6 - 2 * fa["ll"]); kap[i] = fa["kappa"]
        rows.append(dict(generator=gen_name, n_rep=n_rep, observed_gap=obs_gap,
                         gap_median=np.median(gaps), gap_q90=np.percentile(gaps, 90), gap_q99=np.percentile(gaps, 99),
                         share_gap_ge_observed=(gaps >= obs_gap).mean(), share_gap_le_0=(gaps <= 0).mean(),
                         kappa_median=np.nanmedian(kap)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, "delta_aic_reference.csv"), index=False)
    print("(D) AIC gap reference (AIC_GBM - AIC_AR1; observed gap %.2f)" % obs_gap); print(df.round(4).to_string(index=False))
    return df

if __name__ == "__main__":
    t0 = time.time()
    # ---------------- (A) observed sample
    wk = pd.read_csv(MSCI_TXT, sep="\t", header=None, names=["date", "price"], parse_dates=["date"])
    mo = wk.set_index("date")["price"].resample("MS").mean().dropna()
    assert len(mo) == N_MONTHS and abs(mo.iloc[-1] - P0) < 0.01
    x_obs = np.log(mo.values); n_r = N_MONTHS - 1
    if "d" in sys.argv[1:]:          # python <script> d  -> only the fast part (D)
        part_d(x_obs); sys.exit(0)
    fits = fit_all(x_obs)
    rng = np.random.default_rng(SEED)
    rows = []
    for k, f in fits.items():
        s = summarize(project(k, f, N_PATHS_REAL, rng))
        rows.append(dict(model=k, loglik=f["ll"], k=f["k"], AIC=2 * f["k"] - 2 * f["ll"],
                         BIC=f["k"] * np.log(n_r) - 2 * f["ll"], **{f"P10_{a}": b for a, b in s.items()},
                         params=";".join(f"{a}={b:.4f}" for a, b in f.items() if a not in ("ll", "k"))))
    # MS-DR likelihood at the published estimates (P00 exactly 0)
    ll_paper = -msdr_negll([MSDR_PAPER[a] for a in ("mu0", "sig0", "mu1", "sig1", "p00", "p11")], np.diff(x_obs))
    rows.append(dict(model="MS-DR (Table S4 estimates, fixed)", loglik=ll_paper, k=6, AIC=12 - 2 * ll_paper,
                     BIC=6 * np.log(n_r) - 2 * ll_paper, params=";".join(f"{a}={b}" for a, b in MSDR_PAPER.items())))
    dfA = pd.DataFrame(rows)
    dfA.to_csv(os.path.join(OUTDIR, "fit_statistics_real_47m.csv"), index=False)
    print("(A) observed 47-month sample"); print(dfA.drop(columns=["params"]).to_string(index=False))
    for k, f in fits.items():
        print("   ", k, {a: round(float(b), 4) for a, b in f.items()})
    divA = divergence({k: summarize(project(k, f, N_PATHS_REAL, np.random.default_rng(SEED + 1))) for k, f in fits.items()})
    print("    divergence (observed-sample fits):", {a: round(b, 3) for a, b in divA.items()})

    # ---------------- (B)(C) synthetic samples
    out = []
    for gen_name, gen in (("GBM", gen_gbm), ("G-OU", gen_gou)):
        rng_g = np.random.default_rng(SEED)
        for rep in range(R_REP):
            x = gen(rng_g)
            fr = fit_all(x)
            sums = {k: summarize(project(k, f, N_PATHS_SYN, np.random.default_rng(SEED + rep))) for k, f in fr.items()}
            d = divergence(sums)
            rec = dict(generator=gen_name, rep=rep, ar1_b=fr["G-OU"]["b"], ar1_kappa=fr["G-OU"]["kappa"],
                       gbm_mu_ann=fr["GBM"]["mu_ann"], gbm_sig_ann=fr["GBM"]["sig_ann"],
                       msdr_p00=fr["MS-DR"]["p00"], msdr_p11=fr["MS-DR"]["p11"], msdr_mu0=fr["MS-DR"]["mu0"],
                       aic_best=min(FITS, key=lambda k: 2 * fr[k]["k"] - 2 * fr[k]["ll"]))
            for k in FITS:
                rec[f"mean10_{k}"] = sums[k]["mean"]; rec[f"band10_{k}"] = sums[k]["band"]
            rec.update(d); out.append(rec)
        print(f"({gen_name} generator) {R_REP} replicates done, {time.time() - t0:.0f}s")
    dfR = pd.DataFrame(out)
    dfR.to_csv(os.path.join(OUTDIR, "synthetic_refit_replicates.csv"), index=False)
    num = dfR.drop(columns=["rep", "aic_best"]).groupby("generator")
    summ = pd.concat([num.median().add_suffix("_median"), num.quantile(0.25).add_suffix("_q25"),
                      num.quantile(0.75).add_suffix("_q75")], axis=1).T
    summ.to_csv(os.path.join(OUTDIR, "synthetic_refit_summary.csv"))
    print("(B)(C) medians by generator:")
    show = ["ar1_kappa", "mean_GBM_over_GOU", "mean_MSDR_over_GOU", "band_GBM_over_GOU", "band_MSDR_over_GOU",
            "mean_ratio_max_min", "band_ratio_max_min"]
    print(num.median()[show].round(3).to_string())
    print("AIC winner share:"); print(dfR.groupby("generator")["aic_best"].value_counts(normalize=True).round(3).to_string())
    part_d(x_obs)
    print("elapsed", round(time.time() - t0), "s; outputs in", OUTDIR)
