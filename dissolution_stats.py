"""
dissolution_stats - statistical analysis of dissolution data for solid oral IR products.

Implements the calculations and decision rules from:
  * FDA  Guidance: Dissolution Testing of IR Solid Oral Dosage Forms (1997)
  * EMA  Reflection paper on dissolution specification for generic IR products (2017)
  * ICH  M9 BCS-based biowaivers (2019/2020)

Input convention: a "profile" is a table of individual units (rows) by sampling
time in minutes (columns), values in % of label claim dissolved.

Everything here supports regulatory-style analysis but does not replace
validated software or expert judgement; verify against your own SOPs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import optimize, stats

RAPID_THRESHOLD = 85.0  # % dissolved


# --------------------------------------------------------------------------- #
# Data container
# --------------------------------------------------------------------------- #
@dataclass
class Profile:
    """Individual-unit dissolution data (units x time points), t=0 is dropped."""

    times: np.ndarray
    data: np.ndarray
    label: str = ""

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)
        self.data = np.asarray(self.data, dtype=float)
        if self.data.ndim != 2 or self.data.shape[1] != self.times.size:
            raise ValueError("data must be 2-D (units x times) matching len(times)")
        if np.isnan(self.data).any():
            raise ValueError("missing values are not allowed; resolve them first")
        if np.any(np.diff(self.times) <= 0):
            raise ValueError("times must be strictly increasing")
        keep = self.times > 0  # f2/MSD exclude t = 0
        self.times, self.data = self.times[keep], self.data[:, keep]
        if self.data.shape[0] < 2:
            raise ValueError("at least 2 units are required")

    @classmethod
    def from_csv(cls, path: str, label: str = "") -> "Profile":
        df = pd.read_csv(path)
        return cls(times=[float(c) for c in df.columns], data=df.to_numpy(), label=label or path)

    @property
    def n_units(self) -> int:
        return self.data.shape[0]

    @property
    def mean(self) -> np.ndarray:
        return self.data.mean(axis=0)

    @property
    def sd(self) -> np.ndarray:
        return self.data.std(axis=0, ddof=1)

    @property
    def cv(self) -> np.ndarray:
        m = self.mean
        return np.where(m > 0, 100.0 * self.sd / np.where(m > 0, m, 1.0), 0.0)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"mean": self.mean, "sd": self.sd, "cv_%": self.cv,
             "min": self.data.min(axis=0), "max": self.data.max(axis=0)},
            index=pd.Index(self.times, name="time_min"),
        )


def _check_same_times(a: Profile, b: Profile) -> None:
    if a.times.shape != b.times.shape or not np.allclose(a.times, b.times):
        raise ValueError("reference and test must use identical time points")


def _window(r_mean: np.ndarray, t_mean: np.ndarray, threshold: float = RAPID_THRESHOLD) -> int:
    """Number of leading points to use: stop at the first point where BOTH means >= 85%
    (FDA: 'only one measurement should be considered after 85% dissolution of both')."""
    both = (r_mean >= threshold) & (t_mean >= threshold)
    return int(np.argmax(both)) + 1 if both.any() else len(r_mean)


def _very_rapid(p: Profile, limit_min: float = 15.0) -> bool:
    return bool(np.any((p.mean >= RAPID_THRESHOLD) & (p.times <= limit_min)))


def _rapid(p: Profile, limit_min: float = 30.0) -> bool:
    return _very_rapid(p, limit_min)


# --------------------------------------------------------------------------- #
# f1 / f2
# --------------------------------------------------------------------------- #
def f1_factor(r: np.ndarray, t: np.ndarray) -> float:
    r, t = np.asarray(r, float), np.asarray(t, float)
    return float(100.0 * np.abs(r - t).sum() / r.sum())


def f2_factor(r: np.ndarray, t: np.ndarray) -> float:
    r, t = np.asarray(r, float), np.asarray(t, float)
    msd = np.mean((r - t) ** 2)
    return float(50.0 * np.log10(100.0 / np.sqrt(1.0 + msd)))


@dataclass
class F2Result:
    f1: Optional[float]
    f2: Optional[float]
    times_used: list
    similar: Optional[bool]  # None = no conclusion can be drawn
    conclusion: str
    warnings: list = field(default_factory=list)


def compare_f2(ref: Profile, test: Profile, early_cutoff: float = 10.0,
               early_cv_limit: float = 20.0, late_cv_limit: float = 10.0) -> F2Result:
    """Model-independent comparison with the validity rules from FDA V.A and ICH M9 3.2.

    early_cutoff: ICH M9 defines early points as <= 10 min (FDA example: 15 min);
    set to 15 to follow the FDA wording.
    """
    _check_same_times(ref, test)
    warn = []
    for p in (ref, test):
        if p.n_units < 12:
            warn.append(f"{p.label or 'profile'}: {p.n_units} units (12 recommended)")

    if _very_rapid(ref) and _very_rapid(test):
        return F2Result(None, None, [], True,
                        "Both products >=85% dissolved within 15 min: f2 unnecessary, profiles considered similar.",
                        warn)

    n = _window(ref.mean, test.mean)
    r, t = ref.mean[:n], test.mean[:n]
    times = ref.times[:n].tolist()
    f1, f2 = f1_factor(r, t), f2_factor(r, t)

    if n < 3:
        warn.append(f"only {n} usable time point(s); minimum of 3 required")
        return F2Result(f1, f2, times, None, "f2 not valid: fewer than 3 time points.", warn)

    bad = []
    for p in (ref, test):
        cv = p.cv[:n]
        for tm, c in zip(p.times[:n], cv):
            lim = early_cv_limit if tm <= early_cutoff else late_cv_limit
            if c > lim:
                bad.append(f"{p.label or 'profile'} t={tm:g} min: CV {c:.1f}% > {lim:g}%")
    if bad:
        return F2Result(f1, f2, times, None,
                        "Variability too high to use mean data; f2 is unreliable. "
                        "Consider MSD/bootstrap approaches.", warn + bad)

    similar = f2 >= 50.0
    return F2Result(f1, f2, times, similar,
                    f"f2 = {f2:.1f} -> {'similar' if similar else 'NOT similar'} (f2 >= 50).", warn)


def _f2_arr(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Vectorised f2 over the last axis (time points)."""
    return 50.0 * np.log10(100.0 / np.sqrt(1.0 + np.mean((r - t) ** 2, axis=-1)))


def _jackknife(xr: np.ndarray, xt: np.ndarray):
    """Leave-one-unit-out f2 values. xr (..., n1, p), xt (..., n2, p) -> (..., n1), (..., n2)."""
    n1, n2 = xr.shape[-2], xt.shape[-2]
    mr, mt = xr.mean(axis=-2), xt.mean(axis=-2)
    loo_r = (xr.sum(axis=-2, keepdims=True) - xr) / (n1 - 1)
    loo_t = (xt.sum(axis=-2, keepdims=True) - xt) / (n2 - 1)
    return _f2_arr(loo_r, mt[..., None, :]), _f2_arr(mr[..., None, :], loo_t)


def _jackknife_se(jr: np.ndarray, jt: np.ndarray) -> np.ndarray:
    """Jackknife standard error of f2, treating reference and test as two independent samples."""
    n1, n2 = jr.shape[-1], jt.shape[-1]
    var = ((n1 - 1) / n1) * ((jr - jr.mean(-1, keepdims=True)) ** 2).sum(-1) \
        + ((n2 - 1) / n2) * ((jt - jt.mean(-1, keepdims=True)) ** 2).sum(-1)
    return np.sqrt(var)


BOOT_LABELS = {"percentile": "Percentile", "bca": "BCa", "bootstrap_t": "Bootstrap-t"}


def bootstrap_f2(ref: Profile, test: Profile, n_boot: int = 5000, conf: float = 0.90,
                 seed: Optional[int] = 0, include_t: bool = False) -> dict:
    """Bootstrap confidence intervals for f2 (supplementary; not in the guidance texts).

    Whole units are resampled with replacement, independently within reference and test, on the
    analysis window of the observed means. Methods returned under ``"methods"``:
      * percentile   - plain percentiles of the resampled f2 values
      * bca          - bias-corrected and accelerated (z0 from the resamples, a from a jackknife)
      * bootstrap_t  - only if include_t: studentized; each resample is scaled by its own
                       jackknife standard error. Slower to explain, can be wide when CV is high.
    A method calls the profiles similar when its lower bound is >= 50.
    Legacy keys ci_low / ci_high / similar mirror the percentile interval.
    """
    _check_same_times(ref, test)
    n = _window(ref.mean, test.mean)
    rd, td = ref.data[:, :n], test.data[:, :n]
    n1, n2 = rd.shape[0], td.shape[0]
    theta = f2_factor(rd.mean(axis=0), td.mean(axis=0))

    rng = np.random.default_rng(seed)
    idx_r, idx_t = np.empty((n_boot, n1), int), np.empty((n_boot, n2), int)
    for i in range(n_boot):                       # draw order kept stable for reproducible results
        idx_r[i] = rng.integers(0, n1, n1)
        idx_t[i] = rng.integers(0, n2, n2)
    xr, xt = rd[idx_r], td[idx_t]                 # (B, n1, p), (B, n2, p)
    vals = _f2_arr(xr.mean(axis=1), xt.mean(axis=1))

    a = (1 - conf) / 2
    methods = {}

    def put(name, lo, hi):
        hi = min(float(hi), 100.0)                # f2 cannot exceed 100
        methods[name] = {"ci_low": float(lo), "ci_high": hi, "similar": bool(lo >= 50.0)}

    put("percentile", *np.quantile(vals, [a, 1 - a]))

    # --- BCa -----------------------------------------------------------------
    B = n_boot
    p0 = (np.sum(vals < theta) + 0.5 * np.sum(vals == theta)) / B
    z0 = stats.norm.ppf(min(max(p0, 0.5 / B), 1 - 0.5 / B))
    jr, jt = _jackknife(rd, td)
    jack = np.concatenate([jr, jt])
    d = jack.mean() - jack
    den = 6.0 * np.sum(d ** 2) ** 1.5
    acc = float(np.sum(d ** 3) / den) if den > 0 else 0.0

    def adj(alpha):
        u = z0 + stats.norm.ppf(alpha)
        denom = 1.0 - acc * u
        return float(stats.norm.cdf(z0 + u / denom)) if denom > 1e-9 else alpha

    put("bca", *np.quantile(vals, [adj(a), adj(1 - a)]))

    # --- Bootstrap-t (opt-in) ------------------------------------------------
    if include_t:
        se_hat = float(_jackknife_se(jr, jt))
        jrb, jtb = _jackknife(xr, xt)             # leave-one-out inside every resample
        se_b = _jackknife_se(jrb, jtb)
        ok = se_b > 1e-9                          # resample with identical profiles -> SE 0
        if se_hat > 0 and ok.sum() >= 0.5 * B:
            tq = np.quantile((vals[ok] - theta) / se_b[ok], [a, 1 - a])
            put("bootstrap_t", theta - tq[1] * se_hat, theta - tq[0] * se_hat)
        else:                                     # nothing to studentize: no variability at all
            put("bootstrap_t", theta, theta)

    head = methods["percentile"]
    return {"f2_observed": float(theta), "conf": conf, "n_boot": n_boot, "methods": methods,
            "ci_low": head["ci_low"], "ci_high": head["ci_high"], "similar": head["similar"]}


def bootstrap_table(b: dict) -> pd.DataFrame:
    """One row per bootstrap method, for display."""
    return pd.DataFrame([
        {"Method": BOOT_LABELS[k], f"{b['conf']:.0%} CI low": round(m["ci_low"], 1),
         f"{b['conf']:.0%} CI high": round(m["ci_high"], 1),
         "Similar (low >= 50)": "yes" if m["similar"] else "NO"}
        for k, m in b["methods"].items()])


def bootstrap_agreement(b: dict):
    """(state, message) summarising whether the bootstrap methods give the same verdict.
    state: True all similar, False none similar, None methods disagree."""
    verdicts = {k: m["similar"] for k, m in b["methods"].items()}
    if all(verdicts.values()):
        return True, (f"All {len(verdicts)} bootstrap methods agree: lower bound >= 50 "
                      f"(observed f2 = {b['f2_observed']:.1f}).")
    if not any(verdicts.values()):
        return False, (f"All {len(verdicts)} bootstrap methods agree: lower bound < 50 "
                       f"(observed f2 = {b['f2_observed']:.1f}).")
    pass_ = ", ".join(BOOT_LABELS[k] for k, v in verdicts.items() if v)
    fail = ", ".join(BOOT_LABELS[k] for k, v in verdicts.items() if not v)
    return None, (f"Bootstrap methods disagree (similar: {pass_}; not similar: {fail}). "
                  "Treat as inconclusive and look at the interval widths.")



# --------------------------------------------------------------------------- #
# Multivariate confidence region (MSD, FDA V.B)
# --------------------------------------------------------------------------- #
@dataclass
class MSDResult:
    msd_observed: float          # Mahalanobis distance between means
    ci_low: float
    ci_high: float               # upper limit of the confidence interval on true distance
    similarity_limit: float
    similar: bool
    n_points: int


def _ncf_limit(F: float, df1: int, df2: int, prob: float) -> float:
    """Noncentrality lambda such that P(F' <= F | lambda) = prob (cdf is decreasing in lambda)."""
    g = lambda lam: stats.ncf.cdf(F, df1, df2, lam) - prob
    if g(0.0) <= 0:
        return 0.0
    hi = 1.0
    while g(hi) > 0 and hi < 1e7:
        hi *= 2
    return float(optimize.brentq(g, 0.0, hi))


def msd_similarity(ref: Profile, test: Profile, limit_pct: float = 15.0,
                   conf: float = 0.90) -> MSDResult:
    """Multivariate statistical distance approach (Tsong/Shah). Use when within-batch CV is high.

    The similarity limit is the Mahalanobis distance of a constant `limit_pct` difference at
    every time point (the common Shah convention, not fixed by the FDA text, which derives the
    limit from reference-batch variability). Pooled covariance of ref and test is used.
    """
    _check_same_times(ref, test)
    n = _window(ref.mean, test.mean)
    p = n
    n1, n2 = ref.n_units, test.n_units
    df2 = n1 + n2 - p - 1
    if df2 <= 0:
        raise ValueError("too many time points for the number of units (need n1+n2 > p+1)")

    xr, xt = ref.data[:, :n], test.data[:, :n]
    sp = ((n1 - 1) * np.cov(xr, rowvar=False, ddof=1).reshape(p, p)
          + (n2 - 1) * np.cov(xt, rowvar=False, ddof=1).reshape(p, p)) / (n1 + n2 - 2)
    d = xt.mean(axis=0) - xr.mean(axis=0)
    solve = lambda v: np.linalg.solve(sp, v)
    D2 = float(d @ solve(d))
    dmax = float(np.sqrt(np.full(p, limit_pct) @ solve(np.full(p, limit_pct))))

    k = n1 * n2 / (n1 + n2)
    F = df2 / (p * (n1 + n2 - 2)) * k * D2
    a = (1 - conf) / 2
    lam_hi = _ncf_limit(F, p, df2, a)
    lam_lo = _ncf_limit(F, p, df2, 1 - a)
    hi, lo = np.sqrt(lam_hi / k), np.sqrt(lam_lo / k)
    return MSDResult(float(np.sqrt(D2)), float(lo), float(hi), dmax, bool(hi <= dmax), n)


# --------------------------------------------------------------------------- #
# EMA: dissolution specification from the biobatch (decision tree)
# --------------------------------------------------------------------------- #
@dataclass
class SpecRecommendation:
    time_min: Optional[int]
    q_percent: Optional[int]
    multipoint: bool
    rationale: str


def _mean_at(p: Profile, t: float, notes: list) -> float:
    if t in p.times:
        return float(p.mean[list(p.times).index(t)])
    notes.append(f"{t:g} min not sampled; linearly interpolated")
    return float(np.interp(t, p.times, p.mean))


def _nearest_q(target: float) -> int:
    # closest of 75/80/85; ties go to the lower Q
    return min((75, 80, 85), key=lambda q: (abs(q - target), q))


def ema_spec_from_biobatch(biobatch: Profile) -> SpecRecommendation:
    """EMA reflection paper Annex decision tree. Q ~ (biobatch mean - 10%), limited to 75/80/85."""
    notes: list = []
    a = {t: _mean_at(biobatch, t, notes) for t in (15, 30, 45)}
    suffix = ("; " + "; ".join(notes)) if notes else ""
    for t in (15, 30):
        if a[t] >= 85:
            q = _nearest_q(a[t] - 10)
            return SpecRecommendation(t, q, False, f"A{t}={a[t]:.1f}% >= 85% -> Q={q}% at {t} min{suffix}")
    if a[45] >= 85:
        q = _nearest_q(a[45] - 10)
        return SpecRecommendation(45, q, False, f"A45={a[45]:.1f}% >= 85% -> Q={q}% at 45 min{suffix}")
    if a[45] >= 75:
        return SpecRecommendation(45, 75, False,
                                  f"A45={a[45]:.1f}% < 85% -> Q=75% at 45 min (judge feasibility vs S2 compliance){suffix}")
    return SpecRecommendation(None, None, True,
                              f"A45={a[45]:.1f}% < 75% -> specification should use more than one time point{suffix}")


def is_immediate_release_ema(p: Profile) -> bool:
    """EMA scope: IR = at least 75% (Q) dissolved within 45 min."""
    notes: list = []
    return _mean_at(p, 45, notes) >= 75.0


# --------------------------------------------------------------------------- #
# ICH M9: BCS-based biowaiver
# --------------------------------------------------------------------------- #
M9_MEDIA = ("pH 1.2", "pH 4.5", "pH 6.8")


def bcs_high_solubility(highest_dose_mg: float, solubility_mg_per_ml: dict) -> dict:
    """Highly soluble if highest single dose dissolves in <=250 mL at every pH (lowest solubility governs)."""
    missing = [m for m in M9_MEDIA if m not in solubility_mg_per_ml]
    lowest = min(solubility_mg_per_ml.values())
    vol = highest_dose_mg / lowest
    return {"dose_solubility_volume_ml": vol, "high_solubility": vol <= 250.0,
            "missing_required_ph": missing}


def bcs_high_permeability(absolute_bioavailability_pct: Optional[float] = None,
                          urinary_recovery_pct: Optional[float] = None) -> bool:
    """ICH M9: high permeability if absolute BA >= 85% or >= 85% recovered in urine (parent+ox/conj metabolites)."""
    return any(v is not None and v >= 85.0 for v in (absolute_bioavailability_pct, urinary_recovery_pct))


@dataclass
class MediumAssessment:
    medium: str
    ref_class: str
    test_class: str
    f2: Optional[F2Result]
    passed: bool
    reason: str


def _speed(p: Profile) -> str:
    return "very rapid" if _very_rapid(p) else "rapid" if _rapid(p) else "not rapid"


def bcs_biowaiver_dissolution(bcs_class: str, media: dict) -> dict:
    """media: {'pH 1.2': (ref_profile, test_profile), ...} for pH 1.2, 4.5, 6.8.

    Class I : both very rapid, or both rapid AND f2-similar (mixed rapid/very rapid needs f2).
    Class III: both very rapid.
    """
    if bcs_class not in ("I", "III"):
        raise ValueError("BCS-based biowaiver applies to class I or III only")
    missing = [m for m in M9_MEDIA if m not in media]
    results = []
    for name, (ref, test) in media.items():
        rs, ts = _speed(ref), _speed(test)
        f2res = None
        if rs == "very rapid" and ts == "very rapid":
            ok, why = True, "both very rapid (>=85% in 15 min)"
        elif bcs_class == "III":
            ok, why = False, "class III requires both products very rapid"
        elif "not rapid" in (rs, ts):
            ok, why = False, "a product is not rapid (>=85% in 30 min)"
        else:
            f2res = compare_f2(ref, test)
            ok = f2res.similar is True
            why = f2res.conclusion
        results.append(MediumAssessment(name, rs, ts, f2res, ok, why))
    passed = not missing and all(r.passed for r in results)
    return {"eligible_on_dissolution": passed, "missing_media": missing, "media": results}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cli(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Dissolution profile statistics")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare", help="compare reference vs test (f1/f2, MSD, bootstrap)")
    c.add_argument("reference"); c.add_argument("test")
    c.add_argument("--bootstrap-t", action="store_true", help="also compute bootstrap-t intervals")
    s = sub.add_parser("spec", help="EMA specification proposal from a biobatch")
    s.add_argument("biobatch")
    args = ap.parse_args(argv)

    if args.cmd == "compare":
        ref, test = Profile.from_csv(args.reference, "reference"), Profile.from_csv(args.test, "test")
        print(ref.summary().round(1), "\n", test.summary().round(1), sep="\n")
        r = compare_f2(ref, test)
        print(f"\nf1={r.f1}  f2={r.f2}\n{r.conclusion}")
        for w in r.warnings:
            print("  warning:", w)
        try:
            m = msd_similarity(ref, test)
            print(f"\nMSD: D={m.msd_observed:.2f}, 90% CI upper={m.ci_high:.2f}, "
                  f"limit={m.similarity_limit:.2f} -> {'similar' if m.similar else 'NOT similar'}")
        except ValueError as e:
            print("\nMSD not computed:", e)
        b = bootstrap_f2(ref, test, include_t=args.bootstrap_t)
        print("\nBootstrap f2 intervals:")
        print(bootstrap_table(b).to_string(index=False))
        print(bootstrap_agreement(b)[1])
    else:
        rec = ema_spec_from_biobatch(Profile.from_csv(args.biobatch, "biobatch"))
        print(rec.rationale)


if __name__ == "__main__":
    _cli()
