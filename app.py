"""Streamlit UI for dissolution_stats.  Run:  streamlit run app.py"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import dissolution_stats as ds
import session_guard

st.set_page_config(page_title="Dissolution Statistics", page_icon="💊", layout="wide")
session_guard.enforce()   # no-op unless DISSOLUTION_MAX_USERS is set (server use)

TIMES = [5, 10, 15, 20, 30, 45, 60]
MAX_UPLOAD_BYTES = 5_000_000   # protects small hosts; real data files are a few KB


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def sample_profile(means, cv=4.0, n=12, seed=0, label="") -> ds.Profile:
    rng = np.random.default_rng(seed)
    m = np.asarray(means, float)
    data = np.clip(m + rng.normal(0, 1, (n, len(m))) * m * cv / 100, 0, 100)
    return ds.Profile(TIMES[: len(m)], data, label)


SAMPLES = {
    "ref": [22, 40, 55, 65, 80, 92, 97],
    "test": [20, 37, 52, 62, 77, 90, 96],
    "fast": [70, 88, 96, 98, 99, 100, 100],
}


def load_profile(upload, label: str) -> ds.Profile | None:
    """Read a CSV/Excel table: rows = units, columns = time (min), values = % dissolved."""
    if upload is None:
        return None
    if upload.size > MAX_UPLOAD_BYTES:
        st.error(f"**{upload.name}** is {upload.size / 1e6:.1f} MB; the limit is "
                 f"{MAX_UPLOAD_BYTES // 1_000_000} MB. Dissolution tables are normally a few KB.")
        return None
    try:
        df = (pd.read_excel(upload) if upload.name.lower().endswith(("xlsx", "xls"))
              else pd.read_csv(upload))
        return ds.Profile([float(c) for c in df.columns], df.to_numpy(dtype=float), label)
    except Exception as e:  # show a friendly message instead of a traceback
        st.error(f"Could not read **{upload.name}**: {e}")
        return None


def template_csv() -> bytes:
    p = sample_profile(SAMPLES["ref"], seed=1)
    df = pd.DataFrame(p.data.round(1), columns=[int(t) for t in p.times])
    return df.to_csv(index=False).encode()


def data_input(key: str, label: str, sample_key: str, use_sample: bool, seed: int):
    if use_sample:
        return sample_profile(SAMPLES[sample_key], seed=seed, label=label)
    up = st.file_uploader(f"{label} (CSV / Excel)", type=["csv", "xlsx", "xls"], key=key)
    return load_profile(up, label)


def profile_figure(profiles, show_units=False, extra_shapes=None, title="") -> go.Figure:
    fig = go.Figure()
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    for p, c in zip(profiles, colors):
        if show_units:
            for row in p.data:
                fig.add_scatter(x=p.times, y=row, mode="lines", line=dict(color=c, width=0.6),
                                opacity=0.25, showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=p.times, y=p.mean, mode="lines+markers", name=f"{p.label} (mean ± SD)",
                        line=dict(color=c), error_y=dict(type="data", array=p.sd, visible=True))
    fig.add_hline(y=85, line_dash="dot", line_color="grey", annotation_text="85%")
    for s in extra_shapes or []:
        fig.add_shape(**s)
    fig.update_layout(title=title, xaxis_title="Time (min)", yaxis_title="% dissolved",
                      yaxis_range=[0, 110], height=430, margin=dict(t=50, b=40),
                      legend=dict(orientation="h", y=-0.2))
    return fig


def verdict(similar: bool | None, text: str):
    (st.success if similar else st.error if similar is False else st.warning)(text)


def excel_report(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31])
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Page 1: profile comparison
# --------------------------------------------------------------------------- #
def page_compare():
    st.header("Profile comparison (f1 / f2, MSD, bootstrap)")
    st.caption("FDA 1997 §V and ICH M9 §3.2. Rows = units, columns = time points in minutes.")
    use_sample = st.toggle("Use example data", value=False, key="cmp_sample")
    c1, c2 = st.columns(2)
    with c1:
        ref = data_input("cmp_ref", "Reference", "ref", use_sample, 1)
    with c2:
        test = data_input("cmp_test", "Test", "test", use_sample, 2)
    st.download_button("Download data template", template_csv(), "template.csv", "text/csv")

    with st.expander("Settings"):
        s1, s2, s3 = st.columns(3)
        early_cutoff = s1.selectbox("Early time points are ≤ (min)", [10, 15], 0,
                                    help="ICH M9 uses 10 min; FDA's example uses 15 min.")
        early_cv = s2.number_input("Max CV early (%)", 1.0, 100.0, 20.0)
        late_cv = s3.number_input("Max CV later (%)", 1.0, 100.0, 10.0)
        m1, m2, m3 = st.columns(3)
        limit = m1.number_input("MSD similarity limit (% difference)", 1.0, 50.0, 15.0)
        conf = m2.number_input("Confidence level", 0.80, 0.99, 0.90, 0.01)
        n_boot = m3.number_input("Bootstrap resamples", 500, 20000, 5000, 500)

    if ref is None or test is None:
        st.info("Provide both a reference and a test profile to continue.")
        return
    try:
        ds._check_same_times(ref, test)
    except ValueError as e:
        st.error(str(e))
        return

    show_units = st.checkbox("Show individual units", value=False)
    st.plotly_chart(profile_figure([ref, test], show_units), use_container_width=True)

    t1, t2 = st.columns(2)
    t1.subheader("Reference"); t1.dataframe(ref.summary().round(2), use_container_width=True)
    t2.subheader("Test"); t2.dataframe(test.summary().round(2), use_container_width=True)
    for p in (ref, test):
        if p.n_units < 12:
            st.warning(f"{p.label}: {p.n_units} units – 12 are recommended.")

    st.subheader("f2 similarity")
    r = ds.compare_f2(ref, test, early_cutoff, early_cv, late_cv)
    verdict(r.similar, r.conclusion)
    k1, k2, k3 = st.columns(3)
    k1.metric("f1", "–" if r.f1 is None else f"{r.f1:.1f}", help="Difference factor; 0–15 acceptable")
    k2.metric("f2", "–" if r.f2 is None else f"{r.f2:.1f}", help="Similarity factor; ≥ 50 similar")
    k3.metric("Points used (min)", ", ".join(f"{t:g}" for t in r.times_used) or "–")
    for w in r.warnings:
        st.warning(w)

    st.subheader("Multivariate (MSD) confidence region")
    msd = None
    try:
        msd = ds.msd_similarity(ref, test, limit, conf)
        verdict(msd.similar, f"Upper {conf:.0%} limit of true distance {msd.ci_high:.2f} "
                             f"{'≤' if msd.similar else '>'} similarity limit {msd.similarity_limit:.2f}.")
        st.caption(f"Observed distance {msd.msd_observed:.2f}; CI {msd.ci_low:.2f} – {msd.ci_high:.2f}; "
                   f"{msd.n_points} time points. Suited to profiles with CV > 15%.")
    except ValueError as e:
        st.warning(f"MSD not computed: {e}")

    st.subheader("Bootstrap f2 (supplementary)")
    use_t = st.checkbox("Also compute bootstrap-t (studentized)", value=False, key="boot_t",
                        help="Adds a third interval. Unchecked by default; ticking it re-runs the analysis "
                             "with the extra calculation.")
    b = ds.bootstrap_f2(ref, test, int(n_boot), conf, include_t=use_t)
    st.dataframe(ds.bootstrap_table(b), hide_index=True, use_container_width=True)
    state, msg = ds.bootstrap_agreement(b)
    verdict(state, msg)
    st.caption("Similar = lower bound of the interval ≥ 50. BCa corrects the percentile interval for bias and "
               "skew (useful with small samples). Bootstrap-t standardises each resample by its own standard "
               "error (jackknife) and can be wide when variability is high.")

    rows = {"f1": r.f1, "f2": r.f2, "f2 conclusion": r.conclusion, "f2 observed (bootstrap window)": b["f2_observed"]}
    for k, m in b["methods"].items():
        rows[f"{ds.BOOT_LABELS[k]} low"], rows[f"{ds.BOOT_LABELS[k]} high"] = m["ci_low"], m["ci_high"]
        rows[f"{ds.BOOT_LABELS[k]} similar"] = m["similar"]
    if msd:
        rows.update({"MSD observed": msd.msd_observed, "MSD CI upper": msd.ci_high,
                     "MSD limit": msd.similarity_limit, "MSD similar": msd.similar})
    st.download_button("Download Excel report", excel_report({
        "Results": pd.Series(rows, name="value").to_frame(),
        "Reference": ref.summary(), "Test": test.summary(),
        "Reference units": pd.DataFrame(ref.data, columns=ref.times),
        "Test units": pd.DataFrame(test.data, columns=test.times)}),
        "dissolution_comparison.xlsx")


# --------------------------------------------------------------------------- #
# Page 2: EMA specification
# --------------------------------------------------------------------------- #
def page_spec():
    st.header("Dissolution specification from the biobatch")
    st.caption("EMA reflection paper (2017) decision tree: Q ≈ biobatch mean − 10%, limited to 75 / 80 / 85%.")
    use_sample = st.toggle("Use example data", value=False, key="spec_sample")
    bio = data_input("spec_bio", "Biobatch (12 units)", "ref", use_sample, 3)
    if bio is None:
        st.info("Upload the biobatch dissolution data to continue.")
        return

    ir = ds.is_immediate_release_ema(bio)
    (st.success if ir else st.error)(
        "Meets the EMA immediate-release definition (≥ 75% in 45 min)." if ir
        else "Does not reach 75% in 45 min: outside the immediate-release scope of the paper.")
    rec = ds.ema_spec_from_biobatch(bio)
    if rec.multipoint:
        st.warning(rec.rationale)
    else:
        st.metric("Proposed specification", f"Q = {rec.q_percent}% at {rec.time_min} min")
        st.caption(rec.rationale)
    shapes = []
    if not rec.multipoint:
        shapes.append(dict(type="line", x0=rec.time_min, x1=rec.time_min, y0=0, y1=rec.q_percent,
                           line=dict(color="black", dash="dash")))
        shapes.append(dict(type="line", x0=0, x1=rec.time_min, y0=rec.q_percent, y1=rec.q_percent,
                           line=dict(color="black", dash="dash")))
    st.plotly_chart(profile_figure([bio], True, shapes), use_container_width=True)
    st.dataframe(bio.summary().round(2), use_container_width=True)
    st.info("Judgement calls remain (narrow-therapeutic-index drugs, discriminatory power of the method, "
            "compliance with S2). The tool implements the decision tree only.")


# --------------------------------------------------------------------------- #
# Page 3: BCS biowaiver
# --------------------------------------------------------------------------- #
def page_bcs():
    st.header("BCS-based biowaiver assessment")
    st.caption("ICH M9. Covers classification and comparative dissolution. Excipient assessment, "
               "narrow-therapeutic-index exclusion and dosage-form eligibility are not evaluated.")
    cls = st.radio("BCS class claimed", ["I", "III"], horizontal=True)

    st.subheader("1. Solubility")
    dose = st.number_input("Highest single therapeutic dose (mg)", 0.1, 10000.0, 100.0)
    sc = st.columns(3)
    sol = {m: sc[i].number_input(f"Solubility {m} (mg/mL)", 0.0001, 1000.0, [5.0, 1.0, 0.8][i],
                                 format="%.4f") for i, m in enumerate(ds.M9_MEDIA)}
    s = ds.bcs_high_solubility(dose, sol)
    verdict(s["high_solubility"], f"Dose/solubility volume = {s['dose_solubility_volume_ml']:.0f} mL "
                                  f"({'≤' if s['high_solubility'] else '>'} 250 mL) – "
                                  f"{'highly soluble' if s['high_solubility'] else 'not highly soluble'}.")
    st.subheader("2. Permeability")
    pc = st.columns(2)
    ba = pc[0].number_input("Absolute bioavailability (%), 0 if unknown", 0.0, 100.0, 90.0)
    ur = pc[1].number_input("Urinary recovery (%), 0 if unknown", 0.0, 100.0, 0.0)
    hp = ds.bcs_high_permeability(ba or None, ur or None)
    if cls == "I":
        verdict(hp, "High permeability supported." if hp else "High permeability not demonstrated.")
    else:
        verdict(not hp, "Low permeability (consistent with class III)." if not hp
                else "Data indicate high permeability – class I may apply instead of III.")

    st.subheader("3. Comparative dissolution (≥ 12 units, paddle 50 rpm / basket 100 rpm)")
    use_sample = st.toggle("Use example data (very rapid in all media)", value=False, key="bcs_sample")
    media = {}
    cols = st.columns(3)
    for i, m in enumerate(ds.M9_MEDIA):
        with cols[i]:
            st.markdown(f"**{m}**")
            if use_sample:
                media[m] = (sample_profile(SAMPLES["fast"], 2, seed=10 + i, label=f"Ref {m}"),
                            sample_profile(SAMPLES["fast"], 2, seed=20 + i, label=f"Test {m}"))
            else:
                r = load_profile(st.file_uploader("Reference", key=f"bcs_r{i}", type=["csv", "xlsx"]), f"Ref {m}")
                t = load_profile(st.file_uploader("Test", key=f"bcs_t{i}", type=["csv", "xlsx"]), f"Test {m}")
                if r is not None and t is not None:
                    media[m] = (r, t)
    if len(media) < 3:
        st.info("Provide reference and test data for all three media (pH 1.2, 4.5, 6.8).")
        return
    try:
        res = ds.bcs_biowaiver_dissolution(cls, media)
    except ValueError as e:
        st.error(str(e))
        return
    table = pd.DataFrame([{"Medium": a.medium, "Reference": a.ref_class, "Test": a.test_class,
                           "f2": None if a.f2 is None or a.f2.f2 is None else round(a.f2.f2, 1),
                           "Pass": "✅" if a.passed else "❌", "Reason": a.reason} for a in res["media"]])
    st.dataframe(table, hide_index=True, use_container_width=True)
    for m, (r, t) in media.items():
        with st.expander(f"Profiles – {m}"):
            st.plotly_chart(profile_figure([r, t]), use_container_width=True)
    ok = res["eligible_on_dissolution"] and s["high_solubility"] and (hp if cls == "I" else not hp)
    verdict(ok, "Classification and dissolution criteria are met." if ok
            else "One or more classification / dissolution criteria are not met.")
    st.download_button("Download Excel report", excel_report({"Dissolution": table.set_index("Medium")}),
                       "bcs_biowaiver.xlsx")


# --------------------------------------------------------------------------- #
PAGES = {"Profile comparison": page_compare, "Specification (EMA)": page_spec,
         "BCS biowaiver (ICH M9)": page_bcs}
with st.sidebar:
    st.title("💊 Dissolution Statistics")
    choice = st.radio("Analysis", list(PAGES))
    st.caption("Based on FDA 1997 IR dissolution guidance, EMA 2017 reflection paper and ICH M9. "
               "Decision-support only; verify against your SOPs and validated systems.")
PAGES[choice]()
