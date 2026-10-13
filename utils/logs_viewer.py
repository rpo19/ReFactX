"""Tiny Streamlit viewer for ReFactX evaluation logs.

Run it with::

    pip install streamlit
    streamlit run utils/logs_viewer.py

Pick a folder in the sidebar (default: the ``logs/`` next to this file), choose
a file, then inspect the summary and browse/filter individual questions.  The
``Browse`` tab also flags records that ran out of tokens and triples written in
the text that the constrained generator never produced (likely hallucinations).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.log_analysis import (  # noqa: E402
    METRIC_KEYS,
    SENTINEL,
    analyze,
    hallucinated_triples,
    load_log,
    triple_diff,
)

DEFAULT_FOLDER = REPO / "logs"


@st.cache_data(show_spinner=False)
def _load(path_str: str, mtime: float):
    # ``mtime`` only invalidates the cache when the file changes.
    return load_log(path_str)


def _metric_row(metrics: dict):
    cols = st.columns(len(METRIC_KEYS) + 1)
    cols[0].metric("n", int(metrics.get("n", 0)))
    for col, key in zip(cols[1:], METRIC_KEYS):
        col.metric(key, f"{metrics.get(key, 0.0):.3f}")


def _render_record(record: dict):
    ev = record.get("evaluation", {}) or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("correct", f"{ev.get('correct', 0):.0f}")
    c2.metric("f1", f"{ev.get('f1', 0):.3f}")
    c3.metric("answered", "yes" if record.get("prediction") else "no")
    c4.metric("kind", str(record.get("kind")))

    incompl = int(record.get("incomplete_triples", 0) or 0)
    if incompl:
        st.warning(f"Incomplete triples: {incompl} lookup(s) not recorded "
                   f"(terminal cut off). Excluded from the hallucination check.")

    extra, missing = triple_diff(record)
    unknown = hallucinated_triples(record)
    if unknown:
        st.error(f"Hallucinated triples (in the text, never generated): {len(unknown)}")
        st.code("\n".join(unknown), language="text")
    if extra and not unknown:
        st.info(f"Text has {sum(extra.values())} extra triple mention(s) "
                f"(re-stated/quoted generated triples, or incomplete lookups).")
    if missing:
        st.warning(f"Generated triples missing from the text: {sum(missing.values())}")

    st.markdown("**Question**")
    st.write(record.get("question", ""))
    st.markdown(f"**Ground truth:** `{record.get('gt_answer')}`")
    st.markdown(f"**Prediction:** `{record.get('prediction')}`")

    triples = record.get("triples", []) or []
    if triples:
        st.markdown(f"**KG triples ({len(triples)})**")
        st.code("\n".join(triples), language="text")

    st.markdown("**Full prediction**")
    st.markdown(
        (record.get("full_prediction", "") or "")
        .replace(SENTINEL, f":red[**{SENTINEL}**]")
        .replace("\n", "  \n"),
    )
    with st.expander("Raw record JSON"):
        st.json(record)


def main():
    st.set_page_config(page_title="ReFactX log viewer", layout="wide")
    st.title("ReFactX log viewer")

    with st.sidebar:
        st.header("Logs")
        folder = st.text_input("Folder", value=str(DEFAULT_FOLDER))
        show_all = st.checkbox("Show all files (not just *.out*)", value=False)
        if st.button("Refresh"):
            st.cache_data.clear()

    folder_path = Path(folder).expanduser()
    if not folder_path.is_dir():
        st.error(f"Not a folder: {folder_path.resolve()}")
        st.stop()

    files = sorted(p for p in folder_path.iterdir() if p.is_file())
    if not show_all:
        files = [p for p in files if ".out" in p.name]
    if not files:
        st.warning("No matching files in this folder.")
        st.stop()

    labels = {}
    for p in files:
        try:
            stat = p.stat()
            when = pd.Timestamp(stat.st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")
            labels[p] = f"{p.name}  ({stat.st_size/1024:.0f} KB, {when})"
        except OSError:
            labels[p] = p.name

    chosen = st.sidebar.selectbox("File", options=files, format_func=lambda p: labels[p])
    metadata, records, summary = _load(str(chosen), chosen.stat().st_mtime)

    if not records:
        st.error("No evaluation records found in this file (is it a log?).")
        with st.expander("File preview"):
            st.code("\n".join(chosen.read_text(errors="replace").splitlines()[:20]))
        st.stop()

    df, stats = analyze(records, metadata)
    st.caption(f"{chosen} — {len(records)} records")
    with st.expander("Run metadata", expanded=True):
        st.json(metadata or {"(none)": "no header line"})

    summary_tab, browse_tab = st.tabs(["Summary", "Browse"])

    with summary_tab:
        if summary:
            st.subheader("Macro (from file)")
            _metric_row(summary.get("macro", {}))
            by_kind = summary.get("by_kind", {})
            if by_kind:
                st.subheader("By kind")
                st.dataframe(
                    pd.DataFrame(by_kind).T.rename_axis("kind").reset_index(),
                    width="stretch", hide_index=True,
                )

        st.subheader("Error analysis")
        c = st.columns(7)
        c[0].metric(
            "ran out of tokens", stats["hit_max_tokens"],
            help=("Records where generation produced no EOS/pad token, i.e. it hit "
                  "`max_new_tokens`. (Estimated from text length when the log "
                  "predates the `hit_max_tokens` field.)"),
        )
        c[1].metric(
            "incomplete triples", stats["incomplete_triples"],
            help=("Lookups whose terminal started but was cut off before the "
                  "generator recorded them (they exist in the text but not in "
                  "`triples`). Discounted by the hallucination check."),
        )
        c[2].metric(
            "records w/ sentinel", f"{stats['sentinel_rate']:.0%}",
            help="Share of records whose text contains `<no further records>` at least once.",
        )
        c[3].metric("max sentinels/record", stats["max_sentinels"])
        c[4].metric(
            "records w/ count mismatch", stats["triple_mismatch"],
            help=("Records where the number of `<s> <r> <o> .` patterns found in "
                  "`full_prediction` by regex differs from the number of entries in "
                  "that record's `triples` key.\n\n"
                  "Two causes: (1) **re-quotes** — the model restates a triple it "
                  "already generated (e.g. \"the lookup returned <S> <R> <O> .\"), "
                  "so the regex counts it twice; (2) an incomplete lookup cut off "
                  "before it was recorded. Use the hallucination metric for "
                  "made-up facts."),
        )
        c[5].metric(
            "records w/ hallucination", stats["records_with_unknown"],
            help=("Records with a triple in the text that is neither a generated "
                  "triple nor one of the incomplete (cut-off) lookups. The regex "
                  "looks everywhere; re-quotes are discounted by matching against "
                  "`triples`."),
        )
        c[6].metric(
            "hallucinated triples", stats["unknown_triples"],
            help="Distinct triples in the text not explained by `triples` or `incomplete_triples`.",
        )

        st.subheader("Per kind")
        per_kind = df.groupby("kind").agg(
            n=("idx", "size"), f1=("f1", "mean"), correct=("correct", "mean"),
            answered=("answered", "mean"), sentinel=("sentinel", "mean"),
            count_mismatch=("triple_mismatch", "mean"),
            hallucinations=("unknown_triples", "sum"),
            ran_out=("hit_max_tokens", "sum"),
        ).reset_index()
        st.dataframe(per_kind, width="stretch", hide_index=True)

    with browse_tab:
        f1, f2, f3, f4 = st.columns(4)
        kinds = sorted(x for x in df["kind"].dropna().unique())
        sel_kinds = f1.multiselect("kind", kinds, default=kinds)
        outcome = f2.radio("outcome", ["all", "correct", "incorrect"], horizontal=True)
        flags = f3.multiselect("flags", ["sentinel", "ran out of tokens",
                                         "count mismatch", "incomplete", "hallucination"])
        query = f4.text_input("Search question / prediction", "")

        view = df[df["kind"].isin(sel_kinds)]
        if outcome == "correct":
            view = view[view["correct"] > 0]
        elif outcome == "incorrect":
            view = view[view["correct"] <= 0]
        if "sentinel" in flags:
            view = view[view["sentinel"] == 1]
        if "ran out of tokens" in flags:
            view = view[view["hit_max_tokens"] == 1]
        if "count mismatch" in flags:
            view = view[view["triple_mismatch"] == 1]
        if "incomplete" in flags:
            view = view[view["incomplete_triples"] > 0]
        if "hallucination" in flags:
            view = view[view["unknown_triples"] > 0]
        if query:
            mask = view["question"].str.contains(query, case=False, na=False) | \
                   view["prediction"].astype(str).str.contains(query, case=False, na=False)
            view = view[mask]

        st.caption(f"{len(view)} / {len(df)} records")
        st.dataframe(
            view[["idx", "kind", "correct", "f1", "answered", "sentinel",
                  "n_triples", "n_triples_regex", "incomplete_triples",
                  "triple_mismatch", "unknown_triples", "hit_max_tokens", "question"]],
            width="stretch", hide_index=True, height=260,
        )

        if view.empty:
            st.stop()
        pick = st.selectbox(
            "Inspect record",
            options=view["idx"].tolist(),
            format_func=lambda i: f"[{i}] ({records[i].get('kind')}) {records[i].get('question','')[:90]}",
        )
        st.divider()
        _render_record(records[pick])


if __name__ == "__main__":
    main()
