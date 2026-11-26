"""Streamlit tools for ReFactX: evaluation-log viewer + index/KG explorer.

Run it with::

    pip install streamlit
    streamlit run utils/logs_viewer.py

Two modes (sidebar):

* **Logs viewer** — inspect ``utils/eval_kg.py`` logs: metrics, error analysis
  and per-question browsing.  Default folder: the ``logs/`` next to this file.
* **KG / index explorer** — load a trie/index (text file of triples, a pasted
  list, or a PostgreSQL index) and count / enumerate the triples that start
  with a given text prefix or token list.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils import kg_explorer  # noqa: E402
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


def render_logs():
    with st.sidebar:
        st.header("Logs")
        folder = st.text_input("Folder", value=str(DEFAULT_FOLDER))
        show_all = st.checkbox("Show all files (not just *.out*)", value=False)
        if st.button("Refresh logs"):
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


@st.cache_resource(show_spinner="Loading index…")
def _load_index(kind, payload, tokenizer_name, add_special_tokens, configkey, cache, mtime):
    # ``mtime`` only invalidates the cache when the source file changes.
    if kind == "postgres":
        index = kg_explorer.load_postgres_index(payload, configkey=configkey, cache=cache)
        name = getattr(index, "tokenizer_name", None) or tokenizer_name
        tokenizer = kg_explorer.get_tokenizer(name)
        index.set_tokenizer(tokenizer)
    else:
        tokenizer = kg_explorer.get_tokenizer(tokenizer_name)
        if kind == "file":
            index = kg_explorer.load_file_index(
                payload, tokenizer, add_special_tokens=add_special_tokens)
        else:  # pasted list of triples, one per line
            index = kg_explorer.load_lines_index(
                payload.splitlines(), tokenizer, add_special_tokens=add_special_tokens)
    return index, tokenizer


def render_explorer():
    with st.sidebar:
        st.header("Index source")
        label = st.radio("Source", ["Triple file (.txt)", "Triple list (paste)",
                                    "PostgreSQL index URL"])
        tokenizer_name = st.text_input("Tokenizer", value="Qwen/Qwen3.5-4B")
        add_special = st.checkbox("add_special_tokens", value=False)
        with st.expander("PostgreSQL options"):
            configkey = st.number_input("configkey", value=-200, step=1)
            cache = st.text_input("cache", value="simple",
                                  help="'simple', 'none', ...")
        if st.button("Reload"):
            st.cache_resource.clear()

        payload = None
        kind = {"Triple file (.txt)": "file",
                "Triple list (paste)": "lines",
                "PostgreSQL index URL": "postgres"}[label]
        mtime = 0.0
        if kind == "file":
            path = st.text_input("Path to triples (.txt)")
            if path:
                payload = str(Path(path).expanduser())
                mtime = os.path.getmtime(payload) if os.path.isfile(payload) else 0.0
        elif kind == "lines":
            payload = st.text_area("One triple per line", height=160)
        else:
            payload = st.text_input("postgresql://user:pwd@host:port/db?tablename=t")

    st.subheader("Index explorer")
    if not payload or (kind != "lines" and not payload.strip()):
        st.info("Select a source in the sidebar and provide it to load an index.")
        return

    try:
        index, tokenizer = _load_index(
            kind, payload, tokenizer_name, add_special, int(configkey), cache, mtime)
    except Exception as exc:
        st.error(f"Failed to load index: {exc}")
        return

    try:
        size = len(index)
    except Exception:
        size = "?"
    st.success(f"Loaded {type(index).__name__} — {size} triples")

    st.markdown("**Query**")
    c1, c2 = st.columns([2, 2])
    text = c1.text_input("Prefix (text)", value="")
    tokens_json = c2.text_input("…or token list (JSON array of ids)", value="")
    limit = st.slider("Max triples to enumerate", 1, 1000, 50)

    if not (text or tokens_json):
        st.caption("Type a prefix (text or token list) to count/enumerate matching triples.")
        return

    try:
        tokens = kg_explorer.parse_tokens(text, tokens_json, tokenizer, add_special)
    except Exception as exc:
        st.error(f"Bad query: {exc}")
        return

    report = kg_explorer.prefix_report(index, tokens, tokenizer, limit=limit)
    if report["note"]:
        st.warning(report["note"])

    m1, m2, m3 = st.columns(3)
    m1.metric("triples starting with prefix", report["count"])
    m2.metric("tokens", len(tokens))
    m3.metric("shown", len(report["triples"]))
    st.text_input("decoded prefix", value=report["decoded"], disabled=True)
    st.caption(f"tokens: {tokens}")

    if report["next"]:
        st.markdown("**Next tokens**")
        st.dataframe(pd.DataFrame(report["next"]), width="stretch", hide_index=True)

    if report["triples"]:
        st.markdown("**Matching triples**")
        rows = [{"tokens": s, "triple": t} for s, t in report["triples"]]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                     height=320)
    if report.get("truncated"):
        st.caption(f"Showing the first {len(report['triples'])} of {report['count']} triples.")


def main():
    st.set_page_config(page_title="ReFactX tools", layout="wide")
    st.title("ReFactX tools")
    mode = st.sidebar.radio("Mode", ["Logs viewer", "KG / index explorer"])
    if mode == "Logs viewer":
        render_logs()
    else:
        render_explorer()


if __name__ == "__main__":
    main()
