"""
Financial Risk Intelligence Dashboard (Streamlit).

Tabs
  1. Live Events      structured risk signals as they arrive + black-swan injector
  2. Risk Graph       event -> sector -> ticker propagation and portfolio exposure
  3. Index Rebalancer Module A: current weights, weights over time, event replay
  4. Stress Tester    Module B: auto-triggered + manual scenarios, before/after
  5. Model & Results  sentiment evaluation, validation and backtest outputs

The dashboard talks only to the FastAPI backend (API_BASE, default
http://localhost:8000); the Risk Graph tab and the results tab also read the
repository's static data files.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import EVENT_CLASSES, output_path  # noqa: E402

API_BASE = os.getenv("API_BASE", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="Financial Risk Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

GREEN, RED, GREY, AMBER = "#00A878", "#E5484D", "#8B8D98", "#F5A524"


# ─── API helpers ──────────────────────────────────────────────

def api_get(path: str, timeout: float = 10, quiet: bool = False):
    try:
        r = httpx.get(f"{API_BASE}{path}", timeout=timeout)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        if not quiet:
            st.error(f"API error on {path}: {exc}")
        return None


def api_post(path: str, data: dict, timeout: float = 60):
    try:
        r = httpx.post(f"{API_BASE}{path}", json=data, timeout=timeout)
        if r.status_code >= 400:
            detail = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
            st.error(f"API returned {r.status_code}: {detail}")
            return None
        return r.json()
    except Exception as exc:
        st.error(f"API error on {path}: {exc}")
        return None


def load_json(name: str):
    path = output_path(name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def money(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.0f}"


def sentiment_icon(label: str) -> str:
    return {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}.get(label, "⚪")


def novelty_badge(label: str) -> str:
    return {"high": "🆕 high", "medium": "📰 medium", "low": "🔁 low"}.get(label, label or "")


def impact_icon(score: float) -> str:
    if score >= 7:
        return "🔴"
    if score >= 5:
        return "🟠"
    if score >= 3:
        return "🟡"
    return "🟢"


def render_event_detail(event: dict, key_prefix: str) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sentiment", f"{event.get('sentiment_label', '').title()} ({event.get('sentiment_score', 0):+.2f})")
    c2.metric("Event", event.get("event_class", "—"),
              help=f"method: {event.get('event_method')} · confidence {event.get('event_confidence')}")
    c3.metric("Novelty", f"{novelty_badge(event.get('novelty_label'))} ({event.get('novelty_score', 0):.2f})")
    c4.metric("Impact", f"{event.get('impact_score', 0)}/10")

    badges = []
    if event.get("is_duplicate"):
        badges.append("🔁 near-duplicate: downstream modules skipped")
    if event.get("stress_test_triggered"):
        badges.append(f"⚡ triggered stress test #{event.get('stress_scenario_id')} "
                      f"({event.get('stress_loss_pct', 0):+.2f}%)")
    if event.get("event_low_confidence"):
        badges.append("❔ low-confidence classification")
    for update in event.get("module_a_updates", []):
        if "skipped" not in update:
            badges.append(f"📈 {update['ticker']} weight {update['old_weight'] * 100:.2f}% → "
                          f"{update['new_weight'] * 100:.2f}%")
    if badges:
        st.markdown("  \n".join(badges))
    if event.get("affected_tickers"):
        st.write("**Tickers:**", " · ".join(event["affected_tickers"]),
                 " | **Sectors:**", ", ".join(event.get("affected_sectors", [])))
    if event.get("triggering_sentence"):
        st.info(f"💬 {event['triggering_sentence']}")

    contributions = event.get("impact_contributions") or {}
    if contributions:
        df = pd.DataFrame(
            [{"Factor": k.replace("_", " ").title(), "Points": v} for k, v in contributions.items()]
        )
        fig = px.bar(df, x="Points", y="Factor", orientation="h", height=210,
                     title="Impact score decomposition (points out of 10)")
        fig.update_traces(marker_color=AMBER)
        fig.update_layout(margin=dict(t=35, b=5, l=5, r=5))
        st.plotly_chart(fig, key=f"{key_prefix}_impact_{event.get('event_id')}")
    st.caption(
        f"Source: {event.get('source')} · latency {event.get('latency_ms', 0):.0f} ms · "
        f"sentiment model {event.get('sentiment_model', '')} · {str(event.get('timestamp', ''))[:19]}"
    )


# ─── Header ───────────────────────────────────────────────────

st.title("📊 Financial Risk Intelligence Platform")
st.caption("Real-time NLP risk engine → tactical index rebalancing & portfolio stress testing")


@st.fragment(run_every=10)
def header_fragment():
    health = api_get("/", quiet=True)
    if not health:
        st.warning(f"Backend not reachable at {API_BASE}. Start it with "
                   "`python -m uvicorn src.api.main:app`.")
        return
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Signals processed", f"{health.get('total_processed', 0):,}")
    c2.metric("Duplicates suppressed", health.get("duplicates_suppressed", 0))
    c3.metric("Stress tests triggered", health.get("stress_tests_triggered", 0))
    c4.metric("Base portfolio", money(health.get("portfolio_value")))
    c5.metric("Last stress result", f"{health.get('last_stress_loss_pct', 0):+.2f}%")
    model = health.get("sentiment_model", "")
    note = " (fallback)" if health.get("sentiment_model_fallback") else ""
    st.caption(f"News source: {health.get('news_source', '?')} · queue {health.get('queue_size', 0)} · "
               f"sentiment model: {model}{note}")


header_fragment()
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔴 Live Events",
    "🕸️ Risk Graph",
    "📈 Index Rebalancer",
    "💼 Stress Tester",
    "🧪 Model & Results",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — LIVE EVENTS
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Inject a headline")
    col_in, col_btn = st.columns([4, 1])
    with col_in:
        headline = st.text_input(
            "Headline",
            value=("China blockade of Taiwan Strait sparks military standoff; "
                   "TSMC, NVIDIA and Apple shares plunge as sanctions loom"),
            label_visibility="collapsed",
        )
    with col_btn:
        clicked = st.button("⚡ BLACK SWAN", type="primary", width="stretch")
    st.caption("Tip: the same headline twice is treated as a duplicate (no second stress test). "
               "Change a few words between demos.")
    if clicked and headline.strip():
        with st.spinner("Running NLP engine and downstream modules..."):
            result = api_post("/black-swan", {"headline": headline, "source": "news:reuters"})
        if result and result.get("event"):
            st.success(result.get("message", "Processed"))
            render_event_detail(result["event"], "swan")

    st.divider()
    st.subheader("Live signal feed")
    refresh_rate = st.selectbox("Refresh", [5, 10, 30], index=1, format_func=lambda x: f"every {x}s")

    @st.fragment(run_every=refresh_rate)
    def live_events_fragment():
        data = api_get("/events?limit=50", quiet=True) or {}
        events = data.get("events", [])
        if not events:
            st.info("No events yet: the pipeline is warming up (first model load can take a minute).")
            return
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg impact", f"{sum(e['impact_score'] for e in events) / len(events):.1f}/10")
        m2.metric("High impact (≥7)", sum(1 for e in events if e["impact_score"] >= 7))
        m3.metric("Negative sentiment", sum(1 for e in events if e["sentiment_label"] == "negative"))
        m4.metric("Avg latency", f"{sum(e.get('latency_ms', 0) for e in events) / len(events):.0f} ms")

        mix = pd.DataFrame(events)[["event_class", "sentiment_label"]].value_counts().reset_index(name="count")
        fig = px.bar(mix, x="event_class", y="count", color="sentiment_label", height=260,
                     color_discrete_map={"positive": GREEN, "negative": RED, "neutral": GREY},
                     title="Last 50 signals by event class and sentiment")
        fig.update_layout(margin=dict(t=35, b=5), xaxis_title=None, legend_title=None)
        st.plotly_chart(fig, key="mix_chart")

        for event in events[:15]:
            flag = " ⚡" if event.get("stress_test_triggered") else (" 🔁" if event.get("is_duplicate") else "")
            with st.expander(
                f"{impact_icon(event['impact_score'])} {sentiment_icon(event['sentiment_label'])} "
                f"[{event['event_class']}] {event['impact_score']}/10{flag} | {event['headline'][:90]}"
            ):
                render_event_detail(event, "feed")

    live_events_fragment()

# ══════════════════════════════════════════════════════════════
# TAB 2 — RISK GRAPH
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Risk propagation graph")
    st.caption("Event class → affected sectors → named / portfolio tickers")
    col_e, col_t = st.columns(2)
    with col_e:
        graph_event = st.selectbox("Event type", EVENT_CLASSES, key="graph_event")
    with col_t:
        tickers_input = st.text_input("Affected tickers (comma separated)", value="TSM, NVDA, ASML, JPM")
    graph_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    if st.button("🔄 Generate risk graph", type="primary"):
        from src.risk.exposure_calc import calculate_exposure
        from src.risk.risk_graph import traverse

        result = traverse(graph_event, graph_tickers)
        try:
            from pyvis.network import Network

            net = Network(height="500px", width="100%", bgcolor="#0e1117", font_color="white", directed=True)
            colors = {"event": RED, "sector": AMBER, "ticker": GREEN}
            for node, data in result["graph"].nodes(data=True):
                node_type = data.get("node_type", "ticker")
                net.add_node(node, label=data.get("label", node), color=colors.get(node_type, GREEN),
                             size=30 if node_type == "event" else 20 if node_type == "sector" else 14,
                             title=f"{node_type.title()}: {data.get('label', node)}")
            for src, dst in result["graph"].edges():
                net.add_edge(src, dst, color="#666666")
            net.set_options(json.dumps({"physics": {"stabilization": {"iterations": 100}},
                                        "interaction": {"hover": True}}))
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as fh:
                tmp_name = fh.name
            net.save_graph(tmp_name)
            html = Path(tmp_name).read_text()
            os.unlink(tmp_name)
            if hasattr(st, "iframe"):
                st.iframe(html, height=520)
            else:  # older Streamlit
                components.html(html, height=520)
        except Exception as exc:
            st.warning(f"Interactive graph unavailable ({exc}); showing the mapping instead.")
            st.json({"affected_sectors": result["affected_sectors"], "sector_tickers": result["sector_tickers"]})

        exposure = calculate_exposure(graph_tickers, result["affected_sectors"])
        st.subheader("Portfolio exposure")
        e1, e2, e3 = st.columns(3)
        e1.metric("Directly / sector-exposed value", money(exposure["exposed_value"]))
        e2.metric("Exposure", f"{exposure['exposure_pct']}%")
        e3.metric("Direct obligors", exposure["direct_count"])
        exp_df = pd.DataFrame(exposure["positions"])[
            ["id", "obligor", "type", "sector", "value", "exposure_type", "exposure_multiplier"]
        ]
        exp_df["value"] = exp_df["value"].map(money)
        st.dataframe(exp_df, hide_index=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — MODULE A
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Module A · Tactical index rebalancer")
    st.caption("15 S&P 100 names · sentiment-driven tilts · 2–15% bounds · 3% turnover cap · "
               "5-day decay back to equal weight")

    @st.fragment(run_every=10)
    def rebalancer_fragment():
        weights_data = api_get("/module-a/weights", quiet=True)
        history_data = api_get("/module-a/history?limit=500", quiet=True)
        if not weights_data:
            st.info("Waiting for backend...")
            return
        eq = weights_data.get("equal_weight_pct", 6.667)
        df = pd.DataFrame([
            {"Ticker": t, "Weight %": d["weight_pct"], "Sector": d["sector"],
             "vs Equal Weight (pp)": round(d["weight_pct"] - eq, 3)}
            for t, d in weights_data.get("weights", {}).items()
        ]).sort_values("Weight %", ascending=False)

        left, right = st.columns(2)
        with left:
            fig = go.Figure(go.Bar(
                x=df["Ticker"], y=df["Weight %"],
                marker_color=[GREEN if v > 0.0005 else RED if v < -0.0005 else GREY
                              for v in df["vs Equal Weight (pp)"]],
                text=[f"{w:.2f}%" for w in df["Weight %"]], textposition="outside",
            ))
            fig.add_hline(y=eq, line_dash="dash", line_color=GREY, annotation_text=f"Equal weight {eq:.2f}%")
            fig.update_layout(title="Current weights", height=380, yaxis_title="Weight (%)", margin=dict(t=40))
            st.plotly_chart(fig, key="weights_now")

        with right:
            history = (history_data or {}).get("history", [])
            if len(history) > 1:
                rows = [
                    {"time": h["timestamp"], "Ticker": t, "Weight %": w * 100}
                    for h in history for t, w in h["weights"].items()
                ]
                hist_df = pd.DataFrame(rows)
                hist_df["time"] = pd.to_datetime(hist_df["time"])
                fig = px.area(hist_df, x="time", y="Weight %", color="Ticker", height=380,
                              title="Index weights over time")
                fig.update_layout(margin=dict(t=40), legend_title=None, yaxis_range=[0, 100])
                st.plotly_chart(fig, key="weights_history")
            else:
                st.info("Weights-over-time chart appears after the first index-relevant signal.")

        if history:
            st.caption("Recent rebalancing triggers: " + ", ".join(
                h["trigger"] for h in history[-8:] if h["trigger"] != "init"
            ))
        st.dataframe(df, hide_index=True)

    rebalancer_fragment()
    if st.button("↺ Reset index to equal weight"):
        api_post("/module-a/reset", {})

    st.divider()
    st.subheader("Historical event replay vs equal weight")
    st.caption("Same NLP pipeline + rebalancer replayed over 45 dated historical headlines on real prices. "
               "Signals enter at the next trading-day open; 5 bp transaction costs.")
    refresh = st.checkbox("Recompute (otherwise use cached results)", value=False)
    if st.button("▶ Run / load event replay"):
        with st.spinner("Running replay (first run downloads prices and may take a few minutes)..."):
            backtest = api_get(f"/module-a/backtest?days=0&refresh={str(refresh).lower()}", timeout=600)
        if backtest:
            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("Strategy (net)", f"{backtest['sentiment_final'] * 100:+.2f}%")
            b2.metric("Equal weight", f"{backtest['equalweight_final'] * 100:+.2f}%")
            b3.metric("Excess", f"{backtest['excess_return'] * 100:+.2f}%")
            b4.metric("Information ratio", f"{backtest.get('information_ratio', 0):.2f}")
            b5.metric("Max drawdown", f"{backtest.get('max_drawdown', 0) * 100:.1f}%")

            perf = pd.DataFrame({
                "Date": pd.to_datetime(backtest["dates"]),
                "Sentiment-weighted (net)": [r * 100 for r in backtest["sentiment_returns"]],
                "Equal weight": [r * 100 for r in backtest["equalweight_returns"]],
            }).set_index("Date")
            fig = px.line(perf, height=360, title="Cumulative return (%)",
                          color_discrete_map={"Sentiment-weighted (net)": GREEN, "Equal weight": GREY})
            fig.update_layout(legend_title=None, yaxis_title="%")
            st.plotly_chart(fig, key="bt_perf")

            if backtest.get("weights_history"):
                wh = pd.DataFrame(backtest["weights_history"])
                wh.index = pd.to_datetime(backtest["dates"])
                fig = px.line(wh * 100, height=360, title="Replay weights over time (%)")
                fig.update_layout(legend_title=None, yaxis_title="Weight (%)")
                st.plotly_chart(fig, key="bt_weights")
            st.caption(backtest.get("note", ""))

# ══════════════════════════════════════════════════════════════
# TAB 4 — MODULE B
# ══════════════════════════════════════════════════════════════

def render_scenario(result: dict, key_prefix: str) -> None:
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Portfolio before", money(result["portfolio_before"]))
    r2.metric("Portfolio after", money(result["portfolio_after"]), delta=f"{result['loss_pct']:+.2f}%")
    r3.metric("Gross losses", money(-result.get("gross_loss", 0)))
    r4.metric("Hedge / gains offset", money(result.get("gross_gain", 0)))
    st.success(f"📋 {result.get('summary_sentence', '')}")

    before = result.get("value_by_asset_type_before", {})
    after = result.get("value_by_asset_type_after", {})
    left, right = st.columns(2)
    with left:
        if before:
            types = list(before)
            fig = go.Figure([
                go.Bar(name="Before", x=types, y=[before[t] for t in types], marker_color=GREY),
                go.Bar(name="After", x=types, y=[after.get(t, 0) for t in types], marker_color=RED),
            ])
            fig.update_layout(barmode="group", height=340, title="Value by asset type: before vs after",
                              yaxis_title="USD", margin=dict(t=40))
            st.plotly_chart(fig, key=f"{key_prefix}_before_after")
    with right:
        by_sector = result.get("by_sector") or []
        if by_sector:
            fig = go.Figure(go.Bar(
                x=[s["sector"] for s in by_sector],
                y=[s["loss"] for s in by_sector],
                text=[f"{s.get('pct_of_gross_loss', 0):.0f}%" for s in by_sector],
                textposition="outside", marker_color=RED,
            ))
            fig.update_layout(title="Loss attribution by sector", height=340, yaxis_title="Loss (USD)",
                              margin=dict(t=40))
            st.plotly_chart(fig, key=f"{key_prefix}_sector")

    shocks = result.get("shocks_applied", {})
    if shocks:
        st.caption(
            f"Shocks ({result.get('shock_direction', '')}, scale {result.get('scale_factor', '')}): "
            f"equity {shocks.get('equity_shock', 0) * 100:+.1f}% · rates {shocks.get('rates_shock', 0) * 1e4:+.0f} bp · "
            f"credit spreads {shocks.get('credit_spread_shock', 0) * 1e4:+.0f} bp · "
            f"oil {shocks.get('oil_shock', 0) * 100:+.1f}%"
        )
    positions = result.get("positions") or []
    if positions:
        pos_df = pd.DataFrame(positions)[
            ["obligor", "type", "sector", "credit_rating", "exposure_type", "value_before", "pnl", "pnl_pct"]
        ].sort_values("pnl")
        pos_df["value_before"] = pos_df["value_before"].map(money)
        pos_df["pnl"] = pos_df["pnl"].map(money)
        st.dataframe(pos_df, hide_index=True)
    top = result.get("top_positions") or []
    if top:
        with st.expander("Why these losses? (top contributors)"):
            for p in top:
                st.write(f"**{p['obligor']}** ({p['type']}): {money(-p['loss'])}, {p['reason']}")


with tab4:
    st.subheader("Module B · Strategic portfolio stress tester")
    st.caption("Synthetic $100M wholesale banking book. Auto-trigger: impact ≥ 7 and either a systemic "
               "risk-off event (Geopolitical / Macroeconomic / Credit) that is not good news, or clearly "
               "negative company news. Scenarios are what-ifs on the base book.")

    @st.fragment(run_every=10)
    def stress_summary_fragment():
        summary = api_get("/module-b/portfolio", quiet=True)
        if not summary:
            return
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Base portfolio", money(summary["initial_value"]))
        s2.metric("Scenarios run", summary["scenario_count"])
        s3.metric("Last scenario", f"{summary.get('last_loss_pct', 0):+.2f}%",
                  help=str(summary.get("last_event_class")))
        s4.metric("Worst scenario", f"{summary.get('worst_loss_pct', 0):+.2f}%",
                  help=str(summary.get("worst_event_class")))

        latest = api_get("/module-b/latest", quiet=True)
        if latest:
            st.markdown(f"**Latest scenario #{latest['scenario_id']}** · {latest['trigger']} · "
                        f"{latest['event_class']} · impact {latest['impact_score']}/10  \n"
                        f"_{latest.get('headline', '')}_")
            render_scenario(latest, "latest")
        else:
            st.info("No stress scenario yet. Inject a black-swan headline or run a manual test below.")

    stress_summary_fragment()

    st.divider()
    st.subheader("Manual scenario")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        manual_event = st.selectbox("Event type", EVENT_CLASSES, key="manual_event")
    with col_b:
        manual_impact = st.slider("Impact score", 1.0, 10.0, 8.0, 0.5)
    with col_c:
        manual_tickers = st.text_input("Affected tickers", value="TSM, NVDA, ASML", key="manual_tickers")
    with col_d:
        manual_mode = st.radio("Mode", ["baseline", "cumulative"], horizontal=True,
                               help="baseline = what-if on the original book; cumulative = shocks compound")
    if st.button("⚡ Run stress test", type="primary"):
        tickers_list = [t.strip().upper() for t in manual_tickers.split(",") if t.strip()]
        result = api_post("/stress-test", {
            "event_class": manual_event,
            "impact_score": manual_impact,
            "affected_tickers": tickers_list,
            "affected_sectors": [],
            "mode": manual_mode,
        })
        if result:
            render_scenario(result, "manual")

    st.divider()
    st.subheader("Scenario history")

    @st.fragment(run_every=15)
    def history_fragment():
        history = api_get("/module-b/history", quiet=True) or {}
        scenarios = history.get("scenarios", [])
        if not scenarios:
            st.info("No scenarios yet.")
            return
        hist_df = pd.DataFrame(scenarios)
        hist_df["time"] = pd.to_datetime(hist_df["timestamp"]).dt.strftime("%H:%M:%S")
        fig = px.scatter(hist_df, x="time", y="loss_pct", color="event_class", size="impact_score",
                         hover_data=["headline", "trigger", "mode"], height=330,
                         title="Portfolio P&L % per scenario",
                         labels={"loss_pct": "P&L (%)", "time": "Time"})
        st.plotly_chart(fig, key="scenario_history")
        st.dataframe(hist_df[["scenario_id", "time", "trigger", "mode", "event_class",
                              "impact_score", "loss_pct", "headline"]], hide_index=True)
        if st.button("↺ Reset stress history"):
            api_post("/module-b/reset", {})

    history_fragment()

# ══════════════════════════════════════════════════════════════
# TAB 5 — MODEL & RESULTS
# ══════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Sentiment model · locked test set")
    eval_results = load_json("eval_results.json")
    if eval_results and isinstance(eval_results.get("results"), dict):
        rows = [
            {"Model": name, "Accuracy": r.get("accuracy"), "Macro F1": r.get("macro_f1")}
            for name, r in eval_results["results"].items()
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption(f"{eval_results.get('test_rows')} locked test rows · majority-class accuracy "
                   f"{eval_results.get('majority_class_accuracy')}")
    else:
        st.info("Run `python -m src.engine.evaluator --mode all` to populate data/eval_results.json.")

    st.subheader("Fresh unseen headlines (blind-labelled)")
    fresh = load_json("fresh_test_results.json")
    if fresh and fresh.get("summary"):
        fs = fresh["summary"]
        f1, f2, f3, f4 = st.columns(4)
        lo, hi = fs.get("accuracy_95ci", [0, 0])
        f1.metric("Event classification", f"{fs['accuracy'] * 100:.0f}%",
                  help=f"95% CI {lo * 100:.0f}–{hi * 100:.0f}% · n={fs['n_labelled']}")
        f2.metric("Majority baseline", f"{fs['majority_baseline_accuracy'] * 100:.0f}%",
                  help=f"always predicting {fs.get('majority_class')}")
        f3.metric("Macro-F1", f"{fs['macro_f1']:.2f}")
        tr = fs.get("ticker_recall")
        f4.metric("Ticker recall", f"{tr * 100:.0f}%" if tr is not None else "—")
        st.caption(fresh.get("protocol", ""))
        with st.expander("Per-headline predictions"):
            st.dataframe(pd.DataFrame(fresh.get("rows", []))[["headline", "label", "predicted", "method"]],
                         hide_index=True)
    else:
        st.caption("Run `python scripts/fresh_test_set.py collect`, label the CSV, then "
                   "`python scripts/fresh_test_set.py score`.")

    st.subheader("Historical validation")
    for split in ("tuning", "holdout"):
        payload = load_json(f"validation_results_{split}.json")
        if not payload:
            st.caption(f"{split}: run `python -m src.modules.validation_backtest --split {split}`")
            continue
        summary = payload.get("summary", {})
        st.markdown(f"**{split.title()} set** (n={summary.get('n_events')})")
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Event classification", f"{(summary.get('classification_accuracy') or 0) * 100:.0f}%",
                  help=f"majority baseline {(summary.get('classification_majority_baseline') or 0) * 100:.0f}%")
        dm = summary.get("direction_match_rate_abnormal_t5")
        v2.metric("Direction match (T+5 abnormal)", f"{dm * 100:.0f}%" if dm is not None else "—",
                  help=f"n={summary.get('direction_sample_size_abnormal_t5')}")
        v3.metric("High-impact avg |move|", f"{summary.get('high_impact_abs_move_avg_pct') or 0:.2f}%")
        v4.metric("Low-impact avg |move|", f"{summary.get('low_impact_abs_move_avg_pct') or 0:.2f}%")
        confusion = summary.get("classification_confusion_matrix")
        if confusion:
            with st.expander(f"{split} confusion matrix (rows = expected)"):
                st.dataframe(pd.DataFrame(confusion).T)
