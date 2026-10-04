"""
Financial Risk Intelligence Dashboard
4-tab Streamlit dashboard connected to FastAPI backend.

Architecture:
  - Dashboard ONLY talks to localhost:8000 (our FastAPI)
  - Never calls NewsAPI, HuggingFace, or any external API directly
  - st.fragment used for live components (no full page rerun)
  - All data fetched via httpx from the backend
"""

import streamlit as st
import httpx
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ─── Config ───────────────────────────────────────────────────

API_BASE = "http://localhost:8000"
st.set_page_config(
    page_title="Financial Risk Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── API Helper ───────────────────────────────────────────────

def api_get(path: str, timeout: int = 10) -> dict:
    """Safe API call with error handling."""
    try:
        r = httpx.get(f"{API_BASE}{path}", timeout=timeout)
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return {}


def api_post(path: str, data: dict, timeout: int = 30) -> dict:
    """Safe POST with error handling."""
    try:
        r = httpx.post(f"{API_BASE}{path}", json=data, timeout=timeout)
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return {}

# ─── Styling Helpers ──────────────────────────────────────────

def sentiment_color(label: str) -> str:
    return {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}.get(label, "⚪")


def novelty_badge(label: str) -> str:
    return {"high": "🆕 HIGH", "medium": "📰 MED", "low": "🔁 LOW"}.get(label, "")


def impact_color(score: float) -> str:
    if score >= 7:
        return "🔴"
    elif score >= 5:
        return "🟠"
    elif score >= 3:
        return "🟡"
    return "🟢"

# ─── Header ───────────────────────────────────────────────────

st.title("📊 Financial Risk Intelligence Platform")
st.caption("Real-time NLP-driven risk engine | S&P Global & Crisil Hackathon 2026")

# System stats in header
health = api_get("/")
if health:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Events Processed", f"{health.get('total_processed', 0):,}")
    c2.metric("Portfolio Value",
              f"${health.get('portfolio_value', 100_000_000):,.0f}")
    c3.metric("Active Subscribers",
              health.get('active_subscribers', 0))
    c4.metric("Events in Store",
              health.get('events_in_store', 0))

st.divider()

# ─── Tabs ─────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🔴 Live Events",
    "🕸️ Risk Graph",
    "📈 Index Rebalancer",
    "💼 Stress Tester",
])

# ══════════════════════════════════════════════════════════════
# TAB 1: LIVE EVENTS
# ══════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Live Event Feed")

    # Black Swan button
    col_btn, col_input = st.columns([1, 3])
    with col_btn:
        black_swan_clicked = st.button(
            "⚡ BLACK SWAN",
            type="primary",
            help="Inject a high-impact event into the pipeline"
        )
    with col_input:
        black_swan_headline = st.text_input(
            "Custom headline",
            value="Fed announces emergency rate hike of 100 basis points",
            label_visibility="collapsed"
        )

    if black_swan_clicked:
        result = api_post("/black-swan", {
            "headline": black_swan_headline,
            "source": "news:reuters"
        })
        if result:
            st.success(f"⚡ Injected: {black_swan_headline[:80]}")
            st.info("Watch impact scores update in ~10 seconds")

    st.divider()

    # Auto-refresh controls
    col_r1, col_r2 = st.columns([1, 4])
    with col_r1:
        refresh_rate = st.selectbox(
            "Refresh", [5, 10, 30], index=1,
            format_func=lambda x: f"Every {x}s"
        )

    @st.fragment(run_every=refresh_rate)
    def live_events_fragment():
        data = api_get("/events?limit=20")
        events = data.get("events", [])

        if not events:
            st.info("No events yet. Pipeline is warming up...")
            return

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        avg_impact = sum(e["impact_score"] for e in events) / len(events)
        high_impact = sum(1 for e in events if e["impact_score"] >= 7)
        neg_events  = sum(1 for e in events if e["sentiment_label"] == "negative")
        avg_latency = sum(e.get("latency_ms", 0) for e in events) / len(events)

        m1.metric("Avg Impact", f"{avg_impact:.1f}/10")
        m2.metric("High Impact (≥7)", high_impact)
        m3.metric("Negative Sentiment", neg_events)
        m4.metric("Avg Latency", f"{avg_latency:.0f}ms")

        st.divider()

        # Event cards
        for event in events[:10]:
            with st.expander(
                f"{impact_color(event['impact_score'])} "
                f"{sentiment_color(event['sentiment_label'])} "
                f"[{event['event_class']}] "
                f"Impact {event['impact_score']}/10 | "
                f"{event['headline'][:70]}",
                expanded=False
            ):
                c1, c2, c3 = st.columns(3)
                c1.metric("Sentiment",
                          f"{event['sentiment_label'].title()} "
                          f"({event['sentiment_score']:+.2f})")
                c2.metric("Novelty",
                          f"{novelty_badge(event['novelty_label'])} "
                          f"({event['novelty_score']:.2f})")
                c3.metric("Impact", f"{event['impact_score']}/10")

                if event.get("affected_tickers"):
                    st.write("**Affected tickers:**",
                             " · ".join(event["affected_tickers"]))

                if event.get("triggering_sentence"):
                    st.info(f"💬 **Trigger:** {event['triggering_sentence']}")

                # Impact breakdown
                breakdown = event.get("impact_breakdown", {})
                if breakdown:
                    bd_df = pd.DataFrame([
                        {"Factor": k.replace("_", " ").title(),
                         "Score": v}
                        for k, v in breakdown.items()
                    ])
                    fig = px.bar(
                        bd_df, x="Factor", y="Score",
                        title="Impact Score Breakdown",
                        height=200,
                        color="Score",
                        color_continuous_scale="RdYlGn"
                    )
                    fig.update_layout(margin=dict(t=30, b=0),
                                      showlegend=False)
                    st.plotly_chart(fig, use_container_width=True,
                                    key=f"impact_{event['event_id']}")

                st.caption(
                    f"Source: {event['source']} | "
                    f"Latency: {event.get('latency_ms', 0):.0f}ms | "
                    f"{event['timestamp'][:19]}"
                )

    live_events_fragment()

# ══════════════════════════════════════════════════════════════
# TAB 2: RISK GRAPH
# ══════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Risk Propagation Graph")
    st.caption("Shows how events propagate through sectors to portfolio positions")

    # Controls
    col_e, col_t = st.columns(2)
    with col_e:
        event_class = st.selectbox("Event Type", [
            "Geopolitical", "Macroeconomic",
            "Credit Event", "Merger/Acquisition", "Product Launch"
        ])
    with col_t:
        tickers_input = st.text_input(
            "Affected Tickers (comma separated)",
            value="TSM, NVDA, ASML, JPM"
        )

    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    if st.button("🔄 Generate Risk Graph", type="primary"):
        # Build graph from risk engine
        import sys
        sys.path.insert(0, ".")
        from src.risk.risk_graph import traverse

        result = traverse(event_class, tickers)

        try:
            from pyvis.network import Network
            import tempfile
            import os

            G = result["graph"]
            net = Network(height="500px", width="100%",
                         bgcolor="#0e1117", font_color="white")

            # Color scheme by node type
            colors = {
                "event":  "#FF4B4B",   # red
                "sector": "#FFA500",   # orange
                "ticker": "#00CC96",   # green
            }

            for node, data in G.nodes(data=True):
                node_type = data.get("node_type", "ticker")
                net.add_node(
                    node,
                    label=node,
                    color=colors.get(node_type, "#00CC96"),
                    size=30 if node_type == "event" else
                         20 if node_type == "sector" else 15,
                    title=f"{node_type.title()}: {node}"
                )

            for src, dst in G.edges():
                net.add_edge(src, dst, color="#666666", arrows="to")

            net.set_options(json.dumps({
                "physics": {"stabilization": {"iterations": 100}},
                "interaction": {"hover": True}
            }))

            # Save and display
            with tempfile.NamedTemporaryFile(
                suffix=".html", delete=False, mode="w"
            ) as f:
                net.save_graph(f.name)
                html = open(f.name).read()
                os.unlink(f.name)

            st.components.v1.html(html, height=520)

        except Exception as e:
            st.error(f"Graph render error: {e}")
            # Fallback: text representation
            st.write("**Affected sectors:**",
                     result["affected_sectors"])
            st.write("**Sector → Ticker map:**",
                     result["sector_tickers"])

        # Show exposure summary
        from src.risk.exposure_calc import calculate_exposure
        exposure = calculate_exposure(tickers, result["affected_sectors"])

        st.divider()
        st.subheader("Portfolio Exposure")
        col_v, col_p = st.columns(2)
        col_v.metric("Exposed Value",
                     f"${exposure['exposed_value']:,.0f}")
        col_p.metric("Exposure %",
                     f"{exposure['exposure_pct']}%")

        if exposure["positions"]:
            exp_df = pd.DataFrame(exposure["positions"])[
                ["id", "obligor", "type", "sector",
                 "value", "exposure_type"]
            ]
            exp_df["value"] = exp_df["value"].apply(
                lambda x: f"${x:,.0f}"
            )
            st.dataframe(exp_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 3: MODULE A — INDEX REBALANCER
# ══════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Tactical Index Rebalancer")
    st.caption(
        "15-stock sentiment-weighted index vs equal-weight benchmark"
    )

    @st.fragment(run_every=10)
    def rebalancer_fragment():
        weights_data = api_get("/module-a/weights")
        if not weights_data:
            st.info("Loading weights...")
            return

        weights = weights_data.get("weights", {})
        eq_weight = weights_data.get("equal_weight_pct", 6.67)

        if not weights:
            return

        # Current weights bar chart
        df = pd.DataFrame([
            {
                "Ticker": ticker,
                "Weight %": data["weight_pct"],
                "Sector": data["sector"],
                "vs Equal Weight": round(data["weight_pct"] - eq_weight, 4),
            }
            for ticker, data in weights.items()
        ]).sort_values("Weight %", ascending=False).reset_index(drop=True)

        # Color by deviation from equal weight
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["Ticker"],
            y=df["Weight %"],
            marker_color=[
                "#00CC96" if v > 0.001 else "#FF4B4B"
                for v in df["vs Equal Weight"]
            ],
            text=[f"{w:.1f}%" for w in df["Weight %"]],
            textposition="outside",
            name="Current Weight"
        ))
        fig.add_hline(
            y=eq_weight,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Equal Weight ({eq_weight}%)"
        )
        fig.update_layout(
            title="Current Index Weights",
            xaxis_title="Ticker",
            yaxis_title="Weight (%)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # Deviation table
        st.subheader("Weight Deviations from Equal Weight")
        deviation_df = df[["Ticker", "Sector",
                           "Weight %", "vs Equal Weight"]].copy()
        deviation_df = deviation_df.sort_values(
            "vs Equal Weight", ascending=False
        )
        st.dataframe(deviation_df, use_container_width=True)

    rebalancer_fragment()

    st.divider()
    st.subheader("Backtest: Sentiment-Weighted vs Equal-Weight")

    if st.button("▶ Run Backtest (90 days)"):
        with st.spinner("Downloading price data and running backtest..."):
            backtest = api_get("/module-a/backtest?days=90", timeout=60)

        if backtest:
            b1, b2, b3 = st.columns(3)
            b1.metric(
                "Sentiment-Weighted Return",
                f"{backtest['sentiment_final']*100:+.2f}%"
            )
            b2.metric(
                "Equal-Weight Return",
                f"{backtest['equalweight_final']*100:+.2f}%"
            )
            b3.metric(
                "Excess Return",
                f"{backtest['excess_return']*100:+.2f}%",
                delta=f"{backtest['excess_return']*100:+.2f}%"
            )

            # Cumulative return chart
            bt_df = pd.DataFrame({
                "Date": backtest["dates"],
                "Sentiment-Weighted": [
                    r * 100 for r in backtest["sentiment_returns"]
                ],
                "Equal-Weight": [
                    r * 100 for r in backtest["equalweight_returns"]
                ],
            })
            bt_df = bt_df.set_index("Date")

            fig = px.line(
                bt_df,
                title="Cumulative Returns (%)",
                labels={"value": "Return (%)", "variable": "Strategy"},
                color_discrete_map={
                    "Sentiment-Weighted": "#00CC96",
                    "Equal-Weight": "#888888"
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            if "note" in backtest:
                st.caption(f"Note: {backtest['note']}")

# ══════════════════════════════════════════════════════════════
# TAB 4: MODULE B — STRESS TESTER
# ══════════════════════════════════════════════════════════════

with tab4:
    st.subheader("Strategic Portfolio Stress Tester")
    st.caption(
        "Wholesale banking portfolio stress tested against "
        "detected high-impact events"
    )

    @st.fragment(run_every=10)
    def stress_fragment():
        summary = api_get("/module-b/portfolio")
        if not summary:
            return

        # Portfolio summary metrics
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Initial Value",
                  f"${summary['initial_value']:,.0f}")
        s2.metric(
            "Current Value",
            f"${summary['current_value']:,.0f}",
            delta=f"{summary['total_loss_pct']}%"
        )
        s3.metric("Total Loss",
                  f"${abs(summary['total_loss']):,.0f}")
        s4.metric("Scenarios Run",
                  summary['scenario_count'])

    stress_fragment()

    st.divider()

    # Manual stress test
    st.subheader("Manual Stress Test")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        manual_event = st.selectbox("Event Type", [
            "Geopolitical", "Macroeconomic",
            "Credit Event", "Merger/Acquisition", "Product Launch"
        ], key="manual_event")
    with col_b:
        manual_impact = st.slider(
            "Impact Score", 1.0, 10.0, 7.5, 0.5
        )
    with col_c:
        manual_tickers = st.text_input(
            "Affected Tickers",
            value="TSM, NVDA, ASML",
            key="manual_tickers"
        )

    if st.button("⚡ Run Stress Test", type="primary"):
        tickers_list = [
            t.strip().upper()
            for t in manual_tickers.split(",")
            if t.strip()
        ]
        with st.spinner("Running stress test..."):
            result = api_post("/stress-test", {
                "event_class":      manual_event,
                "impact_score":     manual_impact,
                "affected_tickers": tickers_list,
                "affected_sectors": [],
            })

        if result:
            r1, r2, r3 = st.columns(3)
            r1.metric("Portfolio Before",
                      f"${result['portfolio_before']:,.0f}")
            r2.metric(
                "Portfolio After",
                f"${result['portfolio_after']:,.0f}",
                delta=f"{result['loss_pct']}%"
            )
            r3.metric("Total Loss",
                      f"${abs(result['total_loss']):,.0f}")

            # Loss attribution waterfall
            if result.get("by_sector"):
                sectors  = [s["sector"] for s in result["by_sector"]]
                losses   = [abs(s["loss"]) for s in result["by_sector"]]
                pcts     = [s["pct_of_total"] for s in result["by_sector"]]

                fig = go.Figure(go.Bar(
                    x=sectors,
                    y=losses,
                    text=[f"{p:.1f}%" for p in pcts],
                    textposition="outside",
                    marker_color="#FF4B4B",
                    name="Loss by Sector"
                ))
                fig.update_layout(
                    title="Loss Attribution by Sector",
                    xaxis_title="Sector",
                    yaxis_title="Loss ($)",
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)

            # Position P&L table
            if result.get("positions"):
                pos_df = pd.DataFrame(result["positions"])[
                    ["obligor", "type", "sector",
                     "value_before", "pnl", "exposure_type"]
                ]
                pos_df = pos_df.sort_values("pnl")
                pos_df["value_before"] = pos_df["value_before"].apply(
                    lambda x: f"${x:,.0f}"
                )
                pos_df["pnl"] = pos_df["pnl"].apply(
                    lambda x: f"${x:,.0f}"
                )
                st.dataframe(pos_df, use_container_width=True)

            if result.get("summary_sentence"):
                st.success(f"📋 {result['summary_sentence']}")

    st.divider()

    # Scenario history
    st.subheader("Scenario History")

    @st.fragment(run_every=15)
    def history_fragment():
        history = api_get("/module-b/history")
        scenarios = history.get("scenarios", [])

        if not scenarios:
            st.info("No stress scenarios triggered yet.")
            return

        hist_df = pd.DataFrame(scenarios)
        hist_df["timestamp"] = pd.to_datetime(
            hist_df["timestamp"]
        ).dt.strftime("%H:%M:%S")

        fig = px.scatter(
            hist_df,
            x="timestamp",
            y="loss_pct",
            color="event_class",
            size="impact_score",
            title="Stress Test History — Loss % by Event",
            labels={
                "loss_pct": "Portfolio Loss (%)",
                "timestamp": "Time",
            },
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            hist_df[["scenario_id", "timestamp",
                     "event_class", "impact_score", "loss_pct"]],
            use_container_width=True
        )

    history_fragment()
