"""Risk propagation graph: event -> sectors -> affected tickers."""

import json
from functools import lru_cache

import networkx as nx

from src.config import data_path


@lru_cache(maxsize=1)
def _load_graph_config() -> dict:
    return json.loads(data_path("risk_graph_static.json").read_text())


@lru_cache(maxsize=1)
def _load_portfolio_tickers() -> set[str]:
    data = json.loads(data_path("synthetic_portfolio.json").read_text())
    return {
        p["ticker"] for p in data["positions"]
        if p.get("ticker")
    }


def traverse(event_class: str, affected_tickers: list[str] | None = None) -> dict:
    """
    Build a directed propagation graph.

    Event class determines broad affected sectors/risk buckets; explicitly
    detected tickers are also linked to their own sector. The graph only expands
    to names that are in the synthetic portfolio or were directly detected,
    keeping the live visualization readable while preserving exposure logic.
    """
    cfg = _load_graph_config()
    sector_tickers = cfg["sector_tickers"]
    event_sector_impact = cfg["event_sector_impact"]
    portfolio_tickers = _load_portfolio_tickers()

    direct = [t.upper() for t in (affected_tickers or [])]

    sectors = []
    for sector in event_sector_impact.get(event_class, []):
        if sector not in sectors:
            sectors.append(sector)

    ticker_to_sector = {
        ticker: sector
        for sector, tickers in sector_tickers.items()
        for ticker in tickers
    }
    for ticker in direct:
        sector = ticker_to_sector.get(ticker)
        if sector and sector not in sectors:
            sectors.append(sector)

    graph = nx.DiGraph()
    event_node = f"event:{event_class}"
    graph.add_node(event_node, node_type="event", label=event_class)

    sector_ticker_map = {}
    for sector in sectors:
        sector_node = f"sector:{sector}"
        graph.add_node(sector_node, node_type="sector", label=sector)
        graph.add_edge(event_node, sector_node, relation="impacts")

        # Show portfolio names plus explicitly detected tickers.
        candidates = []
        for ticker in sector_tickers.get(sector, []):
            if ticker in portfolio_tickers or ticker in direct:
                candidates.append(ticker)
        for ticker in direct:
            if ticker_to_sector.get(ticker) == sector and ticker not in candidates:
                candidates.append(ticker)

        sector_ticker_map[sector] = candidates
        for ticker in candidates:
            ticker_node = f"ticker:{ticker}"
            graph.add_node(ticker_node, node_type="ticker", label=ticker)
            graph.add_edge(sector_node, ticker_node, relation="exposes")

    return {
        "graph": graph,
        "event_class": event_class,
        "affected_sectors": sectors,
        "sector_tickers": sector_ticker_map,
        "direct_tickers": direct,
    }


if __name__ == "__main__":
    r = traverse("Geopolitical", ["TSM", "NVDA", "JPM"])
    print("Sectors:", r["affected_sectors"])
    print("Sector -> tickers:", r["sector_tickers"])
    print("Nodes:", list(r["graph"].nodes(data=True)))
    print("Edges:", list(r["graph"].edges()))
