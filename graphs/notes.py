"""Aegra graph entry for the Notes agent."""

from app.agents.note_agent.graph import note_graph, build_note_graph

# Precompiled graph export for Aegra
graph = note_graph


def make_graph(config):
    """Factory for building the notes graph with runtime configuration."""
    return build_note_graph()


__all__ = ["graph", "make_graph"]
