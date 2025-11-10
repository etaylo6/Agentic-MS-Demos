"""Utilities for removing trivial edges using standard library types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

from edge_grouper import Edge, normalize_edges

try:  # Optional import for backwards compatibility with ConstraintHG.
    from constrainthg.hypergraph import Hypergraph  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Hypergraph = None  # type: ignore

LogFn = Callable[[str], None]


def _noop_log(_: str) -> None:
    """Fallback logger when callers do not supply a hook."""


@dataclass(frozen=True)
class TrivialTrimOutcome:
    """Result returned by :func:`trim_trivial_edges`."""

    edges: Tuple[Edge, ...]
    removed_edges: Tuple[Edge, ...]
    removed_count: int
    notes: Tuple[str, ...]
    hypergraph: Optional[Any] = None


def _target_matches(target: Any, raw_targets: Set[Any], label_targets: Set[str]) -> bool:
    if target in raw_targets:
        return True
    label = getattr(target, "label", None)
    return isinstance(label, str) and label in label_targets


def _clone_hypergraph_without_edges(hypergraph: Hypergraph, excluded_ids: Set[Any], log: LogFn) -> Hypergraph:
    new_hg = Hypergraph(
        no_weights=hypergraph.no_weights,
        memory_mode=hypergraph.memory_mode,
    )
    new_hg.logging_is_setup = hypergraph.logging_is_setup
    new_hg.nodes.update(hypergraph.nodes)

    for edge_id, edge in hypergraph.edges.items():
        if edge_id in excluded_ids:
            continue
        new_edge = new_hg.add_edge(
            edge.source_nodes,
            edge.target,
            rel=getattr(edge, "rel", None),
            via=getattr(edge, "via", None),
            index_via=getattr(edge, "index_via", None),
            weight=getattr(edge, "weight", 1.0),
            label=getattr(edge, "label", None),
            index_offset=getattr(edge, "index_offset", 0),
            disposable=list(getattr(edge, "disposable", []) or []),
            edge_props=list(getattr(edge, "edge_props", []) or []),
        )
        for attr_name in ("semantic_group", "group"):
            if hasattr(edge, attr_name):
                setattr(new_edge, attr_name, getattr(edge, attr_name))

    log(f"Created hypergraph with {len(new_hg.edges)} edges (removed {len(excluded_ids)})")
    return new_hg


def trim_trivial_edges(
    edges_or_hypergraph: Any,
    edge_groups: Mapping[str, Sequence[Edge]],
    *,
    inputs: Mapping[Any, Any],
    skip_groups: Sequence[str] = (),
    log: Optional[LogFn] = None,
) -> TrivialTrimOutcome:
    """Remove edges whose targets are already satisfied by ``inputs``."""

    log_fn = log or _noop_log
    normalized_edges = list(normalize_edges(edges_or_hypergraph))
    raw_targets = set(inputs.keys())
    label_targets: Set[str] = set()
    for target in raw_targets:
        if isinstance(target, str):
            label_targets.add(target)
            continue
        label = getattr(target, "label", None)
        if isinstance(label, str):
            label_targets.add(label)
    skip_groups_set = set(skip_groups)
    hypergraph_input = (
        edges_or_hypergraph if (Hypergraph is not None and isinstance(edges_or_hypergraph, Hypergraph)) else None
    )

    excluded_edge_ids: Set[Any] = set()
    notes: list[str] = []

    for group_name, group_edges in edge_groups.items():
        if not group_edges or group_name in skip_groups_set:
            continue

        log_fn(f"Scanning group '{group_name}' for trivial removals")
        for edge in group_edges:
            edge_id = edge.metadata.get("edge_id", id(edge))
            if edge_id in excluded_edge_ids:
                continue
            if _target_matches(edge.target, raw_targets, label_targets):
                excluded_edge_ids.add(edge_id)
                label = edge.label or "unknown"
                target_label = getattr(edge.target, "label", edge.target)
                reason = f"Removing '{label}': target '{target_label}' already provided"
                notes.append(reason)
                log_fn(reason)

    if not excluded_edge_ids:
        log_fn("No trivial edges removed; returning original edge list")
        return TrivialTrimOutcome(
            edges=tuple(normalized_edges),
            removed_edges=tuple(),
            removed_count=0,
            notes=tuple(notes),
            hypergraph=hypergraph_input,
        )

    remaining_edges = tuple(edge for edge in normalized_edges if edge.metadata.get("edge_id", id(edge)) not in excluded_edge_ids)
    removed_edges = tuple(edge for edge in normalized_edges if edge.metadata.get("edge_id", id(edge)) in excluded_edge_ids)

    new_hypergraph = None
    if hypergraph_input is not None:
        new_hypergraph = _clone_hypergraph_without_edges(hypergraph_input, excluded_edge_ids, log_fn)

    return TrivialTrimOutcome(
        edges=remaining_edges,
        removed_edges=removed_edges,
        removed_count=len(excluded_edge_ids),
        notes=tuple(notes),
        hypergraph=new_hypergraph,
    )


__all__ = [
    "TrivialTrimOutcome",
    "trim_trivial_edges",
]
