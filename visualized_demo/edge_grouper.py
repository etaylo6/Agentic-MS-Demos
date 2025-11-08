"""Utilities for grouping edge records using standard library primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # Optional import to preserve compatibility with the original project.
    from constrainthg.hypergraph import Hypergraph  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Hypergraph = None  # type: ignore


@dataclass(frozen=True)
class Edge:
    """Lightweight representation of a directed relation."""

    sources: Tuple[Any, ...]
    target: Any
    label: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GroupingRule:
    """Callable-based rule used to assign edges to a group."""

    name: str
    matcher: Callable[[Optional[str]], bool]
    priority: int = 0


def build_prefix_rule(prefix: str, name: str, priority: int = 0, *, case_sensitive: bool = True) -> GroupingRule:
    """Factory to create a :class:`GroupingRule` that matches label prefixes."""

    prefix_check = prefix if case_sensitive else prefix.lower()

    def _matcher(label: Optional[str]) -> bool:
        if not isinstance(label, str):
            return False
        value = label if case_sensitive else label.lower()
        return value.startswith(prefix_check)

    return GroupingRule(name=name, matcher=_matcher, priority=priority)


DEFAULT_PREFIX_MAP: Dict[str, str] = {
    "[MAT]": "material_properties",
    "[GEO-A]": "geometry_area",
    "[GEO-I]": "geometry_moi",
    "[TIM]": "beam_timoshenko",
    "[EB]": "beam_euler_bernoulli",
}


def build_rules_from_prefixes(
    prefix_map: Mapping[str, str],
    *,
    case_sensitive: bool = True,
) -> Tuple[GroupingRule, ...]:
    """Construct grouping rules from a prefix -> group mapping."""

    return tuple(
        build_prefix_rule(prefix, group, priority=index, case_sensitive=case_sensitive)
        for index, (prefix, group) in enumerate(prefix_map.items(), start=len(prefix_map))
    )


_DEFAULT_RULES: List[GroupingRule] = list(build_rules_from_prefixes(DEFAULT_PREFIX_MAP))


def get_default_rules() -> Tuple[GroupingRule, ...]:
    """Return the currently configured default grouping rules."""

    return tuple(_DEFAULT_RULES)


def set_default_rules(rules: Sequence[GroupingRule]) -> None:
    """Override the default rules used by :func:`group_edges`."""

    _DEFAULT_RULES.clear()
    _DEFAULT_RULES.extend(rules)


def set_default_prefix_map(prefix_map: Mapping[str, str], *, case_sensitive: bool = True) -> None:
    """Convenience helper to rebuild defaults from a prefix map."""

    set_default_rules(build_rules_from_prefixes(prefix_map, case_sensitive=case_sensitive))


def _normalize_sources(raw_sources: Any) -> Tuple[Any, ...]:
    if isinstance(raw_sources, Mapping):
        return tuple(raw_sources.values())
    if isinstance(raw_sources, Sequence):
        return tuple(raw_sources)
    return (raw_sources,)


def _coerce_edges_input(edges: Any) -> List[Edge]:
    if isinstance(edges, Sequence) and edges and all(isinstance(edge, Edge) for edge in edges):
        return list(edges)

    if Hypergraph is not None and isinstance(edges, Hypergraph):
        coerced: List[Edge] = []
        for edge_id in edges.edges:
            edge_obj = edges.get_edge(edge_id)
            if edge_obj is None:
                continue
            label = getattr(edge_obj, "_label", getattr(edge_obj, "label", None))
            sources = _normalize_sources(getattr(edge_obj, "source_nodes", ()))
            target = getattr(edge_obj, "target", None)
            metadata = {
                "edge_id": edge_id,
                "hypergraph": edges,
                "original": edge_obj,
            }
            coerced.append(Edge(sources=sources, target=target, label=label, metadata=metadata))
        return coerced

    if isinstance(edges, Sequence):
        return [edge if isinstance(edge, Edge) else Edge(sources=tuple(), target=edge) for edge in edges]  # pragma: no cover

    raise TypeError("Unsupported edge container; expected Sequence[Edge] or Hypergraph.")


def normalize_edges(edges: Any) -> Tuple[Edge, ...]:
    """Return a tuple of :class:`Edge` records from supported inputs."""

    return tuple(_coerce_edges_input(edges))


def group_edges(
    edges: Any,
    *rules: GroupingRule,
    default_group: str = "other",
    label_extractor: Optional[Callable[[Edge], Optional[str]]] = None,
) -> Dict[str, List[Edge]]:
    """Group edges according to the supplied rules."""

    normalized_edges = _coerce_edges_input(edges)
    applied_rules = sorted(rules or get_default_rules(), key=lambda rule: rule.priority, reverse=True)
    extractor = label_extractor or (lambda edge: edge.label)

    grouped: Dict[str, List[Edge]] = {}
    for edge in normalized_edges:
        label = extractor(edge)
        target_group = default_group
        for rule in applied_rules:
            if rule.matcher(label):
                target_group = rule.name
                break
        grouped.setdefault(target_group, []).append(edge)

    return grouped


def group_edges_by_label(
    edges: Any,
    label_to_group: Optional[Mapping[str, str]] = None,
    *,
    case_sensitive: bool = True,
    default_group: str = "other",
) -> Dict[str, List[Edge]]:
    """Backwards-compatible helper that mimics the previous API."""

    normalized_edges = _coerce_edges_input(edges)

    if label_to_group:
        rules = build_rules_from_prefixes(label_to_group, case_sensitive=case_sensitive)
        return group_edges(normalized_edges, *rules, default_group=default_group)

    return group_edges(normalized_edges, default_group=default_group)


def iter_edges(edge_groups: Mapping[str, Iterable[Edge]], group_name: str) -> Iterable[Edge]:
    """Yield edges of a given ``group_name`` without assuming storage type."""

    return tuple(edge_groups.get(group_name, ()))


def list_group_labels(
    edge_groups: Mapping[str, Iterable[Edge]],
    group_name: str,
    *,
    extractor: Optional[Callable[[Edge], Optional[str]]] = None,
) -> list[str]:
    """Return a list with the textual label of each edge in ``group_name``."""

    extractor = extractor or (lambda edge: edge.label or "unknown")
    return [extractor(edge) for edge in edge_groups.get(group_name, [])]


__all__ = [
    "Edge",
    "GroupingRule",
    "build_prefix_rule",
    "DEFAULT_PREFIX_MAP",
    "build_rules_from_prefixes",
    "get_default_rules",
    "set_default_rules",
    "set_default_prefix_map",
    "group_edges",
    "group_edges_by_label",
    "normalize_edges",
    "iter_edges",
    "list_group_labels",
]
