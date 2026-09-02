"""Algorithm-facing derived views and search contracts."""

from .sc_dag import DAGCursor, DAGIndex, build_dag_index, topological_order
from .sabre import (
    ExecuteNode,
    LayoutSnapshot,
    RouteSwap,
    RoutedEvent,
    SabreResult,
    SiteId,
    sabre_map,
)
from .zap import (
    ZapInteraction,
    ZapTrace,
    architecture_sites,
    compile_interactions,
    load_architecture,
)

__all__ = [
    "DAGCursor",
    "DAGIndex",
    "build_dag_index",
    "topological_order",
    "ExecuteNode",
    "LayoutSnapshot",
    "RouteSwap",
    "RoutedEvent",
    "SabreResult",
    "SiteId",
    "sabre_map",
    "ZapInteraction",
    "ZapTrace",
    "architecture_sites",
    "compile_interactions",
    "load_architecture",
]
