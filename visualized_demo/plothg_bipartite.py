"""
Bipartite plotting utilities for hypergraph visualization and animation.

This module provides functions for creating static and animated visualizations
of hypergraph simulations using a bipartite layout, where nodes are arranged
in two columns (source nodes and target nodes) with edges connecting between them.

The bipartite layout provides a cleaner, more organized view of the hypergraph
structure, making it easier to understand the relationships between input and
output variables.
"""

# Standard library imports
import random
from functools import partial
from typing import Dict, List, Optional, Any, Tuple

# Third-party imports
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.axes import Axes
from matplotlib.patches import Patch, Circle, Rectangle
from matplotlib.lines import Line2D
from matplotlib.text import Text
import matplotlib.animation as animation
from functools import partial
import random
import networkx as nx
import numpy as np

# Local imports
from constrainthg.hypergraph import Hypergraph, Edge, Node, TNode

# Set random seed to ensure consistent node positioning across runs
random.seed(2)

class BipartitePlotSettings:
    """Settings for bipartite hypergraph visualization.
    
    This class provides comprehensive configuration options for bipartite
    hypergraph plotting including node colors, edge styles, animation parameters, 
    and layout settings specifically designed for bipartite layouts.
    
    To configure, instantiate the defaults using ``__init__``, then update
    the specific dictionary.

    Example::
        ps = BipartitePlotSettings()
        ps.node_solved.update(facecolor='#ffff00')
    
    """
    
    def __init__(self) -> None:
        # Default node appearance
        self.node_default = dict(
            facecolor='#ababab',
            edgecolor='#666666',
            linewidth=1,
            radius=0.14,  # larger nodes
            zorder=10,
        )

        self.node_source = dict(
            facecolor='#27ae60',  # Green for source nodes
            edgecolor='#1e8449',
            linewidth=2,
            radius=0.14,
            zorder=12,
        )

        self.node_target = dict(
            facecolor='#e67e22',  # Orange for target nodes
            edgecolor='#d35400',
            linewidth=2,
            radius=0.14,
            zorder=12,
        )

        # Function/group nodes (squares) - make these very large
        self.node_function = dict(
            facecolor='#000000',  # black function groups by default
            edgecolor='#000000',  # black outline as requested
            linewidth=2.5,
            radius=0.42,  # much larger squares
            zorder=13,
        )

        self.node_solved = dict(
            facecolor='#3498db',  # Blue for solved nodes
            edgecolor='#2980b9',
            linewidth=2,
            radius=0.14,
            zorder=15,
        )

        self.node_output_solved = dict(
            facecolor='#9b59b6',  # Purple for final output
            edgecolor='#8e44ad',
            linewidth=3,
            radius=0.16,
            zorder=20,
        )

        # Edge styles
        self.edge_default = dict(
            color='#cccccc44',
            linewidth=1,
            zorder=10,
        )

        self.edge_solved = dict(
            color='#00118888',
            linewidth=2,
            zorder=15,
        )

        self.edge_active = dict(
            color='#0033cc',
            linewidth=3,
            zorder=16,
        )

        # Bipartite layout settings
        self.bipartite = dict(
            left_column_x=-2.0,      # X position for variable nodes
            right_column_x=2.0,      # X position for function nodes
            node_spacing=0.45,       # Vertical spacing between nodes
            edge_curvature=0.35,     # Curvature of edges (0 = straight, 1 = curved)
            column_width=0.4,        # Width of each column
            x_jitter=0.25,           # Horizontal jitter for organic look
            layout_driver='spring',   # 'spring' | 'kamada' | 'spectral' | 'bipartite'
        )

        # Animation settings
        self.animation = dict(
            frames_per_edge=30,
            fps=30,
            circle_radius=0.08,
            circle_color='#e74c3c',
            circle_alpha=0.8,
        )

        # Text settings
        self.text = dict(
            fontsize=10,
            color='white',
            weight='bold',
            ha='center',
            va='center',
            zorder=25,
        )

        # Merge settings
        self.merge = dict(
            merge_equivalent_hyperedges=False,  # Default off; semantic groups take precedence
            merged_node_prefix='Merged',        # Prefix for merged node labels
            merged_node_style='diamond',        # Shape for merged nodes: 'circle', 'square', 'diamond'
            # Keyword-based grouping for function nodes
            group_keywords={
                'timoshenko': 'Timoshenko',
                'area': 'Area of Circle',
                'circle': 'Area of Circle',
                'moment': 'Moment/Shear',
            },
            # Explicit semantic groups by edge label: {'edge_label': 'Group Name'}
            semantic_groups={},
            # If True, read edge.semantic_group (or edge.group) attribute when present
            read_edge_attr=True,
        )

        # Alias map for shortened variable labels (filled at layout time)
        self.alias_map: Dict[str, str] = {}


def _build_label_aliases(hg: Hypergraph) -> Dict[str, str]:
    """Build unique short aliases for variable labels (e.g., 'radius' -> 'r')."""
    labels = []
    for node in hg.nodes:
        labels.append(node.label if hasattr(node, 'label') else str(node))

    aliases: Dict[str, str] = {}
    used: set = set()

    for label in sorted(set(labels)):
        base = ''.join([c for c in label if c.isalnum()])
        if not base:
            base = 'v'
        # Start with first character, then grow until unique
        idx = 1
        alias = base[:idx].lower()
        while alias in used and idx <= len(base):
            idx += 1
            alias = base[:idx].lower()
        # If still conflict (identical names), append numeric suffix
        suffix = 1
        final_alias = alias
        while final_alias in used:
            suffix += 1
            final_alias = f"{alias}{suffix}"
        aliases[label] = final_alias
        used.add(final_alias)
    return aliases


def merge_equivalent_hyperedges(hg: Hypergraph, ps: BipartitePlotSettings) -> Dict[str, List[str]]:
    """
    Identify and group hyperedges that have the same set of input/output nodes.
    
    Args:
        hg: Hypergraph to analyze
        ps: Bipartite plot settings
        
    Returns:
        Dictionary mapping merged node labels to lists of original edge labels
    """
    if not ps.merge['merge_equivalent_hyperedges']:
        return {}
    
    # Group edges by their node sets (both source and target nodes)
    edge_groups = {}
    
    for edge in hg.edges.values():
        # Get all nodes involved in this edge (sources + target)
        source_labels = set()
        for source_node in edge.source_nodes.values():
            if hasattr(source_node, 'label'):
                source_labels.add(source_node.label)
            else:
                source_labels.add(str(source_node))
        
        target_label = edge.target.label if hasattr(edge.target, 'label') else str(edge.target)
        
        # Create a key that represents the complete node set
        all_nodes = frozenset(source_labels | {target_label})
        
        if all_nodes not in edge_groups:
            edge_groups[all_nodes] = []
        edge_groups[all_nodes].append(edge.label)
    
    # Ensure alias map exists for compact labels
    if not ps.alias_map:
        ps.alias_map = _build_label_aliases(hg)

    # Create merged node mappings for groups with more than one edge, using readable names
    merged_nodes: Dict[str, List[str]] = {}
    for node_set, edge_labels in edge_groups.items():
        if len(edge_labels) > 1:
            # Build label from variables and target using aliases
            # Extract target (the one that is a target in any of the edges);
            # fallback to sorted list if ambiguous
            example_edge = next(e for e in hg.edges.values() if e.label in edge_labels)
            target_label = example_edge.target.label if hasattr(example_edge.target, 'label') else str(example_edge.target)
            source_labels = []
            for src in example_edge.source_nodes.values():
                source_labels.append(src.label if hasattr(src, 'label') else str(src))
            src_aliases = ','.join(sorted(ps.alias_map[s] for s in set(source_labels)))
            tgt_alias = ps.alias_map.get(target_label, target_label)
            merged_label_base = f"f({src_aliases})→{tgt_alias}"
            # Ensure uniqueness
            ml = merged_label_base
            k = 2
            while ml in merged_nodes:
                ml = f"{merged_label_base}#{k}"
                k += 1
            merged_nodes[ml] = edge_labels
    
    return merged_nodes


def group_function_nodes_semantic(hg: Hypergraph, ps: BipartitePlotSettings) -> Dict[str, str]:
    """
    Map each edge label to a semantic group name using priorities:
    1) ps.merge['semantic_groups'][edge.label]
    2) edge.semantic_group or edge.group attribute (if enabled)
    3) keyword rules from ps.merge['group_keywords']
    4) fallback f(src_aliases)->tgt_alias
    Returns {edge_label: group_name}.
    """
    groups: Dict[str, str] = {}
    keyword_map = ps.merge.get('group_keywords', {}) or {}
    explicit = ps.merge.get('semantic_groups', {}) or {}
    read_attr = bool(ps.merge.get('read_edge_attr', True))
    for edge in hg.edges.values():
        label = edge.label
        # 1) explicit mapping
        if label in explicit:
            groups[label] = explicit[label]
            continue
        # 2) attribute on edge
        if read_attr:
            group_attr = getattr(edge, 'semantic_group', None)
            if group_attr is None:
                group_attr = getattr(edge, 'group', None)
            if group_attr:
                groups[label] = str(group_attr)
                continue
        # 3) keywords
        low = label.lower()
        matched_kw = False
        for kw, name in keyword_map.items():
            if kw in low:
                groups[label] = name
                matched_kw = True
                break
        if matched_kw:
            continue
        # 4) fallback from sources -> target
        if not ps.alias_map:
            ps.alias_map = _build_label_aliases(hg)
        srcs = []
        for src in edge.source_nodes.values():
            s = src.label if hasattr(src, 'label') else str(src)
            srcs.append(ps.alias_map.get(s, s))
        tgt = edge.target.label if hasattr(edge.target, 'label') else str(edge.target)
        tgt_alias = ps.alias_map.get(tgt, tgt)
        src_aliases = ','.join(sorted(set(srcs)))
        groups[label] = f"f({src_aliases})->{tgt_alias}"
    return groups


def create_bipartite_layout(hg: Hypergraph, ps: BipartitePlotSettings) -> Dict[str, Tuple[float, float]]:
    """
    Create a bipartite layout using NetworkX layouts where variables and functions are separate node types.
    
    Args:
        hg: Hypergraph to layout
        ps: Bipartite plot settings
        
    Returns:
        Dictionary mapping node labels to (x, y) positions
    """
    # Build aliases for compact labels
    ps.alias_map = _build_label_aliases(hg)

    # Group functions semantically (highest priority)
    edge_to_group = group_function_nodes_semantic(hg, ps)
    # If semantic grouping is present, disable node-set merging to avoid conflicts
    merged_nodes = {}
    if not edge_to_group and ps.merge.get('merge_equivalent_hyperedges', False):
        merged_nodes = merge_equivalent_hyperedges(hg, ps)
    
    # Collect all variable nodes (from hypergraph nodes)
    variable_nodes = []
    for node in hg.nodes:
        if hasattr(node, 'label'):
            variable_nodes.append(node.label)
        else:
            variable_nodes.append(str(node))
    
    # Collect all function nodes (group names), excluding merged ones (handled later)
    function_nodes = []
    merged_edge_labels = set()
    for merged_label, edge_labels in merged_nodes.items():
        merged_edge_labels.update(edge_labels)
        function_nodes.append(merged_label)  # Add merged node instead
    
    # Add non-merged edge labels
    for edge in hg.edges.values():
        if edge.label not in merged_edge_labels:
            group_name = edge_to_group.get(edge.label, 'Function')
            if group_name not in function_nodes:
                function_nodes.append(group_name)
    
    # Remove duplicates and sort for consistent layout
    variable_nodes = sorted(list(set(variable_nodes)))
    function_nodes = sorted(list(set(function_nodes)))
    
    # Build a NetworkX graph using variable nodes and grouped function nodes
    G = nx.Graph()
    G.add_nodes_from(variable_nodes + function_nodes)

    for edge in hg.edges.values():
        # Use group name for function node
        group_name = edge_to_group.get(edge.label, 'Function')
        # Add edges from sources to group and group to target
        for src in edge.source_nodes.values():
            s = src.label if hasattr(src, 'label') else str(src)
            G.add_edge(s, group_name)
        t = edge.target.label if hasattr(edge.target, 'label') else str(edge.target)
        G.add_edge(group_name, t)

    # Choose layout
    driver = ps.bipartite.get('layout_driver', 'spring')
    if driver == 'kamada':
        raw_pos = nx.kamada_kawai_layout(G)
    elif driver == 'spectral':
        raw_pos = nx.spectral_layout(G)
    elif driver == 'bipartite':
        raw_pos = nx.bipartite_layout(G, variable_nodes)
    else:
        raw_pos = nx.spring_layout(G, k=2, iterations=100)

    # Scale and center the raw layout so nodes are spread across the canvas
    xs = [p[0] for p in raw_pos.values()]
    ys = [p[1] for p in raw_pos.values()]
    x_min, x_max = (min(xs), max(xs)) if xs else (0, 1)
    y_min, y_max = (min(ys), max(ys)) if ys else (0, 1)
    x_range = (x_max - x_min) or 1.0
    y_range = (y_max - y_min) or 1.0
    scale = 3.0 / max(x_range, y_range)

    # Add small jitter to reduce overlaps and a slight bias to separate types
    jitter = ps.bipartite['x_jitter']
    pos: Dict[str, Tuple[float, float]] = {}
    for node, (x, y) in raw_pos.items():
        jx = jitter * 0.6 * np.sin((hash(node) % 97) / 7.0)
        jy = jitter * 0.6 * np.cos((hash(node) % 89) / 5.0)
        # Slight bias to keep function groups generally to the right, but not in a line
        bias = 0.5 if node in function_nodes else -0.2
        cx = ((x - (x_min + x_max) / 2) * scale) + bias
        cy = (y - (y_min + y_max) / 2) * scale
        pos[node] = (cx + jx, cy + jy)

    # Simple overlap resolution: repel nodes based on visual radius
    def get_radius(label: str) -> float:
        if label in function_nodes:
            return ps.node_function['radius']
        return ps.node_default['radius']

    labels = list(pos.keys())
    for _ in range(60):  # iterations
        moved = False
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                ax, ay = pos[a]
                bx, by = pos[b]
                dx = bx - ax
                dy = by - ay
                dist2 = dx * dx + dy * dy
                min_d = get_radius(a) + get_radius(b) + 0.08
                if dist2 == 0:
                    dx, dy = 0.01, 0.01
                    dist2 = dx * dx + dy * dy
                dist = np.sqrt(dist2)
                if dist < min_d:
                    # push apart
                    push = (min_d - dist) * 0.5
                    ux = dx / dist
                    uy = dy / dist
                    pos[a] = (ax - ux * push, ay - uy * push)
                    pos[b] = (bx + ux * push, by + uy * push)
                    moved = True
        if not moved:
            break

    return pos


def plot_bipartite_nodes(hg: Hypergraph, ax: Axes, positions: Dict[str, Tuple[float, float]], 
                        ps: BipartitePlotSettings) -> Dict[str, Any]:
    """
    Plot nodes in bipartite layout with different styles for variables vs functions.
    Variables are circles, functions are squares, merged nodes are diamonds.
    
    Args:
        hg: Hypergraph to plot
        ax: Matplotlib axes object
        positions: Dictionary of node positions
        ps: Bipartite plot settings
        
    Returns:
        Dictionary of plotted shapes {label : Circle/Rectangle/Diamond}
    """
    shapes = {}
    
    # Get grouping; if present, turn off node-set merging to prevent duplicate nodes
    edge_to_group = group_function_nodes_semantic(hg, ps)
    merged_nodes = {}
    if not edge_to_group and ps.merge.get('merge_equivalent_hyperedges', False):
        merged_nodes = merge_equivalent_hyperedges(hg, ps)
    
    # Get variable nodes from hypergraph
    variable_nodes = set()
    for node in hg.nodes:
        if hasattr(node, 'label'):
            variable_nodes.add(node.label)
        else:
            variable_nodes.add(str(node))
    
    # Get function nodes as group names
    function_nodes = set()
    merged_edge_labels = set()
    for merged_label, edge_labels in merged_nodes.items():
        merged_edge_labels.update(edge_labels)
    
    for edge in hg.edges.values():
        if edge.label not in merged_edge_labels:
            function_nodes.add(edge_to_group.get(edge.label, 'Function'))
    
    # Plot all nodes with appropriate styling
    for node_label, (x, y) in positions.items():
        if node_label in variable_nodes:
            # Variable nodes use circles with default styling
            shape = create_circle((x, y), ps.node_default)
        elif node_label in merged_nodes:
            # Merged nodes use diamonds with special styling
            shape = create_diamond((x, y), ps.node_target)
        elif node_label in function_nodes:
            # Function nodes (grouped) use large squares
            shape = create_square((x, y), ps.node_function)
        else:
            # Fallback to default circle
            shape = create_circle((x, y), ps.node_default)
        
        ax.add_patch(shape)
        shapes[node_label] = shape
        
        # Add node label text (shorter aliases for variables)
        display_label = ps.alias_map.get(node_label, node_label)
        ax.text(x, y, display_label, **ps.text)
    
    return shapes


def plot_bipartite_edges(hg: Hypergraph, ax: Axes, positions: Dict[str, Tuple[float, float]], 
                        ps: BipartitePlotSettings) -> Dict[str, Any]:
    """
    Plot edges in bipartite layout: variables -> functions -> variables.
    
    Args:
        hg: Hypergraph to plot
        ax: Matplotlib axes object
        positions: Dictionary of node positions
        ps: Bipartite plot settings
        
    Returns:
        Dictionary of plotted lines {label : Line2D}
    """
    lines = {}
    
    # Get merged nodes and grouping
    merged_nodes = merge_equivalent_hyperedges(hg, ps)
    edge_to_group = group_function_nodes_semantic(hg, ps)
    
    for edge in hg.edges.values():
        # Check if this edge is part of a merged group
        merged_label = None
        for merged_node_label, edge_labels in merged_nodes.items():
            if edge.label in edge_labels:
                merged_label = merged_node_label
                break
        
        # Use merged label if available, otherwise use grouped function label
        function_label = merged_label if merged_label else edge_to_group.get(edge.label, 'Function')
        
        # Create edges from source variables to function
        for source_node in edge.source_nodes.values():
            if hasattr(source_node, 'label'):
                source_label = source_node.label
            else:
                source_label = str(source_node)
            
            if source_label in positions and function_label in positions:
                # Edge from variable to function
                line = create_curved_edge(positions[source_label], positions[function_label], ps)
                ax.add_line(line)
                lines[f"{source_label}->{function_label}"] = line
        
        # Create edge from function to target variable
        if hasattr(edge.target, 'label'):
            target_label = edge.target.label
        else:
            target_label = str(edge.target)
        
        if function_label in positions and target_label in positions:
            # Edge from function to variable
            line = create_curved_edge(positions[function_label], positions[target_label], ps)
            ax.add_line(line)
            lines[f"{function_label}->{target_label}"] = line
    
    return lines


def create_curved_edge(start_pos: Tuple[float, float], end_pos: Tuple[float, float], 
                      ps: BipartitePlotSettings) -> Line2D:
    """
    Create a curved edge between two positions.
    
    Args:
        start_pos: Starting (x, y) position
        end_pos: Ending (x, y) position
        ps: Bipartite plot settings
        
    Returns:
        Line2D object for the curved edge
    """
    x1, y1 = start_pos
    x2, y2 = end_pos
    
    # Create control points for curved edge
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    
    # Add curvature by offsetting the control point
    curvature = ps.bipartite['edge_curvature']
    if x2 > x1:  # Rightward edge
        control_x = mid_x + curvature
    else:  # Leftward edge
        control_x = mid_x - curvature
    control_y = mid_y
    
    # Create quadratic Bezier curve
    t = np.linspace(0, 1, 50)
    x_curve = (1-t)**2 * x1 + 2*(1-t)*t * control_x + t**2 * x2
    y_curve = (1-t)**2 * y1 + 2*(1-t)*t * control_y + t**2 * y2
    
    return Line2D(x_curve, y_curve, **ps.edge_default)


def plot_bipartite_edge(edge: Edge, ax: Axes, positions: Dict[str, Tuple[float, float]], 
                       ps: BipartitePlotSettings) -> Any:
    """
    Plot a single edge with curved connection in bipartite layout.
    
    Args:
        edge: Edge object to plot
        ax: Matplotlib axes object
        positions: Dictionary of node positions
        ps: Bipartite plot settings
        
    Returns:
        Line2D object representing the edge
    """
    # Get source and target positions
    source_positions = []
    for source_node in edge.source_nodes.values():
        # Handle both Node objects and string labels
        if hasattr(source_node, 'label'):
            source_label = source_node.label
        else:
            source_label = str(source_node)
        if source_label in positions:
            source_positions.append(positions[source_label])
    
    # Handle target node
    if hasattr(edge.target, 'label'):
        target_label = edge.target.label
    else:
        target_label = str(edge.target)
    
    if target_label not in positions:
        return None
    
    target_pos = positions[target_label]
    
    # Create curved edge from multiple sources to single target
    for source_pos in source_positions:
        line = create_curved_edge(source_pos, target_pos, ps)
        ax.add_line(line)
    
    # Return the first line (or create a representative line)
    if source_positions:
        return create_curved_edge(source_positions[0], target_pos, ps)
    return None


def create_curved_edge(start_pos: Tuple[float, float], end_pos: Tuple[float, float], 
                      ps: BipartitePlotSettings) -> Line2D:
    """
    Create a curved edge between two points.
    
    Args:
        start_pos: Starting position (x, y)
        end_pos: Ending position (x, y)
        ps: Bipartite plot settings
        
    Returns:
        Line2D object representing the curved edge
    """
    x1, y1 = start_pos
    x2, y2 = end_pos
    
    # Create control points for curved edge
    curvature = ps.bipartite['edge_curvature']
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    
    # Add curvature perpendicular to the line
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    
    if length > 0:
        # Perpendicular vector
        perp_x = -dy / length
        perp_y = dx / length
        
        # Control point offset
        offset = curvature * 0.5
        control_x = mid_x + perp_x * offset
        control_y = mid_y + perp_y * offset
        
        # Create quadratic Bezier curve
        t = np.linspace(0, 1, 50)
        x_curve = (1-t)**2 * x1 + 2*(1-t)*t * control_x + t**2 * x2
        y_curve = (1-t)**2 * y1 + 2*(1-t)*t * control_y + t**2 * y2
        
        line = Line2D(x_curve, y_curve, **ps.edge_default)
    else:
        # Fallback to straight line
        line = Line2D([x1, x2], [y1, y2], **ps.edge_default)
    
    return line


def create_circle(center: Tuple[float, float], props: Dict[str, Any]) -> Any:
    """
    Create a circle patch with given properties.
    
    Args:
        center: Center position (x, y)
        props: Circle properties
        
    Returns:
        Circle patch object
    """
    circle = plt.Circle(center, **props)
    return circle


def create_square(center: Tuple[float, float], props: Dict[str, Any]) -> Any:
    """
    Create a square patch with given properties.
    
    Args:
        center: Center position (x, y)
        props: Square properties
        
    Returns:
        Rectangle patch object
    """
    # Extract size from props (assuming it's the radius for circles)
    size = props.get('radius', 0.1)
    # Create rectangle centered at the given point
    square = Rectangle(
        (center[0] - size/2, center[1] - size/2),  # bottom-left corner
        size, size,  # width, height
        **{k: v for k, v in props.items() if k != 'radius'}
    )
    return square


def create_diamond(center: Tuple[float, float], props: Dict[str, Any]) -> Any:
    """
    Create a diamond patch with given properties.
    
    Args:
        center: Center position (x, y)
        props: Diamond properties
        
    Returns:
        Polygon patch object
    """
    from matplotlib.patches import Polygon
    
    # Extract size from props (assuming it's the radius for circles)
    size = props.get('radius', 0.1)
    
    # Create diamond vertices (rotated square)
    x, y = center
    vertices = [
        (x, y + size),      # top
        (x + size, y),      # right
        (x, y - size),      # bottom
        (x - size, y),      # left
    ]
    
    diamond = Polygon(
        vertices,
        **{k: v for k, v in props.items() if k != 'radius'}
    )
    return diamond


def color_patch(settings_label: str, ps: BipartitePlotSettings, patch: Any) -> Any:
    """
    Color a patch according to plot settings.
    
    Args:
        settings_label: Label for the plot settings
        ps: Bipartite plot settings object
        patch: Patch object to color
        
    Returns:
        The colored patch object
    """
    props = getattr(ps, settings_label)
    patch.set(**props)
    return patch


def set_function_node_color(shapes: Dict[str, Any], group_label: str, facecolor: str,
                            edgecolor: Optional[str] = None) -> Optional[Any]:
    """
    Set the color of a function (group) node by its label.
    Args:
        shapes: Dict of plotted shapes from plot_bipartite_nodes
        group_label: Label of the function/group node to recolor
        facecolor: New face color (e.g., '#ff0000')
        edgecolor: Optional edge color
    Returns:
        The updated patch if found, else None
    """
    shape = shapes.get(group_label)
    if shape is None:
        return None
    try:
        if hasattr(shape, 'set_facecolor'):
            shape.set_facecolor(facecolor)
        if edgecolor is not None and hasattr(shape, 'set_edgecolor'):
            shape.set_edgecolor(edgecolor)
        return shape
    except Exception:
        return None


def set_all_function_nodes_color(shapes: Dict[str, Any], facecolor: str,
                                 edgecolor: Optional[str] = None) -> None:
    """
    Set the color for all function (square) nodes at once.
    This detects function nodes by rectangle type.
    Args:
        shapes: Dict of plotted shapes from plot_bipartite_nodes
        facecolor: New face color
        edgecolor: Optional edge color
    """
    for shape in shapes.values():
        if isinstance(shape, Rectangle):
            try:
                shape.set_facecolor(facecolor)
                if edgecolor is not None:
                    shape.set_edgecolor(edgecolor)
            except Exception:
                continue


def get_node_label(node) -> str:
    """
    Get the label of the node for various types.
    
    Args:
        node: Node object of various types
        
    Returns:
        String label of the node
    """
    if isinstance(node, Node):
        return node.label
    if isinstance(node, TNode):
        return node.node_label
    return str(node)


def get_line_label(tnode: TNode) -> str:
    """
    Get the label of the edge from a TNode.
    
    Args:
        tnode: TNode object
        
    Returns:
        Edge label string
    """
    if tnode.gen_edge_label is None:
        return None
    label = tnode.gen_edge_label.split('#')[0]
    return label


def get_edge_label_from_tnode(t: TNode, lines: Dict[str, Any], merged_nodes: Dict[str, List[str]] = None) -> Optional[str]:
    """
    Get edge label from tnode for line matching, handling merged hyperedges.
    
    Args:
        t: TNode object
        lines: Dictionary of line objects
        merged_nodes: Dictionary of merged node mappings
        
    Returns:
        Edge label string or None if not found
    """
    if t.gen_edge_label is None:
        return None
    
    # If we have merged nodes, check if this TNode's edge is part of a merged group
    if merged_nodes:
        for merged_label, edge_labels in merged_nodes.items():
            if t.gen_edge_label in edge_labels:
                # Find the line that connects to/from the merged node
                target_node = t.node_label
                for line_label in lines.keys():
                    if line_label.endswith(f'->{target_node}') and merged_label in line_label:
                        return line_label
                    elif line_label.startswith(f'{target_node}->') and merged_label in line_label:
                        return line_label
                # If no specific line found, return the merged label
                return merged_label
    
    # Original logic for non-merged edges
    # Extract source and target node information from the TNode
    if hasattr(t, 'children') and t.children:
        # Locate the edge connecting to this calculated node
        target_node = t.node_label
        # Find which source nodes connect to this target
        for line_label in lines.keys():
            if line_label.endswith(f'->{target_node}'):
                return line_label
    else:
        # This is an input node, find edges that start from it
        source_node = t.node_label
        for line_label in lines.keys():
            if line_label.startswith(f'{source_node}->'):
                return line_label
    
    return None


def create_bipartite_animation(hg: Hypergraph, tnodes: List[TNode], inputs: List[str], 
                              output: str, shapes: Dict[str, Any], lines: Dict[str, Any],
                              positions: Dict[str, Tuple[float, float]], ps: BipartitePlotSettings,
                              fig: Any, ax: Axes, text_box: Any) -> Any:
    """
    Create a bipartite animation for hypergraph visualization.
    
    Args:
        hg: Hypergraph object
        tnodes: List of TNode objects from solution
        inputs: List of input node labels
        output: Target node label
        shapes: Dictionary of shape patches (circles and squares)
        lines: Dictionary of line objects
        positions: Dictionary of node positions
        ps: Bipartite plot settings
        fig: Matplotlib figure
        ax: Matplotlib axes
        text_box: Text box for status updates
        
    Returns:
        Animation object
    """
    # Get merged nodes for animation
    merged_nodes = merge_equivalent_hyperedges(hg, ps)
    
    # Initialize moving circle
    moving_circle = plt.Circle((0, 0), ps.animation['circle_radius'], 
                              color=ps.animation['circle_color'], 
                              alpha=0.0, zorder=100)
    ax.add_patch(moving_circle)
    
    # Filter out source-node entries (no generating edge) for path traversal
    path_tnodes = [t for t in tnodes if getattr(t, 'gen_edge_label', None) is not None]

    # Calculate animation parameters
    frames_per_edge = ps.animation['frames_per_edge']
    input_display_frames = len(inputs) * 20
    pathfinding_frames = len(path_tnodes) * frames_per_edge
    final_result_frames = 60
    completion_frames = 30
    
    total_frames = input_display_frames + pathfinding_frames + final_result_frames + completion_frames
    interval = 1000 / ps.animation['fps']
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, partial(animate_bipartite_frame,
                     tnodes=tnodes,
                     path_tnodes=path_tnodes,
                     inputs=inputs,
                     output=output,
                     shapes=shapes,
                     lines=lines,
                     positions=positions,
                     text=text_box,
                     moving_circle=moving_circle,
                     frames_per_edge=frames_per_edge,
                     input_display_frames=input_display_frames,
                     pathfinding_frames=pathfinding_frames,
                     final_result_frames=final_result_frames,
                     merged_nodes=merged_nodes,
                     ps=ps),
        frames=total_frames, interval=interval, blit=False, repeat=False)
    
    return ani


def animate_bipartite_frame(frame: int, tnodes: List[TNode], inputs: List[str], 
                           output: str, shapes: Dict[str, Any], lines: Dict[str, Any],
                           positions: Dict[str, Tuple[float, float]], text: Any,
                           moving_circle: Any, frames_per_edge: int,
                           input_display_frames: int, pathfinding_frames: int,
                           final_result_frames: int, merged_nodes: Dict[str, List[str]],
                           ps: BipartitePlotSettings, path_tnodes: List[TNode]) -> List[Any]:
    """
    Animate a single frame of the bipartite visualization.
    
    Args:
        frame: Current frame number
        tnodes: List of TNode objects
        inputs: List of input node labels
        output: Target node label
        shapes: Dictionary of shape patches (circles and squares)
        lines: Dictionary of line objects
        positions: Dictionary of node positions
        text: Text box for status updates
        moving_circle: Moving circle patch
        frames_per_edge: Frames per edge traversal
        input_display_frames: Total frames for input display
        pathfinding_frames: Total frames for pathfinding
        final_result_frames: Total frames for final result
        ps: Bipartite plot settings
        
    Returns:
        List of modified patches
    """
    mod_patches = []
    
    # Phase 1: Highlight source nodes with their input values
    if frame < input_display_frames:
        input_index = frame // 20
        sub_frame = frame % 20
        
        if input_index < len(inputs):
            input_node = inputs[input_index]
            text.set_text(f'Setting source node: {input_node}')
            mod_patches.append(text)
            
            if sub_frame == 0 and input_node in shapes:
                mod_patches.append(color_patch('node_source', ps, shapes[input_node]))
        else:
            text.set_text('Source nodes set. Starting pathfinding...')
            mod_patches.append(text)
    
    # Phase 2: Animate solution path traversal
    elif frame < input_display_frames + pathfinding_frames:
        pathfinding_frame = frame - input_display_frames
        tnode_index = pathfinding_frame // frames_per_edge
        sub_frame = pathfinding_frame % frames_per_edge
        
        if tnode_index < len(path_tnodes):
            t = path_tnodes[tnode_index]
            text.set_text(f'Calculating: {t.node_label}')
            mod_patches.append(text)
            
            # Highlight current node: avoid reverting solved (green) nodes to orange
            if t.node_label in shapes:
                shape = shapes[t.node_label]
                # Only apply target style if the node hasn't been marked solved yet
                fc = getattr(shape, 'get_facecolor', lambda: None)()
                # If facecolor exists and is already green-ish, skip
                apply_target = True
                try:
                    if fc is not None and len(fc) >= 3:
                        # simple check: green channel dominant
                        if fc[1] > fc[0] and fc[1] > fc[2]:
                            apply_target = False
                except Exception:
                    pass
                if apply_target:
                    mod_patches.append(color_patch('node_target', ps, shape))
            
            # Animate moving circle along edge
            if t.node_label in positions:
                current_pos = positions[t.node_label]
                moving_circle.center = current_pos
                moving_circle.set_alpha(ps.animation['circle_alpha'])
                mod_patches.append(moving_circle)
            
            # Highlight active edge
            edge_label = get_edge_label_from_tnode(t, lines, merged_nodes)
            if edge_label and edge_label in lines:
                mod_patches.append(color_patch('edge_active', ps, lines[edge_label]))
            
            # Mark previously solved edges
            for i in range(tnode_index):
                prev_t = path_tnodes[i]
                if prev_t.gen_edge_label is not None:
                    prev_edge_label = get_edge_label_from_tnode(prev_t, lines, merged_nodes)
                    if prev_edge_label and prev_edge_label in lines:
                        mod_patches.append(color_patch('edge_solved', ps, lines[prev_edge_label]))
    
    # Phase 3: Display final result
    elif frame < input_display_frames + pathfinding_frames + final_result_frames:
        text.set_text(f'Final result: {output}')
        mod_patches.append(text)
        
        # Highlight final output node
        if output in shapes:
            mod_patches.append(color_patch('node_output_solved', ps, shapes[output]))
        
        # Hide moving circle
        moving_circle.set_alpha(0.0)
        mod_patches.append(moving_circle)
    
    # Phase 4: Completion
    else:
        text.set_text('Animation complete!')
        mod_patches.append(text)
        moving_circle.set_alpha(0.0)
        mod_patches.append(moving_circle)
    
    return mod_patches


def plot_bipartite_hypergraph(hg: Hypergraph, ax: Axes, ps: Optional[BipartitePlotSettings] = None) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Tuple[float, float]]]:
    """
    Plot a complete bipartite hypergraph visualization.
    
    Args:
        hg: Hypergraph to plot
        ax: Matplotlib axes object
        ps: Optional bipartite plot settings
        
    Returns:
        Tuple of (circles, lines, positions) dictionaries
    """
    if ps is None:
        ps = BipartitePlotSettings()
    
    # Create bipartite layout
    positions = create_bipartite_layout(hg, ps)
    
    # Plot nodes and edges
    shapes = plot_bipartite_nodes(hg, ax, positions, ps)
    lines = plot_bipartite_edges(hg, ax, positions, ps)
    
    # Set axis properties
    ax.set_aspect('equal')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.axis('off')
    
    return shapes, lines, positions


# Example usage and testing
if __name__ == "__main__":
    # This would be used for testing the bipartite plotting functionality
    pass
