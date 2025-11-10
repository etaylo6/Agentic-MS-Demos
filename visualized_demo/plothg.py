"""
Plotting utilities for hypergraph visualization and animation.

This module provides functions for creating static and animated visualizations
of hypergraph simulations, including moving circle animations along solution paths.
"""

# Standard library imports
import random
from functools import partial
from typing import Dict, List, Optional, Any, Tuple

# Third-party imports
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.axes import Axes
from matplotlib.patches import Patch
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

class PlotSettings:
    """Settings for hypergraph visualization.
    
    This class provides comprehensive configuration options for hypergraph
    plotting including node colors, edge styles, animation parameters, and layout settings.
    
    To configure, instantiate the defaults using ``__init__``, then update
    the specific dictionary.

    Example::
        ps = PlotSettings()
        ps.node_solved.update(facecolor='#ffff00')
    
    """
    
    def __init__(self) -> None:
        # Default node appearance
        self.node_default = dict(
            facecolor='#ababab',
            edgecolor='k',
            linewidth=0,
            linestyle='-',
            radius=0.2,
            zorder=30,
        )

        # Node states
        self.node_solved = dict(
            facecolor='#00ffff',
            zorder=32,
        )

        self.node_source = dict(
            facecolor='#00aa00',  # Green
            zorder=33,
        )

        self.node_target = dict(
            facecolor='#ffaa00',  # Yellow
            zorder=35,
        )

        self.node_input = dict(
            facecolor='#00ffff',
            edgecolor='k',
            linewidth=2.0,
            zorder=40,
        )

        self.node_output = dict(
            facecolor='#00bb66',
            edgecolor='k',
            linewidth=0.0,
            zorder=50,
        )

        self.node_output_solved = self.node_output | dict(
            linewidth=2.0,
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
            linewidth=2,
            zorder=16,
        )

        # Layout and spacing
        self.spacing = dict(
            x_spacing=0.8,
            y_spacing=0.8,
            num_rows=15,
            jitter_rate=0.1
        )

        # Text settings
        self.text = dict(
            x=0, 
            y=0,
            s='',
            ha='left',
            va='bottom',
            fontsize=10,
        )

        # Figure settings
        self.settings = dict(
            figsize=(10, 8),
            layout='constrained'
        )
        
        # Animation parameters for solution path visualization
        self.animation = dict(
            frames_per_edge=10,  # Number of frames per edge traversal
            fps=30,              # Frames per second
            circle_radius=0.05,  # Radius of moving circle
            circle_color='#ff0000',  # Color of moving circle
            circle_alpha=0.8,    # Alpha of moving circle
            # Visuals for BFS jumps between nodes without a hyperedge
            jump_color='#888888',
            jump_alpha=0.6,
            jump_linewidth=1.5
        )

def plot_simulation(hg: Hypergraph, ps: PlotSettings, inputs: Dict[str, float], 
                    output: Node, **kwargs) -> None:
    """Animates a simulation of the hypergraph.
    
    Parameters
    ----------
    hg : Hypergraph
        The hypergraph to simulate.
    ps : PlotSettings
        Settings for the plot.
    inputs : Dict[str, float]
        Inputs to pass to ``hg.solve``, of the form {label : value}.
    output : Node | str
        The node (or node label) to solve for.
    **kwargs
        Other arguments to pass to the solver. See documentation for 
        `Hypergraph.solve() <https://constrainthg.readthedocs.io/en/latest/constrainthg.html#constrainthg.hypergraph.Hypergraph.solve>`_ 
        for more information.
    """
    fig, ax = plt.subplots(**ps.settings)
    text_box = create_textbox(ax, ps)
    tnodes = sim_hg(hg, inputs, output, **kwargs)
    circles, lines = initialize_hg(hg, ax, inputs, output, ps)
    ani = animate_hg(fig, ax, tnodes, list(inputs.keys()), get_node_label(output), 
                     circles, lines, text_box, ps)
    show_plot(ax)

def animate_simulation_with_moving_circle(hg: Hypergraph, ps: PlotSettings, 
                                          inputs: Dict[str, float], output: Node, **kwargs) -> None:
    """Animates a simulation with a moving circle along the solution path.
    
    Parameters
    ----------
    hg : Hypergraph
        The hypergraph to simulate.
    ps : PlotSettings
        Settings for the plot.
    inputs : Dict[str, float]
        Inputs to pass to ``hg.solve``, of the form {label : value}.
    output : Node | str
        The node (or node label) to solve for.
    **kwargs
        Other arguments to pass to the solver.
    """
    fig, ax = plt.subplots(**ps.settings)
    text_box = create_textbox(ax, ps)
    tnodes = sim_hg(hg, inputs, output, **kwargs)
    circles, lines = initialize_hg(hg, ax, inputs, output, ps)
    ani = animate_hg_with_moving_circle(fig, ax, tnodes, list(inputs.keys()), get_node_label(output), 
                                       circles, lines, text_box, ps, hg)
    show_plot(ax)

def create_textbox(ax: Axes, ps: PlotSettings) -> Text:
    """Creates a text box below the plot.
    
    Args:
        ax: Matplotlib axes object
        ps: Plot settings object
        
    Returns:
        Text object for status display
    """
    props = ps.text.copy()
    props.update(s='', transform=ax.transAxes)
    text_box = ax.text(**props)
    return text_box

def show_plot(ax: Axes) -> None:
    """Configures and displays the plot.
    
    Args:
        ax: Matplotlib axes object to configure and display
    """
    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.show()  

def sim_hg(hg: Hypergraph, inputs: Dict[str, float], target: Node, **kwargs) -> List[TNode]:
    """Simulates the hypergraph and returns every found TNode.
    
    Args:
        hg: Hypergraph to simulate
        inputs: Dictionary of input values
        target: Target node to solve for
        **kwargs: Additional arguments for the solver
        
    Returns:
        List of TNode objects representing the solution path
        
    Raises:
        Exception: If no solutions are found
    """
    hg.memory_mode = True
    end = hg.solve(target, inputs, **kwargs)
    if end is None:
        raise Exception("No solutions found")
    
    tnodes = trim_unneeded_tnodes(hg, target)
    
    return tnodes

def trim_unneeded_tnodes(hg: Hypergraph, target: Node) -> List[TNode]:
    """Removes all TNodes after the one representing the successful search.
    
    Args:
        hg: Hypergraph object
        target: Target node
        
    Returns:
        Trimmed list of TNode objects
    """
    tnodes = hg.solved_tnodes
    target_label = hg.get_node(target).label

    target_index = next((i for i, node in enumerate(tnodes) 
                         if node.node_label == target_label), -1)
    
    tnodes = tnodes[:target_index+1]
    return tnodes

def initialize_hg(hg: Hypergraph, ax: Axes, inputs: Dict[str, float], 
                  output: Node, ps: PlotSettings) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Initializes the Hypergraph visualization.
    
    Args:
        hg: Hypergraph to visualize
        ax: Matplotlib axes object
        inputs: Dictionary of input values
        output: Output node
        ps: Plot settings
        
    Returns:
        Tuple of (circles, lines) dictionaries
    """
    circles = plot_nodes(hg, ax, inputs, output, ps)
    lines = plot_edges(hg, ax, circles, ps)
    return circles, lines

def plot_nodes(hg: Hypergraph, ax: Axes, inputs: Dict[str, float], 
               output: Node, ps: PlotSettings) -> Dict[str, Any]:
    """Adds the nodes to the axes as patches.
    
    Args:
        hg: Hypergraph to plot
        ax: Matplotlib axes object
        inputs: Dictionary of input values
        output: Output node
        ps: Plot settings
        
    Returns:
        Dictionary mapping node labels to circle patches
    """
    index = 0
    plotted_nodes = {}

    # Plot input nodes first
    for input_label in inputs:
        center, index = get_next_center(ps, index)
        plotted_nodes[input_label] = ax.add_patch(plot_circle('node_input', ps, center))
        # Display node name and value on the circular node representation
        ax.text(center[0], center[1], input_label, 
                ha='center', va='center', 
                fontsize=8, fontweight='bold',
                color='white', zorder=ps.node_default['zorder'] + 1)
    
    # Plot remaining nodes
    unplotted = [label for label in hg.nodes if label not in inputs]
    output_label = get_node_label(output)
    if output_label in unplotted:
        unplotted.remove(output_label)
    curr_node = list(inputs.keys())[-1] if inputs else None

    while len(unplotted) > 0:
        center, index = get_next_center(ps, index)
        curr_node = pop_close_node(curr_node, hg, unplotted)
        circle = plot_circle('node_default', ps, center)
        plotted_nodes[curr_node] = ax.add_patch(circle)
        # Display node name and value on the circular node representation
        ax.text(center[0], center[1], curr_node, 
                ha='center', va='center', 
                fontsize=8, fontweight='bold',
                color='white', zorder=ps.node_default['zorder'] + 1)

    # Plot output node
    if plotted_nodes:
        x_centers, y_centers = zip(*[circle.center for circle in plotted_nodes.values()])
        output_center = (max(x_centers) + ps.spacing['x_spacing'], 
                         sum(y_centers) / len(y_centers))
        output_circle = plot_circle('node_output', ps, output_center)
        plotted_nodes[get_node_label(output)] = ax.add_patch(output_circle)
        # Display target node name and calculated result
        ax.text(output_center[0], output_center[1], get_node_label(output), 
                ha='center', va='center', 
                fontsize=8, fontweight='bold',
                color='white', zorder=ps.node_default['zorder'] + 1)

    return plotted_nodes

def get_next_center(ps: PlotSettings, index: int = 0) -> Tuple[Tuple[float, float], int]:
    """Returns the next center position for node placement.
    
    Args:
        ps: Plot settings object
        index: Current index for position calculation
        
    Returns:
        Tuple of (center_position, next_index)
    """
    x_space, y_space = ps.spacing['x_spacing'], ps.spacing['y_spacing']
    height = ps.spacing['num_rows']

    jiggle = (x_space + y_space) / 2 * 0.1 
    y = (index % height) * y_space + (jiggle * random.choice((1, -1)))
    x = (index // height) * x_space + (jiggle * random.choice((1, -1)))

    index += 1

    return (x, y), index
    
def pop_close_node(node_label: Optional[str], hg: Hypergraph, unplotted: List[str]) -> str:
    """Returns a close-ish node to the node with the given label and 
    removes the node from unplotted.
    
    Args:
        node_label: Label of the current node (can be None)
        hg: Hypergraph object
        unplotted: List of unplotted node labels
        
    Returns:
        Label of the selected close node
    """
    if node_label is None:
        # If no current node, just pick a random one
        out_node = hg.get_node(random.choice(unplotted))
        out = out_node.label
        unplotted.remove(out)
        return out
    
    node = hg.get_node(node_label)
    target_node = get_a_target_of_node(node, hg, unplotted)
    if target_node is not None:
        out_node = target_node
    else:
        source_node = get_a_source_of_node(node, hg, unplotted)
        if source_node is not None:
            out_node = source_node
        else:
            out_node = hg.get_node(random.choice(unplotted))
    out = out_node.label
    unplotted.remove(out)
    return out

def get_a_target_of_node(node: Node, hg: Hypergraph, unplotted: List[str]) -> Optional[Node]:
    """Returns the first encountered node for which the node with the 
    given node_label is a source node.
    
    Args:
        node: Source node
        hg: Hypergraph object
        unplotted: List of unplotted node labels
        
    Returns:
        Target node or None if not found
    """
    for edge in node.leading_edges:
        target = edge.target
        if target.label in unplotted:
            return target
    return None

def get_a_source_of_node(node: Node, hg: Hypergraph, unplotted: List[str]) -> Optional[Node]:
    """Returns the first encountered node for which the node with the 
    given node_label is a target node.
    
    Args:
        node: Target node
        hg: Hypergraph object
        unplotted: List of unplotted node labels
        
    Returns:
        Source node or None if not found
    """
    for edge in node.generating_edges:
        for sn in edge.source_nodes.values():
            if getattr(sn, 'label', '') in unplotted:
                return sn
    return None

def plot_edges(hg: Hypergraph, ax: Axes, plotted_nodes: Dict[str, Any], 
               ps: PlotSettings) -> Dict[str, Any]:
    """Plots the edges of the hypergraph.
    
    Args:
        hg: Hypergraph to plot
        ax: Matplotlib axes object
        plotted_nodes: Dictionary of plotted node circles
        ps: Plot settings
        
    Returns:
        Dictionary of plotted lines {label : Line2D}
    """
    lines = {}
    for edge in hg.edges.values():
        line = plot_edge(edge, ax, plotted_nodes, ps)
        lines[edge.label] = line
    return lines

def plot_edge(edge: Edge, ax: Axes, plotted_nodes: Dict[str, Any], ps: PlotSettings) -> Any:
    """Adds the edge to the axis.
    
    Args:
        edge: Edge object to plot
        ax: Matplotlib axes object
        plotted_nodes: Dictionary of plotted node circles
        ps: Plot settings
        
    Returns:
        Line2D object representing the edge
    """
    sn_circles = [plotted_nodes[sn.label] for sn in edge.source_nodes.values()]
    target_circle = plotted_nodes[edge.target.label]
    circles = sn_circles + [target_circle]

    t_center = target_circle.center
    x_data, y_data = [t_center[0]], [t_center[1]]
    for circle in circles:
        x_data.extend([circle.center[0], t_center[0]])
        y_data.extend([circle.center[1], t_center[1]])
    
    lines = ax.plot(x_data, y_data, **ps.edge_default)
    return lines[0]


###############################################################################
###   ANIMATION   #############################################################
###############################################################################


def animate_hg(fig, ax: Axes, tnodes: list, inputs: list, output: str, 
               circles: dict, lines: dict, text_box: Text, ps: PlotSettings):
    """Animates a simulation of the hypergraph."""
    interval = ps.spacing.get('interval', 500 if len(tnodes) < 50 else 100)
    solved_nodes = []

    ani = animation.FuncAnimation(
        fig, partial(color_active_tnode,
                     tnodes=tnodes,
                     inputs=inputs,
                     output=output,
                     circles=circles, 
                     lines=lines,
                     text=text_box,
                     solved_nodes=solved_nodes,
                     ps=ps),
        frames=len(tnodes)+1, interval=interval, blit=False, repeat=False)
    return ani

def animate_hg_with_moving_circle(fig, ax: Axes, tnodes: list, inputs: list, output: str, 
                                 circles: dict, lines: dict, text_box: Text, ps: PlotSettings, hg: Hypergraph):
    """Animates a simulation with a moving circle along the solution path."""
    # Calculate total frames based on animation settings
    frames_per_edge = ps.animation['frames_per_edge']
    total_frames = len(tnodes) * frames_per_edge
    
    # Initialize animated circle for solution path visualization
    moving_circle = plt.Circle((0, 0), ps.animation['circle_radius'], 
                              color=ps.animation['circle_color'], 
                              alpha=0.0,  # Start invisible
                              zorder=100)
    ax.add_patch(moving_circle)
    
    # Initialize a reusable dashed line to indicate BFS jumps without hyperedges
    jump_line = Line2D([], [], linestyle='--',
                       color=ps.animation.get('jump_color', '#888888'),
                       linewidth=ps.animation.get('jump_linewidth', 1.5),
                       alpha=0.0,  # hidden until used
                       zorder=95)
    ax.add_line(jump_line)
    
    # Calculate interval based on fps
    interval = 1000 / ps.animation['fps']  # Convert fps to milliseconds
    
    ani = animation.FuncAnimation(
        fig, partial(animate_with_moving_circle_smooth,
                     tnodes=tnodes,
                     inputs=inputs,
                     output=output,
                     circles=circles, 
                     lines=lines,
                     text=text_box,
                     moving_circle=moving_circle,
                     jump_line=jump_line,
                     frames_per_edge=frames_per_edge,
                     ps=ps,
                     hg=hg),
        frames=total_frames, interval=interval, blit=False, repeat=False)
    return ani

def animate_with_moving_circle_smooth(frame: int, tnodes: list, inputs: list, output: str, 
                                     circles: dict, lines: dict, text: Text,
                                     moving_circle: plt.Circle, jump_line: Line2D, frames_per_edge: int, 
                                     ps: PlotSettings, hg: Hypergraph):
    """Animates the moving circle smoothly along the solution path."""
    mod_patches = []
    
    # Calculate which tnode and sub-frame we're on
    tnode_index = frame // frames_per_edge
    sub_frame = frame % frames_per_edge
    
    if tnode_index >= len(tnodes):
        # Animation sequence finished - remove visual circle
        moving_circle.set_alpha(0.0)
        # hide jump line as well
        jump_line.set_alpha(0.0)
        text.set_text('Animation complete!')
        mod_patches.append(text)
        mod_patches.append(moving_circle)  # Add the hidden circle to update it
        mod_patches.append(jump_line)
        return mod_patches
    
    t = tnodes[tnode_index]
    
    # Update status text to reflect current animation phase
    if sub_frame == 0:
        # Display current node location in the solution path
        if t.gen_edge_label is not None:
            edge_name = get_line_label(t)
            text.set_text(f'At node: {t.node_label} (via {edge_name})')
        else:
            text.set_text(f'Starting at: {t.node_label}')
    else:
        # Display edge traversal progress during animation
        if t.gen_edge_label is not None:
            edge_name = get_line_label(t)
            text.set_text(f'Traveling on: {edge_name}')
        else:
            text.set_text(f'At: {t.node_label}')
    mod_patches.append(text)
    
    # Highlight the currently active node in the solution sequence
    mod_patches.append(color_patch('node_target', ps, circles[t.node_label]))
    
    # Move circle from centroid of sources to target for hyperedge steps,
    # or from previous node to current node for BFS jumps (no hyperedge)
    if sub_frame > 0:
        # Make circle visible when moving
        moving_circle.set_alpha(ps.animation['circle_alpha'])

        target_circle = circles[t.node_label]
        target_x, target_y = target_circle.center

        # Determine start position strategy
        prev_label = tnodes[tnode_index - 1].node_label if tnode_index > 0 else None
        prev_is_source_of_current = any(
            ch.node_label == prev_label for ch in getattr(t, 'children', [])
        ) if prev_label is not None else False

        # Jump if previous node is not one of the current node's sources
        is_jump = (tnode_index > 0) and (not prev_is_source_of_current)

        if is_jump:
            # Start at previous node center for BFS jump
            prev_circle = circles[prev_label]
            start_x, start_y = prev_circle.center
        else:
            # Hyperedge traversal: centroid of sources if available
            if hasattr(t, 'children') and t.children:
                src_points = [circles[ch.node_label].center for ch in t.children if ch.node_label in circles]
            else:
                src_points = []

            if len(src_points) > 0:
                start_x = float(np.mean([p[0] for p in src_points]))
                start_y = float(np.mean([p[1] for p in src_points]))
            else:
                start_x, start_y = target_x, target_y

        # Ease-in-out for smoother motion
        linear_p = sub_frame / frames_per_edge
        ease_p = 0.5 - 0.5 * np.cos(np.pi * linear_p)

        x = start_x + ease_p * (target_x - start_x)
        y = start_y + ease_p * (target_y - start_y)

        moving_circle.center = (x, y)
        mod_patches.append(moving_circle)

        # Update jump line visibility and geometry
        if is_jump and tnode_index > 0:
            jump_line.set_data([start_x, target_x], [start_y, target_y])
            jump_line.set_alpha(ps.animation.get('jump_alpha', 0.6))
            mod_patches.append(jump_line)
        else:
            jump_line.set_alpha(0.0)
            mod_patches.append(jump_line)
    else:
        # Hide circle when not moving (sub_frame == 0)
        moving_circle.set_alpha(0.0)
        mod_patches.append(moving_circle)  # Add the hidden circle to update it
        # Also hide jump line at keyframes
        jump_line.set_alpha(0.0)
        mod_patches.append(jump_line)
    
    # Color solved nodes and edges
    for i in range(tnode_index):
        prev_t = tnodes[i]
        # Mark previously calculated nodes as solved
        mod_patches.append(color_patch('node_solved', ps, circles[prev_t.node_label]))
        
        # Try to color any edge that connects to this node (simplified)
        if prev_t.gen_edge_label is not None:
            for edge_label, line in lines.items():
                if prev_t.node_label in edge_label:
                    mod_patches.append(color_patch('edge_solved', ps, line))
                    break
    
    return mod_patches


def color_active_tnode(frame: int, tnodes: list, inputs: list, output: str, 
                       circles: dict, lines: dict, text: Text,
                       solved_nodes: list, ps: PlotSettings)-> list:
    """Colors the path to the TNode in the plot."""
    mod_patches = []
    try:
        # solved_nodes = [t.node_label for t in tnodes[:max(0,frame-1)]]
        mod_patches.extend(restore_plot(circles, lines, inputs, tnodes[max(0,frame-1)], ps))
        t = tnodes[frame]
    except IndexError:
        mod_patches.extend(color_path(tnodes[-1], circles, lines, ps))
        mod_patches.append(color_patch('node_output_solved', ps, circles[output]))
        text.set_text(f'Output solved.')
        mod_patches.append(text)
        return mod_patches

    mod_patches.append(text)
    mod_patches.append(color_patch('node_target', ps, circles[t.node_label]))

    if t.gen_edge_label is not None:
        mod_patches.extend(color_path(t, circles, lines, ps))
        text.set_text(f'Current Node: {t.node_label}, Current Edge: {get_line_label(t)}') 
    else:
        text.set_text(f'Input Node: {t.node_label}')

    return mod_patches

def restore_plot(circles: dict, lines: dict, inputs: list, 
                 prev_t: TNode, ps: PlotSettings):
    """Clears active lines and nodes from plot."""
    mod_patches = []
    for line in lines.values():
        mod_patches.append(color_patch('edge_default', ps, line))

    mod_patches.append(color_patch('node_solved', ps, circles[prev_t.node_label]))
    mod_patches.extend(color_node_children(prev_t, circles, lines, ps, color_edge=False))
    
    return mod_patches
    
def get_node_label(node)-> str:
    """Gets the label of the node for various types."""
    if isinstance(node, Node):
        return node.label
    if isinstance(node, TNode):
        return node.node_label
    return str(node)

def get_line_label(tnode: TNode)-> str:
    """Returns the label of the edge as stored in the edge."""
    label = tnode.gen_edge_label.split('#')[0]
    return label

def get_edge_label_from_tnode(t: TNode, lines: dict) -> str:
    """Get edge label from tnode for line matching."""
    if t.gen_edge_label is None:
        return None
    
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

def color_path(t:TNode, circles: dict, lines: dict, ps: PlotSettings)-> list:
    mod_patches = [color_patch('edge_active', ps, lines[get_line_label(t)])]

    for child in t.children:
        mod_patches.append(color_patch('node_source', ps, circles[child.node_label]))
        mod_patches.extend(color_node_children(child, circles, lines, ps))

    return mod_patches

def color_node_children(tnode: TNode, circles, lines, ps, seen: list=None, 
                        color_edge: bool=True)-> list:
    """Recursive caller coloring the children of `tnode` as found nodes."""
    if seen is None: 
        seen = []
    elif tnode.node_label in seen:
        return []
    seen.append(tnode.node_label)
    out = []

    if tnode.gen_edge_label is not None:
        if color_edge:
            out.append(color_patch('edge_solved', ps, lines[get_line_label(tnode)]))

        for child in tnode.children:
            out.append(color_patch('node_solved', ps, circles[child.node_label]))
            out.extend(color_node_children(child, circles, lines, ps, seen, color_edge))
    
    return out

def plot_circle(settings_label: str, ps: PlotSettings, center)-> plt.Circle:
    """Creates and returns the Circle according to the values in PlotSettings."""
    props = getattr(ps, settings_label)
    props = ps.node_default | props
    circle = plt.Circle(center, **props)
    return circle

def color_patch(settings_label: str, ps: PlotSettings, patch)-> Patch:
    """Colors and returns the patch according to the values in PlotSettings."""
    props = getattr(ps, settings_label)
    patch.set(**props)
    return patch