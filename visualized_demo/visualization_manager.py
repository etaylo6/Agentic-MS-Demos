"""
Visualization Management Module

This module handles all aspects of hypergraph visualization, including node and edge
plotting, layout management, color coordination, and real-time updates. It provides
a clean interface between the hypergraph data and the matplotlib visualization,
ensuring consistent styling and smooth user interactions.

Key responsibilities:
- Hypergraph layout calculation and management
- Node and edge visualization with proper styling
- Real-time color updates based on node states
- Text label management and formatting
- Layout position storage and consistency
- Integration with matplotlib for smooth rendering

The module maintains a consistent visual representation of the hypergraph while
allowing for dynamic updates during simulations and animations. It handles the
complexity of positioning nodes, drawing edges, and managing visual states.

Usage:
    from visualization_manager import VisualizationManager
    viz = VisualizationManager(fig, ax, canvas)
    viz.setup_initial_view(hypergraph, node_vars, toggle_switches)
    viz.update_node_colors()
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from gui_constants import (
    CARD_BG_COLOR, SUCCESS_COLOR, WARNING_COLOR, NODE_TEXT_FONT,
    NODE_TEXT_COLOR, NODE_TEXT_ZORDER
)


class VisualizationManager:
    """
    Manages hypergraph visualization and real-time updates.
    
    This class handles all aspects of visualizing the hypergraph, including
    layout calculation, node and edge rendering, color management, and text
    label handling. It provides a clean interface for updating the visualization
    based on user interactions and simulation states.
    
    Features:
    - Automatic layout calculation using NetworkX
    - Node and edge rendering with consistent styling
    - Real-time color updates based on node states
    - Text label management with value formatting
    - Layout position storage for consistency
    - Integration with matplotlib for smooth rendering
    
    Attributes:
        fig: Matplotlib figure object
        ax: Matplotlib axes object
        canvas: Matplotlib canvas for rendering
        layout_positions: Dictionary storing node positions
        circles: Dictionary of node circle patches
        lines: Dictionary of edge line objects
        hypergraph: Reference to the hypergraph model
    """
    
    def __init__(self, fig: plt.Figure, ax: plt.Axes, canvas: Any):
        """
        Initialize the visualization manager.
        
        Args:
            fig: Matplotlib figure object
            ax: Matplotlib axes object
            canvas: Matplotlib canvas for rendering
        """
        self.fig = fig
        self.ax = ax
        self.canvas = canvas
        
        # Visualization state
        self.layout_positions: Optional[Dict[str, Tuple[float, float]]] = None
        self.circles: Dict[str, plt.Circle] = {}
        self.lines: Dict[str, plt.Line2D] = {}
        self.hypergraph: Optional[Any] = None
        self.alias_map: Dict[str, str] = {}
        self.inverse_alias_map: Dict[str, str] = {}
        self.function_group_labels: Dict[str, str] = {}
        # Representation mode
        self.use_bipartite: bool = False
        
        # Configure plot appearance
        self._configure_plot()
    
    def _configure_plot(self) -> None:
        """
        Configure the matplotlib plot appearance.
        """
        # Ensure no borders
        self.fig.patch.set_facecolor(CARD_BG_COLOR)
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.ax.set_facecolor(CARD_BG_COLOR)
        
        # Configure main plot
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)
        
        # Add subtle grid
        self.ax.grid(True, alpha=0.1, linestyle='-', linewidth=0.5)
    
    def set_alias_map(self, alias_map: Dict[str, str]) -> None:
        """
        Provide a mapping from canonical node labels to display aliases.
        """
        self.alias_map = dict(alias_map) if alias_map else {}
        self.inverse_alias_map = {}
        if self.alias_map:
            for canonical, alias in self.alias_map.items():
                self.inverse_alias_map.setdefault(alias, canonical)
                self.inverse_alias_map.setdefault(canonical, canonical)
    
    def set_function_group_labels(self, group_labels: Dict[str, str]) -> None:
        """
        Provide a mapping of edge labels to function group display names.
        """
        self.function_group_labels = dict(group_labels) if group_labels else {}
    
    def setup_initial_view(self, hypergraph: Any, node_vars: Dict[str, Any], 
                          toggle_switches: Dict[str, Any]) -> None:
        """
        Set up the initial static view of the hypergraph.
        
        Args:
            hypergraph: The hypergraph model
            node_vars: Dictionary of node StringVar objects
            toggle_switches: Dictionary of toggle switch widgets
        """
        self.hypergraph = hypergraph
        
        # Reset visualization area to prepare for new hypergraph display
        self.ax.clear()
        self._configure_plot()
        
        if self.use_bipartite:
            from plothg_bipartite import plot_bipartite_hypergraph, BipartitePlotSettings
            ps = BipartitePlotSettings()
            if self.alias_map:
                ps.alias_map = dict(self.alias_map)
            if self.function_group_labels:
                ps.merge['semantic_groups'] = dict(self.function_group_labels)
            ps.merge['read_edge_attr'] = True
            shapes, lines, positions = plot_bipartite_hypergraph(self.hypergraph, self.ax, ps)
            self.circles = shapes
            self.lines = lines
            self.layout_positions = positions
        else:
            # Calculate layout
            self.layout_positions = self._calculate_layout()
            
            # Create visualization elements
            self._create_nodes(node_vars, toggle_switches)
            self._create_edges()
        
        # Update node colors based on current toggle states
        self.update_node_colors(toggle_switches)
        
        self.canvas.draw()
    
    def setup_animation_view(self, hypergraph: Any, node_vars: Dict[str, Any], 
                           toggle_switches: Dict[str, Any]) -> None:
        """
        Set up the initial view for animation (all nodes gray).
        
        Args:
            hypergraph: The hypergraph model
            node_vars: Dictionary of node StringVar objects
            toggle_switches: Dictionary of toggle switch widgets
        """
        self.hypergraph = hypergraph
        
        # Reset visualization area to prepare for new hypergraph display
        self.ax.clear()
        self._configure_plot()
        
        if self.use_bipartite:
            from plothg_bipartite import plot_bipartite_hypergraph, BipartitePlotSettings
            ps = BipartitePlotSettings()
            if self.alias_map:
                ps.alias_map = dict(self.alias_map)
            if self.function_group_labels:
                ps.merge['semantic_groups'] = dict(self.function_group_labels)
            ps.merge['read_edge_attr'] = True
            shapes, lines, positions = plot_bipartite_hypergraph(self.hypergraph, self.ax, ps)
            self.circles = shapes
            self.lines = lines
            self.layout_positions = positions
            # Set nodes gray except function squares (rectangles), which we keep colored
            for shape in self.circles.values():
                if isinstance(shape, Rectangle):
                    continue
                if hasattr(shape, 'set_facecolor'):
                    shape.set_facecolor('#ababab')
        else:
            # Calculate layout
            self.layout_positions = self._calculate_layout()
            
            # Create visualization elements
            self._create_nodes(node_vars, toggle_switches)
            self._create_edges()
            
            # Initialize all nodes with neutral gray color for animation sequence
            self._set_all_nodes_gray()
        
        self.canvas.draw()
    
    def _calculate_layout(self) -> Dict[str, Tuple[float, float]]:
        """
        Calculate node positions using NetworkX spring layout.
        
        Returns:
            Dict[str, Tuple[float, float]]: Dictionary mapping node labels to positions
        """
        # Build NetworkX graph structure to calculate optimal node positioning
        G = nx.Graph()
        
        # Add nodes
        node_labels = []
        for node in self.hypergraph.nodes:
            if hasattr(node, 'label'):
                node_labels.append(node.label)
            else:
                node_labels.append(str(node))
        
        G.add_nodes_from(node_labels)
        
        # Add edges
        for edge in self.hypergraph.edges.values():
            source_labels = []
            for sn in edge.source_nodes.values():
                if hasattr(sn, 'label'):
                    source_labels.append(sn.label)
                else:
                    source_labels.append(str(sn))
            target_label = edge.target.label if hasattr(edge.target, 'label') else str(edge.target)
            for source_label in source_labels:
                G.add_edge(source_label, target_label)
        
        # Apply spring force algorithm to distribute nodes evenly across visualization area
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        return pos
    
    def _create_nodes(self, node_vars: Dict[str, Any], toggle_switches: Dict[str, Any]) -> None:
        """
        Create node circles and text labels.
        
        Args:
            node_vars: Dictionary of node StringVar objects
            toggle_switches: Dictionary of toggle switch widgets
        """
        from plothg import PlotSettings
        ps = PlotSettings()
        
        # Generate circular visual representations for all hypergraph nodes
        for node_label, (x, y) in self.layout_positions.items():
            circle = plt.Circle((x, y), ps.node_default['radius'], 
                              facecolor=ps.node_default['facecolor'],
                              edgecolor=ps.node_default['edgecolor'],
                              linewidth=ps.node_default['linewidth'],
                              zorder=ps.node_default['zorder'])
            self.circles[node_label] = self.ax.add_patch(circle)
            
            # Display node name and current value based on user input state
            self._create_node_text(node_label, x, y, node_vars, toggle_switches)
    
    def _create_node_text(self, node_label: str, x: float, y: float, 
                         node_vars: Dict[str, Any], toggle_switches: Dict[str, Any]) -> None:
        """
        Create text label for a node.
        
        Args:
            node_label: Label of the node
            x, y: Position coordinates
            node_vars: Dictionary of node StringVar objects
            toggle_switches: Dictionary of toggle switch widgets
        """
        display_label = self.alias_map.get(node_label, node_label)
        toggle = toggle_switches.get(display_label)
        var = node_vars.get(display_label)
        if toggle is not None and var is not None:
            toggle_state = toggle.get()
            value = var.get()
            
            if toggle_state == "source":
                if value:
                    formatted_value = self._format_value_3sf(value)
                    label_text = f"{display_label}\n{formatted_value}"
                else:
                    label_text = f"{display_label}\n?"
            elif toggle_state == "target":
                label_text = f"{display_label}\n?"
            else:
                label_text = f"{display_label}\n?"
        else:
            label_text = display_label
        
        text = self.ax.text(x, y, label_text, 
                          ha='center', va='center', 
                          fontsize=NODE_TEXT_FONT[1], fontweight=NODE_TEXT_FONT[2],
                          color=NODE_TEXT_COLOR, zorder=NODE_TEXT_ZORDER)
        text._node_label = node_label  # Attach canonical node identifier for text updates during animation
    
    def _create_edges(self) -> None:
        """
        Create edge lines between nodes.
        """
        from plothg import PlotSettings
        ps = PlotSettings()
        
        # Draw connection lines between related hypergraph nodes
        for edge in self.hypergraph.edges.values():
            source_labels = []
            for sn in edge.source_nodes.values():
                if hasattr(sn, 'label'):
                    source_labels.append(sn.label)
                else:
                    source_labels.append(str(sn))
            target_label = edge.target.label if hasattr(edge.target, 'label') else str(edge.target)
            
            for source_label in source_labels:
                if source_label in self.layout_positions and target_label in self.layout_positions:
                    x_data = [self.layout_positions[source_label][0], self.layout_positions[target_label][0]]
                    y_data = [self.layout_positions[source_label][1], self.layout_positions[target_label][1]]
                    
                    line, = self.ax.plot(x_data, y_data, 
                                       color=ps.edge_default['color'],
                                       linewidth=ps.edge_default['linewidth'],
                                       zorder=ps.edge_default['zorder'])
                    self.lines[f"{source_label}->{target_label}"] = line
    
    def update_node_colors(self, toggle_switches: Dict[str, Any]) -> None:
        """
        Update node colors based on toggle states.
        
        Args:
            toggle_switches: Dictionary of toggle switch widgets
        """
        for alias_label, toggle in toggle_switches.items():
            canonical_label = self.inverse_alias_map.get(alias_label, alias_label)
            if canonical_label in self.circles:
                circle = self.circles[canonical_label]
                toggle_state = toggle.get()
                
                if toggle_state == "source":
                    color = SUCCESS_COLOR
                elif toggle_state == "target":
                    color = WARNING_COLOR
                else:
                    color = "#ababab"
                
                circle.set_facecolor(color)
    
    def update_node_text(self, node_label: str, node_vars: Dict[str, Any], 
                        toggle_switches: Dict[str, Any]) -> None:
        """
        Update the text display for a specific node.
        
        Args:
            node_label: The label of the node to update
            node_vars: Dictionary of node StringVar objects
            toggle_switches: Dictionary of toggle switch widgets
        """
        canonical_label = self.inverse_alias_map.get(node_label, node_label)
        if canonical_label not in self.layout_positions:
            return
            
        x, y = self.layout_positions[canonical_label]
        display_label = self.alias_map.get(canonical_label, canonical_label)
        
        # Find and update existing text for this node
        text_found = False
        for text_obj in self.ax.texts[:]:
            if (hasattr(text_obj, '_node_label') and text_obj._node_label == canonical_label) or \
               (abs(text_obj.get_position()[0] - x) < 0.01 and abs(text_obj.get_position()[1] - y) < 0.01):
                
                # Generate new text based on toggle status
                if display_label in node_vars and display_label in toggle_switches:
                    toggle_state = toggle_switches[display_label].get()
                    value = node_vars[display_label].get()
                    
                    if toggle_state == "source":
                        if value:
                            formatted_value = self._format_value_3sf(value)
                            label_text = f"{display_label}\n{formatted_value}"
                        else:
                            label_text = f"{display_label}\n?"
                    elif toggle_state == "target":
                        label_text = f"{display_label}\n?"
                    else:  # "none"
                        label_text = f"{display_label}\n?"
                else:
                    label_text = display_label
                
                # Update existing text object
                text_obj.set_text(label_text)
                text_found = True
                break
        
        # If no existing text found, create new one
        if not text_found:
            self._create_node_text(canonical_label, x, y, node_vars, toggle_switches)
    
    def _format_value_3sf(self, value: Any) -> str:
        """
        Format a numeric value to 3 significant figures.
        
        Args:
            value: The value to format (can be string or numeric)
            
        Returns:
            str: The formatted value as a string
        """
        try:
            num_value = float(value)
            if num_value == 0:
                return "0"
            elif abs(num_value) >= 1000:
                return f"{num_value:.2e}"
            elif abs(num_value) >= 1:
                return f"{num_value:.3g}"
            else:
                return f"{num_value:.2e}"
        except (ValueError, TypeError):
            return str(value)
    
    def recreate_graph(self, node_vars: Dict[str, Any], toggle_switches: Dict[str, Any]) -> None:
        """
        Recreate the graph with the same layout.
        
        Args:
            node_vars: Dictionary of node StringVar objects
            toggle_switches: Dictionary of toggle switch widgets
        """
        # Clear existing elements
        self.ax.clear()
        self._configure_plot()
        self.circles.clear()
        self.lines.clear()
        
        # Recreate visualization elements
        if self.layout_positions:
            self._create_nodes(node_vars, toggle_switches)
            self._create_edges()
            self.update_node_colors(toggle_switches)
    
    def get_layout_positions(self) -> Optional[Dict[str, Tuple[float, float]]]:
        """
        Get the current layout positions.
        
        Returns:
            Optional[Dict[str, Tuple[float, float]]]: Dictionary of node positions
        """
        return self.layout_positions
    
    def get_circles(self) -> Dict[str, plt.Circle]:
        """
        Get dictionary of node circles.
        
        Returns:
            Dict[str, plt.Circle]: Dictionary mapping node labels to circle patches
        """
        return self.circles
    
    def get_lines(self) -> Dict[str, plt.Line2D]:
        """
        Get dictionary of edge lines.
        
        Returns:
            Dict[str, plt.Line2D]: Dictionary mapping edge labels to line objects
        """
        return self.lines
    
    def update_canvas(self) -> None:
        """
        Update the canvas to reflect current changes.
        """
        self.canvas.draw()
    
    def clear_visualization(self) -> None:
        """
        Clear all visualization elements.
        """
        self.ax.clear()
        self._configure_plot()
        self.circles.clear()
        self.lines.clear()
        self.layout_positions = None
    
    def _set_all_nodes_gray(self) -> None:
        """
        Set all nodes to gray color for animation start.
        """
        for circle in self.circles.values():
            circle.set_facecolor('#ababab')  # Gray color
