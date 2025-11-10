"""
Animation Engine Module

This module handles all animation-related functionality for the Timoshenko Beam
GUI application. It manages moving circle animations, frame-by-frame animation
logic, animation state handling, and smooth transitions during hypergraph
simulation visualization.

Key responsibilities:
- Moving circle animation along solution paths
- Frame-by-frame animation management
- Animation state and timing control
- Smooth transitions between animation phases
- Animation resource management and cleanup
- Integration with matplotlib animation framework

The module provides a sophisticated animation system that visualizes the
solution process through the hypergraph, showing how values propagate from
source nodes to the target node. It handles complex timing, state management,
and visual effects to create an engaging and informative user experience.

Usage:
    from animation_engine import AnimationEngine
    engine = AnimationEngine(fig, ax, canvas)
    animation = engine.create_animation(tnodes, inputs, target_node, text_box)
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial
from typing import Dict, List, Optional, Tuple, Any
from gui_constants import (
    DEFAULT_FRAMES_PER_EDGE, DEFAULT_FPS, DEFAULT_CIRCLE_RADIUS,
    DEFAULT_CIRCLE_COLOR, DEFAULT_CIRCLE_ALPHA, SUCCESS_COLOR, CLEANUP_DELAY
)


class AnimationEngine:
    """
    Manages animation creation and execution for hypergraph visualization.
    
    This class handles the creation of sophisticated animations that show the
    solution process through the hypergraph. It manages moving circles, frame
    timing, state transitions, and visual effects to create an engaging
    visualization of the constraint solving process.
    
    Features:
    - Moving circle animation along solution paths
    - Multi-phase animation (traversal, graph update, completion)
    - Smooth frame transitions and timing control
    - Animation resource management and cleanup
    - Integration with matplotlib animation framework
    - Customizable animation parameters
    
    Attributes:
        fig: Matplotlib figure object
        ax: Matplotlib axes object
        canvas: Matplotlib canvas for rendering
        moving_circle: Moving circle patch for animation
        animation_running: Flag indicating if animation is active
    """
    
    def __init__(self, fig: plt.Figure, ax: plt.Axes, canvas: Any):
        """
        Initialize the animation engine.
        
        Args:
            fig: Matplotlib figure object
            ax: Matplotlib axes object
            canvas: Matplotlib canvas for rendering
        """
        self.fig = fig
        self.ax = ax
        self.canvas = canvas
        self.moving_circle: Optional[plt.Circle] = None
        self.animation_running = False
    
    def create_custom_animation(self, tnodes: List[Any], inputs: List[str], 
                               target_node: str, text_box: Any, 
                               circles: Dict[str, Any], lines: Dict[str, Any],
                               layout_positions: Dict[str, Tuple[float, float]],
                               ps: Any) -> Any:
        """
        Create custom animation with new behavior:
        1. Start with all nodes gray
        2. Show input nodes turning green with values (no red dot)
        3. Red dot moves during pathfinding
        
        Args:
            tnodes: List of TNode objects from the simulation
            inputs: List of input node labels
            target_node: Label of the target node
            text_box: Text box for animation status
            circles: Dictionary of circle patches
            lines: Dictionary of line objects
            layout_positions: Dictionary of node positions
            ps: Plot settings object
            
        Returns:
            Animation object
        """
        # Initialize animated circle for visualizing solution path traversal
        self.moving_circle = plt.Circle((0, 0), ps.animation['circle_radius'], 
                                      color=ps.animation['circle_color'], 
                                      alpha=0.0,  # Start invisible
                                      zorder=100)
        self.ax.add_patch(self.moving_circle)
        
        # Reset all edges to default state at the beginning of animation
        self._reset_all_edges_to_default(lines, ps)
        
        # Remove source nodes from animation sequence since they're already highlighted
        pathfinding_tnodes = [t for t in tnodes if t.node_label not in inputs]
        
        # Calculate total frames and interval
        frames_per_edge = ps.animation['frames_per_edge']
        input_display_frames = len(inputs) * 20  # 20 frames per input node (2x slower)
        pathfinding_frames = len(pathfinding_tnodes) * frames_per_edge
        final_result_frames = 60  # 60 frames to show final result
        completion_frames = 30  # 30 frames for completion
        
        total_frames = input_display_frames + pathfinding_frames + final_result_frames + completion_frames
        interval = 1000 / ps.animation['fps']
        
        # Create animation
        ani = animation.FuncAnimation(
            self.fig, partial(self._animate_new_behavior,
                             tnodes=tnodes,
                             pathfinding_tnodes=pathfinding_tnodes,
                             inputs=inputs,
                             output=target_node,
                             circles=circles,
                             lines=lines,
                             text=text_box,
                             moving_circle=self.moving_circle,
                             frames_per_edge=frames_per_edge,
                             input_display_frames=input_display_frames,
                             pathfinding_frames=pathfinding_frames,
                             final_result_frames=final_result_frames,
                             layout_positions=layout_positions,
                             ps=ps),
            frames=total_frames, interval=interval, blit=False, repeat=False)
        
        # Schedule cleanup after animation completes
        total_duration = total_frames * interval
        self._schedule_cleanup(total_duration)
        
        return ani
    
    def _animate_new_behavior(self, frame: int, tnodes: List[Any], 
                             pathfinding_tnodes: List[Any], inputs: List[str], output: str, 
                             circles: Dict[str, Any], lines: Dict[str, Any], 
                             text: Any, moving_circle: Any, 
                             frames_per_edge: int, input_display_frames: int,
                             pathfinding_frames: int, final_result_frames: int,
                             layout_positions: Dict[str, Tuple[float, float]],
                             ps: Any) -> List[Any]:
        """
        New animation behavior:
        1. Start with all nodes gray
        2. Show input nodes turning green with values (no red dot)
        3. Red dot moves during pathfinding
        
        Args:
            frame: Current animation frame number
            tnodes: List of TNode objects from simulation
            inputs: List of input node labels
            output: Target node label
            circles: Dictionary of circle patches
            lines: Dictionary of line objects
            text: Text box for status updates
            moving_circle: Moving circle patch
            frames_per_edge: Number of frames per edge traversal
            input_display_frames: Total frames for input display phase
            pathfinding_frames: Total frames for pathfinding phase
            layout_positions: Dictionary of node positions
            ps: Plot settings object
            
        Returns:
            List of modified patches for animation
        """
        mod_patches = []
        
        # Phase 1: Highlight source nodes with their input values
        if frame < input_display_frames:
            self._handle_input_display_phase(frame, inputs, circles, text, mod_patches)
        
        # Phase 2: Animate solution path traversal with moving circle
        elif frame < input_display_frames + pathfinding_frames:
            pathfinding_frame = frame - input_display_frames
            tnode_index = pathfinding_frame // frames_per_edge
            sub_frame = pathfinding_frame % frames_per_edge
            self._handle_pathfinding_phase(pathfinding_frame, tnode_index, sub_frame, pathfinding_tnodes, 
                                         inputs, output, circles, lines, text, moving_circle, 
                                         frames_per_edge, layout_positions, ps, mod_patches)
        
        # Phase 3: Display calculated target node value
        elif frame < input_display_frames + pathfinding_frames + final_result_frames:
            final_result_frame = frame - input_display_frames - pathfinding_frames
            self._handle_final_result_phase(final_result_frame, tnodes, circles, text, moving_circle, mod_patches)
        
        # Phase 4: Complete animation sequence
        else:
            print("Animation complete!")
            text.set_text('Animation complete!')
            mod_patches.append(text)
            # Remove animated circle from visualization
            moving_circle.set_alpha(0.0)
            mod_patches.append(moving_circle)
            return mod_patches
        
        return mod_patches
    
    def _handle_edge_traversal(self, frame: int, tnode_index: int, sub_frame: int, 
                              tnodes: List[Any], inputs: List[str], output: str, 
                              circles: Dict[str, Any], lines: Dict[str, Any], 
                              text: Any, moving_circle: Any, frames_per_edge: int,
                              layout_positions: Dict[str, Tuple[float, float]],
                              ps: Any, mod_patches: List[Any]) -> None:
        """
        Handle the edge traversal phase of animation.
        
        Args:
            frame: Current animation frame
            tnode_index: Index of current TNode
            sub_frame: Sub-frame within current edge traversal
            tnodes: List of TNode objects
            inputs: List of input node labels
            output: Target node label
            circles: Dictionary of circle patches
            lines: Dictionary of line objects
            text: Text box for status
            moving_circle: Moving circle patch
            frames_per_edge: Frames per edge traversal
            layout_positions: Dictionary of node positions
            ps: Plot settings
            mod_patches: List to append modified patches to
        """
        t = tnodes[tnode_index]
        
        # Update status text
        self._update_status_text(t, sub_frame, text, mod_patches)
        
        # Handle moving circle animation
        self._update_moving_circle(t, tnode_index, sub_frame, tnodes, circles, 
                                 moving_circle, frames_per_edge, layout_positions, ps, mod_patches)
        
        # Color solved edges and update calculated nodes
        self._update_solved_elements(tnode_index, sub_frame, tnodes, circles, lines, 
                                   frames_per_edge, ps, mod_patches)
    
    def _handle_graph_update(self, graph_update_index: int, tnodes: List[Any], 
                            circles: Dict[str, Any], mod_patches: List[Any]) -> None:
        """
        Handle the graph update phase where nodes turn green and show values.
        
        Args:
            graph_update_index: Index for graph update phase
            tnodes: List of TNode objects
            circles: Dictionary of circle patches
            mod_patches: List to append modified patches to
        """
        if graph_update_index < len(tnodes):
            t = tnodes[graph_update_index]
            if t.node_label in circles:
                # Change node color to green
                circles[t.node_label].set_facecolor(SUCCESS_COLOR)
                mod_patches.append(circles[t.node_label])
                
                # Update node text with calculated value
                self._update_node_text_with_value(t, mod_patches)
    
    def _update_status_text(self, t: Any, sub_frame: int, text: Any, 
                           mod_patches: List[Any]) -> None:
        """
        Update the status text based on animation state.
        
        Args:
            t: Current TNode object
            sub_frame: Current sub-frame within edge traversal
            text: Text box for status updates
            mod_patches: List to append modified patches to
        """
        if sub_frame == 0:
            # At the beginning of a frame - show we're at a node
            if t.gen_edge_label is not None:
                edge_name = t.gen_edge_label.split("#")[0]
                text.set_text(f'At node: {t.node_label} (via {edge_name})')
            else:
                text.set_text(f'Starting at: {t.node_label}')
        else:
            # During traversal - show we're traveling on an edge
            if t.gen_edge_label is not None:
                edge_name = t.gen_edge_label.split("#")[0]
                text.set_text(f'Traveling on: {edge_name}')
            else:
                text.set_text(f'At: {t.node_label}')
        mod_patches.append(text)
    
    def _update_moving_circle(self, t: Any, tnode_index: int, sub_frame: int, 
                            tnodes: List[Any], circles: Dict[str, Any], 
                            moving_circle: Any, frames_per_edge: int,
                            layout_positions: Dict[str, Tuple[float, float]],
                            ps: Any, mod_patches: List[Any]) -> None:
        """
        Update the moving circle position and visibility.
        
        Args:
            t: Current TNode object
            tnode_index: Index of current TNode
            sub_frame: Current sub-frame within edge traversal
            tnodes: List of TNode objects
            circles: Dictionary of circle patches
            moving_circle: Moving circle patch
            frames_per_edge: Frames per edge traversal
            layout_positions: Dictionary of node positions
            ps: Plot settings
            mod_patches: List to append modified patches to
        """
        if sub_frame > 0:
            # Make circle visible when moving
            moving_circle.set_alpha(ps.animation['circle_alpha'])
            
            # Get current node position
            if t.node_label in layout_positions:
                current_x, current_y = layout_positions[t.node_label]
                
                # If this is not the first tnode, move from previous node to current
                if tnode_index > 0:
                    prev_t = tnodes[tnode_index - 1]
                    if prev_t.node_label in layout_positions:
                        prev_x, prev_y = layout_positions[prev_t.node_label]
                        
                        # Calculate progress along the path (0 to 1)
                        progress = sub_frame / frames_per_edge
                        
                        # Interpolate between previous and current node
                        x = prev_x + progress * (current_x - prev_x)
                        y = prev_y + progress * (current_y - prev_y)
                        
                        moving_circle.center = (x, y)
                        mod_patches.append(moving_circle)
                else:
                    # First node - just position at current node
                    moving_circle.center = (current_x, current_y)
                    mod_patches.append(moving_circle)
        else:
            # Hide circle when not moving (sub_frame == 0)
            moving_circle.set_alpha(0.0)
            mod_patches.append(moving_circle)
    
    def _update_solved_elements(self, tnode_index: int, sub_frame: int, 
                              tnodes: List[Any], circles: Dict[str, Any], 
                              lines: Dict[str, Any], frames_per_edge: int, 
                              ps: Any, mod_patches: List[Any]) -> None:
        """
        Update solved edges and calculated nodes.
        
        Args:
            tnode_index: Index of current TNode
            sub_frame: Current sub-frame within edge traversal
            tnodes: List of TNode objects
            circles: Dictionary of circle patches
            lines: Dictionary of line objects
            frames_per_edge: Frames per edge traversal
            ps: Plot settings
            mod_patches: List to append modified patches to
        """
        for i in range(tnode_index):
            prev_t = tnodes[i]
            
            # Color solved edges
            if prev_t.gen_edge_label is not None:
                prev_edge_label = self._get_edge_label_from_tnode(prev_t, lines)
                if prev_edge_label in lines:
                    mod_patches.append(self._color_patch('edge_solved', ps, lines[prev_edge_label]))
            
            # Update calculated nodes
            should_update_node = (i < tnode_index) or (i == tnode_index and sub_frame == frames_per_edge - 1)
            
            if should_update_node and prev_t.node_label in circles:
                # Change node color to green (calculated)
                circles[prev_t.node_label].set_facecolor(SUCCESS_COLOR)
                mod_patches.append(circles[prev_t.node_label])
                
                # Update the node text to show calculated value
                self._update_node_text_with_value(prev_t, mod_patches)
    
    def _update_node_text_with_value(self, t: Any, mod_patches: List[Any]) -> None:
        """
        Update a node's text to show its calculated value.
        
        Args:
            t: TNode object with calculated value
            mod_patches: List to append modified patches to
        """
        calculated_value = None
        # Try different ways to get the computed value from TNode
        if hasattr(t, 'value') and t.value is not None:
            calculated_value = t.value
        elif hasattr(t, 'result') and t.result is not None:
            calculated_value = t.result
        
        if calculated_value is not None:
            formatted_value = self._format_value_3sf(calculated_value)
            
            # Find and update the text for this node
            text_found = False
            for text_obj in self.ax.texts[:]:
                # Check if this text object is at the right position for this node
                if hasattr(text_obj, '_node_label') and text_obj._node_label == t.node_label:
                    text_obj.set_text(f"{t.node_label}\n{formatted_value}")
                    mod_patches.append(text_obj)
                    text_found = True
                    # print(f"Found text object with _node_label for {t.node_label}")  # Commented out to reduce debug spam
                    break
            
            # if not text_found:
            #     print(f"Could not find text object for {t.node_label}")  # Commented out to reduce debug spam
    
    def _get_edge_label_from_tnode(self, t: Any, lines: Dict[str, Any]) -> Optional[str]:
        """
        Get edge label from tnode for line matching.
        
        Args:
            t: TNode object
            lines: Dictionary of line objects
            
        Returns:
            Edge label string or None if not found
        """
        if t.gen_edge_label is None:
            return None
        
        # Get the source and target nodes from the tnode
        if hasattr(t, 'children') and t.children:
            # This is a generated node, find the edge that connects to it
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
    
    def _color_patch(self, settings_label: str, ps: Any, patch: Any) -> Any:
        """
        Color a patch according to plot settings.
        
        Args:
            settings_label: Label for the plot settings
            ps: Plot settings object
            patch: Patch object to color
            
        Returns:
            The colored patch object
        """
        props = getattr(ps, settings_label)
        patch.set(**props)
        return patch
    
    def _reset_all_edges_to_default(self, lines: Dict[str, Any], ps: Any) -> None:
        """
        Reset all edges to their default appearance at the start of animation.
        
        Args:
            lines: Dictionary of line objects
            ps: Plot settings object
        """
        for line in lines.values():
            self._color_patch('edge_default', ps, line)
    
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
    
    def _schedule_cleanup(self, total_duration: int) -> None:
        """
        Schedule cleanup after animation completes.
        
        Args:
            total_duration: Total animation duration in milliseconds
        """
        # This would need to be implemented with proper timing mechanism
        # For now, we'll handle cleanup in the main GUI
        pass
    
    def cleanup_animation(self) -> None:
        """
        Clean up animation resources after completion.
        """
        if self.moving_circle:
            try:
                self.moving_circle.remove()
                print("Cleaned up moving circle")
            except Exception as e:
                print(f"Error cleaning up moving circle: {e}")
            self.moving_circle = None
    
    def is_animation_running(self) -> bool:
        """
        Check if animation is currently running.
        
        Returns:
            bool: True if animation is running, False otherwise
        """
        return self.animation_running
    
    def set_animation_running(self, running: bool) -> None:
        """
        Set the animation running state.
        
        Args:
            running: True if animation is running, False otherwise
        """
        self.animation_running = running
    
    def _handle_input_display_phase(self, frame: int, inputs: List[str], 
                                   circles: Dict[str, Any], text: Any, 
                                   mod_patches: List[Any]) -> None:
        """
        Handle the input display phase where input nodes turn green with values.
        
        Args:
            frame: Current frame number
            inputs: List of input node labels
            circles: Dictionary of circle patches
            text: Text box for status updates
            mod_patches: List to append modified patches to
        """
        # Calculate which input node to highlight (20 frames per input)
        input_index = frame // 20
        sub_frame = frame % 20
        
        if input_index < len(inputs):
            input_node = inputs[input_index]
            
            # Update status text
            text.set_text(f'Setting static values for source node: {input_node}')
            mod_patches.append(text)
            
            # Turn node green when we reach it
            if sub_frame == 0 and input_node in circles:
                circles[input_node].set_facecolor('#27ae60')  # Green
                mod_patches.append(circles[input_node])
                
                # Update text to show the input value
                self._update_input_node_text(input_node, mod_patches)
        else:
            # All inputs displayed, show ready message
            text.set_text('Static values set. Starting pathfinding...')
            mod_patches.append(text)
    
    def _handle_pathfinding_phase(self, pathfinding_frame: int, tnode_index: int, 
                                 sub_frame: int, pathfinding_tnodes: List[Any], inputs: List[str], 
                                 output: str, circles: Dict[str, Any], lines: Dict[str, Any], 
                                 text: Any, moving_circle: Any, frames_per_edge: int,
                                 layout_positions: Dict[str, Tuple[float, float]], 
                                 ps: Any, mod_patches: List[Any]) -> None:
        """
        Handle the pathfinding phase with red dot movement (simplified).
        
        Args:
            pathfinding_frame: Current pathfinding frame number
            tnode_index: Index of current TNode
            sub_frame: Sub-frame within current edge traversal
            pathfinding_tnodes: List of TNode objects (filtered, excluding source nodes)
            inputs: List of input node labels
            output: Target node label
            circles: Dictionary of circle patches
            lines: Dictionary of line objects
            text: Text box for status updates
            moving_circle: Moving circle patch
            frames_per_edge: Frames per edge traversal
            layout_positions: Dictionary of node positions
            ps: Plot settings object
            mod_patches: List to append modified patches to
        """
        if tnode_index < len(pathfinding_tnodes):
            t = pathfinding_tnodes[tnode_index]
            
            # Simple status text
            text.set_text(f'Pathfinding: {t.node_label}')
            mod_patches.append(text)
            
            # Handle moving circle animation - always visible during pathfinding
            moving_circle.set_alpha(ps.animation['circle_alpha'])
            
            # Get current node position
            if t.node_label in layout_positions:
                current_x, current_y = layout_positions[t.node_label]
                
                # If this is not the first tnode, move from previous node to current
                if tnode_index > 0:
                    prev_t = pathfinding_tnodes[tnode_index - 1]
                    if prev_t.node_label in layout_positions:
                        prev_x, prev_y = layout_positions[prev_t.node_label]
                        
                        # Calculate progress along the path (0 to 1)
                        progress = sub_frame / frames_per_edge
                        
                        # Interpolate between previous and current node
                        x = prev_x + progress * (current_x - prev_x)
                        y = prev_y + progress * (current_y - prev_y)
                        
                        moving_circle.center = (x, y)
                        mod_patches.append(moving_circle)
                else:
                    # First node - just position at current node
                    moving_circle.center = (current_x, current_y)
                    mod_patches.append(moving_circle)
            
            # Color solved edges and update calculated nodes (but not the final target)
            for i in range(tnode_index + 1):  # Include current node to color its edge
                prev_t = pathfinding_tnodes[i]
                
                # Color solved edges
                if prev_t.gen_edge_label is not None:
                    prev_edge_label = self._get_edge_label_from_tnode(prev_t, lines)
                    if prev_edge_label in lines:
                        # Use edge_solved for previously calculated edges
                        mod_patches.append(self._color_patch('edge_solved', ps, lines[prev_edge_label]))
                
                # Update calculated nodes (but not the final target - that will be shown in final phase)
                if i < tnode_index and prev_t.node_label in circles:
                    # Change node color to green (calculated)
                    circles[prev_t.node_label].set_facecolor('#27ae60')
                    mod_patches.append(circles[prev_t.node_label])
                    
                    # Update the node text to show calculated value
                    self._update_node_text_with_value(prev_t, mod_patches)
            
            # Highlight the currently active edge (the one being traversed)
            if tnode_index < len(pathfinding_tnodes):
                current_t = pathfinding_tnodes[tnode_index]
                if current_t.gen_edge_label is not None:
                    current_edge_label = self._get_edge_label_from_tnode(current_t, lines)
                    if current_edge_label in lines:
                        # Use edge_active for the currently active edge
                        mod_patches.append(self._color_patch('edge_active', ps, lines[current_edge_label]))
    
    def _update_input_node_text(self, node_label: str, mod_patches: List[Any]) -> None:
        """
        Update text for an input node to show its value.
        
        Args:
            node_label: Label of the input node
            mod_patches: List to append modified patches to
        """
        # Find and update the text for this node
        for text_obj in self.ax.texts[:]:
            if hasattr(text_obj, '_node_label') and text_obj._node_label == node_label:
                # For input nodes, show the node label and value
                # The value should already be in the text from the initial setup
                mod_patches.append(text_obj)
                break
    
    def _handle_final_result_phase(self, final_result_frame: int, tnodes: List[Any], 
                                  circles: Dict[str, Any], text: Any, moving_circle: Any, 
                                  mod_patches: List[Any]) -> None:
        """
        Handle the final result phase to show the target node's calculated value.
        
        Args:
            final_result_frame: Current frame in final result phase
            tnodes: List of TNode objects
            circles: Dictionary of circle patches
            text: Text box for status updates
            moving_circle: Moving circle patch
            mod_patches: List to append modified patches to
        """
        if tnodes:
            # Get the final target node (last in the list)
            final_tnode = tnodes[-1]
            
            # Update status text
            text.set_text(f'Final result: {final_tnode.node_label}')
            mod_patches.append(text)
            
            # Remove animated circle from visualization
            moving_circle.set_alpha(0.0)
            mod_patches.append(moving_circle)
            
            # Show final result on the target node
            if final_tnode.node_label in circles:
                # Change target node color to green
                circles[final_tnode.node_label].set_facecolor('#27ae60')
                mod_patches.append(circles[final_tnode.node_label])
                
                # Update the node text to show calculated value
                self._update_node_text_with_value(final_tnode, mod_patches)
