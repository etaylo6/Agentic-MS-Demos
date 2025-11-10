"""
Main Timoshenko Beam GUI Application

This is the main orchestrator for the Timoshenko Beam GUI application. It coordinates
between all the modular components to provide a cohesive user experience for
interacting with Timoshenko beam hypergraph simulations.

Key responsibilities:
- Application lifecycle management
- Component coordination and integration
- High-level event handling
- Demo management and execution
- Main application entry point

The main GUI class serves as the central coordinator, bringing together the layout
manager, input handler, visualization manager, animation engine, and simulation
controller to create a seamless user experience. It maintains the overall application
state and handles high-level user interactions.

Usage:
    from beam_gui_main import TimoshenkoBeamGUI
    root = tk.Tk()
    app = TimoshenkoBeamGUI(root)
    root.mainloop()
"""

import tkinter as tk
from typing import Dict, List, Optional, Tuple, Any

# Local imports
from gui_constants import (
    DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_TITLE, DEFAULT_BG_COLOR,
    DEMO1_CONFIG, DEMO2_CONFIG, DEMO3_CONFIG
)
from gui_layout import LayoutManager
from input_handler import InputHandler
from visualization_manager import VisualizationManager
from animation_engine import AnimationEngine
from simulation_controller import SimulationController
from beam_model_def import create_beam_model
import edge_grouper
from trivial_trimmer import trim_trivial_edges


NODE_LABEL_ALIASES = {
    'P': 'point load',
    'k': 'kappa',
    'E': 'youngs modulus',
    'I': 'moment of inertia',
    'G': 'shear modulus',
    'A': 'area',
    'L': 'length',
    'w': 'theta',
    'radius': 'radius',
    'V': 'poisson',
    'S': 'slenderness ratio',
    'heuristic': 'slenderness heuristic',
}

TRIM_SKIP_GROUPS = ('other',)


class TimoshenkoBeamGUI:
    """
    Main GUI application for Timoshenko beam simulation.
    
    This class serves as the central coordinator for the entire application,
    bringing together all the modular components to provide a cohesive user
    experience. It manages the application lifecycle, coordinates between
    components, and handles high-level user interactions.
    
    Features:
    - Modular architecture with clear separation of concerns
    - Comprehensive demo system with predefined scenarios
    - Real-time visualization with animated simulations
    - Robust error handling and user feedback
    - Professional user interface with consistent styling
    - Integration with hypergraph constraint solving
    
    Attributes:
        root: Main Tkinter root window
        hypergraph: Timoshenko beam hypergraph model
        layout_manager: Manages GUI layout and styling
        input_handler: Handles user input and validation
        visualization_manager: Manages hypergraph visualization
        animation_engine: Handles animation creation and execution
        simulation_controller: Controls simulation execution
        animation_running: Flag indicating if animation is active
        computed_results: Dictionary of computed simulation results
    """
    
    def __init__(self, root: tk.Tk) -> None:
        """
        Initialize the GUI application.
        
        Args:
            root: The main Tkinter root window
        """
        self.root = root
        self.root.title(DEFAULT_WINDOW_TITLE)
        self.root.geometry(f"{DEFAULT_WINDOW_SIZE[0]}x{DEFAULT_WINDOW_SIZE[1]}")
        self.root.configure(bg=DEFAULT_BG_COLOR)
        
        # Create hypergraph model
        self.base_hypergraph, self.nodes = create_beam_model()
        (
            self.node_alias_map,
            self.node_display_lookup,
            self.node_label_to_alias,
        ) = self._build_alias_maps(self.nodes)
        self.hypergraph = self.base_hypergraph
        self.current_sim_hypergraph = self.base_hypergraph
        self.current_trim_notes: List[str] = []
        self.function_group_labels: Dict[str, str] = self._build_group_label_map(
            edge_grouper.group_edges_by_label(self.base_hypergraph)
        )
        
        # Initialize components
        self.layout_manager = LayoutManager(self.root)
        self.animation_running = False
        self.computed_results: Dict[str, float] = {}
        
        # Create main layout
        self.control_frame, self.viz_frame = self.layout_manager.create_main_layout()
        
        # Initialize component managers
        self._initialize_components()
        
        # Create GUI elements
        self._create_gui_elements()
        
        # Set up initial view
        self._setup_initial_view()
    
    def _initialize_components(self) -> None:
        """
        Initialize all component managers.
        """
        # Create input handler
        self.input_handler = InputHandler(
            self.control_frame,
            self,
            self.hypergraph,
            node_aliases=self.node_alias_map,
        )
        
        # Create visualization components
        self.fig, self.ax, self.canvas = self.layout_manager.create_visualization_panel(self.viz_frame)
        self.visualization_manager = VisualizationManager(self.fig, self.ax, self.canvas)
        # Use bipartite representation in the GUI
        self.visualization_manager.use_bipartite = True
        self.visualization_manager.set_alias_map(self.node_label_to_alias)
        self.visualization_manager.set_function_group_labels(self.function_group_labels)
        
        # Create animation engine
        self.animation_engine = AnimationEngine(self.fig, self.ax, self.canvas)
        
        # Create simulation controller
        self.simulation_controller = SimulationController(self, self.hypergraph)
    
    def _build_alias_maps(
        self, nodes: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[Any, str], Dict[str, str]]:
        """
        Construct alias mappings without mutating the underlying hypergraph nodes.
        
        Args:
            nodes: Dictionary of nodes returned by the beam model definition
        
        Returns:
            Tuple containing:
                - alias_to_node: Mapping of display labels to Node objects
                - node_to_alias: Mapping of Node objects to display labels
                - canonical_to_alias: Mapping of canonical node labels to display labels
        """
        alias_to_node: Dict[str, Any] = {}
        node_to_alias: Dict[Any, str] = {}
        canonical_to_alias: Dict[str, str] = {}
        
        for key, node in nodes.items():
            canonical_label = getattr(node, 'label', str(node))
            display_label = NODE_LABEL_ALIASES.get(key, canonical_label)
            
            alias_to_node[display_label] = node
            node_to_alias[node] = display_label
            node_to_alias[id(node)] = display_label
            canonical_to_alias[canonical_label] = display_label
        
        # Ensure every node in the hypergraph has an alias entry
        for node in self.base_hypergraph.nodes:
            canonical_label = getattr(node, 'label', str(node))
            if canonical_label not in canonical_to_alias:
                canonical_to_alias[canonical_label] = canonical_label
                node_to_alias[node] = canonical_label
                node_to_alias[id(node)] = canonical_label
                alias_to_node.setdefault(canonical_label, node)
        
        return alias_to_node, node_to_alias, canonical_to_alias
    
    def _create_gui_elements(self) -> None:
        """
        Create all GUI elements and layout sections.
        """
        # Create control panel layout first
        layout_sections = self.layout_manager.create_control_panel_layout(self.control_frame)
        
        # Initialize input fields for beam parameters in the designated input area
        self.input_handler.parent = layout_sections['inputs_section']
        self.input_handler.create_node_inputs()
        
        # Create control buttons
        self._create_control_buttons(layout_sections['action_frame'])
        
        # Create test button
        self._create_test_button(layout_sections['action_frame'])
        
        # Create demo buttons
        self._create_demo_buttons(layout_sections['demo_section'])
        
        # Create status display
        self._create_status_display(layout_sections['status_frame'])
    
    def _create_control_buttons(self, parent: tk.Widget) -> None:
        """
        Create main control buttons.
        
        Args:
            parent: Parent widget for the buttons
        """
        from gui_constants import ACCENT_COLOR, GRAY_COLOR, TEXT_COLOR, BUTTON_FONT
        
        self.run_button = tk.Button(parent, text="Run Simulation", 
                                   command=self.run_simulation,
                                   font=BUTTON_FONT,
                                   foreground=TEXT_COLOR, background=ACCENT_COLOR,
                                   relief='flat', bd=0, padx=20, pady=8)
        self.run_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 12))
        
        self.reset_button = tk.Button(parent, text="Reset", 
                                     command=self.reset_inputs,
                                     font=BUTTON_FONT,
                                     foreground=TEXT_COLOR, background=GRAY_COLOR,
                                     relief='flat', bd=0, padx=20, pady=8)
        self.reset_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(12, 0))
    
    def _create_test_button(self, parent: tk.Widget) -> None:
        """
        Create test button for verifying calculations.
        """
        from gui_constants import WARNING_COLOR, TEXT_COLOR, BUTTON_FONT
        
        # Create test button
        self.test_button = tk.Button(parent, text="Test Calculations", 
                                    command=self.test_calculations,
                                    font=BUTTON_FONT,
                                    foreground=TEXT_COLOR, background=WARNING_COLOR,
                                    relief='flat', bd=0, padx=20, pady=8)
        self.test_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(12, 0))
    
    def _create_demo_buttons(self, parent: tk.Widget) -> None:
        """
        Create demo buttons.
        
        Args:
            parent: Parent widget for the buttons
        """
        from gui_constants import GRAY_COLOR, TEXT_COLOR, DEMO_BUTTON_FONT
        
        self.demo1_button = tk.Button(parent, text="Demo 1: Basic Deflection", 
                                     command=lambda: self.run_demo(1),
                                     font=DEMO_BUTTON_FONT,
                                     foreground=TEXT_COLOR, background=GRAY_COLOR,
                                     relief='flat', bd=0, padx=15, pady=6)
        self.demo1_button.pack(fill=tk.X, pady=3)
        
        self.demo2_button = tk.Button(parent, text="Demo 2: Load from Deflection", 
                                     command=lambda: self.run_demo(2),
                                     font=DEMO_BUTTON_FONT,
                                     foreground=TEXT_COLOR, background=GRAY_COLOR,
                                     relief='flat', bd=0, padx=15, pady=6)
        self.demo2_button.pack(fill=tk.X, pady=3)
        
        self.demo3_button = tk.Button(parent, text="Demo 3: Poisson's Ratio", 
                                     command=lambda: self.run_demo(3),
                                     font=DEMO_BUTTON_FONT,
                                     foreground=TEXT_COLOR, background=GRAY_COLOR,
                                     relief='flat', bd=0, padx=15, pady=6)
        self.demo3_button.pack(fill=tk.X, pady=3)
    
    def _create_status_display(self, parent: tk.Widget) -> None:
        """
        Create status display.
        
        Args:
            parent: Parent widget for the status display
        """
        from gui_constants import STATUS_TEXT_COLOR, CARD_BG_COLOR, STATUS_FONT
        
        self.status_label = tk.Label(parent, text="Ready", font=STATUS_FONT, 
                                    foreground=STATUS_TEXT_COLOR, background=CARD_BG_COLOR)
        self.status_label.pack(side=tk.LEFT)
    
    def _setup_initial_view(self) -> None:
        """
        Set up the initial view of the hypergraph.
        """
        self.visualization_manager.set_alias_map(self.node_label_to_alias)
        self.visualization_manager.set_function_group_labels(self.function_group_labels)
        # Set up initial visualization
        self.visualization_manager.setup_initial_view(
            self.hypergraph,
            self.input_handler.get_node_vars(),
            self.input_handler.get_toggle_switches()
        )
    
    def run_simulation(self) -> None:
        """
        Run the simulation with animation.
        """
        # Get simulation inputs
        inputs, target_node = self.input_handler.get_simulation_inputs()
        
        # Validate inputs
        is_valid, error_message = self.input_handler.validate_inputs()
        if not is_valid:
            self.status_label.config(text=f"Error: {error_message}")
            return
        
        # Prepare simulation hypergraph using edge grouping and trimming
        simulation_hypergraph, removed_edges = self._prepare_simulation_hypergraph(inputs)
        self.current_sim_hypergraph = simulation_hypergraph
        self.simulation_controller.hypergraph = simulation_hypergraph
        
        if removed_edges:
            self.status_label.config(text=f"Pruned {removed_edges} redundant edges; running simulation...")
        
        # Execute hypergraph simulation and handle results via callback function
        self.simulation_controller.run_simulation(
            inputs, target_node, self._on_simulation_complete
        )
    
    def _on_simulation_complete(self, tnodes: List[Any], inputs: Dict[Any, float], 
                               target_node: Any, computed_results: Dict[str, float]) -> None:
        """
        Handle simulation completion and create animation.
        
        Args:
            tnodes: List of TNode objects from simulation
            inputs: Dictionary of input values
            target_node: Target node object
            computed_results: Dictionary of computed results
        """
        try:
            # Store computed results
            self.computed_results = computed_results
            
            # Create animation
            self._create_animation(
                tnodes,
                inputs,
                target_node,
                computed_results,
                hypergraph=self.current_sim_hypergraph
            )
            
            # Update status
            self.status_label.config(text="Animation complete!")
            
        except Exception as e:
            self.status_label.config(text=f"Animation error: {str(e)}")
        finally:
            self.run_button.config(state="normal")
    
    def _create_animation(self, tnodes: List[Any], inputs: Dict[Any, float], 
                         target_node: Any, computed_results: Optional[Dict[str, float]] = None,
                         *, hypergraph: Optional[Any] = None) -> None:
        """
        Create the animation for the simulation results.
        
        Args:
            tnodes: List of TNode objects from simulation
            inputs: Dictionary of input values
            target_node: Target node object
            computed_results: Optional dictionary of computed results
            hypergraph: Optional hypergraph to use for visualization
        """
        active_hypergraph = hypergraph or self.current_sim_hypergraph or self.hypergraph
        
        # Validate inputs
        if not tnodes:
            raise ValueError("No solution nodes provided for animation")
        if not inputs:
            raise ValueError("No input values provided for animation")
        if target_node is None:
            raise ValueError("No target node provided for animation")
        
        # Prepare visualization for animation by setting all nodes to neutral gray color
        self.visualization_manager.set_alias_map(self.node_label_to_alias)
        self.visualization_manager.set_function_group_labels(self.function_group_labels)
        self.visualization_manager.setup_animation_view(
            active_hypergraph,
            self.input_handler.get_node_vars(),
            self.input_handler.get_toggle_switches()
        )
        
        # Create bipartite plot settings
        from plothg_bipartite import BipartitePlotSettings, create_bipartite_animation
        from gui_constants import SEMANTIC_GROUPS
        bps = BipartitePlotSettings()
        bps.animation['frames_per_edge'] = 30
        bps.animation['fps'] = 30
        bps.animation['circle_radius'] = 0.08
        bps.animation['circle_color'] = '#ff3333'
        bps.animation['circle_alpha'] = 0.9
        semantic_map: Dict[str, str] = {}
        if isinstance(SEMANTIC_GROUPS, dict) and SEMANTIC_GROUPS:
            semantic_map.update(SEMANTIC_GROUPS)
        if self.function_group_labels:
            semantic_map.update(self.function_group_labels)
        if semantic_map:
            bps.merge['semantic_groups'] = semantic_map
        bps.merge['read_edge_attr'] = True

        # Add status display widget to show animation progress and current operation
        text_box = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                              verticalalignment='top', fontsize=10,
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Generate bipartite animation using current visualization elements
        input_labels = [self._get_canonical_label(node) for node in inputs.keys()]
        target_label = self._get_canonical_label(target_node)
        bps.alias_map = dict(self.node_label_to_alias)
        ani = create_bipartite_animation(
            active_hypergraph,
            tnodes,
            input_labels,
            target_label,
            self.visualization_manager.get_circles(),
            self.visualization_manager.get_lines(),
            self.visualization_manager.get_layout_positions(),
            bps,
            self.fig,
            self.ax,
            text_box,
        )
        
        # Update canvas
        self.canvas.draw()
    
    def run_demo(self, demo_number: int) -> None:
        """
        Run a predefined demo scenario.
        
        Args:
            demo_number: Number of the demo to run (1, 2, or 3)
        """
        try:
            # Set up demo
            self.input_handler.setup_demo(demo_number)
            
            # Update visualization
            self.visualization_manager.update_node_colors(
                self.input_handler.get_toggle_switches()
            )
            self.canvas.draw()
            
            # Execute the beam deflection calculation with current demo parameters
            self.run_simulation()
            
        except Exception as e:
            self.status_label.config(text=f"Demo error: {str(e)}")
    
    def test_calculations(self) -> None:
        """
        Test the calculations to verify they are working correctly.
        """
        try:
            # Import and run the test function
            from test_calculations import test_timoshenko_calculations, test_with_different_values
            
            print("\n" + "="*60)
            print("RUNNING CALCULATION TESTS FROM GUI")
            print("="*60)
            
            test_timoshenko_calculations()
            test_with_different_values()
            
            print("\n" + "="*60)
            print("TESTS COMPLETED - CHECK CONSOLE FOR RESULTS")
            print("="*60)
            
            # Update status
            self._update_status("Tests completed - check console for results")
            
        except Exception as e:
            print(f"Error running tests: {e}")
            self._update_status(f"Test error: {e}")
    
    def reset_inputs(self) -> None:
        """
        Reset all inputs and toggles to default state.
        """
        self.input_handler.reset_inputs()
        
        # Reset to initial view
        self._setup_initial_view()
        self.status_label.config(text="Reset complete")
        self.current_sim_hypergraph = self.base_hypergraph
        self.function_group_labels = self._build_group_label_map(
            edge_grouper.group_edges_by_label(self.base_hypergraph)
        )
        self.visualization_manager.set_function_group_labels(self.function_group_labels)
    
    def update_node_colors(self) -> None:
        """
        Update node colors based on toggle states.
        """
        self.visualization_manager.update_node_colors(
            self.input_handler.get_toggle_switches()
        )
    
    def _update_node_text(self, node_label: str) -> None:
        """
        Update the text display for a specific node.
        
        Args:
            node_label: The label of the node to update
        """
        self.visualization_manager.update_node_text(
            node_label,
            self.input_handler.get_node_vars(),
            self.input_handler.get_toggle_switches()
        )

    def _prepare_simulation_hypergraph(self, inputs: Dict[Any, float]) -> Tuple[Any, int]:
        """
        Prepare a simulation-ready hypergraph by grouping edges and trimming trivial ones.
        
        Args:
            inputs: Dictionary of source node inputs selected by the user
        
        Returns:
            Tuple[Any, int]: The hypergraph to use for simulation and the count of trimmed edges
        """
        if not inputs:
            return self.base_hypergraph, 0
        
        base_group_map: Dict[str, List[edge_grouper.Edge]] = {}
        try:
            edge_groups = edge_grouper.group_edges_by_label(self.base_hypergraph)
            base_group_map = edge_groups
            trim_outcome = trim_trivial_edges(
                self.base_hypergraph,
                edge_groups,
                inputs=inputs,
                skip_groups=TRIM_SKIP_GROUPS,
                log=self._log_trim_message,
            )
            trimmed_hg = trim_outcome.hypergraph or self.base_hypergraph
            self.current_trim_notes = list(trim_outcome.notes)
            self.function_group_labels = self._build_group_label_map(
                edge_grouper.group_edges_by_label(trimmed_hg)
            )
            self.visualization_manager.set_function_group_labels(self.function_group_labels)
            return trimmed_hg, trim_outcome.removed_count
        except Exception as exc:
            self._log_trim_message(f"Trimming disabled due to error: {exc}")
            self.current_trim_notes = []
            if base_group_map:
                self.function_group_labels = self._build_group_label_map(base_group_map)
            else:
                self.function_group_labels = {}
            self.visualization_manager.set_function_group_labels(self.function_group_labels)
            return self.base_hypergraph, 0

    def _log_trim_message(self, message: str) -> None:
        """
        Simple logger hook for edge trimming operations.
        
        Args:
            message: Message emitted by the trimmer
        """
        print(f"[Trimmer] {message}")

    def _build_group_label_map(self, grouped_edges: Dict[str, List[edge_grouper.Edge]]) -> Dict[str, str]:
        """
        Convert grouped edge data into a mapping of edge label -> friendly group name.
        """
        label_map: Dict[str, str] = {}
        for group_name, edges in grouped_edges.items():
            if not edges:
                continue
            friendly = self._format_group_name(group_name)
            for edge in edges:
                if edge.label:
                    label_map[edge.label] = friendly
        return label_map

    @staticmethod
    def _format_group_name(raw_name: str) -> str:
        """
        Format a raw group identifier into a user-friendly name.
        """
        cleaned = raw_name.replace('_', ' ').strip()
        if not cleaned:
            return "Function"
        return cleaned.title()

    def _get_display_label(self, node: Any) -> str:
        """
        Retrieve the display label for a given node if available.
        """
        if node is None:
            return ""
        return (
            self.node_display_lookup.get(node)
            or self.node_display_lookup.get(id(node))
            or getattr(node, 'label', str(node))
        )

    @staticmethod
    def _get_canonical_label(node: Any) -> str:
        """
        Retrieve the canonical label for a node (used internally by the solver/visualizer).
        """
        if node is None:
            return ""
        return getattr(node, 'label', str(node))


def main() -> None:

    root = tk.Tk()
    app = TimoshenkoBeamGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
