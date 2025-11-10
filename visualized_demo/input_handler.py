"""
Input Handling and Management Module

This module manages all user input interactions, validation, and state management
for the Timoshenko Beam GUI application. It handles node input fields, toggle
switches, demo configurations, and input validation to ensure data integrity
and provide a smooth user experience.

Key responsibilities:
- Node input field creation and management
- Toggle switch state handling and validation
- Input value formatting and validation
- Demo configuration management
- Input change event handling
- State synchronization between inputs and visualization

The module provides a clean interface between user input and the application
logic, ensuring that all inputs are properly validated and formatted before
being used in simulations or visualizations.

Usage:
    from input_handler import InputHandler
    handler = InputHandler(parent, gui_ref)
    handler.create_node_inputs()
    values = handler.get_simulation_inputs()
"""

import tkinter as tk
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from gui_constants import (
    CARD_BG_COLOR, TEXT_COLOR, DEFAULT_BG_COLOR, GRAY_COLOR,
    DEFAULT_BEAM_VALUES, DEFAULT_TOGGLE_STATES, MECHANICAL_PROPERTIES,
    GEOMETRY_PROPERTIES, LOADING_OUTPUT_PROPERTIES, DEMO1_CONFIG,
    DEMO2_CONFIG, DEMO3_CONFIG, NODE_LABEL_FONT, MAX_SIGNIFICANT_FIGURES
)
from gui_components import AnimatedToggleSwitch


class InputHandler:
    """
    Handles all user input interactions and state management.
    
    This class manages the creation, validation, and processing of user inputs
    for the Timoshenko Beam application. It provides a centralized way to handle
    node input fields, toggle switches, demo configurations, and input validation.
    
    Features:
    - Dynamic node input field creation
    - Toggle switch state management
    - Input validation and formatting
    - Demo configuration management
    - Real-time input change handling
    - State synchronization with visualization
    
    Attributes:
        parent: Parent widget for input fields
        gui_ref: Reference to main GUI for callbacks
        node_vars: Dictionary of StringVar objects for input values
        toggle_switches: Dictionary of AnimatedToggleSwitch widgets
        input_entries: Dictionary of Entry widgets
        hypergraph: Reference to the hypergraph model
    """
    
    def __init__(self, parent: tk.Widget, gui_ref: Any, hypergraph: Any, *, node_aliases: Optional[Dict[str, Any]] = None):
        """
        Initialize the input handler.
        
        Args:
            parent: Parent widget for input fields
            gui_ref: Reference to main GUI for callbacks
            hypergraph: Reference to the hypergraph model
        """
        self.parent = parent
        self.gui_ref = gui_ref
        self.hypergraph = hypergraph
        self.node_alias_map: Dict[str, Any] = node_aliases or {}
        
        # Input state management
        self.node_vars: Dict[str, tk.StringVar] = {}
        self.toggle_switches: Dict[str, AnimatedToggleSwitch] = {}
        self.input_entries: Dict[str, tk.Entry] = {}
        
        # Demo configurations
        self.demo_configs = {
            1: DEMO1_CONFIG,
            2: DEMO2_CONFIG,
            3: DEMO3_CONFIG
        }
    
    def create_node_inputs(self) -> None:
        """
        Create input fields for each node organized into logical groups.
        
        Creates input fields for all nodes in the hypergraph, organized into
        logical sections (Mechanical Properties, Geometry, Loading & Output)
        with appropriate default values and toggle states.
        """
        # Extract node labels from the Timoshenko beam hypergraph model
        nodes = self._get_node_labels()
        
        # Organize input fields into logical categories for better user experience
        self._create_mechanical_properties_section(nodes)
        self._create_geometry_section(nodes)
        self._create_loading_output_section(nodes)
    
    def _get_node_labels(self) -> List[str]:
        """
        Get sorted list of node labels from hypergraph.
        
        Returns:
            List[str]: Sorted list of node labels
        """
        labels: Set[str] = set(self.node_alias_map.keys())
        if not labels:
            for node in self.hypergraph.nodes:
                if hasattr(node, 'label'):
                    labels.add(node.label)
                else:
                    labels.add(str(node))
        return sorted(labels)
    
    def _create_mechanical_properties_section(self, nodes: List[str]) -> None:
        """
        Create input section for mechanical properties.
        
        Args:
            nodes: List of all node labels
        """
        self._create_section_header("Mechanical Properties")
        for node_label in nodes:
            if node_label in MECHANICAL_PROPERTIES:
                self._create_node_input(node_label)
    
    def _create_geometry_section(self, nodes: List[str]) -> None:
        """
        Create input section for geometry properties.
        
        Args:
            nodes: List of all node labels
        """
        self._create_section_header("Geometry")
        for node_label in nodes:
            if node_label in GEOMETRY_PROPERTIES:
                self._create_node_input(node_label)
    
    def _create_loading_output_section(self, nodes: List[str]) -> None:
        """
        Create input section for loading and output properties.
        
        Args:
            nodes: List of all node labels
        """
        self._create_section_header("Loading & Output")
        for node_label in nodes:
            if node_label in LOADING_OUTPUT_PROPERTIES:
                self._create_node_input(node_label)
    
    def _create_section_header(self, title: str) -> None:
        """
        Create a section header with consistent styling.
        
        Args:
            title: The title text for the section
        """
        # Add some spacing before the header
        spacer = tk.Frame(self.parent, height=12, bg=CARD_BG_COLOR)
        spacer.pack(fill=tk.X)
        
        # Simple header label with consistent styling
        header_label = tk.Label(self.parent, text=title, 
                               font=('Arial', 12, 'bold'), 
                               foreground=TEXT_COLOR, background=CARD_BG_COLOR)
        header_label.pack(pady=(12, 8), padx=15)
        
        # Add a subtle separator line
        separator = tk.Frame(self.parent, height=1, bg=GRAY_COLOR)
        separator.pack(fill=tk.X, pady=(0, 12), padx=15)
    
    def _create_node_input(self, node_label: str) -> None:
        """
        Create input field for a single node.
        
        Args:
            node_label: The label for the node
        """
        
        # Frame for each node with unified background
        node_frame = tk.Frame(self.parent, bg=CARD_BG_COLOR, relief='flat', bd=0, height=45)
        node_frame.pack(fill=tk.X, pady=4, padx=10)
        node_frame.pack_propagate(False)  # Prevent frame from shrinking to fit content
        
        # Node label with consistent styling
        label = tk.Label(node_frame, text=node_label, width=25, font=NODE_LABEL_FONT, 
                        foreground=TEXT_COLOR, background=CARD_BG_COLOR, anchor='w')
        label.pack(side=tk.LEFT, padx=(10, 15))
        
        # Input entry with default value - unified styling
        entry_var = tk.StringVar()
        default_val = DEFAULT_BEAM_VALUES.get(node_label, "")
        entry_var.set(default_val)
        
        entry = tk.Entry(node_frame, textvariable=entry_var, width=25, font=NODE_LABEL_FONT,
                        foreground=DEFAULT_BG_COLOR, background=TEXT_COLOR, relief='flat', bd=1)
        entry.pack(side=tk.LEFT, padx=(0, 15))
        
        # Toggle switch with default state
        default_toggle = DEFAULT_TOGGLE_STATES.get(node_label, "none")
        toggle = AnimatedToggleSwitch(node_frame, 
                                    callback=lambda val, n=node_label: self._on_toggle_change(n, val),
                                    gui_ref=self.gui_ref)
        toggle.set(default_toggle)
        toggle.pack(side=tk.LEFT, padx=(0, 10), pady=5)
        
        # Store references
        self.node_vars[node_label] = entry_var
        self.toggle_switches[node_label] = toggle
        self.input_entries[node_label] = entry
        
        # Debug: Print toggle creation (commented out to reduce spam)
        # print(f"Created toggle for {node_label}: {toggle.get()}")
        
        # Bind entry change
        entry_var.trace('w', lambda *args, n=node_label: self._on_entry_change(n))
    
    def _on_toggle_change(self, node_label: str, value: str) -> None:
        """
        Handle toggle switch changes.
        
        Args:
            node_label: The label of the node whose toggle changed
            value: The new toggle value ("none", "target", or "source")
        """
        # Prevent user input modifications while animation is running
        if self.gui_ref.animation_running:
            return
            
        # Validate source nodes have values and reset empty source nodes to inactive state
        self._revert_empty_source_nodes()
            
        entry = self.input_entries[node_label]
        entry_value = self.node_vars[node_label].get().strip()
        
        # Update entry state based on toggle
        if value == "target":
            # Target can be selected even with blank field, and field should be editable
            entry.config(state="normal")
        elif value == "none":
            # None means field is disabled
            entry.config(state="disabled")
        else:  # source
            # Source requires a value, so field is editable
            entry.config(state="normal")
        
        # Notify GUI of changes
        if hasattr(self.gui_ref, 'update_node_colors'):
            self.gui_ref.update_node_colors()
        if hasattr(self.gui_ref, '_update_node_text'):
            self.gui_ref._update_node_text(node_label)
        if hasattr(self.gui_ref, 'canvas'):
            self.gui_ref.canvas.draw()
    
    def _on_entry_change(self, node_label: str) -> None:
        """
        Handle input entry changes.
        
        Args:
            node_label: The label of the node whose entry changed
        """
        # Prevent user input modifications while animation is running
        if self.gui_ref.animation_running:
            return
            
        # If field becomes blank and toggle is source, set to none
        entry_value = self.node_vars[node_label].get().strip()
        toggle_state = self.toggle_switches[node_label].get()
        
        if not entry_value and toggle_state == "source":
            self.toggle_switches[node_label].set("none")
            # Update field state to disabled since we switched to "none"
            self.input_entries[node_label].config(state="disabled")
            
        # Notify GUI of changes
        if hasattr(self.gui_ref, 'update_node_colors'):
            self.gui_ref.update_node_colors()
        if hasattr(self.gui_ref, '_update_node_text'):
            self.gui_ref._update_node_text(node_label)
        if hasattr(self.gui_ref, 'canvas'):
            self.gui_ref.canvas.draw()
    
    def _revert_empty_source_nodes(self) -> None:
        """
        Revert any nodes with 'source' toggle but empty fields to 'none'.
        """
        for node_label, toggle in self.toggle_switches.items():
            toggle_state = toggle.get()
            entry_value = self.node_vars[node_label].get().strip()
            
            if toggle_state == "source" and not entry_value:
                toggle.set("none")
                self.input_entries[node_label].config(state="disabled")
    
    def get_simulation_inputs(self) -> Tuple[Dict[Any, float], Optional[Any]]:
        """
        Get inputs for simulation based on toggle states.
        
        Returns:
            Tuple containing:
                - Dictionary of input values for source nodes (with Node objects as keys)
                - Target node object (or None if not found)
        """
        inputs = {}
        target_node = None
        
        for node_label, toggle in self.toggle_switches.items():
            toggle_state = toggle.get()
            entry_value = self.node_vars[node_label].get().strip()
            
            if toggle_state == "source" and entry_value:
                try:
                    # Handle fraction like "5/6"
                    if '/' in entry_value:
                        numerator, denominator = entry_value.split('/')
                        value = float(numerator) / float(denominator)
                    else:
                        value = float(entry_value)
                    
                    # Find the corresponding Node object for this label
                    node_obj = self.node_alias_map.get(node_label)
                    if node_obj is None:
                        for node in self.hypergraph.nodes:
                            candidate = getattr(node, 'label', str(node))
                            if candidate == node_label or str(node) == node_label:
                                node_obj = node
                                break
                    
                    if node_obj is not None:
                        inputs[node_obj] = value
                        # print(f"Added input: {node_label} = {value} (node: {node_obj})")  # Debug
                    else:
                        print(f"Warning: Could not find node object for label: {node_label}")  # Debug
                        print(f"Available nodes: {[str(n) for n in self.hypergraph.nodes]}")  # Debug
                except ValueError:
                    pass
            elif toggle_state == "target":
                # Find the target node
                target_node = self.node_alias_map.get(node_label)
                if target_node is None:
                    for node in self.hypergraph.nodes:
                        candidate = getattr(node, 'label', str(node))
                        if candidate == node_label or str(node) == node_label:
                            target_node = node
                            break
                
                if target_node is None:
                    # Try to get the node directly
                    try:
                        target_node = self.hypergraph.get_node(node_label)
                    except:
                        continue
                
                if target_node is not None:
                    # print(f"Set target node: {node_label} (node: {target_node})")  # Debug
                    pass
                else:
                    print(f"Warning: Could not find target node for label: {node_label}")  # Debug
                    print(f"Available nodes: {[str(n) for n in self.hypergraph.nodes]}")  # Debug
        
        return inputs, target_node
    
    def format_value_3sf(self, value: Any) -> str:
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
    
    def setup_demo(self, demo_number: int) -> None:
        """
        Set up a predefined demo scenario.
        
        Args:
            demo_number: Number of the demo to run (1, 2, or 3)
        """
        if demo_number not in self.demo_configs:
            raise ValueError(f"Demo {demo_number} not found. Available demos: {list(self.demo_configs.keys())}")
        
        config = self.demo_configs[demo_number]
        self._apply_demo_config(config)
    
    def _apply_demo_config(self, config: Dict[str, Any]) -> None:
        """
        Apply demo configuration to input fields.
        
        Args:
            config: Demo configuration dictionary
        """
        # Reset all input field values to empty state
        for var in self.node_vars.values():
            var.set("")
        
        # Reset all toggle switches to inactive state before applying demo configuration
        for toggle in self.toggle_switches.values():
            toggle.set("none")
        
        # Apply demo configuration
        for node_label, value in config['values'].items():
            if node_label in self.node_vars:
                self.node_vars[node_label].set(value)
        
        for node_label, toggle_state in config['toggles'].items():
            if node_label in self.toggle_switches:
                self.toggle_switches[node_label].set(toggle_state)
    
    def reset_inputs(self) -> None:
        """
        Reset all inputs and toggles to default state.
        """
        for node_label in self.node_vars:
            self.node_vars[node_label].set("")
            self.toggle_switches[node_label].set("none")
    
    def get_node_vars(self) -> Dict[str, tk.StringVar]:
        """
        Get dictionary of node StringVar objects.
        
        Returns:
            Dict[str, tk.StringVar]: Dictionary mapping node labels to StringVar objects
        """
        return self.node_vars
    
    def get_toggle_switches(self) -> Dict[str, AnimatedToggleSwitch]:
        """
        Get dictionary of toggle switch widgets.
        
        Returns:
            Dict[str, AnimatedToggleSwitch]: Dictionary mapping node labels to toggle switches
        """
        return self.toggle_switches
    
    def get_input_entries(self) -> Dict[str, tk.Entry]:
        """
        Get dictionary of input entry widgets.
        
        Returns:
            Dict[str, tk.Entry]: Dictionary mapping node labels to entry widgets
        """
        return self.input_entries
    
    def validate_inputs(self) -> Tuple[bool, str]:
        """
        Validate current input state for simulation.
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        inputs, target_node = self.get_simulation_inputs()
        
        if not inputs:
            return False, "No source nodes with values"
        
        if target_node is None:
            return False, "No target node selected"
        
        if len(inputs) < 3:
            return False, "Need at least 3 source nodes"
        
        return True, ""
