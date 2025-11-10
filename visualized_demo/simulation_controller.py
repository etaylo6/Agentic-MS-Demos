"""
Simulation Controller Module

This module handles all simulation-related functionality for the Timoshenko Beam
GUI application. It manages simulation execution, threading, result processing,
and error handling to ensure robust and reliable simulation performance.

Key responsibilities:
- Simulation execution and management
- Threading for non-blocking simulations
- Result extraction and processing
- Error handling and validation
- Integration with hypergraph solver
- Status updates and user feedback

The module provides a clean interface between the GUI and the underlying
simulation engine, handling the complexity of asynchronous execution while
providing clear feedback to the user about simulation progress and results.

Usage:
    from simulation_controller import SimulationController
    controller = SimulationController(gui_ref)
    controller.run_simulation(inputs, target_node, callback)
"""

import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from gui_constants import MIN_SOURCE_NODES


class SimulationController:
    """
    Controls simulation execution and result processing.
    
    This class manages the execution of hypergraph simulations, including
    threading for non-blocking operation, result extraction, error handling,
    and status updates. It provides a robust interface for running simulations
    while maintaining responsive user interaction.
    
    Features:
    - Non-blocking simulation execution using threading
    - Comprehensive error handling and validation
    - Result extraction from TNode objects
    - Status updates and user feedback
    - Integration with hypergraph solver
    - Callback support for result handling
    
    Attributes:
        gui_ref: Reference to main GUI for status updates
        hypergraph: Reference to the hypergraph model
        simulation_thread: Current simulation thread
        simulation_running: Flag indicating if simulation is active
    """
    
    def __init__(self, gui_ref: Any, hypergraph: Any):
        """
        Initialize the simulation controller.
        
        Args:
            gui_ref: Reference to main GUI for callbacks
            hypergraph: Reference to the hypergraph model
        """
        self.gui_ref = gui_ref
        self.hypergraph = hypergraph
        self.simulation_thread: Optional[threading.Thread] = None
        self.simulation_running = False
    
    def run_simulation(self, inputs: Dict[Any, float], target_node: Any, 
                      callback: Optional[Callable] = None) -> None:
        """
        Run the simulation with animation.
        
        Args:
            inputs: Dictionary of input values for the simulation
            target_node: The target node to solve for
            callback: Optional callback function for result handling
        """
        # Check input validity and target node selection before execution
        validation_result, error_message = self._validate_simulation_inputs(inputs, target_node)
        if not validation_result:
            self._update_status(f"Error: {error_message}")
            return
        
        # Update status and disable controls
        self._update_status("Running simulation...")
        self._set_controls_enabled(False)
        self.simulation_running = True
        
        # Run simulation in separate thread
        self.simulation_thread = threading.Thread(
            target=self._run_simulation_thread, 
            args=(inputs, target_node, callback)
        )
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
    
    def _validate_simulation_inputs(self, inputs: Dict[Any, float], target_node: Any) -> Tuple[bool, str]:
        """
        Validate simulation inputs before execution.
        
        Args:
            inputs: Dictionary of input values
            target_node: Target node object
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if not inputs:
            return False, "No source nodes with values"
        
        if target_node is None:
            return False, "No target node selected"
        
        if len(inputs) < MIN_SOURCE_NODES:
            return False, f"Need at least {MIN_SOURCE_NODES} source nodes"
        
        # Validate input values
        for node_obj, value in inputs.items():
            if not isinstance(value, (int, float)):
                return False, f"Invalid value for {str(node_obj)}: {value}"
            
            if not (isinstance(value, (int, float)) and not isinstance(value, bool)):
                return False, f"Value for {str(node_obj)} must be numeric"
        
        return True, ""
    
    def _run_simulation_thread(self, inputs: Dict[Any, float], target_node: Any, 
                              callback: Optional[Callable] = None) -> None:
        """
        Run simulation in separate thread.
        
        Args:
            inputs: Dictionary of input values for the simulation
            target_node: The target node to solve for
            callback: Optional callback function for result handling
        """
        try:
            # Update status
            self._update_status("Running simulation...")
            
            # Execute hypergraph constraint solving to find solution path
            print(f"Solving for target: {target_node}")
            print(f"Input values: {[(str(k), v) for k, v in inputs.items()]}")
            
            self.hypergraph.memory_mode = True
            # Try with target node as string label and inputs as string-keyed dict
            target_label = str(target_node) if target_node else None
            inputs_dict = {str(k): v for k, v in inputs.items()}
            # print(f"Target label: {target_label}")  # Commented out to reduce debug spam
            # print(f"Inputs dict: {inputs_dict}")  # Commented out to reduce debug spam
            end_result = self.hypergraph.solve(target_label, inputs_dict, search_depth=1000)
            
            if end_result is None:
                raise ValueError("No solution path found")
            
            # Extract the sequence of nodes traversed during solution
            tnodes = self.hypergraph.solved_tnodes
            
            if not tnodes:
                raise ValueError("No solution path found")
            
            # Collect calculated values from all nodes in the solution sequence
            computed_results = self._extract_computed_results(tnodes)
            print(f"Computed results: {computed_results}")  # Keep this for debugging calculation issues
            
            # Update status
            self._update_status("Creating animation...")
            
            # Notify main thread with simulation results via callback
            if callback:
                self._call_callback(callback, tnodes, inputs, target_node, computed_results)
            
        except ValueError as ve:
            error_msg = f"Validation error: {str(ve)}"
            self._update_status(error_msg)
            self._set_controls_enabled(True)
        except ImportError as ie:
            error_msg = f"Import error: {str(ie)}"
            self._update_status(error_msg)
            self._set_controls_enabled(True)
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self._update_status(error_msg)
            self._set_controls_enabled(True)
        finally:
            self.simulation_running = False
    
    def _extract_computed_results(self, tnodes: List[Any]) -> Dict[str, float]:
        """
        Extract computed results from TNode objects.
        
        Args:
            tnodes: List of TNode objects from simulation
            
        Returns:
            Dict[str, float]: Dictionary of computed results
        """
        computed_results = {}
        
        def extract_values_from_path(tnode: Any, visited: Optional[set] = None) -> None:
            """
            Recursively extract values from TNode path.
            
            Args:
                tnode: Current TNode object
                visited: Set of visited node labels to prevent cycles
            """
            if visited is None:
                visited = set()
            
            if tnode.node_label in visited:
                return
            visited.add(tnode.node_label)
            
            # Check if this TNode has a computed value
            if hasattr(tnode, 'value') and tnode.value is not None:
                computed_results[tnode.node_label] = tnode.value
            elif hasattr(tnode, 'result') and tnode.result is not None:
                computed_results[tnode.node_label] = tnode.result
            
            # Recursively collect values from child nodes in the solution tree
            if hasattr(tnode, 'children'):
                for child in tnode.children:
                    extract_values_from_path(child, visited)
        
        # Begin value extraction from the final target node
        if tnodes:
            # Extract values starting from the final calculated target node
            final_tnode = tnodes[-1]
            extract_values_from_path(final_tnode)
        
        return computed_results
    
    def _call_callback(self, callback: Callable, tnodes: List[Any], inputs: Dict[Any, float], 
                      target_node: Any, computed_results: Dict[str, float]) -> None:
        """
        Call the callback function with simulation results.
        
        Args:
            callback: Callback function to call
            tnodes: List of TNode objects from simulation
            inputs: Dictionary of input values
            target_node: Target node object
            computed_results: Dictionary of computed results
        """
        try:
            # Execute callback function in the main GUI thread
            if hasattr(self.gui_ref, 'root'):
                self.gui_ref.root.after(0, lambda: callback(tnodes, inputs, target_node, computed_results))
            else:
                callback(tnodes, inputs, target_node, computed_results)
        except Exception as e:
            error_msg = f"Callback error: {str(e)}"
            self._update_status(error_msg)
            self._set_controls_enabled(True)
    
    def _update_status(self, message: str) -> None:
        """
        Update the status message in the GUI.
        
        Args:
            message: Status message to display
        """
        if hasattr(self.gui_ref, 'status_label'):
            if hasattr(self.gui_ref, 'root'):
                self.gui_ref.root.after(0, lambda: self.gui_ref.status_label.config(text=message))
            else:
                self.gui_ref.status_label.config(text=message)
    
    def _set_controls_enabled(self, enabled: bool) -> None:
        """
        Enable or disable GUI controls.
        
        Args:
            enabled: True to enable controls, False to disable
        """
        if hasattr(self.gui_ref, 'run_button'):
            if hasattr(self.gui_ref, 'root'):
                self.gui_ref.root.after(0, lambda: self.gui_ref.run_button.config(state="normal" if enabled else "disabled"))
            else:
                self.gui_ref.run_button.config(state="normal" if enabled else "disabled")
    
    def is_simulation_running(self) -> bool:
        """
        Check if simulation is currently running.
        
        Returns:
            bool: True if simulation is running, False otherwise
        """
        return self.simulation_running
    
    def stop_simulation(self) -> None:
        """
        Stop the current simulation if running.
        
        Note: This is a basic implementation. For more robust stopping,
        consider implementing proper thread cancellation mechanisms.
        """
        if self.simulation_running and self.simulation_thread:
            # Note: Python threads cannot be forcefully stopped
            # This is a placeholder for future implementation
            self.simulation_running = False
            self._update_status("Simulation stopped")
            self._set_controls_enabled(True)
    
    def get_simulation_status(self) -> str:
        """
        Get the current simulation status.
        
        Returns:
            str: Current status message
        """
        if hasattr(self.gui_ref, 'status_label'):
            return self.gui_ref.status_label.cget('text')
        return "Unknown"
    
    def validate_inputs_for_simulation(self, inputs: Dict[Any, float], target_node: Any) -> Tuple[bool, str]:
        """
        Validate inputs for simulation without running the simulation.
        
        Args:
            inputs: Dictionary of input values
            target_node: Target node object
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        return self._validate_simulation_inputs(inputs, target_node)
    
    def get_required_inputs(self) -> List[str]:
        """
        Get list of required input parameters for simulation.
        
        Returns:
            List[str]: List of required input parameter names
        """
        # This could be made more sophisticated by analyzing the hypergraph
        # For now, return a basic list
        return ["point load", "length", "youngs modulus", "shear modulus", "area", "kappa"]
    
    def get_optional_inputs(self) -> List[str]:
        """
        Get list of optional input parameters for simulation.
        
        Returns:
            List[str]: List of optional input parameter names
        """
        # This could be made more sophisticated by analyzing the hypergraph
        # For now, return a basic list
        return ["moment of inertia", "radius", "poisson"]
