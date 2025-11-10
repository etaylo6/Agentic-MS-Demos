"""
Custom GUI Components for Timoshenko Beam Application

This module contains all custom GUI components and widgets used in the application.
These components provide specialized functionality beyond standard Tkinter widgets,
including animated toggle switches and other interactive elements.

Key responsibilities:
- AnimatedToggleSwitch: Custom toggle widget for node state selection
- Component styling and theming
- Event handling and user interaction
- Animation and visual feedback
- Component lifecycle management

The components are designed to be reusable and maintain consistent styling
throughout the application. They integrate seamlessly with the main GUI
framework while providing enhanced user experience through animations and
visual feedback.

Usage:
    from gui_components import AnimatedToggleSwitch
    toggle = AnimatedToggleSwitch(parent, callback=my_callback)
"""

import tkinter as tk
from tkinter import ttk
from typing import List, Optional, Any, Callable
from gui_constants import (
    SUCCESS_COLOR, WARNING_COLOR, GRAY_COLOR, CARD_BG_COLOR, 
    TEXT_COLOR, TOGGLE_FONT, ANIMATION_INTERVAL, ERROR_COLOR
)


class AnimatedToggleSwitch(tk.Frame):
    """
    Custom animated toggle switch widget for node state selection.
    
    This widget provides a visual toggle between three states: none, target, and source.
    Each state has a distinct color and the widget includes smooth animations for
    better user experience. The toggle cycles through states on click and provides
    visual feedback during transitions.
    
    Features:
    - Three-state toggle: none, target, source
    - Smooth color transitions
    - Click animation with debouncing
    - Customizable colors and styling
    - Callback support for state changes
    - Integration with GUI animation state
    
    Attributes:
        options: List of available toggle states
        current_index: Index of currently selected state
        callback: Function called when state changes
        animation_running: Flag to prevent rapid clicking
        gui_ref: Reference to main GUI for animation state checking
        colors: Color mapping for each state
    
    Example:
        def on_toggle_change(value):
            print(f"Toggle changed to: {value}")
        
        toggle = AnimatedToggleSwitch(
            parent=frame,
            callback=on_toggle_change,
            gui_ref=main_gui
        )
        toggle.set("source")
    """
    
    def __init__(self, parent: tk.Widget, options: Optional[List[str]] = None, 
                 callback: Optional[Callable[[str], None]] = None, 
                 gui_ref: Optional[Any] = None, **kwargs):
        """
        Initialize the animated toggle switch.
        
        Args:
            parent: Parent widget to contain the toggle
            options: List of toggle states (default: ["none", "target", "source"])
            callback: Function called when state changes
            gui_ref: Reference to main GUI for animation state checking
            **kwargs: Additional arguments passed to Canvas
        """
        super().__init__(parent, width=120, height=30, bg=CARD_BG_COLOR, relief='raised', bd=2, **kwargs)
        self.pack_propagate(False)  # Prevent frame from shrinking
        
        # Configuration
        self.options = options or ["none", "target", "source"]
        self.current_index = 0
        self.callback = callback
        self.animation_running = False
        self.gui_ref = gui_ref
        
        # Define color mapping for toggle states matching the application theme
        self.colors = {
            "source": SUCCESS_COLOR,  # Green - consistent with theme
            "target": WARNING_COLOR,  # Orange - consistent with theme
            "none": GRAY_COLOR        # Gray - consistent with theme
        }
        
        # Initialize toggle switch display label
        self.label = tk.Label(self, text="", font=TOGGLE_FONT, 
                             foreground='white', background=CARD_BG_COLOR,
                             width=15, height=1, relief='flat', bd=1)
        self.label.pack(fill=tk.BOTH, expand=True)
        
        # Bind click event and draw initial state
        self.bind("<Button-1>", self._on_click)
        self.label.bind("<Button-1>", self._on_click)
        self._draw_switch()
    
    def _on_click(self, event: tk.Event) -> None:
        """
        Handle click to cycle through options.
        
        Prevents rapid clicking during animations and checks if the main GUI
        is currently running an animation to avoid conflicts.
        
        Args:
            event: The mouse click event
        """
        # Prevent interaction during animations
        if self.animation_running or (self.gui_ref and self.gui_ref.animation_running):
            return
            
        self.animation_running = True
        self._animate_to_next()
    
    def _animate_to_next(self) -> None:
        """
        Animate to the next option with smooth transition.
        
        Cycles through the available options and triggers the callback
        if one is provided. Includes a delay to prevent rapid clicking.
        """
        # Cycle to next option
        self.current_index = (self.current_index + 1) % len(self.options)
        self._draw_switch()
        
        # Trigger callback function to notify parent of state change
        if self.callback:
            self.callback(self.get())
        
        # Reset animation flag after delay to prevent rapid clicking
        self.after(ANIMATION_INTERVAL, lambda: setattr(self, 'animation_running', False))
    
    def _draw_switch(self) -> None:
        """
        Draw the toggle switch with current state styling.
        
        Updates the label with the current option text and color.
        The color changes based on the current state to provide
        clear visual feedback to the user.
        """
        # Get current option and its color
        current_option = self.options[int(self.current_index)]
        color = self.colors.get(current_option, GRAY_COLOR)
        
        # Refresh label display to show current toggle state
        self.label.config(text=current_option.upper(), background=color, foreground='white')
        self.config(background=color)
    
    def get(self) -> str:
        """
        Get current option.
        
        Returns:
            str: The current option name
        """
        return self.options[int(self.current_index)]
    
    def set(self, option: str) -> None:
        """
        Set current option.
        
        Args:
            option: The option to set as current
            
        Raises:
            ValueError: If option is not in available options
        """
        if option not in self.options:
            raise ValueError(f"Option '{option}' not in available options: {self.options}")
        
        self.current_index = self.options.index(option)
        self._draw_switch()
    
    def is_animating(self) -> bool:
        """
        Check if the toggle is currently animating.
        
        Returns:
            bool: True if animation is running, False otherwise
        """
        return self.animation_running
    
    def get_available_options(self) -> List[str]:
        """
        Get list of available toggle options.
        
        Returns:
            List[str]: List of available option names
        """
        return self.options.copy()
    
    def set_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        """
        Set or update the callback function.
        
        Args:
            callback: Function to call when state changes, or None to remove
        """
        self.callback = callback
    
    def reset_to_default(self) -> None:
        """
        Reset the toggle to the first option (typically "none").
        """
        self.current_index = 0
        self._draw_switch()
        if self.callback:
            self.callback(self.get())


class StatusIndicator(tk.Frame):
    """
    Custom status indicator widget for displaying application state.
    
    Provides a visual indicator of the current application status with
    color-coded states and optional progress indication.
    
    Features:
    - Color-coded status states
    - Optional progress bar
    - Customizable status messages
    - Smooth state transitions
    """
    
    def __init__(self, parent: tk.Widget, **kwargs):
        """
        Initialize the status indicator.
        
        Args:
            parent: Parent widget
            **kwargs: Additional arguments for Frame
        """
        super().__init__(parent, **kwargs)
        
        # Status label
        self.status_label = tk.Label(
            self, 
            text="Ready", 
            font=('Arial', 10),
            foreground='#b0b0b0', 
            background=CARD_BG_COLOR
        )
        self.status_label.pack(side=tk.LEFT)
        
        # Status colors
        self.status_colors = {
            'ready': '#b0b0b0',
            'running': SUCCESS_COLOR,
            'error': ERROR_COLOR,
            'warning': WARNING_COLOR
        }
    
    def set_status(self, message: str, status_type: str = 'ready') -> None:
        """
        Set the status message and color.
        
        Args:
            message: Status message to display
            status_type: Type of status ('ready', 'running', 'error', 'warning')
        """
        self.status_label.config(
            text=message,
            foreground=self.status_colors.get(status_type, '#b0b0b0')
        )


class ProgressIndicator(tk.Frame):
    """
    Custom progress indicator for long-running operations.
    
    Provides a visual progress bar with percentage display for
    operations like simulation and animation.
    """
    
    def __init__(self, parent: tk.Widget, **kwargs):
        """
        Initialize the progress indicator.
        
        Args:
            parent: Parent widget
            **kwargs: Additional arguments for Frame
        """
        super().__init__(parent, **kwargs)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self, 
            variable=self.progress_var, 
            maximum=100,
            length=200
        )
        self.progress_bar.pack(side=tk.LEFT, padx=(0, 10))
        
        # Percentage label
        self.percent_label = tk.Label(
            self,
            text="0%",
            font=('Arial', 9),
            foreground=TEXT_COLOR,
            background=CARD_BG_COLOR
        )
        self.percent_label.pack(side=tk.LEFT)
    
    def set_progress(self, percentage: float) -> None:
        """
        Set the progress percentage.
        
        Args:
            percentage: Progress percentage (0-100)
        """
        percentage = max(0, min(100, percentage))  # Clamp to 0-100
        self.progress_var.set(percentage)
        self.percent_label.config(text=f"{percentage:.0f}%")
    
    def reset(self) -> None:
        """Reset progress to 0%."""
        self.set_progress(0)
    
    def complete(self) -> None:
        """Set progress to 100%."""
        self.set_progress(100)
