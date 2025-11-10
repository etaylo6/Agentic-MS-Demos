"""
GUI Layout and UI Management Module

This module handles all aspects of GUI layout, window management, and UI organization
for the Timoshenko Beam application. It provides a clean separation between the
visual layout and the application logic, making the interface easier to maintain
and modify.

Key responsibilities:
- Main window setup and configuration
- Layout creation and organization
- Styling and theme management
- Cantilever beam diagram creation
- Widget positioning and sizing
- Style configuration for ttk widgets

The module creates a responsive layout with a control panel on the left and
visualization area on the right. It maintains consistent styling throughout
the application and provides a professional, cohesive user interface.

Usage:
    from gui_layout import LayoutManager
    layout = LayoutManager(root)
    layout.create_main_layout()
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from typing import Any, Optional

from gui_constants import (
    DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_TITLE, DEFAULT_BG_COLOR, CARD_BG_COLOR,
    TEXT_COLOR, ACCENT_COLOR, GRAY_COLOR, STATUS_TEXT_COLOR, SEPARATOR_COLOR,
    CONTROL_PANEL_WIDTH, MAIN_FIGURE_SIZE, MAIN_FIGURE_DPI, MAIN_PADDING, 
    CONTROL_PADDING, SECTION_SPACING, TITLE_FONT, SECTION_FONT, BUTTON_FONT, 
    DEMO_BUTTON_FONT, STATUS_FONT
)


class LayoutManager:
    """
    Manages the overall layout and styling of the GUI application.
    
    This class handles window setup, layout creation, styling configuration,
    and the creation of specialized UI components like the cantilever beam
    diagram. It provides a centralized way to manage the visual appearance
    and organization of the entire application.
    
    Features:
    - Responsive window layout
    - Consistent styling and theming
    - Professional cantilever beam diagram
    - Organized control panel layout
    - Integrated visualization area
    - Style configuration for all widgets
    
    Attributes:
        root: Main Tkinter root window
        main_container: Main container frame
        content_frame: Content area frame
        control_frame: Left control panel frame
        viz_frame: Right visualization frame
        beam_fig: Matplotlib figure for beam diagram
        beam_ax: Matplotlib axes for beam diagram
        beam_canvas: Canvas for beam diagram
    """
    
    def __init__(self, root: tk.Tk):
        """
        Initialize the layout manager.
        
        Args:
            root: Main Tkinter root window
        """
        self.root = root
        self.main_container = None
        self.content_frame = None
        self.control_frame = None
        self.viz_frame = None
        self.beam_fig = None
        self.beam_ax = None
        self.beam_canvas = None
        
        # Configure window and styles
        self._setup_window()
        self._configure_styles()
    
    def _setup_window(self) -> None:
        """
        Set up the main window configuration.
        
        Configures window size, title, background color, and basic properties.
        """
        self.root.title(DEFAULT_WINDOW_TITLE)
        self.root.geometry(f"{DEFAULT_WINDOW_SIZE[0]}x{DEFAULT_WINDOW_SIZE[1]}")
        self.root.configure(bg=DEFAULT_BG_COLOR)
        
        # Set minimum window size
        self.root.minsize(800, 600)
    
    def _configure_styles(self) -> None:
        """
        Configure custom styles for ttk widgets.
        
        Sets up a cohesive dark theme with consistent colors and fonts
        for all ttk widgets throughout the application.
        """
        style = ttk.Style()
        
        # Apply consistent dark theme styling to all ttk widgets
        style.configure('Title.TLabel', 
                       font=TITLE_FONT, 
                       foreground=TEXT_COLOR, 
                       background=CARD_BG_COLOR)
        
        style.configure('Section.TLabel', 
                       font=SECTION_FONT, 
                       foreground=TEXT_COLOR, 
                       background=CARD_BG_COLOR)
        
        style.configure('Status.TLabel', 
                       font=STATUS_FONT, 
                       foreground=STATUS_TEXT_COLOR, 
                       background=CARD_BG_COLOR)
        
        style.configure('Custom.TButton', 
                       font=BUTTON_FONT, 
                       foreground=TEXT_COLOR, 
                       background=ACCENT_COLOR)
        
        style.configure('Demo.TButton', 
                       font=DEMO_BUTTON_FONT, 
                       foreground=TEXT_COLOR, 
                       background=GRAY_COLOR)
        
        # Set up frame styling to match the dark theme
        style.configure('Card.TFrame', 
                       relief='flat', 
                       borderwidth=0, 
                       background=CARD_BG_COLOR)
        
        style.configure('Input.TFrame', 
                       relief='flat', 
                       borderwidth=0, 
                       background=CARD_BG_COLOR)
    
    def create_main_layout(self) -> tuple:
        """
        Create the main application layout.
        
        Creates the primary layout structure with control panel on the left
        and visualization area on the right.
        
        Returns:
            tuple: (control_frame, viz_frame) for further customization
        """
        # Create main application container with minimal border spacing
        self.main_container = tk.Frame(self.root, bg=DEFAULT_BG_COLOR)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=MAIN_PADDING, pady=MAIN_PADDING)
        
        # Main content frame
        self.content_frame = tk.Frame(self.main_container, bg=DEFAULT_BG_COLOR)
        self.content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create left control panel with fixed width for input fields
        self.control_frame = tk.Frame(self.content_frame, bg=CARD_BG_COLOR, relief='flat', bd=0)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        self.control_frame.configure(width=CONTROL_PANEL_WIDTH)
        self.control_frame.pack_propagate(False)
        
        # Create right panel for hypergraph visualization display
        self.viz_frame = tk.Frame(self.content_frame, bg=CARD_BG_COLOR, relief='flat', bd=0)
        self.viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(0, 0), pady=0)
        
        return self.control_frame, self.viz_frame
    
    def create_control_panel_layout(self, parent: tk.Widget) -> dict:
        """
        Create the control panel layout structure.
        
        Creates organized sections for node inputs, control buttons, demo buttons,
        and status display.
        
        Args:
            parent: Parent widget for the control panel
            
        Returns:
            dict: Dictionary containing references to layout sections
        """
        layout_sections = {}
        
        # Create dedicated container for beam parameter input fields
        inputs_section = tk.Frame(parent, bg=CARD_BG_COLOR, relief='flat', bd=0)
        inputs_section.pack(fill=tk.BOTH, expand=True, padx=CONTROL_PADDING, pady=(10, 15))
        layout_sections['inputs_section'] = inputs_section
        
        # Control buttons section
        button_section = tk.Frame(parent, bg=CARD_BG_COLOR, relief='flat', bd=0)
        button_section.pack(fill=tk.X, padx=CONTROL_PADDING, pady=(0, 15))
        
        # Controls header
        controls_header = tk.Label(button_section, text="Controls", 
                                 font=SECTION_FONT, 
                                 foreground=TEXT_COLOR, background=CARD_BG_COLOR)
        controls_header.pack(pady=(0, 15))
        
        # Main action buttons frame
        action_frame = tk.Frame(button_section, bg=CARD_BG_COLOR)
        action_frame.pack(fill=tk.X, pady=(0, 20))
        layout_sections['action_frame'] = action_frame
        
        # Demo buttons section
        demo_header = tk.Label(button_section, text="Demos", 
                              font=SECTION_FONT, 
                              foreground=TEXT_COLOR, background=CARD_BG_COLOR)
        demo_header.pack(pady=(0, 10))
        layout_sections['demo_section'] = button_section
        
        # Status section
        status_frame = tk.Frame(parent, bg=CARD_BG_COLOR)
        status_frame.pack(fill=tk.X, padx=CONTROL_PADDING, pady=(0, 20))
        layout_sections['status_frame'] = status_frame
        
        return layout_sections
    
    
    def create_visualization_panel(self, parent: tk.Widget) -> tuple:
        """
        Create the matplotlib visualization panel.
        
        Sets up the main visualization area for the hypergraph display
        with proper styling and configuration.
        
        Args:
            parent: Parent widget for the visualization panel
            
        Returns:
            tuple: (fig, ax, canvas) for matplotlib operations
        """
        # Initialize matplotlib figure with optimal dimensions for the layout
        fig = Figure(figsize=MAIN_FIGURE_SIZE, dpi=MAIN_FIGURE_DPI, facecolor=CARD_BG_COLOR)
        
        # Remove figure padding and borders
        fig.patch.set_facecolor(CARD_BG_COLOR)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Create primary hypergraph visualization area
        ax = fig.add_subplot(111, facecolor=CARD_BG_COLOR)
        ax.set_facecolor(CARD_BG_COLOR)
        
        # Create matplotlib canvas without padding to prevent visual borders
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # Configure main plot
        ax.set_aspect('equal', adjustable='box')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        # Add subtle grid
        ax.grid(True, alpha=0.1, linestyle='-', linewidth=0.5)
        
        return fig, ax, canvas
    
    def create_section_header(self, parent: tk.Widget, title: str) -> None:
        """
        Create a section header with consistent styling.
        
        Args:
            parent: Parent widget to contain the header
            title: The title text for the section
        """
        # Insert vertical spacing before section header
        spacer = tk.Frame(parent, height=SECTION_SPACING, bg=CARD_BG_COLOR)
        spacer.pack(fill=tk.X)
        
        # Create section header with consistent typography
        header_label = tk.Label(parent, text=title, 
                               font=SECTION_FONT, 
                               foreground=TEXT_COLOR, background=CARD_BG_COLOR)
        header_label.pack(pady=(SECTION_SPACING, 8), padx=15)
        
        # Add a subtle separator line
        separator = tk.Frame(parent, height=1, bg=SEPARATOR_COLOR)
        separator.pack(fill=tk.X, pady=(0, SECTION_SPACING), padx=15)
    
    def get_control_frame(self) -> tk.Widget:
        """
        Get the control frame for adding widgets.
        
        Returns:
            tk.Widget: The control frame
        """
        return self.control_frame
    
    def get_visualization_frame(self) -> tk.Widget:
        """
        Get the visualization frame for adding widgets.
        
        Returns:
            tk.Widget: The visualization frame
        """
        return self.viz_frame
