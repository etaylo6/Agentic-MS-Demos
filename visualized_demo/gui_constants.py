"""
GUI Constants and Configuration Settings

This module contains all the constants, configuration values, and default settings
used throughout the Timoshenko Beam GUI application. Centralizing these values
makes the application easier to maintain, customize, and modify.

Key responsibilities:
- Color scheme definitions for consistent theming
- Animation parameters for smooth visual effects
- Default values for beam model parameters
- Layout and sizing constants
- Demo configuration templates

Usage:
    from gui_constants import DEFAULT_WINDOW_SIZE, SUCCESS_COLOR
"""

# ============================================================================
# WINDOW AND LAYOUT CONSTANTS
# ============================================================================

# Main window configuration
DEFAULT_WINDOW_SIZE = (1400, 800)
DEFAULT_WINDOW_TITLE = "Timoshenko Beam Hypergraph Simulation"

# Layout spacing and dimensions
CONTROL_PANEL_WIDTH = 800
MAIN_FIGURE_SIZE = (5, 4)
MAIN_FIGURE_DPI = 100

# Padding and margins
MAIN_PADDING = 5
CONTROL_PADDING = 20
SECTION_SPACING = 12

# ============================================================================
# COLOR SCHEME CONSTANTS
# ============================================================================

# Primary colors for dark theme
DEFAULT_BG_COLOR = '#1a1a1a'      # Main background (very dark gray)
CARD_BG_COLOR = '#2d2d2d'         # Card/panel background (dark gray)
TEXT_COLOR = '#ffffff'            # Primary text color (white)

# Accent colors
ACCENT_COLOR = '#4a90e2'          # Primary accent (blue)
SUCCESS_COLOR = '#27ae60'         # Success/positive (green)
WARNING_COLOR = '#e67e22'         # Warning/attention (orange)
ERROR_COLOR = '#e74c3c'           # Error/danger (red)
GRAY_COLOR = '#666666'            # Neutral gray

# Status colors
STATUS_TEXT_COLOR = '#b0b0b0'     # Secondary text (light gray)
SEPARATOR_COLOR = '#666666'       # Separator lines

# Node state colors
NODE_SOURCE_COLOR = SUCCESS_COLOR  # Green for source nodes
NODE_TARGET_COLOR = WARNING_COLOR  # Orange for target nodes
NODE_NONE_COLOR = '#ababab'        # Gray for inactive nodes

# ============================================================================
# ANIMATION CONSTANTS
# ============================================================================

# Animation timing and performance
DEFAULT_FRAMES_PER_EDGE = 30      # Frames for each edge traversal
DEFAULT_FPS = 30                  # Frames per second
ANIMATION_INTERVAL = 200          # Toggle switch animation delay (ms)

# Moving circle animation
DEFAULT_CIRCLE_RADIUS = 0.08      # Radius of moving animation circle
DEFAULT_CIRCLE_COLOR = '#ff3333'  # Color of moving circle (red)
DEFAULT_CIRCLE_ALPHA = 0.9        # Transparency of moving circle

# Animation cleanup
CLEANUP_DELAY = 100               # Delay before animation cleanup (ms)

# ============================================================================
# BEAM MODEL CONSTANTS
# ============================================================================

# Note: Cantilever beam diagram has been removed from the GUI

# ============================================================================
# DEFAULT BEAM PARAMETERS
# ============================================================================

# Default values for Timoshenko beam model (realistic engineering values)
DEFAULT_BEAM_VALUES = {
    "point load": "1000",          # Applied load (N) - 1 kN, typical structural load
    "length": "2.0",               # Beam length (m) - 2m span, common in construction
    "youngs modulus": "200000000000.0",  # Young's modulus (Pa) - structural steel
    "moment of inertia": "0.0001", # Moment of inertia (m^4) - more realistic value
    "shear modulus": "80000000000.0",     # Shear modulus (Pa) - structural steel
    "area": "0.01",                # Cross-sectional area (m^2) - more realistic value
    "kappa": "5/6",                # Shear correction factor (5/6 = 0.833 for rectangular)
    "radius": "0.056",             # Radius (m) - calculated from I=0.0001
    "poisson": "0.3"               # Poisson's ratio - structural steel
}

# Default toggle states for nodes
DEFAULT_TOGGLE_STATES = {
    "point load": "source",
    "length": "source", 
    "youngs modulus": "source",
    "moment of inertia": "source",  # Provide I directly to avoid R→I calculation
    "shear modulus": "none",       # Don't provide G directly to avoid conflicts
    "area": "source",              # Provide A directly to avoid R→A calculation
    "kappa": "source",
    "radius": "none",              # Don't provide radius to avoid R→I and R→A calculations
    "poisson": "source",
    "theta": "target",             # Target variable: beam deflection angle
    "slenderness ratio": "none",
    "slenderness heuristic": "none",
}

# ============================================================================
# DEMO CONFIGURATIONS
# ============================================================================

# Demo 1: Basic deflection calculation
DEMO1_CONFIG = {
    'values': {
        "point load": "1000.0",        # 1000 N point load
        "length": "2.0",               # 2 m beam length
        "youngs modulus": "200000000000.0",  # 200 GPa
        "moment of inertia": "0.0001",       # 0.0001 m^4 (more realistic)
        "shear modulus": "80000000000.0",     # 80 GPa
        "area": "0.01",                # 0.01 m^2 cross-sectional area
        "kappa": "5/6",                # Shear correction factor
        "radius": "0.056",             # 0.056 m radius (for I=0.0001)
        "poisson": "0.3"               # Poisson's ratio
    },
    'toggles': {
        "point load": "source",
        "length": "source", 
        "youngs modulus": "source",
        "moment of inertia": "source",
        "shear modulus": "none",       # Calculate G from E and V
        "area": "source",
        "kappa": "source",
        "radius": "none",              # Don't provide radius to avoid R→I and R→A calculations
        "poisson": "source",
        "theta": "target"              # Calculate beam deflection from given parameters
    },
    'status': "Demo 1: Basic deflection calculation setup"
}

# Demo 2: Calculate point load from known deflection
DEMO2_CONFIG = {
    'values': {
        "theta": "0.001",             # Known deflection (1 mm)
        "length": "2.5",              # 2.5 m beam length
        "youngs modulus": "200000000000.0",  # 200 GPa
        "moment of inertia": "0.00012",      # 0.00012 m^4 (more realistic)
        "shear modulus": "80000000000.0",    # 80 GPa
        "area": "0.012",              # 0.012 m^2 cross-sectional area
        "kappa": "5/6",               # Shear correction factor
        "radius": "0.062",            # 0.062 m radius (for I=0.00012)
        "poisson": "0.3"              # Poisson's ratio
    },
    'toggles': {
        "theta": "source",            # Known deflection
        "length": "source", 
        "youngs modulus": "source",
        "moment of inertia": "source",
        "shear modulus": "none",       # Calculate G from E and V
        "area": "source",
        "kappa": "source",
        "radius": "none",              # Don't provide radius to avoid R→I and R→A calculations
        "poisson": "source",
        "point load": "target"        # Calculate required point load for given deflection
    },
    'status': "Demo 2: Point load from deflection setup"
}

# Demo 3: Poisson's ratio analysis
DEMO3_CONFIG = {
    'values': {
        "point load": "500.0",        # 500 N point load
        "length": "3.0",              # 3 m beam length
        "youngs modulus": "190000000000.0",  # 190 GPa
        "moment of inertia": "1.5e-06",      # 1.5e-6 m^4
        "shear modulus": "76000000000.0",    # 76 GPa
        "area": "0.0015",             # 0.0015 m^2 cross-sectional area
        "kappa": "0.85",              # Shear correction factor
        "radius": "0.012",            # 0.012 m radius
        "theta": "0.0005"             # Known deflection (0.5 mm)
    },
    'toggles': {
        "point load": "source",
        "length": "source", 
        "youngs modulus": "source",
        "moment of inertia": "source",
        "shear modulus": "source",
        "area": "source",
        "kappa": "source",
        "radius": "source",
        "theta": "source",            # Known deflection
        "poisson": "target"           # Calculate Poisson's ratio from material properties
    },
    'status': "Demo 3: Poisson's ratio analysis setup"
}

# ============================================================================
# NODE GROUPING FOR ORGANIZATION
# ============================================================================

# Logical grouping of nodes for better UI organization
MECHANICAL_PROPERTIES = ["youngs modulus", "shear modulus", "poisson", "kappa", "slenderness heuristic"]
GEOMETRY_PROPERTIES = ["length", "area", "moment of inertia", "radius", "slenderness ratio"]
LOADING_OUTPUT_PROPERTIES = ["point load", "theta"]

# ============================================================================
# VALIDATION CONSTANTS
# ============================================================================

# Minimum requirements for simulation
MIN_SOURCE_NODES = 3               # Minimum number of source nodes required
MAX_SIGNIFICANT_FIGURES = 3        # Precision limit for displaying numerical results

# ============================================================================
# FONT AND STYLING CONSTANTS
# ============================================================================

# Font configurations
TITLE_FONT = ('Arial', 16, 'bold')
SECTION_FONT = ('Arial', 12, 'bold')
BUTTON_FONT = ('Arial', 10, 'bold')
DEMO_BUTTON_FONT = ('Arial', 9)
STATUS_FONT = ('Arial', 10)
NODE_LABEL_FONT = ('Arial', 11)
TOGGLE_FONT = ('Arial', 9, 'bold')

# Node text styling
NODE_TEXT_FONT = ('Arial', 7, 'bold')
NODE_TEXT_COLOR = 'white'
NODE_TEXT_ZORDER = 1000

# ============================================================================
# SEMANTIC GROUPING FOR FUNCTION NODES (BIPARTITE VIEW)
# ============================================================================

# Map exact edge labels to semantic group names for visualization merging.
# Leave empty to disable explicit grouping. Example:
# SEMANTIC_GROUPS = {
#     'TimoshenkoDeflection': 'Timoshenko',
#     'AreaFromRadius': 'Area of Circle',
# }
SEMANTIC_GROUPS = {}
