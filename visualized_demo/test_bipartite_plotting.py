"""
Test script for bipartite hypergraph plotting functionality.

This script creates a test hypergraph and demonstrates the bipartite
plotting capabilities, including static visualization and animation.
"""

import matplotlib.pyplot as plt
import numpy as np
from variant_beam_model import create_timoshenko_hypergraph
from plothg_bipartite import (
    BipartitePlotSettings, 
    plot_bipartite_hypergraph, 
    create_bipartite_animation
)

def test_bipartite_static_plot():
    """
    Test the static bipartite plotting functionality.
    """
    print("Testing bipartite static plotting...")
    
    # Create hypergraph
    hg = create_timoshenko_hypergraph()
    
    # Create plot settings
    ps = BipartitePlotSettings()
    
    # Create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle('Bipartite Hypergraph Visualization - Timoshenko Beam Model', fontsize=16, fontweight='bold')
    
    # Plot the bipartite hypergraph
    shapes, lines, positions = plot_bipartite_hypergraph(hg, ax, ps)
    
    # Add title and labels
    ax.text(-1.5, 2.0, 'Source Nodes\n(Input Variables)', ha='center', va='center', 
            fontsize=14, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    ax.text(1.5, 2.0, 'Target Nodes\n(Output Variables)', ha='center', va='center', 
            fontsize=14, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.7))
    
    # Add legend
    legend_elements = [
        plt.Circle((0, 0), 0.1, facecolor='#ababab', edgecolor='#666666', label='Default Node'),
        plt.Circle((0, 0), 0.1, facecolor='#27ae60', edgecolor='#1e8449', label='Source Node'),
        plt.Circle((0, 0), 0.1, facecolor='#e67e22', edgecolor='#d35400', label='Target Node'),
        plt.Line2D([0], [0], color='#cccccc44', linewidth=1, label='Default Edge'),
        plt.Line2D([0], [0], color='#0033cc', linewidth=2, label='Active Edge'),
        plt.Line2D([0], [0], color='#00118888', linewidth=2, label='Solved Edge')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.show()
    
    print(f"Static plot created with {len(shapes)} nodes and {len(lines)} edges")
    print("Node positions:")
    for node_label, (x, y) in positions.items():
        print(f"  {node_label}: ({x:.2f}, {y:.2f})")

def test_bipartite_animation():
    """
    Test the bipartite animation functionality.
    """
    print("\nTesting bipartite animation...")
    
    # Create hypergraph
    hg = create_timoshenko_hypergraph()
    
    # Create plot settings
    ps = BipartitePlotSettings()
    ps.animation['frames_per_edge'] = 20  # Faster animation for testing
    ps.animation['fps'] = 15  # Slower FPS for testing
    
    # Create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle('Bipartite Hypergraph Animation - Timoshenko Beam Model', fontsize=16, fontweight='bold')
    
    # Plot the static bipartite hypergraph first
    shapes, lines, positions = plot_bipartite_hypergraph(hg, ax, ps)
    
    # Add title and labels
    ax.text(-1.5, 2.0, 'Source Nodes\n(Input Variables)', ha='center', va='center', 
            fontsize=14, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    ax.text(1.5, 2.0, 'Target Nodes\n(Output Variables)', ha='center', va='center', 
            fontsize=14, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.7))
    
    # Create a text box for status updates
    text_box = ax.text(0, -2.2, '', ha='center', va='center', fontsize=12,
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
    
    # Simulate a solution path (this would normally come from hypergraph.solve)
    # For testing, we'll create a mock solution
    mock_tnodes = []
    mock_inputs = ['point load', 'length', 'youngs modulus', 'moment of inertia', 'area', 'kappa', 'poisson']
    mock_output = 'theta'
    
    # Create mock TNode objects for testing
    class MockTNode:
        def __init__(self, node_label, gen_edge_label=None):
            self.node_label = node_label
            self.gen_edge_label = gen_edge_label
            self.children = []
    
    # Create a simple solution path
    mock_tnodes.append(MockTNode('shear modulus', 'E,V->G'))
    mock_tnodes.append(MockTNode('theta', 'Timoshenko'))
    
    # Create animation
    animation = create_bipartite_animation(
        hg, mock_tnodes, mock_inputs, mock_output,
        shapes, lines, positions, ps, fig, ax, text_box
    )
    
    print("Animation created successfully!")
    print("Close the plot window to continue...")
    
    plt.tight_layout()
    plt.show()
    
    return animation


def test_merged_hyperedges_animation():
    """
    Test the bipartite animation functionality with merged hyperedges.
    """
    print("\nTesting merged hyperedges animation...")
    
    # Create hypergraph
    hg = create_timoshenko_hypergraph()
    
    # Create plot settings with merge enabled
    ps = BipartitePlotSettings()
    ps.merge['merge_equivalent_hyperedges'] = True
    ps.merge['merged_node_prefix'] = 'Timoshenko'
    ps.merge['merged_node_style'] = 'diamond'
    ps.animation['frames_per_edge'] = 20  # Faster animation for testing
    ps.animation['fps'] = 15  # Slower FPS for testing
    
    # Customize merged node appearance
    ps.node_target.update(facecolor='#9b59b6', edgecolor='#8e44ad', linewidth=3)  # Purple for merged nodes
    
    # Create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle('Bipartite Animation with Merged Timoshenko Hyperedges', fontsize=16, fontweight='bold')
    
    # Plot the static bipartite hypergraph first
    shapes, lines, positions = plot_bipartite_hypergraph(hg, ax, ps)
    
    # Add title and labels
    ax.text(-1.5, 2.0, 'Source Nodes\n(Input Variables)', ha='center', va='center', 
            fontsize=14, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    ax.text(1.5, 2.0, 'Target Nodes\n(Output Variables)', ha='center', va='center', 
            fontsize=14, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.7))
    
    # Create a text box for status updates
    text_box = ax.text(0, -2.2, '', ha='center', va='center', fontsize=12,
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
    
    # Simulate a solution path (this would normally come from hypergraph.solve)
    # For testing, we'll create a mock solution
    mock_tnodes = []
    mock_inputs = ['point load', 'length', 'youngs modulus', 'moment of inertia', 'area', 'kappa', 'poisson']
    mock_output = 'theta'
    
    # Create mock TNode objects for testing
    class MockTNode:
        def __init__(self, node_label, gen_edge_label=None):
            self.node_label = node_label
            self.gen_edge_label = gen_edge_label
            self.children = []
    
    # Create a simple solution path that includes Timoshenko edges
    mock_tnodes.append(MockTNode('shear modulus', 'E,V->G'))
    mock_tnodes.append(MockTNode('theta', 'Timoshenko'))  # This should use the merged node
    
    # Create animation
    animation = create_bipartite_animation(
        hg, mock_tnodes, mock_inputs, mock_output,
        shapes, lines, positions, ps, fig, ax, text_box
    )
    
    print("Merged hyperedges animation created successfully!")
    print("Close the plot window to continue...")
    
    plt.tight_layout()
    plt.show()
    
    return animation

def test_bipartite_layout_analysis():
    """
    Test and analyze the bipartite layout generation.
    """
    print("\nAnalyzing bipartite layout...")
    
    # Create hypergraph
    hg = create_timoshenko_hypergraph()
    
    # Create plot settings
    ps = BipartitePlotSettings()
    
    # Import the layout function
    from plothg_bipartite import create_bipartite_layout
    
    # Create layout
    positions = create_bipartite_layout(hg, ps)
    
    # Analyze the layout
    print(f"Total nodes: {len(positions)}")
    
    # Separate into source and target nodes
    source_nodes = []
    target_nodes = []
    
    for node_label, (x, y) in positions.items():
        if x < 0:  # Left column (source nodes)
            source_nodes.append(node_label)
        else:  # Right column (target nodes)
            target_nodes.append(node_label)
    
    print(f"Source nodes ({len(source_nodes)}): {source_nodes}")
    print(f"Target nodes ({len(target_nodes)}): {target_nodes}")
    
    # Analyze edges
    print(f"\nEdge analysis:")
    for edge in hg.edges.values():
        source_labels = [sn.label for sn in edge.source_nodes.values()]
        target_label = edge.target.label
        print(f"  {edge.label}: {source_labels} -> {target_label}")
    
    return positions

def test_plot_settings_customization():
    """
    Test customizing the plot settings.
    """
    print("\nTesting plot settings customization...")
    
    # Create hypergraph
    hg = create_timoshenko_hypergraph()
    
    # Create custom plot settings
    ps = BipartitePlotSettings()
    
    # Customize colors
    ps.node_source.update(facecolor='#2ecc71', edgecolor='#27ae60')  # Different green
    ps.node_target.update(facecolor='#e74c3c', edgecolor='#c0392b')  # Different red
    ps.edge_active.update(color='#f39c12', linewidth=4)  # Orange active edges
    
    # Customize layout
    ps.bipartite.update(
        left_column_x=-2.0,  # Wider separation
        right_column_x=2.0,
        node_spacing=0.5,    # More spacing
        edge_curvature=0.5   # More curved edges
    )
    
    # Create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.suptitle('Customized Bipartite Hypergraph', fontsize=16, fontweight='bold')
    
    # Plot with custom settings
    shapes, lines, positions = plot_bipartite_hypergraph(hg, ax, ps)
    
    # Add title and labels
    ax.text(-2.0, 2.5, 'Source Nodes\n(Input Variables)', ha='center', va='center', 
            fontsize=14, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    ax.text(2.0, 2.5, 'Target Nodes\n(Output Variables)', ha='center', va='center', 
            fontsize=14, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    print("Customized plot created successfully!")


def test_merged_hyperedges():
    """
    Test the merged hyperedges functionality.
    """
    print("\nTesting merged hyperedges functionality...")
    
    # Create hypergraph
    hg = create_timoshenko_hypergraph()
    
    # Create plot settings with merge enabled
    ps = BipartitePlotSettings()
    ps.merge['merge_equivalent_hyperedges'] = True
    ps.merge['merged_node_prefix'] = 'Timoshenko'
    ps.merge['merged_node_style'] = 'diamond'
    
    # Customize merged node appearance
    ps.node_target.update(facecolor='#9b59b6', edgecolor='#8e44ad', linewidth=3)  # Purple for merged nodes
    
    # Create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.suptitle('Bipartite Hypergraph with Merged Timoshenko Hyperedges', fontsize=16, fontweight='bold')
    
    # Plot with merged settings
    shapes, lines, positions = plot_bipartite_hypergraph(hg, ax, ps)
    
    # Add title and labels
    ax.text(-1.5, 2.0, 'Source Nodes\n(Input Variables)', ha='center', va='center', 
            fontsize=14, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    ax.text(1.5, 2.0, 'Target Nodes\n(Output Variables)', ha='center', va='center', 
            fontsize=14, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.7))
    
    # Add legend including merged nodes
    legend_elements = [
        plt.Circle((0, 0), 0.1, facecolor='#ababab', edgecolor='#666666', label='Variable Node'),
        plt.Rectangle((0, 0), 0.1, 0.1, facecolor='#e67e22', edgecolor='#d35400', label='Function Node'),
        plt.Polygon([(0, 0.1), (0.1, 0), (0, -0.1), (-0.1, 0)], facecolor='#9b59b6', edgecolor='#8e44ad', label='Merged Node'),
        plt.Line2D([0], [0], color='#cccccc44', linewidth=1, label='Edge'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.show()
    
    print("Merged hyperedges plot created successfully!")
    print(f"Total nodes in plot: {len(shapes)}")
    print(f"Total edges in plot: {len(lines)}")
    
    # Print information about merged nodes
    from plothg_bipartite import merge_equivalent_hyperedges
    merged_nodes = merge_equivalent_hyperedges(hg, ps)
    if merged_nodes:
        print(f"\nMerged nodes found: {len(merged_nodes)}")
        for merged_label, edge_labels in merged_nodes.items():
            print(f"  {merged_label}: {len(edge_labels)} edges merged")
            print(f"    Original edges: {edge_labels}")
    else:
        print("No merged nodes found.")

def main():
    """
    Main test function that runs all tests.
    """
    print("=" * 60)
    print("BIPARTITE HYPERGRAPH PLOTTING TESTS")
    print("=" * 60)
    
    try:
        # Test 1: Static plotting
        test_bipartite_static_plot()
        
        # Test 2: Layout analysis
        test_bipartite_layout_analysis()
        
        # Test 3: Customized settings
        test_plot_settings_customization()
        
        # Test 4: Merged hyperedges (NEW FEATURE)
        print("\n" + "=" * 60)
        print("MERGED HYPEREEDGES TEST")
        print("=" * 60)
        test_merged_hyperedges()
        
        # Test 5: Merged hyperedges animation (NEW FEATURE)
        print("\n" + "=" * 60)
        print("MERGED HYPEREEDGES ANIMATION TEST (Close plot window to continue)")
        print("=" * 60)
        test_merged_hyperedges_animation()
        
        # Test 6: Regular animation (optional - comment out if you don't want to see animation)
        print("\n" + "=" * 60)
        print("REGULAR ANIMATION TEST (Close plot window to continue)")
        print("=" * 60)
        test_bipartite_animation()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
