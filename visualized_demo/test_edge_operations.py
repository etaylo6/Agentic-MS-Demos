import logging
import numpy as np
from constrainthg.hypergraph import Hypergraph, Node
import beam_model_def as bmd
import edge_grouper
import trivial_trimmer
def main():
    hg, nodes = bmd.create_beam_model()
    # Based on the label defined in the hg definition
    edge_groups = edge_grouper.group_edges_by_label(hg)
    source_dict = {
        nodes['P']: 1000.0,
        nodes['E']: 200e9,
        nodes['L']: 5.0,
        nodes['I']: 0.002,
    }
    solve_target = nodes['w']
    # Fundamental groups are property relationships that form tight bidirectional cycles
    # These are NOT the engineering model - just material/geometry properties
    # We remove edges from these groups where sources aren't fully defined
    # This breaks cycles while keeping the actual beam theory equations intact
    fundamental_groups = ['material_properties', 'geometry_area', 'geometry_moi']
    # Perform a trivial trim: remove edges that solve for already-known nodes
    trivial_result = trivial_trimmer.trim_trivial_edges(
        hg,
        edge_groups,
        inputs=source_dict,
        skip_groups=('other',),
    )
    hg2 = trivial_result.hypergraph
    edge_groups = edge_grouper.group_edges_by_label(hg2)
    hg2.set_logging_level(logging.DEBUG)
    result = hg2.solve(
        solve_target,
        source_dict,
        to_print=True,
    )

    
if __name__ == "__main__":
    main()

