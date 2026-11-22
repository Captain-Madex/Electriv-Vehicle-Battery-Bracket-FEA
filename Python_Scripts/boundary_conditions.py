"""Boundary and load assembly for the EV bracket 2-D membrane model."""
import numpy as np


def build_force_vector(
    nodes,
    elements,
    load_nodes,
    total_force=294.3,
    direction=(0.0, -1.0)
):
    """
    Apply a consistent force only to external Q4 edges
    belonging to the selected BROWN_LOAD_EDGE component.
    """

    selected = set(int(i) for i in load_nodes)

    # Count how many elements share each edge.
    edge_count = {}

    for element in elements:
        for a, b in zip(element, np.roll(element, -1)):
            a = int(a)
            b = int(b)

            if a == b:
                raise ValueError(
                    f"Invalid element edge with repeated node {a}."
                )

            edge = tuple(sorted((a, b)))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    # An external boundary edge belongs to exactly one element.
    external_edges = {
        edge for edge, count in edge_count.items()
        if count == 1
    }

    # Select external edges whose two nodes belong to
    # BROWN_LOAD_EDGE.
    loaded_edges = sorted(
        edge for edge in external_edges
        if edge[0] in selected and edge[1] in selected
    )

    if not loaded_edges:
        raise ValueError(
            "No external Q4 edges found inside BROWN_LOAD_EDGE. "
            "Check the ANSYS component and mesh connectivity."
        )

    # Check for non-manifold edges.
    non_manifold_edges = [
        edge for edge, count in edge_count.items()
        if count > 2
    ]

    if non_manifold_edges:
        raise ValueError(
            f"Mesh contains {len(non_manifold_edges)} edges "
            "shared by more than two elements."
        )

    # Calculate external loaded-edge lengths.
    lengths = np.array([
        np.linalg.norm(nodes[a] - nodes[b])
        for a, b in loaded_edges
    ])

    total_length = float(lengths.sum())

    if total_length <= 0:
        raise ValueError("Loaded boundary has zero length.")

    # Normalize force direction.
    direction = np.asarray(direction, dtype=float)
    direction_norm = np.linalg.norm(direction)

    if direction_norm <= 0:
        raise ValueError("Force direction cannot be zero.")

    direction = direction / direction_norm

    # Global force vector: two DOFs per node.
    F = np.zeros(2 * len(nodes))

    # Constant line load along the selected external boundary.
    line_load = total_force / total_length

    # Consistent equivalent nodal force distribution.
    for (a, b), length in zip(loaded_edges, lengths):

        nodal_force = (
            line_load * length * 0.5 * direction
        )

        F[2*a:2*a+2] += nodal_force
        F[2*b:2*b+2] += nodal_force

    return F, {
        "segment_count": len(loaded_edges),
        "external_edge_count": len(external_edges),
        "loaded_edge_length_mm": total_length,
        "line_load_N_per_mm": line_load,
        "resultant_x_N": float(F[0::2].sum()),
        "resultant_y_N": float(F[1::2].sum()),
        "total_force_N": float(total_force)
    }

def build_fixed_dofs(components, n_nodes):
    required=('LEFT_FIXED','BOTTOM_FIXED')
    for name in required:
        if name not in components: raise KeyError(f'Missing required CDB component: {name}')
    fixed_nodes=np.unique(np.concatenate([components[n] for n in required])).astype(int)
    fixed=np.sort(np.ravel(np.column_stack((2*fixed_nodes,2*fixed_nodes+1))))
    if np.any(fixed<0) or np.any(fixed>=2*n_nodes): raise ValueError('Invalid fixed DOF index.')
    return fixed, fixed_nodes
