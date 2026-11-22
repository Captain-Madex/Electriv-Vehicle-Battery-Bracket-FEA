"""
ANSYS 18.2 CDB mesh reader for the EV battery bracket FEM project.

Supported model:
    - 2D X-Y mesh
    - Four-node SHELL181 quadrilateral elements
    - ANSYS element type IDs 1, 2 and 3
    - Material IDs and section IDs
    - Named ANSYS node components
    - Shell section thickness extraction

Excluded:
    - Contact elements
    - Target elements
    - Surface-effect elements
    - Non-quadrilateral elements

The resulting mesh is used by the project's 2D Q4 plane-stress
FEM solver. This reader does not implement shell rotational DOFs
or shell bending behaviour.
"""

from pathlib import Path
from collections import defaultdict

import numpy as np


# ANSYS element type IDs mapped to SHELL181 in this project.
SHELL_ELEMENT_TYPES = {1, 2, 3}


# ------------------------------------------------------------
# FILE READER
# ------------------------------------------------------------

def PathLikeRead(filename):
    """Read an ANSYS CDB file and return its lines."""

    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(
            f"ANSYS CDB mesh file not found: {filename}"
        )

    with filename.open(
        "r",
        encoding="utf-8",
        errors="replace"
    ) as file:
        return file.readlines()


# ------------------------------------------------------------
# MAIN ANSYS CDB READER
# ------------------------------------------------------------

def read_ansys_cdb(filename):

    lines = PathLikeRead(filename)

    # Locate ANSYS mesh blocks.
    nblock = next(
        (i for i, line in enumerate(lines)
         if line.startswith("NBLOCK")),
        None
    )

    eblock = next(
        (i for i, line in enumerate(lines)
         if line.startswith("EBLOCK")),
        None
    )

    if nblock is None or eblock is None:
        raise ValueError(
            "CDB file must contain both NBLOCK and EBLOCK."
        )

    if nblock >= eblock:
        raise ValueError(
            "Invalid CDB block order: NBLOCK must precede EBLOCK."
        )

    # --------------------------------------------------------
    # 1. READ NODES
    # --------------------------------------------------------

    nodes = {}

    for line_number, line in enumerate(
        lines[nblock + 2:eblock],
        start=nblock + 3
    ):

        if not line.strip():
            continue

        if line.startswith("N,"):
            continue

        try:
            nid = int(line[:9].strip())

            x = float(line[27:48].strip())

            # ANSYS may omit trailing zero coordinate fields.
            y = (
                float(line[48:69].strip())
                if len(line.rstrip("\n")) >= 69
                else 0.0
            )

            z = (
                float(line[69:90].strip())
                if len(line.rstrip("\n")) >= 90
                else 0.0
            )

        except (ValueError, IndexError):
            # Ignore non-node records and formatting lines.
            continue

        # Important: validate Z outside the parsing try/except.
        # Otherwise this ValueError could be swallowed.
        if abs(z) > 1e-9:
            raise ValueError(
                f"Node {nid} has nonzero Z={z}. "
                "This reader supports only planar X-Y meshes."
            )

        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError(
                f"Node {nid} contains invalid coordinates."
            )

        if nid in nodes:
            raise ValueError(
                f"Duplicate ANSYS node ID found: {nid}"
            )

        nodes[nid] = (x, y)

    if not nodes:
        raise ValueError(
            "No valid nodes were found in the NBLOCK."
        )

    # --------------------------------------------------------
    # 2. READ ELEMENTS
    # --------------------------------------------------------

    elements = []
    element_ids = []
    material_ids = []
    section_ids = []

    degenerate_shell_ids = []

    unsupported_element_types = defaultdict(int)
    unsupported_node_counts = defaultdict(int)

    for line in lines[eblock + 2:]:

        if line.strip() == "-1":
            break

        # Read the fixed-width, 9-character EBLOCK fields.
        vals = []

        record = line.rstrip("\r\n")

        for j in range(0, len(record), 9):

            field = record[j:j + 9]

            if not field.strip():
                continue

            try:
                vals.append(int(field.strip()))
            except ValueError:
                continue

        if len(vals) < 15:
            continue

        # ----------------------------------------------------
        # EBLOCK FIELD MAPPING
        # ----------------------------------------------------

        etype = vals[0]
        material_id = vals[1]
        real_id = vals[2]
        section_id = vals[3]

        nnode = vals[8]

        # ----------------------------------------------------
        # ELEMENT TYPE FILTER
        # ----------------------------------------------------

        # Keep only SHELL181 element types 1, 2 and 3.
        if etype not in SHELL_ELEMENT_TYPES:

            unsupported_element_types[etype] += 1
            continue

        # ----------------------------------------------------
        # NODE COUNT FILTER
        # ----------------------------------------------------

        # The current Q4 solver supports four-node elements.
        if nnode != 4:

            unsupported_node_counts[nnode] += 1
            continue

        # ----------------------------------------------------
        # ELEMENT ID AND CONNECTIVITY
        # ----------------------------------------------------

        eid = vals[10]

        conn = vals[11:15]

        if len(conn) != 4:
            raise ValueError(
                f"Element {eid} does not contain four node IDs."
            )

        # Check for repeated nodes.
        if len(set(conn)) != 4:

            degenerate_shell_ids.append(eid)
            continue

        # Check that every connectivity node exists.
        missing_nodes = [
            node_id for node_id in conn
            if node_id not in nodes
        ]

        if missing_nodes:
            raise ValueError(
                f"Element {eid} references missing nodes: "
                f"{missing_nodes}"
            )

        # Store valid element information.
        elements.append(conn)
        element_ids.append(eid)

        material_ids.append(material_id)
        section_ids.append(section_id)

    if not elements:
        raise ValueError(
            "No valid four-node SHELL181 quadrilateral elements found."
        )

    # --------------------------------------------------------
    # 3. CONVERT ANSYS NODE IDs TO PYTHON INDICES
    # --------------------------------------------------------

    node_ids = np.array(
        sorted(nodes.keys()),
        dtype=int
    )

    coords = np.array(
        [nodes[nid] for nid in node_ids],
        dtype=float
    )

    id_to_idx = {
        int(node_id): index
        for index, node_id in enumerate(node_ids)
    }

    conn = np.array(
        [
            [id_to_idx[node_id] for node_id in element]
            for element in elements
        ],
        dtype=int
    )

    element_ids = np.array(
        element_ids,
        dtype=int
    )

    material_ids = np.array(
        material_ids,
        dtype=int
    )

    section_ids = np.array(
        section_ids,
        dtype=int
    )

    # --------------------------------------------------------
    # 4. CHECK ELEMENT ORIENTATION
    # --------------------------------------------------------

    # Ensure counter-clockwise ordering.
    # This supports positive Jacobian values in the Q4 solver.

    for k, element in enumerate(conn):

        xy = coords[element]

        x = xy[:, 0]
        y = xy[:, 1]

        area = 0.5 * np.sum(
            x * np.roll(y, -1)
            -
            np.roll(x, -1) * y
        )

        if abs(area) < 1e-12:
            raise ValueError(
                f"Element {element_ids[k]} has zero or "
                "near-zero area."
            )

        if area < 0:
            conn[k] = element[::-1]

    # --------------------------------------------------------
    # 5. READ ANSYS NODE COMPONENTS
    # --------------------------------------------------------

    components = read_components(lines)

    component_indices = {}

    for name, ids in components.items():

        missing = [
            node_id for node_id in ids
            if node_id not in id_to_idx
        ]

        if missing:
            raise ValueError(
                f"Component '{name}' references missing "
                f"node IDs: {missing[:10]}"
            )

        component_indices[name] = np.array(
            [id_to_idx[node_id] for node_id in ids],
            dtype=int
        )

    # --------------------------------------------------------
    # 6. READ SHELL SECTION THICKNESSES
    # --------------------------------------------------------

    thickness = parse_shell_thickness(lines)

    # --------------------------------------------------------
    # 7. PRINT MESH SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print("ANSYS CDB MESH READER SUMMARY")
    print("=" * 55)

    print(f"Mesh file                  : {filename}")
    print(f"Number of nodes            : {len(node_ids)}")
    print(f"Valid Q4 shell elements    : {len(conn)}")
    print(f"Degenerate shell elements  : {len(degenerate_shell_ids)}")
    print(f"Number of node components  : {len(components)}")

    print("\nMaterial IDs:")

    for mid in np.unique(material_ids):
        count = np.count_nonzero(material_ids == mid)
        print(f"  Material {mid}: {count} elements")

    print("\nSection IDs:")

    for sid in np.unique(section_ids):
        count = np.count_nonzero(section_ids == sid)
        print(f"  Section {sid}: {count} elements")

    if thickness:
        print("\nSection thicknesses:")

        for sid, value in thickness.items():
            print(f"  Section {sid}: {value}")

    if unsupported_element_types:
        print("\nExcluded element types:")

        for etype, count in sorted(
            unsupported_element_types.items()
        ):
            print(f"  Type {etype}: {count} elements")

    if unsupported_node_counts:
        print("\nExcluded node counts:")

        for nnode, count in sorted(
            unsupported_node_counts.items()
        ):
            print(f"  {nnode} nodes: {count} elements")

    if degenerate_shell_ids:
        print(
            "\nWarning: degenerate shell elements were excluded:"
        )
        print(degenerate_shell_ids[:20])

    print("=" * 55 + "\n")

    # --------------------------------------------------------
    # 8. RETURN MESH DATA
    # --------------------------------------------------------

    return {

        "nodes": coords,

        "node_ids": node_ids,

        "elements": conn,

        "element_ids": element_ids,

        "material_ids": material_ids,

        "section_ids": section_ids,

        "degenerate_shell_element_ids": np.array(
            degenerate_shell_ids,
            dtype=int
        ),

        "components": component_indices,

        "component_node_ids": components,

        "section_thickness": thickness,

        "unsupported_element_types": dict(
            unsupported_element_types
        ),

        "unsupported_node_counts": dict(
            unsupported_node_counts
        )
    }


# ------------------------------------------------------------
# READ ANSYS NODE COMPONENTS
# ------------------------------------------------------------

def read_components(lines):

    components = {}

    i = 0

    while i < len(lines):

        line = lines[i]

        if not line.startswith("CMBLOCK,"):
            i += 1
            continue

        parts = line.split(",")

        if len(parts) < 4:
            raise ValueError(
                f"Invalid CMBLOCK definition: {line.strip()}"
            )

        name = parts[1].strip()

        try:
            count = int(parts[3].split()[0])
        except (ValueError, IndexError):
            raise ValueError(
                f"Invalid node count in component '{name}'."
            )

        ids = []

        # Skip the CMBLOCK header and format declaration.
        j = i + 2

        while j < len(lines) and len(ids) < count:

            current = lines[j].strip()

            if not current:
                j += 1
                continue

            if current.startswith(
                ("CMBLOCK", "NBLOCK", "EBLOCK")
            ):
                break

            if current.startswith("("):
                j += 1
                continue

            try:
                values = [
                    int(value)
                    for value in current.split()
                ]

                ids.extend(values)

            except ValueError:
                pass

            j += 1

        if len(ids) < count:
            raise ValueError(
                f"Component '{name}' declares {count} nodes, "
                f"but only {len(ids)} were read."
            )

        components[name] = ids[:count]

        i = j

    return components


# ------------------------------------------------------------
# PARSE SHELL SECTION THICKNESS
# ------------------------------------------------------------

def parse_shell_thickness(lines):

    thickness = {}

    i = 0

    while i < len(lines):

        line = lines[i]

        if not line.startswith("SECTYPE,"):
            i += 1
            continue

        parts = line.split(",")

        try:
            section_id = int(parts[1].strip())
        except (ValueError, IndexError):
            i += 1
            continue

        # Find the associated SECBLOCK.
        for j in range(
            i + 1,
            min(i + 10, len(lines))
        ):

            current = lines[j].strip()

            if current.startswith("SECTYPE,"):
                break

            if current.startswith("SECBLOCK"):

                # Read the first numeric field after SECBLOCK.
                if j + 1 < len(lines):

                    try:
                        value = float(
                            lines[j + 1].split(",")[0].strip()
                        )

                        if value > 0:
                            thickness[section_id] = value

                    except (ValueError, IndexError):
                        pass

                break

        i += 1

    return thickness