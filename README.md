# EV Battery Bracket FEM — cleaned Q4 pipeline

This project runs a single, ordered FEM workflow for the supplied ANSYS 18.2
CDB mesh. It solves the in-plane membrane response using four-node Q4
quadrilaterals extracted from the SHELL181 mesh.

## Pipeline

1. `main.py` reads the selected CDB mesh.
2. `Python_Scripts/mesh_io.py` reads node coordinates, Q4 shell connectivity,
   material/section IDs, named components and shell thicknesses. Six collapsed
   shell records with repeated node indices are excluded and listed in
   `summary.json`; they are not silently treated as valid Q4 elements.
3. `Python_Scripts/boundary_conditions.py` constructs the consistent edge load
   and applies the fixed supports.
4. `Python_Scripts/fem_solver.py` assembles the global stiffness matrix,
   applies the bonded-interface translation constraints, solves `K u = F`,
   and recovers element strains and stresses from the solved displacement.
5. `main.py` writes the numerical result files and the figures.

Only these Python modules are part of the active pipeline:
- `main.py`
- `run_all_meshes.py`
- `Python_Scripts/mesh_io.py`
- `Python_Scripts/boundary_conditions.py`
- `Python_Scripts/fem_solver.py`
- `Python_Scripts/material.py`
- `Python_Scripts/__init__.py`

The older uploaded/original copies, duplicate visualization modules, standalone
plotting utilities and legacy solver modules have been removed. The active
visualization code is centralized in `main.py`.

## Model setup currently encoded in the project

- Applied resultant: **294.3 N downward (global -Y)** on `BROWN_LOAD_EDGE`.
- Loaded edge: 50 mm total length; 25 boundary segments; uniform line load
  5.886 N/mm.
- `LEFT_FIXED`: Ux = Uy = 0.
- `BOTTOM_FIXED`: Ux = Uy = 0.
- Bonded translation ties: steel wall (MAT 3) to aluminium bracket (MAT 2)
  at x = 0; aluminium bracket (MAT 2) to steel clamp (MAT 1) at y = 8 mm.
- Material definitions and section thicknesses are read/mapped from the CDB.
  Current material mapping: MAT 1 steel (E=200000 MPa, nu=0.30), MAT 2
  aluminium (E=70000 MPa, nu=0.33), MAT 3 steel (E=200000 MPa, nu=0.30).
- The current CDB contains 1498 nodes, 1288 usable Q4 elements and 6
  degenerate shell records excluded from the stiffness assembly.

## Results generated

`Results/Mesh_n/Numerical_Results/` contains:
- `summary.json`: model setup, mesh counts, load/support data, result extrema,
  equilibrium checks and formulation limitations.
- `parameter_summary.csv`: minimum, maximum and maximum-absolute values for
  all requested parameters.
- `nodal_displacements.csv`: nodal Ux, Uy and total displacement.
- `element_results_all_parameters.csv`: element-level displacement, stress
  and strain results.
- `element_stress.csv`: retained compatibility export with stress-only fields.
- `status.json`: completion state and formulation limitation.

`Results/Mesh_n/Figures/` contains mesh/support/load visualization, deformed
shape, and all requested contour figures:
1. Total deformation
2. Directional deformation X
3. Directional deformation Y
4. Equivalent (von Mises) stress
5. Normal stress X
6. Normal stress Y
7. Shear stress XY
8. Equivalent elastic strain
9. Maximum principal stress
10. Minimum principal stress

## Visualization correction

Element result contours are now drawn directly over the actual Q4 polygons
using one constant recovered value per element. The previous Delaunay
triangulation of element centroids was removed because it could interpolate
across holes, disconnected parts and empty regions, producing misleading
contours. The deformed-shape figure overlays undeformed and deformed meshes.
Its magnification is automatically selected for visibility and explicitly
printed in the title; the reported displacement values remain unscaled.

## Run

```bash
pip install -r requirements.txt
python main.py
```

To run all compatible CDB files placed directly in `ANSYS_Meshes/`:

```bash
python run_all_meshes.py
```

To select a mesh and output folder:

```bash
python main.py --mesh EV_BRACKET_MESH_1.CDB --result-name Mesh_1
```

Each run clears and regenerates the selected result folder, so old plots and
CSV files cannot be mistaken for the current run.

## Important formulation limitation

The source mesh uses SHELL181. This Python implementation uses only its
four-node in-plane connectivity and approximates the structure as a 2D
Q4 plane-stress membrane. It solves Ux and Uy only. It does not implement
SHELL181 rotations, bending stiffness, transverse shear, or through-thickness
stress. Results are therefore membrane-only estimates, not a full reproduction
of ANSYS Mechanical SHELL181.

Six collapsed/repeated-node shell records are excluded. Repair/remesh these
elements in ANSYS before relying on local results near those locations.
The model assumptions (load, supports, tie locations, material mapping and
section thickness) must also be checked against the intended physical setup
before treating the output as validated engineering results.
