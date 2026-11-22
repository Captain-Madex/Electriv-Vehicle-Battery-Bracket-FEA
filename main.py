"""Run the EV battery-bracket Q4 membrane FEM solution for one ANSYS CDB mesh.

The nodal displacement vector is solved first. Element strains and stresses,
principal stresses, equivalent stress and equivalent strain are then recovered
from that displacement vector. Results are written to Results/<result-name>/.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from Python_Scripts.mesh_io import read_ansys_cdb
from Python_Scripts.boundary_conditions import build_force_vector, build_fixed_dofs
from Python_Scripts.fem_solver import (
    assemble,
    build_tie_constraints,
    recover_detailed_results,
    solve,
)
from Python_Scripts.material import MATERIALS

ROOT = Path(__file__).resolve().parent
MESH_DIRECTORY = ROOT / "ANSYS_Meshes"
RESULTS_DIRECTORY = ROOT / "Results"

# These are the prescribed loading conditions in the supplied ANSYS model.
TOTAL_FORCE_N = 294.3
LOAD_DIRECTION = (0.0, -1.0)


def _resolve_mesh(mesh_argument: str) -> Path:
    """Accept either a CDB filename in ANSYS_Meshes or an explicit path."""
    supplied = Path(mesh_argument).expanduser()
    candidates = [supplied] if supplied.is_absolute() else [ROOT / supplied, MESH_DIRECTORY / supplied]
    for candidate in candidates:
        if candidate.is_file():
            if candidate.suffix.lower() != ".cdb":
                raise ValueError(f"Mesh must be an ANSYS .CDB file: {candidate}")
            return candidate.resolve()
    raise FileNotFoundError(
        f"ANSYS mesh '{mesh_argument}' was not found. Put it in {MESH_DIRECTORY} "
        "or pass its full path with --mesh."
    )


def _prepare_output_folders(result_name: str) -> tuple[Path, Path]:
    if not result_name or Path(result_name).name != result_name or result_name in {".", ".."}:
        raise ValueError("--result-name must be a single folder name.")

    output = RESULTS_DIRECTORY / result_name
    figures = output / "Figures"
    numerical = output / "Numerical_Results"

    # Remove old generated files so no result from an earlier run is mistaken
    # for output from the current mesh/solver run.
    for folder in (figures, numerical):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
    return figures, numerical


def _extrema(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Cannot summarize an empty or non-finite result field.")
    return {
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "maximum_absolute": float(np.max(np.abs(values))),
    }


def _write_csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _write_result_files(mesh, output_numeric: Path, nodes, elements, mids, sids,
                        displacement, fields, field_specs, total_displacement,
                        summary) -> None:
    ux_uy = displacement.reshape(-1, 2)
    _write_csv(
        output_numeric / "nodal_displacements.csv",
        ["node_id", "x_mm", "y_mm", "ux_mm", "uy_mm", "total_deformation_mm"],
        (
            [int(node_id), float(x), float(y), float(ux), float(uy), float(total)]
            for node_id, (x, y), (ux, uy), total in zip(
                mesh["node_ids"], nodes, ux_uy, total_displacement
            )
        ),
    )

    stress = fields["stress"]
    vm = fields["von_mises"]
    centroids = fields["centroids"]
    rows = []
    for i, element in enumerate(elements):
        rows.append([
            int(mesh["element_ids"][i]), int(mids[i]), int(sids[i]),
            float(centroids[i, 0]), float(centroids[i, 1]),
            float(field_specs["total_deformation"][0][i]),
            float(field_specs["directional_deformation_x"][0][i]),
            float(field_specs["directional_deformation_y"][0][i]),
            float(stress[i, 0]), float(stress[i, 1]), float(stress[i, 2]),
            float(vm[i]), float(fields["equivalent_elastic_strain"][i]),
            float(fields["maximum_principal_stress"][i]),
            float(fields["minimum_principal_stress"][i]),
        ])
    _write_csv(
        output_numeric / "element_results_all_parameters.csv",
        [
            "element_id", "material_id", "section_id", "centroid_x_mm",
            "centroid_y_mm", "total_deformation_mm",
            "directional_deformation_x_mm", "directional_deformation_y_mm",
            "sigma_x_longitudinal_MPa", "sigma_y_vertical_MPa", "tau_xy_MPa",
            "von_mises_MPa", "equivalent_elastic_strain",
            "maximum_principal_stress_MPa", "minimum_principal_stress_MPa",
        ],
        rows,
    )

    # Retain the previous filename for compatibility with existing workflows.
    _write_csv(
        output_numeric / "element_stress.csv",
        [
            "element_id", "material_id", "section_id", "centroid_x_mm",
            "centroid_y_mm", "sigma_x_MPa", "sigma_y_MPa", "tau_xy_MPa",
            "von_mises_MPa", "equivalent_elastic_strain",
            "maximum_principal_stress_MPa", "minimum_principal_stress_MPa",
        ],
        (
            [
                int(mesh["element_ids"][i]), int(mids[i]), int(sids[i]),
                float(centroids[i, 0]), float(centroids[i, 1]),
                float(stress[i, 0]), float(stress[i, 1]), float(stress[i, 2]),
                float(vm[i]), float(fields["equivalent_elastic_strain"][i]),
                float(fields["maximum_principal_stress"][i]),
                float(fields["minimum_principal_stress"][i]),
            ]
            for i in range(len(elements))
        ),
    )

    # This file was present in previous results but was not refreshed by the
    # earlier main.py. Rebuild it from the current run's summary every time.
    parameter_rows = [
        ("Total Deformation / Overall displacement", "total_deformation_mm", "mm"),
        ("Directional Deformation X / Horizontal displacement", "directional_deformation_x_mm", "mm"),
        ("Directional Deformation Y / Vertical displacement", "directional_deformation_y_mm", "mm"),
        ("Equivalent Stress / von Mises stress", "equivalent_stress_von_mises_MPa", "MPa"),
        ("Normal Stress X / Longitudinal stress", "normal_stress_x_longitudinal_MPa", "MPa"),
        ("Normal Stress Y / Vertical stress", "normal_stress_y_vertical_MPa", "MPa"),
        ("Shear Stress XY", "shear_stress_xy_MPa", "MPa"),
        ("Equivalent Elastic Strain", "equivalent_elastic_strain", "-"),
        ("Maximum Principal Stress / tensile stress", "maximum_principal_tensile_stress_MPa", "MPa"),
        ("Minimum Principal Stress / compressive stress", "minimum_principal_compressive_stress_MPa", "MPa"),
    ]
    _write_csv(
        output_numeric / "parameter_summary.csv",
        ["ANSYS_result_parameter", "minimum", "maximum", "maximum_absolute", "unit"],
        (
            [label, *[summary["results"][key][name] for name in (
                "minimum", "maximum", "maximum_absolute"
            )], unit]
            for label, key, unit in parameter_rows
        ),
    )


def _write_figures(figures_dir: Path, mesh, nodes, elements, fields,
                   field_specs, displacement, total_displacement) -> None:
    """Create topology-faithful Q4 plots without Delaunay interpolation.

    Element quantities are displayed as constant values over their actual Q4
    polygons. This avoids the false bridges/filled holes produced by applying
    a Delaunay triangulation to element centroids.
    """
    polygons = [nodes[element] for element in elements]

    # Mesh and boundary conditions.
    fig, ax = plt.subplots(figsize=(10, 8))
    mesh_collection = PolyCollection(
        polygons, facecolors="none", edgecolors="0.35", linewidths=0.35
    )
    ax.add_collection(mesh_collection)
    component_styles = {
        "LEFT_FIXED": ("tab:blue", "Left fixed"),
        "BOTTOM_FIXED": ("tab:green", "Bottom fixed"),
        "BROWN_LOAD_EDGE": ("tab:red", "Loaded edge"),
    }
    for component_name, (color, label) in component_styles.items():
        points = nodes[mesh["components"][component_name]]
        ax.scatter(points[:, 0], points[:, 1], s=8, color=color,
                   label=label, zorder=3)
    _set_mesh_limits(ax, nodes)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("ANSYS Q4 mesh, supports and applied-load boundary")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures_dir / "mesh_and_boundary_conditions.png", dpi=240)
    plt.close(fig)

    # Deformed geometry: magnify only for visibility and state the factor.
    nodal_displacement = displacement.reshape(-1, 2)
    max_u = float(np.max(total_displacement))
    diagonal = float(np.linalg.norm(np.ptp(nodes, axis=0)))
    scale = 1.0 if max_u <= 0 else max(1.0, 0.08 * diagonal / max_u)
    deformed_nodes = nodes + scale * nodal_displacement
    deformed_polygons = [deformed_nodes[element] for element in elements]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.add_collection(PolyCollection(
        polygons, facecolors="none", edgecolors="0.75", linewidths=0.3,
        label="Undeformed"
    ))
    ax.add_collection(PolyCollection(
        deformed_polygons, facecolors="none", edgecolors="tab:red",
        linewidths=0.55, label="Deformed"
    ))
    combined = np.vstack((nodes, deformed_nodes))
    _set_mesh_limits(ax, combined)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(
        "Deformed shape (visual magnification "
        f"{scale:.3g}x; actual max displacement={max_u:.6g} mm)"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures_dir / "deformed_shape.png", dpi=240)
    plt.close(fig)

    # Plot each result as Q4 element patches. No centroid triangulation.
    for key, (values, label, filename) in field_specs.items():
        values = np.asarray(values, dtype=float)
        if values.shape != (len(elements),):
            raise ValueError(
                f"Plot field '{key}' has shape {values.shape}; "
                f"expected one value for each of {len(elements)} Q4 elements."
            )
        collection = PolyCollection(
            polygons, array=values, cmap="turbo", edgecolors="none"
        )
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.add_collection(collection)
        _set_mesh_limits(ax, nodes)
        colorbar = fig.colorbar(collection, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label(label)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title(f"{label} — Q4 element values")
        ax.grid(False)
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=240)
        plt.close(fig)


def _set_mesh_limits(ax, points: np.ndarray) -> None:
    """Set padded limits while preserving the true model aspect ratio."""
    points = np.asarray(points, dtype=float)
    xmin, ymin = np.min(points, axis=0)
    xmax, ymax = np.max(points, axis=0)
    dx = max(float(xmax - xmin), 1e-9)
    dy = max(float(ymax - ymin), 1e-9)
    padx, pady = 0.04 * dx, 0.04 * dy
    ax.set_xlim(xmin - padx, xmax + padx)
    ax.set_ylim(ymin - pady, ymax + pady)
    ax.set_aspect("equal", adjustable="box")

def run_analysis(mesh_argument: str, result_name: str) -> dict:
    """Solve one mesh and write all displacement-derived results."""
    mesh_path = _resolve_mesh(mesh_argument)
    figures_dir, numerical_dir = _prepare_output_folders(result_name)

    mesh = read_ansys_cdb(mesh_path)
    nodes = mesh["nodes"]
    elements = mesh["elements"]
    material_ids = mesh["material_ids"]
    section_ids = mesh["section_ids"]

    missing_materials = set(np.unique(material_ids)) - set(MATERIALS)
    if missing_materials:
        raise ValueError(f"Material IDs not mapped in material.py: {sorted(missing_materials)}")

    required_components = ("LEFT_FIXED", "BOTTOM_FIXED", "BROWN_LOAD_EDGE")
    missing_components = [name for name in required_components if name not in mesh["components"]]
    if missing_components:
        raise ValueError(f"Required ANSYS node components are missing: {missing_components}")

    # 1. Assemble the prescribed load vector and boundary conditions.
    force, load_info = build_force_vector(
        nodes, elements, mesh["components"]["BROWN_LOAD_EDGE"],
        TOTAL_FORCE_N, LOAD_DIRECTION,
    )
    fixed_dofs, fixed_nodes = build_fixed_dofs(mesh["components"], len(nodes))

    # 2. Assemble and solve K u = F, including the bonded-interface ties.
    stiffness = assemble(
        nodes, elements, material_ids, mesh["section_thickness"], section_ids
    )
    constraints, tie_info = build_tie_constraints(nodes, elements, material_ids)
    displacement, reactions = solve(stiffness, force, fixed_dofs, constraints)

    # 3. Recover all other fields from the solved nodal displacement vector.
    fields = recover_detailed_results(
        nodes, elements, material_ids, section_ids,
        mesh["section_thickness"], displacement,
    )
    stress = fields["stress"]
    von_mises = fields["von_mises"]
    nodal_displacement = displacement.reshape(-1, 2)
    ux = nodal_displacement[:, 0]
    uy = nodal_displacement[:, 1]
    total_displacement = np.linalg.norm(nodal_displacement, axis=1)

    field_specs = {
        "total_deformation": (
            np.array([total_displacement[element].mean() for element in elements]),
            "Total deformation (mm)", "total_deformation.png",
        ),
        "directional_deformation_x": (
            np.array([nodal_displacement[element, 0].mean() for element in elements]),
            "Directional deformation X (mm)", "directional_deformation_x.png",
        ),
        "directional_deformation_y": (
            np.array([nodal_displacement[element, 1].mean() for element in elements]),
            "Directional deformation Y (mm)", "directional_deformation_y.png",
        ),
        "equivalent_stress": (
            von_mises, "Equivalent (von Mises) stress (MPa)", "equivalent_stress.png",
        ),
        "normal_stress_x": (
            stress[:, 0], "Normal stress X / longitudinal stress (MPa)", "normal_stress_x.png",
        ),
        "normal_stress_y": (
            stress[:, 1], "Normal stress Y / vertical stress (MPa)", "normal_stress_y.png",
        ),
        "shear_stress_xy": (
            stress[:, 2], "Shear stress XY (MPa)", "shear_stress_xy.png",
        ),
        "equivalent_elastic_strain": (
            fields["equivalent_elastic_strain"], "Equivalent elastic strain (-)",
            "equivalent_elastic_strain.png",
        ),
        "maximum_principal_stress": (
            fields["maximum_principal_stress"], "Maximum principal stress (MPa)",
            "maximum_principal_stress.png",
        ),
        "minimum_principal_stress": (
            fields["minimum_principal_stress"], "Minimum principal stress (MPa)",
            "minimum_principal_stress.png",
        ),
    }

    material_summary = {}
    for material_id in sorted(np.unique(material_ids)):
        mask = material_ids == material_id
        material = MATERIALS[int(material_id)]
        material_summary[str(int(material_id))] = {
            "material": material["name"],
            "elements": int(mask.sum()),
            "E_MPa": material["E_MPa"],
            "nu": material["nu"],
            "section_ids": sorted(int(value) for value in np.unique(section_ids[mask])),
        }

    reaction_info = {
        "reaction_x_N": float(reactions[0::2].sum()),
        "reaction_y_N": float(reactions[1::2].sum()),
        "force_balance_x_N": float(reactions[0::2].sum() + force[0::2].sum()),
        "force_balance_y_N": float(reactions[1::2].sum() + force[1::2].sum()),
    }
    results = {
        "total_deformation_mm": _extrema(total_displacement),
        "overall_displacement_mm": _extrema(total_displacement),
        "directional_deformation_x_mm": _extrema(ux),
        "horizontal_displacement_mm": _extrema(ux),
        "directional_deformation_y_mm": _extrema(uy),
        "vertical_displacement_mm": _extrema(uy),
        "equivalent_stress_von_mises_MPa": _extrema(von_mises),
        "normal_stress_x_longitudinal_MPa": _extrema(stress[:, 0]),
        "normal_stress_y_vertical_MPa": _extrema(stress[:, 1]),
        "shear_stress_xy_MPa": _extrema(stress[:, 2]),
        "equivalent_elastic_strain": _extrema(fields["equivalent_elastic_strain"]),
        "maximum_principal_tensile_stress_MPa": _extrema(fields["maximum_principal_stress"]),
        "minimum_principal_compressive_stress_MPa": _extrema(fields["minimum_principal_stress"]),
    }

    summary = {
        "mesh_file": mesh_path.name,
        "result_folder": result_name,
        "model": "2-D Q4 plane-stress membrane approximation using SHELL181 in-plane connectivity",
        "nodes": int(len(nodes)),
        "valid_q4_elements": int(len(elements)),
        "degenerate_elements_excluded": int(len(mesh["degenerate_shell_element_ids"])),
        "excluded_element_ids": mesh["degenerate_shell_element_ids"].tolist(),
        "geometry_bounds_mm": {
            "x": [float(nodes[:, 0].min()), float(nodes[:, 0].max())],
            "y": [float(nodes[:, 1].min()), float(nodes[:, 1].max())],
        },
        "materials": material_summary,
        "section_thickness_mm": {
            str(key): float(value) for key, value in mesh["section_thickness"].items()
        },
        "load": load_info,
        "supports": {
            "left_fixed_nodes": int(len(mesh["components"]["LEFT_FIXED"])),
            "bottom_fixed_nodes": int(len(mesh["components"]["BOTTOM_FIXED"])),
            "unique_fixed_nodes": int(len(fixed_nodes)),
            "fixed_dofs": int(len(fixed_dofs)),
            "condition": "Ux=Uy=0 on LEFT_FIXED and BOTTOM_FIXED",
        },
        "results": results,
        "bonded_interfaces": tie_info,
        "equilibrium": reaction_info,
        "definitions": {
            "principal_stresses": "In-plane principal stresses from the 2D stress tensor.",
            "equivalent_elastic_strain": (
                "3D von-Mises equivalent strain using plane-stress epsilon_z "
                "inferred from isotropic elasticity."
            ),
            "stress_sampling": (
                "Element-average of 2x2 Gauss-point stress/strain; von Mises is "
                "maximum across the four Gauss points."
            ),
            "displacement_sampling": "Nodal values; plots show element-average values.",
        },
        "limitations": (
            "Not a full SHELL181 solution. Only in-plane Ux/Uy membrane response is "
            "represented; shell rotations, bending, transverse shear and "
            "through-thickness stress are not solved."
        ),
    }

    # 4. Write this run's numerical results and figures.
    _write_result_files(
        mesh, numerical_dir, nodes, elements, material_ids, section_ids,
        displacement, fields, field_specs, total_displacement, summary,
    )
    (numerical_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    status = {
        "status": "COMPLETED",
        "mesh_file": mesh_path.name,
        "result_folder": result_name,
        "load_N": TOTAL_FORCE_N,
        "load_direction": "global -Y",
        "supports": "LEFT_FIXED and BOTTOM_FIXED; Ux=Uy=0",
        "result_parameters": list(results.keys()),
        "formulation_limitations": summary["limitations"],
    }
    (numerical_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    _write_figures(
        figures_dir, mesh, nodes, elements, fields, field_specs,
        displacement, total_displacement,
    )

    print(f"Mesh: {mesh_path.name}")
    print(f"Nodes: {len(nodes)} | Q4 elements solved: {len(elements)}")
    print(f"Applied load: {load_info['resultant_y_N']:.6g} N (global -Y)")
    print(f"Maximum total deformation: {results['total_deformation_mm']['maximum']:.8g} mm")
    print(f"Maximum von Mises stress: {results['equivalent_stress_von_mises_MPa']['maximum']:.8g} MPa")
    print(f"Force-balance error Y: {reaction_info['force_balance_y_N']:.3e} N")
    print(f"Results updated: {RESULTS_DIRECTORY / result_name}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve one ANSYS CDB mesh and update all displacement-derived results."
    )
    parser.add_argument("--mesh", default="EV_BRACKET_MESH_1.CDB",
                        help="CDB filename in ANSYS_Meshes or full/relative path")
    parser.add_argument("--result-name", default="Mesh_1",
                        help="Output folder name under Results (default: Mesh_1)")
    arguments = parser.parse_args()
    run_analysis(arguments.mesh, arguments.result_name)


if __name__ == "__main__":
    main()
