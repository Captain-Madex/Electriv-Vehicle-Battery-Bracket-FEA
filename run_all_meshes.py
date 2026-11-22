"""Run main.py once for every ANSYS .CDB mesh in ANSYS_Meshes."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MESH_DIRECTORY = ROOT / "ANSYS_Meshes"
MAIN_SCRIPT = ROOT / "main.py"


def main() -> int:
    meshes = sorted(
        (path for path in MESH_DIRECTORY.iterdir()
         if path.is_file() and path.suffix.lower() == ".cdb"),
        key=lambda path: path.name.lower(),
    ) if MESH_DIRECTORY.is_dir() else []

    if not meshes:
        print(f"No ANSYS .CDB meshes found in: {MESH_DIRECTORY}", file=sys.stderr)
        return 1

    failures = []
    for index, mesh_path in enumerate(meshes, start=1):
        result_name = f"Mesh_{index}"
        print(
            f"\n=== [{index}/{len(meshes)}] {mesh_path.name} -> Results/{result_name} ===",
            flush=True,
        )
        command = [
            sys.executable,
            str(MAIN_SCRIPT),
            "--mesh", str(mesh_path),
            "--result-name", result_name,
        ]
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as error:
            failures.append((mesh_path.name, error.returncode))
            print(f"FAILED: {mesh_path.name} (exit code {error.returncode})", file=sys.stderr)

    completed = len(meshes) - len(failures)
    print(f"\nMesh runs completed: {completed}/{len(meshes)}")
    if failures:
        print("Failed meshes:")
        for mesh_name, return_code in failures:
            print(f"  {mesh_name}: exit code {return_code}")
        return 1

    print("All mesh solutions and result files have been generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
