from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

mesh_file = (
    PROJECT_ROOT
    / "ANSYS_Meshes"
    / "EV_BRACKET_MESH_1.CDB"
)

lines = mesh_file.read_text(
    encoding="utf-8",
    errors="replace"
).splitlines()

eblock_index = next(
    i for i, line in enumerate(lines)
    if line.startswith("EBLOCK")
)

print("EBLOCK header:")
print(lines[eblock_index])

print("\nFirst 5 EBLOCK records:")

count = 0

for line in lines[eblock_index + 2:]:

    if line.strip() == "-1":
        break

    values = []

    for j in range(0, len(line), 9):
        field = line[j:j+9].strip()

        if field:
            try:
                values.append(int(field))
            except ValueError:
                pass

    if values:
        print(values)
        count += 1

    if count == 5:
        break
print("\nElement type definitions:")

for line in lines:
    if line.startswith("ET,"):
        print(line)

print("\nMaterial definitions:")

for line in lines:
    if line.startswith("MP,"):
        print(line)

print("\nSection definitions:")

for line in lines:
    if line.startswith("SECTYPE,"):
        print(line)