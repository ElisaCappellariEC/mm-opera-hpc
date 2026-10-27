"""
Cubes for two tests of the RVE of the runs for MMM, written by generate() of merope_pore_pressure.py, on the reference run
NFIR of mmm_inputs (the NFIR disc before the anneal, the log-normal fit of its measured pores) at unit pressure in every pore:
- the size of the cube: SEEDS cubes of side SIDE (small, to speed up the runs) and, optionally, cubes of other sides
  (EXTRA_SIDES) to check that D^2 scales as 1/V and that the mean does not drift with the side (Kanit 2003, Section 5);
- the size of the mesh, on the pores of side<SIDE>_seed01: the same pores meshed uniformly at twice the mesh of the runs
  (_meshdouble) and with a size field of gmsh, WALL_SIZES on the pore walls and BULK_SIZE away from them
  (_wall<h>_bulk<H>). The largest stress of a pore sits on its wall, mostly in the ligament towards its nearest neighbour
  (gaps down to distMin = 0.3 um), so the wall size is what the test refines.
This script writes the geometry only (bubbles, .geo, summary) with Mérope. Folders already there are kept: run it again to
add cubes (more seeds, another side, another wall size) next to the ones already solved.

Both tests follow Kanit et al., Int. J. Solids Struct. 40 (2003) 3647: the size of the cube is Sections 5.2 and 6 (Eq. 62,
68-70), the size of the mesh is Section 4.1.2. analyse_rve.py reads the results once MMM has solved the folders.

Usage
    python3 test_rve.py [OUT]     cubes in OUT/run_<RUN> (default OUT below), then ./run_gmsh.sh with FOLDER=OUT

@author E. Cappellari
@date   October 2026
"""

import os
import re
import shutil
import sys

import merope_pore_pressure

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "test_rve2"  # folder of the cubes, in the current directory

# Choices
RUN = "NFIR"                 # run of mmm_inputs: the reference, the NFIR disc before the anneal (pore_distribution.py)
SIDE = 10.0                  # um: side of the cubes of the size and mesh tests, small to speed up (the runs: mpp.dimensions_side)
SEEDS = 10                   # cubes of side SIDE, as the ten seeds of OperaHPC D5.2
EXTRA_SIDES = {}             # {side_um: seeds}, other sides for the scaling of Kanit, e.g. {15.0: 4}; meshed with the size
                             # field (wall mpp.mesh_size, bulk BULK_SIZE), once the mesh test shows it equals the uniform mesh
PRESSURE = 1.0               # unit pressure of every pore (sigma_I / p), no distribution
WALL_SIZES = (0.15, 0.075)   # um: sizes on the pore walls of the mesh test (0.10 is a lighter alternative to 0.075)
BULK_SIZE = 0.30             # um: size away from the walls, as the uniform mesh of _meshdouble
WALL_DIST = (0.05, 0.40)     # um: the wall size up to 0.05 um from the walls, growing linearly to BULK_SIZE at 0.40 um


def write_run(path, radius, pore_fraction, title):
    """A run file in the format of mmm_inputs (radius, porosity, pressure, confinement), with PRESSURE in every class."""
    lines = [f"# {title}", f"# pressure {PRESSURE:g} in every class, no confinement",
             f"# {'radius_um':>10s}{'porosity':>14s}{'pressure_MPa':>14s}{'confinement_MPa':>17s}"]
    lines += [f"{r:12.6f}{pf:14.6e}{PRESSURE:14.4f}{0.0:17.4f}" for r, pf in zip(radius, pore_fraction)]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_file(mpp, run):
    """The run file of the run: where merope_pore_pressure.py takes the runs (RUN_FILES), or mmm_inputs of this folder."""
    for folder in (os.path.dirname(mpp.RUN_FILES), os.path.join(HERE, "mmm_inputs"), os.path.join(HERE, "..", "mmm_inputs")):
        path = os.path.normpath(os.path.join(folder, f"run_{run}.txt"))
        if os.path.exists(path):
            return path
    raise SystemExit(f"run_{run}.txt is not in the folder of RUN_FILES of merope_pore_pressure.py nor in mmm_inputs: "
                     "python3 pore_distribution.py")


def fresh(out, folder):
    """True if out/folder does not exist yet: a cube already written (maybe solved) is never written over."""
    if os.path.isdir(os.path.join(out, folder)):
        print(f"{folder}: already there, kept")
        return False
    return True


def with_size_field(folder, name, wall, bulk):
    """Replaces the uniform mesh size of mesh_<name>.geo of Mérope by a size field of gmsh: wall (um) up to WALL_DIST[0]
    from the pore walls, growing linearly to bulk (um) at WALL_DIST[1]. The pore walls are all the physical surfaces but
    the periodic envelope (the group with the largest tag). The field goes where Mérope sets the size, before its
    'Mesh 2; Mesh 3;', so run_gmsh.sh meshes the .geo as it is (second order kept). The summary gets a line with the
    sizes, which analyse_rve.py reads."""
    geo = os.path.join(folder, f"mesh_{name}.geo")
    text = open(geo).read()
    phys = {int(k): [s.strip() for s in l.split(",")]
            for k, l in re.findall(r"Physical Surface\((\d+)\)\s*=\s*\{([^}]*)\};", text)}
    envelope = set(phys[max(phys)])
    walls = [s for k, l in phys.items() if not envelope & set(l) for s in l]
    field = (f"// size field (test_rve.py): {wall:g} um on the pore walls, {bulk:g} um from {WALL_DIST[1]:g} um away\n"
             "Mesh.MeshSizeFromPoints = 0;\nMesh.MeshSizeFromCurvature = 0;\nMesh.MeshSizeExtendFromBoundary = 0;\n"
             f"Field[1] = Distance;\nField[1].SurfacesList = {{{','.join(walls)}}};\nField[1].Sampling = 20;\n"
             f"Field[2] = Threshold;\nField[2].InField = 1;\nField[2].SizeMin = {wall:g};\nField[2].SizeMax = {bulk:g};\n"
             f"Field[2].DistMin = {WALL_DIST[0]:g};\nField[2].DistMax = {WALL_DIST[1]:g};\nBackground Field = 2;\n")
    text, n = re.subn(r"MeshSize \{:\}\s*=[^;]*;", lambda _: field, text)
    if n != 1:
        raise SystemExit(f"{geo}: one 'MeshSize {{:}} = ...;' of Mérope expected, found {n}")
    with open(geo, "w") as f:
        f.write(text)
    with open(os.path.join(folder, f"summary_{name}.txt"), "a") as f:
        f.write(f"# mesh (test_rve.py): size field, wall {wall:g} um, bulk {bulk:g} um, "
                f"from {WALL_DIST[0]:g} to {WALL_DIST[1]:g} um off the pore walls\n")
    print(f"{folder}: size field on {len(walls)} pore surfaces, wall {wall:g} um, bulk {bulk:g} um")


def write_cubes(mpp, run, out_root):
    """The cubes of the run not written yet, with mpp the module merope_pore_pressure, and the list of all of them in
    cubes.txt."""
    path = run_file(mpp, run)
    radius, pore_fraction = mpp.read_run(path)[:2]
    out = os.path.abspath(os.path.join(out_root, f"run_{run}"))
    os.makedirs(out, exist_ok=True)
    name = f"unit_{run}"  # generate() names the files of a cube after the run file
    unit = os.path.join(out, f"{name}.txt")
    write_run(unit, radius, pore_fraction, f"classes of {os.path.basename(path)} for the tests of the RVE (test_rve.py)")
    h = mpp.mesh_size
    rows = []

    # size of the cube: the classes of the run at unit pressure, SEEDS cubes of SIDE and the EXTRA_SIDES
    for side, seeds in [(SIDE, SEEDS)] + sorted(EXTRA_SIDES.items()):
        expected = mpp.pores_in_cube(radius, pore_fraction, side)[1]
        print(f"run_{run}, side {side:g} um: {expected.sum():.0f} pores expected, {expected.min():.2f} in the rarest class")
        if expected.min() < 2.0:
            print("  Mérope places the integer part of the expected number of pores of a class: the rarest classes are "
                  "missing or have one pore in a cube, analyse_rve.py cannot test them")
        bulk = h if side == SIDE else BULK_SIZE
        for seed in range(1, seeds + 1):
            folder = f"side{side:g}_seed{seed:02d}"
            if fresh(out, folder):
                mpp.generate(unit, side=side, seed=seed, mesh=h, folder=os.path.join(out, folder))
                if bulk != h:
                    with_size_field(os.path.join(out, folder), name, h, bulk)
            rows.append(("size", side, seed, h, bulk, folder))

    # size of the mesh: the pores of side<SIDE>_seed01 meshed again
    base = os.path.join(out, f"side{SIDE:g}_seed01")
    folder = f"side{SIDE:g}_seed01_meshdouble"
    if fresh(out, folder):
        mpp.generate(unit, side=SIDE, seed=1, mesh=2.0 * h, folder=os.path.join(out, folder))
    rows.append(("mesh", SIDE, 1, 2.0 * h, 2.0 * h, folder))
    for wall in WALL_SIZES:
        folder = f"side{SIDE:g}_seed01_wall{wall:g}_bulk{BULK_SIZE:g}"
        if fresh(out, folder):
            os.makedirs(os.path.join(out, folder))
            for file in (f"bubbles_{name}.txt", f"summary_{name}.txt", f"mesh_{name}.geo"):  # not the mesh nor MMM output
                shutil.copy(os.path.join(base, file), os.path.join(out, folder, file))
            with_size_field(os.path.join(out, folder), name, wall, BULK_SIZE)
        rows.append(("mesh", SIDE, 1, wall, BULK_SIZE, folder))

    lines = [f"# cubes of run_{run} for the tests of the RVE (python3 test_rve.py): MMM solves every folder with the pressure "
             f"{PRESSURE:g} in all the pores, analyse_rve.py reads the results",
             f"# {'test':>6s}{'side_um':>9s}{'seed':>6s}{'wall_um':>9s}{'bulk_um':>9s}  folder"]
    lines += [f"{t:>8s}{L:9g}{n:6d}{w:9g}{b:9g}  {f}" for t, L, n, w, b, f in rows]
    with open(os.path.join(out, "cubes.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(rows)} cubes of run_{run} listed in {os.path.join(out, 'cubes.txt')}")


def main():
    """The cubes not written yet, in the folder given as argument or OUT."""
    write_cubes(merope_pore_pressure, RUN, sys.argv[1] if len(sys.argv) > 1 else OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
