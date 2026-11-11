"""
The cubes of a tree of Mérope at unit pressure, with the gas pressure p_k of every class of their run file: the second
load of plan B (sciantix-official/utilities/HBFF, "Inputs for MMM" of pore_distribution.py).

MMM solves every cube twice, on the same mesh:
    unit/   pressure 1 in every pore (merope_pore_pressure.py or test_rve.py of Merope/mmm_files): s_hat_1, the factor of the
            geometry, also needed for the confinement P_h and to bring the run with the class pressures to zero
            macroscopic stress
    pk/     the pressure p_k of its class in every pore (this script): s_hat_B, with the load of the neighbours
The pores do not depend on the pressure, so a pk cube is its unit cube with column 6 of the bubbles file (the pressure
read by MMM, times --pref) replaced: no Mérope and no gmsh, it also runs on a cluster. The pk cube has no mesh: its
mesh_case.txt names the unit cube (path relative to the pk cube), whose mesh and split files run_mmm.sh and
the batch jobs of a cluster use, and whose MMM run plot_bubbles.py reads to bring the pk run to zero macroscopic stress.

Layout (UNIT_ROOT holds the run_<id>/ folders; PK_ROOT is UNIT_ROOT/../pk if UNIT_ROOT is called unit, UNIT_ROOT/pk
otherwise)
    UNIT_ROOT/run_<id>/side<L>_seed<NN>/bubbles_unit_<id>.txt, summary_unit_<id>.txt, mesh_unit_<id>.msh
    PK_ROOT/run_<id>/side<L>_seed<NN>/bubbles_pk_<id>.txt     pores of the unit cube, pressure p_k of their class
    PK_ROOT/run_<id>/side<L>_seed<NN>/summary_pk_<id>.txt     summary of the unit cube with p_k in its class rows
    PK_ROOT/run_<id>/side<L>_seed<NN>/mesh_case.txt           first line: the unit cube, relative to this folder
The class pressures are those of run_<id>.txt in UNIT_ROOT/run_<id> (copied there by merope_pore_pressure.py), else in --runs.
Only the cubes side<L>_seed<NN> are written, not the variants of the mesh test (side<L>_seed01_<variant>).
A pk cube already written is kept if it is the same, and the script stops if it differs (unit cube or run file changed).

Usage (anywhere, numpy only)
    python3 set_pressures.py lhs_side30_mesh0.15_graded_rfine0.45/unit            every run of the tree
    python3 set_pressures.py lhs_side30_mesh0.15_graded_rfine0.45/unit R05 R13    these runs only
    python3 set_pressures.py test_rve_side30_mesh0.15_graded_rfine0.45_delta0.15 --runs ~/mmm/Merope/mmm_inputs

@author E. Cappellari
@date   November 2026
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

RUNS = os.path.expanduser("~/mmm/sciantix-official/utilities/HBFF/mmm_inputs")  # run files of pore_distribution.py
CUBE = re.compile(r"side\d+(?:\.\d+)?_seed\d+$")  # cubes of the runs; the variants of the mesh test are left out
RADIUS_TOL = 2e-6                                 # um: radius of a pore against the radius of its class (6 decimals)
SOLVED = "bubbles_and_stresses_selected.txt"      # output of MMM


def read_run(path):
    """Radius, porosity, pressure and confinement (MPa) of every class of a run file of pore_distribution.py."""
    data = np.atleast_2d(np.loadtxt(path, comments="#"))
    if data.shape[1] != 4:
        raise SystemExit(f"{path}: 4 columns expected (radius, porosity, pressure, confinement), found {data.shape[1]}")
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


def run_file(run_dir, run, runs):
    """run_<id>.txt next to the cubes (merope_pore_pressure.py copies it there), else in runs."""
    for folder in (run_dir, runs):
        path = os.path.join(folder, f"run_{run}.txt")
        if os.path.isfile(path):
            return path
    raise SystemExit(f"run_{run}.txt is neither in {run_dir} nor in {runs}: give the folder of the run files with --runs")


def check_classes(run_dir, run, path, radius, porosity):
    """The classes of the run file must be those the cubes were made with (unit_<id>.txt of merope_pore_pressure.py, test_rve.py)."""
    unit = os.path.join(run_dir, f"unit_{run}.txt")
    if not os.path.isfile(unit):
        print(f"run_{run}: no {os.path.basename(unit)} in {run_dir}, the classes of {path} are not checked against the cubes")
        return
    r, x = read_run(unit)[:2]
    a, b = np.argsort(r), np.argsort(radius)  # unit_<id>.txt has the classes in the order of read_run() of Mérope
    if len(r) != len(radius) or not (np.allclose(r[a], radius[b], rtol=0, atol=RADIUS_TOL)
                                     and np.allclose(x[a], porosity[b], rtol=1e-5)):
        raise SystemExit(f"{path} has other classes than {unit}, the input of the cubes: the design of pore_distribution.py "
                         "was written again after the cubes? Use the run file of the cubes (--runs)")


def pk_lines(bubbles, radius, pressure):
    """Lines of the bubbles file with the pressure of its class in every pore (column 6, the rest kept as it is), and the
    class of every pore."""
    lines, classes = [], []
    with open(bubbles) as f:
        for n, line in enumerate(f, 1):
            words = line.split()
            if not words:
                continue
            if len(words) != 6:
                raise SystemExit(f"{bubbles}, line {n}: 6 columns expected (id, x, y, z, radius, pressure)")
            if float(words[5]) != 1.0:
                raise SystemExit(f"{bubbles}, line {n}: pressure {words[5]}, the unit cube must have pressure 1 in every pore")
            k = int(np.argmin(np.abs(radius - float(words[4]))))
            if abs(radius[k] - float(words[4])) > RADIUS_TOL:
                raise SystemExit(f"{bubbles}, line {n}: radius {words[4]} um is in no class of the run file "
                                 f"({', '.join(f'{r:g}' for r in radius)})")
            lines.append(" ".join(words[:5] + [repr(float(pressure[k]))]) + "\n")
            classes.append(k)
    return lines, np.array(classes, dtype=int)


def pk_summary(summary, run, radius, pressure, confinement, unit_rel, counts):
    """The summary of the unit cube with p_k in its class rows and pk_<id> as name; checks the pores per class."""
    out = []
    with open(summary) as f:
        text = f.read().splitlines()
    for n, line in enumerate(text):
        words = line.split()
        if n == 0:
            line = line.replace(f"# unit_{run}:", f"# pk_{run}:", 1)
            out.append(line)
            out.append(f"# pores and mesh of {unit_rel} (mesh_case.txt), pressure p_k of the class of run_{run}.txt in "
                       "every pore (set_pressures.py)")
            continue
        if not line.lstrip().startswith("#") and len(words) == 6:
            k = int(np.argmin(np.abs(radius - float(words[0]))))
            if abs(radius[k] - float(words[0])) > RADIUS_TOL:
                raise SystemExit(f"{summary}: class {words[0]} um not in the run file")
            if int(words[5]) != counts[k]:
                raise SystemExit(f"{summary}: {words[5]} pores of {words[0]} um, {counts[k]} in the bubbles file")
            r, _, asked, got, expected, placed = words
            line = (f"{float(r):12.6f}{pressure[k]:14.4f}{float(asked):16.6e}{float(got):14.6e}"
                    f"{float(expected):16.1f}{int(placed):8d}")
        elif line.startswith("# confining pressure") and confinement != 0.0:
            line += (f" (run_{run}.txt: P_h = {confinement:g} MPa, applied by SCIANTIX through s_hat_1 of the unit run, "
                     "not by MMM)")
        out.append(line)
    return "\n".join(out) + "\n"


def write_cube(unit_cube, pk_cube, run, radius, pressure, confinement):
    """The pk cube of one unit cube. Returns 'written', 'kept' or 'solved'."""
    bubbles = os.path.join(unit_cube, f"bubbles_unit_{run}.txt")
    summary = os.path.join(unit_cube, f"summary_unit_{run}.txt")
    for path in (bubbles, summary):
        if not os.path.isfile(path):
            raise SystemExit(f"{path} not found: is {unit_cube} a cube of merope_pore_pressure.py or test_rve.py?")
    unit_rel = os.path.relpath(unit_cube, pk_cube)
    lines, classes = pk_lines(bubbles, radius, pressure)
    counts = np.bincount(classes, minlength=len(radius))
    files = {
        f"bubbles_pk_{run}.txt": "".join(lines),
        f"summary_pk_{run}.txt": pk_summary(summary, run, radius, pressure, confinement, unit_rel, counts),
        "mesh_case.txt": (f"{unit_rel}\n# the cube of the line above (relative to this folder) gives the pores and the "
                          f"mesh of this case, solved by MMM with the class pressures of bubbles_pk_{run}.txt: "
                          "run_mmm.sh and the batch jobs of a cluster use its mesh (and split files), plot_bubbles.py its unit "
                          "run (set_pressures.py)\n"),
    }
    if os.path.isdir(pk_cube):
        old = {name: open(os.path.join(pk_cube, name)).read() if os.path.isfile(os.path.join(pk_cube, name)) else None
               for name in files}
        if old == files:
            return "solved" if os.path.exists(os.path.join(pk_cube, SOLVED)) else "kept"
        if any(text is not None for text in old.values()):
            changed = [name for name in files if old[name] != files[name]]
            raise SystemExit(f"{pk_cube} exists with other {', '.join(changed)}: the unit cube or run_{run}.txt changed "
                             "after it was written. Delete the pk cube (and its MMM run) to write it again")
    os.makedirs(pk_cube, exist_ok=True)
    for name, text in files.items():
        with open(os.path.join(pk_cube, name), "w") as f:
            f.write(text)
    return "written"


def set_run(unit_root, pk_root, run, runs):
    """The pk cubes of one run."""
    run_dir = os.path.join(unit_root, f"run_{run}")
    path = run_file(run_dir, run, runs)
    radius, porosity, pressure, confinement = read_run(path)
    if np.ptp(confinement) > 0.0:
        raise SystemExit(f"{path}: the confining pressure differs between the rows")
    check_classes(run_dir, run, path, radius, porosity)
    cubes = sorted(d for d in glob.glob(os.path.join(run_dir, "side*_seed*")) if CUBE.match(os.path.basename(d)))
    if not cubes:
        print(f"run_{run}: no cube side<L>_seed<NN> in {run_dir}")
        return
    status = [write_cube(c, os.path.join(pk_root, f"run_{run}", os.path.basename(c)), run, radius, pressure,
                         confinement[0]) for c in cubes]
    share = porosity / porosity.sum()
    p_ref = np.sum(share * pressure)
    print(f"run_{run} ({path}): p_k {pressure.min():.1f}-{pressure.max():.1f} MPa, p_ref = sum(V p) / sum(V) = "
          f"{p_ref:.1f} MPa; {len(cubes)} cubes: " + ", ".join(f"{status.count(s)} {s}" for s in ("written", "kept", "solved")
                                                              if s in status))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cubes with the class pressures (pk/) from the cubes at unit pressure")
    parser.add_argument("unit_root", help="folder of the run_<id>/ folders of the unit cubes")
    parser.add_argument("runs", nargs="*", help="ids of the runs (default: every run_<id> of unit_root)")
    parser.add_argument("--pk", help="folder of the pk cubes (default: ../pk if unit_root is called unit, unit_root/pk)")
    parser.add_argument("--runs", dest="run_files", default=RUNS,
                        help=f"folder of the run files if run_<id>.txt is not next to the cubes (default {RUNS})")
    args = parser.parse_args(argv)
    unit_root = os.path.abspath(args.unit_root)
    if not os.path.isdir(unit_root):
        raise SystemExit(f"{unit_root} is not a folder")
    pk_root = os.path.abspath(args.pk or (os.path.join(os.path.dirname(unit_root), "pk")
                                          if os.path.basename(unit_root) == "unit" else os.path.join(unit_root, "pk")))
    found = sorted(os.path.basename(d)[4:] for d in glob.glob(os.path.join(unit_root, "run_*")) if os.path.isdir(d))
    runs = args.runs or found
    unknown = [r for r in runs if r not in found]
    if unknown or not runs:
        raise SystemExit(f"no folder run_<id> in {unit_root} for: {', '.join(unknown) or 'any run'}")
    print(f"{len(runs)} runs of {unit_root} -> {pk_root}")
    for run in runs:
        set_run(unit_root, pk_root, run, os.path.expanduser(args.run_files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
