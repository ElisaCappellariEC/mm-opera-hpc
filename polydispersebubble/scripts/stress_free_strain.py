"""
A pk cube solved at zero macroscopic strain (MMM default) has the mean hydrostatic stress of the matrix
    <sigma_h>_1 = -p_m p_ref                                   (sig_hydro.txt, plot_bubbles.py)
with p_ref = sum(V p) / sum(V); the pores are boundary conditions. The mean of the matrix answers a hydrostatic strain
eps (eps_xx = eps_yy = eps_zz, same pressures) with 3 (1 - p_m_1) K_m eps, so
    eps = p_m p_ref / (3 (1 - p_m_1) K_m)
gives zero mean of the matrix (the zero stress of D5.2): p_m_1 the p_m of the unit run of the same cube (pressure 1
in the pores and on the boundary is the matrix at -1, at the strain -1 / (3 K_m)), K_m = E / (3 (1 - 2 nu)) the bulk
modulus of the matrix of Bubble.cxx. eps is in the units of the run (pressures pref * p_i in N/um2, as E). Checked on
the 4 cubes of test_free_side10_mesh0.3_graded_rfine0.45_nu0.3116 (the pk_matrix runs of
~/mmm/analysis_d52_rescaling).

Layout (PK_ROOT as written by set_pressures.py; the unit run of a cube is the folder of its mesh_case.txt)
    PK_ROOT/run_<id>/side<L>_seed<NN>/eps_hydro.txt    first line eps, then the inputs as comments
Only the cubes side<L>_seed<NN> with both runs solved are done; the outputs of the pk cube are those at zero strain, so
the run with -eh must go in another folder.

Usage: python3 stress_free_strain.py PK_ROOT
    e.g. python3 stress_free_strain.py ~/mmm/results/lhs_side30_mesh0.15_graded_rfine0.45/pk

@ author: E.Cappellari, POLIMI-CEA 2026
@ date:   November 2026
"""
# %% ------------------------------------------------------------------------------------------------------------------
# Libraries and paths
import glob
import os
import sys
import plot_bubbles as pb

# the matrix of Bubble.cxx: NEA/NSC/R(2024)1 Eqs. 7.1 and 7.4 at T = 903.15 K (NFIR), p = Pu = x = 0 (since 23/11/2026,
# E = 205.0 GPa, nu = 0.3116; before: 150e-3, 0.3). Must be those printed by test-polybubble for the cube.
YOUNG, POISSON = 205.004641e-3, 0.311602357     # N/um2 and /
K_MATRIX = YOUNG / (3.0 * (1.0 - 2.0 * POISSON))  # N/um2, bulk modulus of the matrix
OUTPUT = "eps_hydro.txt"                        # written in every pk cube

# %% ------------------------------------------------------------------------------------------------------------------
# Main

def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 stress_free_strain.py PK_ROOT (the folder of the run_<id>/ folders of the pk cubes)")
    pk_root = os.path.abspath(os.path.expanduser(sys.argv[1]))
    folders = sorted(d for d in glob.glob(os.path.join(pk_root, "run_*", "side*_seed*"))
                     if os.path.isdir(d) and (m := pb.CASE.match(os.path.basename(d))) and m["variant"] is None)
    if not folders:
        raise SystemExit(f"no cube run_<id>/side<L>_seed<NN> in {pk_root}")

    print(f"{'run':8s}{'cube':16s}{'xi':>8s}{'p_ref':>10s}{'p_m':>9s}{'p_m_1':>9s}{'eps':>14s}")
    written, skipped = 0, []
    for folder in folders:
        # %% the pk run at zero strain and the unit run of the same cube ---------------------------------------------
        pk = pb.read_cube(folder)
        if pk is None or pk["load"] != "pk":
            skipped.append(os.path.relpath(folder, pk_root))
            continue
        unit = pb.read_cube(pk["unit"])
        if unit is None or unit["load"] != "unit":
            skipped.append(os.path.relpath(folder, pk_root))
            continue

        # %% the strain of zero mean of the matrix ---------------------------------------------------------------------
        eps = pk["p_m"] * pk["p_ref"] / (3.0 * (1.0 - unit["p_m"]) * K_MATRIX)
        with open(os.path.join(folder, OUTPUT), "w") as f:
            f.write(f"{eps:.10e}\n"
                    f"# -eh of test-polybubble for zero mean of the matrix of this cube (stress_free_strain.py); the "
                    f"outputs of this folder are at zero strain\n"
                    f"# eps = p_m p_ref / (3 (1 - p_m_1) K_m), p_m p_ref = -<sigma_h> of this run, p_m_1 of the unit run "
                    f"{os.path.relpath(pk['unit'], folder)}\n"
                    f"# xi = {pk['xi']:.6f}, p_ref = {pk['p_ref']:.6g}, p_m = {pk['p_m']:.6f}, p_m_1 = {unit['p_m']:.6f}, "
                    f"K_m = {K_MATRIX:g} N/um2\n")
        written += 1
        print(f"{os.path.basename(os.path.dirname(folder)):8s}{os.path.basename(folder):16s}{pk['xi']:8.4f}"
              f"{pk['p_ref']:10.3f}{pk['p_m']:9.4f}{unit['p_m']:9.4f}{eps:14.6e}")

    print(f"\n{written} cubes: {OUTPUT} written in {pk_root}/run_*/side*_seed*")
    if skipped:
        print(f"{len(skipped)} cubes left out (pk or unit run not solved): " + ", ".join(skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
