"""
A pk cube solved at zero macroscopic strain (MMM default) has the macroscopic hydrostatic stress
    Sigma_1 = (1 - xi) <sigma_h> - xi p_ref = -h p_ref         (matrix and pores at -p, plot_bubbles.py)
with <sigma_h> the mean hydrostatic stress of the matrix (sig_hydro.txt) and p_ref = sum(V p) / sum(V). The cube
answers a hydrostatic strain eps (eps_xx = eps_yy = eps_zz, same pressures) with 3 K_eff eps, so
    eps = -Sigma_1 / (3 K_eff) = h p_ref / (3 (1 - h_1) K_m)
gives zero macroscopic stress: K_eff = (1 - h_1) K_m exactly, h_1 the h of the unit run of the same cube (its Biot
coefficient), K_m = E / (3 (1 - 2 nu)) the bulk modulus of the matrix of Bubble.cxx. eps is in the units of the run
(pressures pref * p_i in N/um2, as E). Checked on test_single_pore/regular.

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

    print(f"{'run':8s}{'cube':16s}{'xi':>8s}{'p_ref':>10s}{'h':>9s}{'h_1':>9s}{'eps':>14s}")
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

        # %% the strain of zero macroscopic stress --------------------------------------------------------------------
        sigma_1 = -pk["h"] * pk["p_ref"]  # RVE vs matrix mean: Sigma of matrix and pores, not the Average of sig_hydro.txt
        k_eff = (1.0 - unit["h"]) * K_MATRIX
        eps = -sigma_1 / (3.0 * k_eff)
        with open(os.path.join(folder, OUTPUT), "w") as f:
            f.write(f"{eps:.10e}\n"
                    f"# -eh of test-polybubble for zero macroscopic stress of this cube (stress_free_strain.py); the "
                    f"outputs of this folder are at zero strain\n"
                    f"# eps = -Sigma_1 / (3 K_eff), Sigma_1 = -h p_ref (this run), K_eff = (1 - h_1) K_m (unit run "
                    f"{os.path.relpath(pk['unit'], folder)})\n"
                    f"# xi = {pk['xi']:.6f}, p_ref = {pk['p_ref']:.6g}, h = {pk['h']:.6f}, h_1 = {unit['h']:.6f}, "
                    f"K_m = {K_MATRIX:g} N/um2, Sigma_1 = {sigma_1:.6g}, K_eff = {k_eff:.6g}\n")
        written += 1
        print(f"{os.path.basename(os.path.dirname(folder)):8s}{os.path.basename(folder):16s}{pk['xi']:8.4f}"
              f"{pk['p_ref']:10.3f}{pk['h']:9.4f}{unit['h']:9.4f}{eps:14.6e}")

    print(f"\n{written} cubes: {OUTPUT} written in {pk_root}/run_*/side*_seed*")
    if skipped:
        print(f"{len(skipped)} cubes left out (pk or unit run not solved): " + ", ".join(skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
