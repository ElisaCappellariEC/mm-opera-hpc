#!/usr/bin/env python3
"""Fractional gas release (FGR) of a bubble population, from test-bubble output.

For a rupture stress sigma_R, a bubble i "releases" its gas if the maximum
first principal stress found around it is larger than sigma_R (see the
README of mm-opera-hpc):

    FGR(sigma_R) = sum_i V_i [sigma_I,i > sigma_R] / sum_i V_i,  V_i = 4/3 pi r_i^3

Input: the CSV written by test-bubble (default bubbles_and_stresses_selected.txt)
with the columns  bubble_id,radius,pressure,sigma_I_max,x,y,z.
Run test-bubble with -sf 0 (and -dl > 0 for polydisperse bubbles) so that
every bubble has a value; bubbles written as "nan" are counted as not released.

Optional --shift-hydro: test-bubble imposes a zero macroscopic strain, so the
whole RVE is under a mean compressive stress (sig_hydro.txt, column Average).
Subtracting it (sigma_I - Average) gives the stress field of the same RVE
without that confinement, as explained in the README, BUT for bubble pressures
p_i - |Average| instead of p_i. It is exact only in this sense; with
different pressures per bubble it is an approximation.

Output: <prefix>.csv (sigma_R, FGR) and <prefix>.png.
"""

import argparse
import csv
import sys

import numpy as np


def read_bubbles(path):
    """Return radius, pressure, sigma_I_max arrays (nan for missing values)."""
    radius, pressure, sigma = [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        expected = {"radius", "pressure", "sigma_I_max"}
        if not expected.issubset(reader.fieldnames or []):
            sys.exit(f"{path}: expected the columns {sorted(expected)}, "
                     f"found {reader.fieldnames} (old test-bubble output?)")
        for row in reader:
            radius.append(float(row["radius"]))
            pressure.append(float(row["pressure"]))
            sigma.append(float(row["sigma_I_max"]))  # float("nan") is fine
    return np.array(radius), np.array(pressure), np.array(sigma)


def read_mean_hydrostatic_stress(path):
    """Return the 'Average' column of sig_hydro.txt (last data line)."""
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r and r[0].strip()]
    header = [h.strip() for h in rows[0]]
    values = rows[-1]
    return float(values[header.index("Average")])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="?", default="bubbles_and_stresses_selected.txt",
                        help="CSV written by test-bubble")
    parser.add_argument("--sig-hydro", default="sig_hydro.txt",
                        help="file with the mean hydrostatic stress (for --shift-hydro)")
    parser.add_argument("--shift-hydro", action="store_true",
                        help="subtract the mean hydrostatic stress from sigma_I (see above)")
    parser.add_argument("--sigma-r", nargs=3, type=float, metavar=("MIN", "MAX", "N"),
                        help="rupture stresses to test (default: range of the data, 200 values)")
    parser.add_argument("--output", default="fgr",
                        help="prefix of the output files (default: fgr)")
    args = parser.parse_args()

    radius, pressure, sigma = read_bubbles(args.csv)
    volume = 4.0 / 3.0 * np.pi * radius**3

    missing = np.isnan(sigma)
    if missing.any():
        print(f"WARNING: {missing.sum()} bubble(s) without value "
              f"({100 * volume[missing].sum() / volume.sum():.1f}% of the pore volume), "
              "counted as not released. Run test-bubble with -sf 0 and -dl > 0.")

    if args.shift_hydro:
        mean_sh = read_mean_hydrostatic_stress(args.sig_hydro)
        print(f"mean hydrostatic stress of the RVE: {mean_sh:g} -> sigma_I - ({mean_sh:g})")
        sigma = sigma - mean_sh

    valid = sigma[~missing]
    if args.sigma_r:
        smin, smax, n = args.sigma_r
        sigma_r = np.linspace(smin, smax, int(n))
    else:
        sigma_r = np.linspace(valid.min(), valid.max(), 200)

    # nan > s is False: bubbles without value never release
    with np.errstate(invalid="ignore"):
        fgr = np.array([volume[sigma > s].sum() for s in sigma_r]) / volume.sum()

    np.savetxt(f"{args.output}.csv", np.column_stack([sigma_r, fgr]),
               delimiter=",", header="sigma_R,FGR", comments="")
    print(f"{len(radius)} bubbles, pressures in [{pressure.min():g}, {pressure.max():g}], "
          f"sigma_I_max in [{valid.min():g}, {valid.max():g}]")
    print(f"wrote {args.output}.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available: no plot")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sigma_r, fgr)
    ax.set_xlabel(r"rupture stress $\sigma_R$ (unit of the bubble pressures)")
    ax.set_ylabel("FGR (released pore volume fraction)")
    ax.set_ylim(0, 1.02)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"{args.output}.png", dpi=150)
    print(f"wrote {args.output}.png")


if __name__ == "__main__":
    main()
