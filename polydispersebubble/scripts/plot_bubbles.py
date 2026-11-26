"""
Size of the RVE and fission gas release (FGR) of the bubble cubes of test_rve.py, as in OperaHPC D5.2 (Section 2.2, Fig. 7).

Folders of ROOT, all of the same side L, each with bubbles_<run>.txt and summary_<run>.txt (Mérope),
bubbles_and_stresses_selected.txt and sig_hydro.txt (MMM):
    side<L>_seed<NN>             one cube per seed
    side<L>_seed01_<variant>     the pores of side<L>_seed01 with another mesh of gmsh (the mesh test)

Stress factor of a pore, p its pressure, p_ref = sum(V p) / sum(V) the volume mean of the pressures of the cube:
    sigma = sigma_I_max / p          zero macroscopic strain (MMM)
    s_hat = (sigma + p_m) / (1 - p_m)
                                     zero stress (D5.2), same pressure in every pore (the unit runs): p_m p = -<sigma_h>,
                                     <sigma_h> the mean hydrostatic stress of the matrix (sig_hydro.txt; the pores are
                                     boundary conditions); a uniform p_m p brings it to zero
    s_hat = sigma + p_m p_ref (1 + s_hat_1) / p
                                     zero stress, pressure of its class in every pore (the pk cubes of set_pressures.py,
                                     plan B): MMM gives <sigma_h> = -p_m p_ref at zero strain; the state with no pressure
                                     in the pores and mean p_m p_ref of the matrix, p_m p_ref (I + S_1), brings it to zero,
                                     S_1 the field of the unit run of the same cube (its folder in mesh_case.txt) at zero
                                     stress, whose largest principal stress at the pore is s_hat_1. Exact when the
                                     pressures are equal (then s_hat = s_hat_1); otherwise it adds the maxima of the two
                                     runs as if they were at the same point with the same direction (the distance between
                                     the two points is printed)
A cube without s_hat (no sig_hydro.txt, or a pk cube whose unit run is not solved) is left out.
A pore breaks when s_hat p > sigma_R. All the pressures scale with the load x = p_ref / sigma_R: a pore breaks at
x_c = p_ref / (s_hat p), 1 / s_hat at one pressure; FGR = volume of the broken pores / volume of all the pores, a
function of x only. sigma_R is fitted as in D5.2: least squares of the mean FGR of the cubes against the FGR of the NFIR anneal against the bubble pressure (D5.2 Fig. 6 right). Biot: s_hat = 1/2 + xi p_ref / p, the isolated pore plus a stress of about xi p_ref (D5.2 footnote 2).

Questions, with Kanit et al., Int. J. Solids Struct. 40 (2003) 3647 (variance D^2 between cubes ~ V^-alpha, alpha = 1 for
independent pores; one cube is an RVE if 2 D / mean <= EPS_R):
 1. Is the mesh fine enough?                     mesh_test.txt/png
 2. Does a cube hold the whole population?       top of rve_size.txt
 3. How much does the FGR change between seeds?  rve_fgr.txt/png
 4. Which side gives the precision asked?        rve_fgr.txt, rve_size.txt/png

Usage: python3 plot_bubbles.py [ROOT]
Exit 1 if one cube is not an RVE, or if the mesh of the runs is more than EPS_MESH from the reference.

@author E. Cappellari
@date   October 2026
"""
# %% ------------------------------------------------------------------------------------------------------------------
# Libraries and paths
import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.expanduser(os.path.join("~", "mmm", "results", "test_rve_side30_mesh0.15_graded_rfine0.45_delta0.15", "run_NFIR"))  # default: the latest size test (the results live in ~/mmm/results, not in the build tree)
EPS_R = 0.05             # precision asked on a mean (relative, 95 %)
EPS_MESH = 0.02          # tolerance on the class means against the reference mesh
PREF = 1.0               # --pref of the MMM runs
FGR_LEVELS = (0.01, 0.015, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 0.95)  # levels of FGR at which the cubes are compared
VERDICT_LEVELS = (0.05, 0.95)                      # levels of FGR in the verdict (10-90 % until 11/11/2026); the
                                                   # levels below are the tail of the first pores, shown but left out
MIN_VOLUME_SHARE = 0.05  # classes with less of the pore volume stay out of the verdict
D52_SIGMA_R = 275.0      # MPa, sigma_R of D5.2 (monodisperse cubes): quoted next to the one fitted here
D52_SIDE = 30.0          # um, side of the RVE of D5.2
# NFIR anneal of D5.2 Fig. 6 right (digitised): bubble pressure (MPa, their Xe EoS at the measured gas density, 868-1185 C) and cumulative Kr-85 release (%)
D52_P_MPA = np.array([182.88, 192.10, 198.72, 205.09, 210.69, 216.32, 223.83, 229.43, 235.83, 240.06, 245.50])
D52_FGR = np.array([0.000, 0.329, 0.696, 1.353, 2.427, 3.771, 5.646, 7.589, 11.823, 16.754, 22.854])

STRESSES, HYDRO = "bubbles_and_stresses_selected.txt", "sig_hydro.txt"
MESH_CASE = "mesh_case.txt"  # in a pk cube of set_pressures.py: the folder of its unit run (same pores and mesh)
CASE = re.compile(r"side(?P<side>\d+(?:\.\d+)?)_seed(?P<seed>\d+)(?:_(?P<variant>.+))?$")
IN_VERDICT = {lv for lv in FGR_LEVELS if VERDICT_LEVELS[0] - 1e-9 <= lv <= VERDICT_LEVELS[1] + 1e-9}
S_HAT = r"$\hat s$"
SEQ_BLUE, GRID = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"), "#e1e0d9"
REFERENCE_LINE, D52_COLOR = "#8a8a85", "#c2410c"


# %% ------------------------------------------------------------------------------------------------------------------
# Functions 

def read_csv(path):
    """Columns of a csv with a header, as float arrays; None if the file is missing or has no rows."""
    if path is None or not os.path.isfile(path) or os.path.getsize(path) == 0:
        return None
    with open(path) as f:
        rows = [line.strip().split(",") for line in f if line.strip()]
    if len(rows) < 2:
        return None
    data = np.array([[float(x) for x in row] for row in rows[1:]])
    return {name.strip(): data[:, k] for k, name in enumerate(rows[0])}


def read_cube(folder):
    """A solved cube: mesh, classes of the summary and, per pore, id, R, V, gap, sigma, s_hat. None if MMM has not run or
    s_hat cannot be computed."""
    case = CASE.match(os.path.basename(folder))
    side = float(case["side"])

    # bubbles_and_stresses_selected.txt in the folder or in its shallowest subfolder
    hits = sorted(glob.glob(os.path.join(folder, "**", STRESSES), recursive=True), key=lambda p: (p.count(os.sep), p))
    if len(hits) > 1:
        print(f"{folder}: {len(hits)} {STRESSES}, reading {os.path.relpath(hits[0], folder)}")
    stresses = hits[0] if hits else None
    table = read_csv(stresses)
    if table is None:
        return None
    for column in ("bubble_id", "radius", "pressure", "sigma_I_max"):
        if column not in table:
            raise SystemExit(f"{stresses}: no column {column} (columns: {', '.join(table)})")

    pores = [p for p in glob.glob(os.path.join(folder, "bubbles_*.txt")) if os.path.basename(p) != STRESSES]
    if len(pores) != 1:
        raise SystemExit(f"{folder}: one bubbles_<run>.txt of Mérope expected, found {len(pores)}")
    bubbles = np.atleast_2d(np.loadtxt(pores[0]))  # id, x, y, z, radius, pressure
    row = {int(i): k for k, i in enumerate(bubbles[:, 0])}
    ids = table["bubble_id"].astype(int)
    if set(ids) != set(row):
        raise SystemExit(f"{folder}: the pores of {STRESSES} ({len(ids)}) are not those of {os.path.basename(pores[0])} "
                         f"({len(row)})")

    # summary of Mérope: mesh sizes, then one row per class (radius, pressure, porosity asked, porosity placed,
    # pores expected, pores placed)
    summaries = glob.glob(os.path.join(folder, "summary_*.txt"))
    text = open(summaries[0]).read() if summaries else ""
    # graded mesh (refine() of merope_pore_pressure.py): "mesh size <wall> um next to the pores, <bulk> um after ...",
    # and "x R / <r_fine> um" if the size next to the pores below r_fine scales with their radius; uniform: "mesh size <h>"
    m = (re.search(r"wall ([\d.eE+-]+) um, bulk ([\d.eE+-]+) um", text)
         or re.search(r"mesh size ([\d.eE+-]+) um next to the pores, ([\d.eE+-]+) um after", text))
    if m:
        wall, bulk = float(m.group(1)), float(m.group(2))
    else:
        m = re.search(r"mesh size ([\d.eE+-]+) um", text)
        wall = bulk = float(m.group(1)) if m else np.nan
    m = re.search(r"x R / ([\d.eE+-]+) um", text)
    r_fine = float(m.group(1)) if m else 0.0  # 0: no size scaled with the radius
    rows = []
    for line in text.splitlines():
        words = line.split()
        if line.lstrip().startswith("#") or len(words) != 6:
            continue
        try:
            rows.append([float(w) for w in words])
        except ValueError:
            continue
    a = np.array(rows)
    classes = ({"R": np.round(a[:, 0], 6), "asked": a[:, 2], "placed": a[:, 3], "expected": a[:, 4], "pores": a[:, 5]}
               if rows else None)

    # gap from the wall of every pore to the wall of its nearest neighbour, in the periodic cube
    centres, radii = bubbles[:, 1:4], bubbles[:, 4]
    d = centres[:, None, :] - centres[None, :, :]
    d -= side * np.round(d / side)
    gaps = np.linalg.norm(d, axis=2) - radii[:, None] - radii[None, :]
    np.fill_diagonal(gaps, np.inf)
    gap = gaps.min(axis=1)[[row[i] for i in ids]]

    p = PREF * table["pressure"]
    sigma = table["sigma_I_max"] / p
    R = np.round(table["radius"], 6)
    V = 4.0 / 3.0 * np.pi * R**3
    p_ref = np.sum(V * p) / np.sum(V)  # all the pores: those of the table are those of the bubbles file
    unit_folder = None
    if os.path.isfile(os.path.join(folder, MESH_CASE)):
        with open(os.path.join(folder, MESH_CASE)) as f:
            unit_folder = os.path.normpath(os.path.join(folder, f.readline().strip()))
    load = "pk" if unit_folder else ("unit" if np.ptp(p) == 0.0 else "mixed")
    max_point = np.column_stack([table[k] for k in "xyz"]) if set("xyz") <= set(table) else None
    cube = {"folder": os.path.basename(folder), "side": side, "seed": int(case["seed"]), "variant": case["variant"],
            "wall": wall, "bulk": bulk, "r_fine": r_fine, "classes": classes, "id": ids, "R": R,
            "V": V, "gap": gap, "gap_R": gap / R, "sigma": sigma, "s_hat": None,
            "p": p, "p_ref": p_ref, "load": load, "unit": unit_folder, "s_hat_1": None, "max_shift": None,
            # centre of every pore and point of its maximum (x, y, z of MMM), in the order of the rows above
            "centre": centres[[row[i] for i in ids]], "max_point": max_point}
    if not np.isfinite(sigma).all():
        print(f"{folder}: {np.count_nonzero(~np.isfinite(sigma))} pores of {len(sigma)} without a stress (nan), left out")

    hydro = read_csv(os.path.join(os.path.dirname(stresses), HYDRO))
    if hydro is None or not {"Volume", "Average"} <= set(hydro):
        print(f"{folder}: no {HYDRO}, no s_hat: left out")
        return None
    # Volume and Average of sig_hydro.txt are over the matrix only: 1 - Volume / L^3 is the porosity
    xi = 1.0 - hydro["Volume"][0] / side**3
    xi_pores = np.sum(4.0 / 3.0 * np.pi * radii**3) / side**3
    if abs(xi - xi_pores) > 0.01:
        print(f"{folder}: porosity {xi:.4f} from the volume of {HYDRO}, {xi_pores:.4f} from the pores: wrong side or "
              "output of another run? s_hat of this cube is suspect")
    p_m = -hydro["Average"][0] / p_ref  # mean hydrostatic stress of the matrix at zero strain, over -p_ref
    cube.update(xi=xi, p_m=p_m)
    if load == "unit":
        cube["s_hat"] = (sigma + p_m) / (1.0 - p_m)
        return cube
    if load == "mixed":
        print(f"{folder}: pores at different pressures and no {MESH_CASE} (the unit run of the same cube, "
              "set_pressures.py): no s_hat, left out")
        return None

    # pk cube: s_hat_1 of its unit run, in the order of ids, and the distance between the points of the two maxima
    unit = read_cube(unit_folder) if os.path.isdir(unit_folder) else None
    if unit is None or unit["load"] != "unit":
        print(f"{folder}: the unit run {unit_folder} of {MESH_CASE} is not there or not solved (or not one pressure): "
              "no s_hat, left out")
        return None
    k = {int(i): j for j, i in enumerate(unit["id"])}
    if set(k) != set(int(i) for i in ids):
        raise SystemExit(f"{folder}: other pores than its unit run {unit_folder}")
    order = [k[int(i)] for i in ids]
    mine = bubbles[[row[i] for i in ids], 1:5]
    theirs = np.column_stack([unit["centre"], unit["R"]])[order]
    if not np.allclose(mine, theirs, rtol=0.0, atol=1e-6):
        raise SystemExit(f"{folder}: the pores differ from those of its unit run {unit_folder} (set_pressures.py)")
    if unit["max_point"] is not None and max_point is not None:
        d = max_point - unit["max_point"][order]
        d -= side * np.round(d / side)
        shift = np.linalg.norm(d, axis=1)
        ok = np.isfinite(shift)
        print(f"{folder}: point of the maximum {np.median(shift[ok]):.3f} um from that of the unit run (median), "
              f"{np.quantile(shift[ok], 0.95):.3f} um (95 %), "
              f"{100 * np.mean(shift[ok] > 0.5 * unit['R'][order][ok]):.1f} % of the pores beyond R / 2")
        cube["max_shift"] = shift
    s_hat_1 = unit["s_hat"][order]
    cube.update(s_hat=sigma + p_m * p_ref * (1.0 + s_hat_1) / p, s_hat_1=s_hat_1)
    return cube


def kanit(Z, side):
    """Kanit statistics of a quantity measured once per cube (nan below 3 cubes): mean, D^2, half-width 2 D / sqrt(n),
    eps = 2 D / mean, cubes needed for EPS_R, side of one cube that is an RVE (alpha = 1) and its 95 % interval."""
    Z = np.asarray(Z, dtype=float)
    Z = Z[np.isfinite(Z)]
    n = len(Z)
    r = {"n": n, "side": side, "mean": Z.mean() if n else np.nan, "D2": np.nan, "half": np.nan, "eps": np.nan,
         "cubes": np.nan, "rve": np.nan, "rve_low": np.nan, "rve_high": np.nan}
    if n < 3:
        return r
    D2 = Z.var(ddof=1)
    eps = 2.0 * np.sqrt(D2) / abs(r["mean"])
    cubes = (eps / EPS_R) ** 2
    # 97.5 % and 2.5 % quantiles of chi-square with k = n - 1 degrees of freedom (Wilson-Hilferty)
    k = n - 1
    chi2_high, chi2_low = (k * (1.0 - 2.0 / (9.0 * k) + z * np.sqrt(2.0 / (9.0 * k))) ** 3 for z in (1.959964, -1.959964))
    low, high = k / chi2_high, k / chi2_low
    r.update(D2=D2, half=2.0 * np.sqrt(D2 / n), eps=eps, cubes=cubes, rve=side * np.cbrt(cubes),
             rve_low=side * np.cbrt(cubes * low), rve_high=side * np.cbrt(cubes * high))
    r["interval"] = f"{r['rve_low']:.0f}-{r['rve_high']:.0f}"
    return r


def percent(share):
    """A share as a percentage, with one decimal below 10 %."""
    return f"{100 * share:.2f}" if share < 0.1 else f"{100 * share:.0f}"


def class_ticks(radii, shares):
    """Tick labels of the classes: radius and share of the pore volume."""
    return [f"{R:.3g}\n{percent(s)}% vol" for R, s in zip(radii, shares)]


def class_color(k, n):
    """Colour of the class k of n (viridis)."""
    return plt.get_cmap("viridis")(k / max(n - 1, 1))


def completeness(cubes):
    """Classes of the measured distribution against the pores placed in the cubes; None without classes."""
    cubes = [c for c in cubes if c["classes"] is not None]
    if not cubes:
        return None
    ref = cubes[0]["classes"]
    placed = np.array([c["classes"]["pores"] for c in cubes])
    porosity = np.array([c["classes"]["placed"] for c in cubes])
    real = ref["asked"] / ref["asked"].sum()
    return {"side": cubes[0]["side"], "n": len(cubes), "R": ref["R"], "real": real,
            "cube": (porosity / porosity.sum(axis=1, keepdims=True)).mean(axis=0), "expected": ref["expected"],
            "placed_min": placed.min(axis=0), "placed_max": placed.max(axis=0), "porosity_asked": ref["asked"].sum(),
            "porosity": porosity.sum(axis=1).mean(),
            "missing": [k for k in range(len(ref["R"])) if placed.max(axis=0)[k] == 0 and real[k] > 0.0]}


def side_for(info, k, pores):
    """Side (um) of a cube expected to hold the given number of pores of class k."""
    return info["side"] * np.cbrt(pores / info["expected"][k]) if info["expected"][k] > 0.0 else np.inf


def critical_x(cube):
    """x_c = p_ref / (s_hat p) of every pore of the cube: the load x = p_ref / sigma_R at which s_hat p x / p_ref reaches
    1 (all the pressures scale together); 1 / s_hat at one pressure, infinite for s_hat <= 0 or nan."""
    s = cube["s_hat"]
    ok = np.isfinite(s) & (s > 0.0)
    return np.where(ok, 1.0 / np.where(ok, s * cube["p"] / cube["p_ref"], 1.0), np.inf)


def critical_pressures(cube):
    """x_c (critical_x) of the pores with a value of s_hat, and their volumes."""
    ok = np.isfinite(cube["s_hat"])
    return critical_x(cube)[ok], cube["V"][ok]


def fgr_curve(cube, x):
    """FGR of a cube at the normalised loads x = p_ref / sigma_R."""
    xc, V = critical_pressures(cube)
    return (xc[None, :] < x[:, None]) @ V / V.sum()


def x_levels(cube):
    """x_c at which the FGR of a cube reaches every level of FGR_LEVELS."""
    xc, V = critical_pressures(cube)
    order = np.argsort(xc)
    released = np.cumsum(V[order]) / V.sum()
    return np.array([xc[order][min(np.searchsorted(released, lv - 1e-12), len(xc) - 1)] for lv in FGR_LEVELS])


def draw_fgr(ax, cubes, x, legend=True, scale=1.0):
    """FGR curves of every cube (thin) and their mean with the band +/- 2 D of one cube, in %, against scale * x."""
    cubes = sorted(cubes, key=lambda c: c["seed"])
    curves = np.array([fgr_curve(c, x) for c in cubes])
    for j, (c, f) in enumerate(zip(cubes, curves)):
        ax.plot(scale * x, 100 * f, lw=0.9, color=plt.get_cmap("tab10")(j % 10), alpha=0.85,
                label=f"seed {c['seed']:02d}" if legend else None)
    mean = curves.mean(axis=0)
    if len(curves) > 1:
        D = curves.std(axis=0, ddof=1)
        ax.fill_between(scale * x, 100 * np.clip(mean - 2 * D, 0, 1), 100 * np.clip(mean + 2 * D, 0, 1), color="black",
                        alpha=0.10, lw=0, label=r"mean $\pm\,2D$ (spread of one cube)" if legend else None)
    ax.plot(scale * x, 100 * mean, color="black", lw=2.2, label=f"mean of {len(cubes)} cubes" if legend else None)


def fit_sigma_r(cubes):
    """sigma_R (MPa) as in D5.2: least squares of the mean FGR of the cubes at x = p / sigma_R against the FGR of the NFIR
    anneal (D52_P_MPA, D52_FGR), on a grid of 0.1 MPa; and its rms distance (points of FGR)."""
    parts = [critical_pressures(c) for c in cubes]
    xc = np.concatenate([x for x, _ in parts])
    w = np.concatenate([V / V.sum() / len(cubes) for _, V in parts])
    order = np.argsort(xc)
    xc, released = xc[order], np.concatenate([[0.0], 100.0 * np.cumsum(w[order])])  # mean FGR (%) below every x_c
    finite = xc[np.isfinite(xc)]
    trial = np.arange(D52_P_MPA.min() / finite.max(), D52_P_MPA.max() / finite.min(), 0.1)
    rms = np.array([np.sqrt(np.mean((released[np.searchsorted(xc, D52_P_MPA / s)] - D52_FGR) ** 2)) for s in trial])
    return trial[rms.argmin()], rms.min()


def smallest_rve_level(stats, side, key):
    """Smallest level of FGR from which the RVE side `key` ('rve' or 'rve_high') of stats {level: kanit} stays below the
    side of the cubes, linear in log(RVE / side) between the two levels around the last crossing; text."""
    levels = [lv for lv in sorted(stats) if np.isfinite(stats[lv][key])]
    if not levels:
        return "unknown (fewer than 3 cubes)"
    f = [np.log(stats[lv][key] / side) for lv in levels]
    failing = [k for k, v in enumerate(f) if v > 0.0]
    if not failing:
        return f"every level from {100 * levels[0]:g} %"
    k = failing[-1]
    if k == len(levels) - 1:
        return f"none up to {100 * levels[-1]:g} %"
    lv = levels[k] + (levels[k + 1] - levels[k]) * f[k] / (f[k] - f[k + 1])
    return f"{100 * lv:.1f} %"


def class_stats(cubes, q, info):
    """Kanit statistics of the class means of q (s_hat or sigma) over the cubes, one entry per class of radius."""
    side = cubes[0]["side"]
    total = sum(c["V"].sum() for c in cubes)
    results = []
    for R in np.unique(np.concatenate([c["R"] for c in cubes])):
        groups = [g for g in (c[q][(c["R"] == R) & np.isfinite(c[q])] for c in cubes) if len(g)]
        pooled = np.concatenate(groups)
        r = kanit([g.mean() for g in groups], side)
        S = pooled.std(ddof=1) if len(pooled) > 1 else np.nan
        pores = np.mean([len(g) for g in groups])
        k = np.flatnonzero(np.isclose(info["R"], R, atol=1e-6)) if info is not None else []
        r.update(R=R, n_all=len(cubes), pores=pores, S=S,
                 A_x_dens=r["D2"] * pores / S**2 if S > 0.0 else np.nan,
                 vol_cube=sum(c["V"][c["R"] == R].sum() for c in cubes) / total,
                 vol_real=info["real"][k[0]] if len(k) else np.nan)
        r["share"] = r["vol_real"] if np.isfinite(r["vol_real"]) else r["vol_cube"]
        results.append(r)
    return results


# %% ------------------------------------------------------------------------------------------------------------------
# Main

def main():
    """The four questions on the folders of ROOT (argument or default). Returns the worst status."""
    root = sys.argv[1] if len(sys.argv) > 1 else ROOT
    folders = sorted(f for f in glob.glob(os.path.join(root, "side*_seed*"))
                     if os.path.isdir(f) and CASE.match(os.path.basename(f)))
    if not folders:
        print(f"no side<L>_seed<NN> folder in {root}: nothing to analyse (python3 test_rve.py writes them)")
        return 0
    cubes, unsolved = [], []
    for folder in folders:
        cube = read_cube(folder)
        (cubes if cube else unsolved).append(cube or os.path.basename(folder))
    print(f"{root}: {len(cubes)} folders solved" + (f", not solved yet: {', '.join(unsolved)}" if unsolved else ""))
    if not cubes:
        return 0
    if len({c["side"] for c in cubes}) > 1:
        raise SystemExit(f"{root}: cubes of several sides ({', '.join(sorted({format(c['side'], 'g') for c in cubes}))}): "
                         "one side per folder")

    # %% 1. Is the mesh fine enough? side<L>_seed01 on the meshes of gmsh of the mesh test -------------------------------
    print("\n== 1. mesh ==")
    family = sorted((c for c in cubes if c["seed"] == 1), key=lambda c: (c["wall"], c["bulk"]))
    mesh_status, mesh_summary = 0, "# - mesh: not tested (no variant of side<L>_seed01 solved)"
    if len(family) < 2 or not any(c["variant"] for c in family):
        print(f"{root}: no mesh variant of a seed01 solved, no mesh test")
    else:
        side = family[0]["side"]
        finest, runs = family[0], next((c for c in family if c["variant"] is None), None)
        ids = sorted(set.intersection(*(set(c["id"][np.isfinite(c["s_hat"])]) for c in family)))
        value = {c["folder"]: dict(zip(c["id"], c["s_hat"])) for c in family}
        R = dict(zip(finest["id"], finest["R"]))
        gap_R = dict(zip(finest["id"], finest["gap_R"]))
        radii = sorted({R[i] for i in ids})
        shares = np.array([finest["V"][finest["R"] == Rk].sum() for Rk in radii]) / finest["V"].sum()
        used = np.append(shares >= MIN_VOLUME_SHARE, False)  # the classes of the verdict, not "all"
        means = {c["folder"]: np.array([np.mean([value[c["folder"]][i] for i in ids if R[i] == Rk]) for Rk in radii] +
                                       [np.mean([value[c["folder"]][i] for i in ids])]) for c in family}
        pore = {c["folder"]: np.array([100.0 * (value[c["folder"]][i] / value[finest["folder"]][i] - 1.0) for i in ids])
                for c in family}  # every pore against the finest mesh (%)

        # reference: Richardson on three wall sizes or more, of meshes alike: uniform meshes, graded ones that scale as a
        # whole (same bulk / wall and r_fine, as the MESH_SIZES of test_rve.py), or graded ones with one bulk size; it has
        # to hold on every class of the verdict (all pores if none), the other classes keep the finest mesh when their fit
        # fails; otherwise the finest mesh
        reference, how, extrapolated = means[finest["folder"]], f"the finest mesh of gmsh, {finest['folder']}", False
        failed = None  # (group, sizes, classes of the verdict with their reason) of the last Richardson fit not possible
        members = lambda k: [i for i in ids if k == len(radii) or R[i] == radii[k]]  # pores of the class k (len(radii): all)
        verdict = [k for k in range(len(radii)) if used[k]] or [len(radii)]
        label = lambda k: (f"{radii[k]:.3f} ({100 * shares[k]:.1f} % vol, {len(members(k))} pores)" if k < len(radii)
                           else f"all ({len(ids)} pores)")
        uniform = lambda c: bool(np.isclose(c["wall"], c["bulk"], rtol=1e-3))
        groups = [("uniform meshes", [c for c in family if uniform(c)])]
        alike = []  # graded meshes by bulk / wall and r_fine; the summary rounds the bulk size, hence the tolerance
        for c in (c for c in family if not uniform(c)):
            ratio = c["bulk"] / c["wall"]
            same = next((g for g in alike if np.isclose(g[0], ratio, rtol=5e-3) and np.isclose(g[1], c["r_fine"])), None)
            if same:
                same[2].append(c)
            else:
                alike.append((ratio, c["r_fine"], [c]))
        groups += [(f"graded meshes, bulk = {k:.3g} x wall" + (f", x R/{r:g} below R = {r:g} um" if r > 0.0 else ""), g)
                   for k, r, g in alike]
        groups += [(f"bulk {bulk:g} um", [c for c in family if not uniform(c) and c["bulk"] == bulk])
                   for bulk in sorted({c["bulk"] for c in family if not uniform(c)})]
        for where, group in groups:
            if len({c["wall"] for c in group}) < 3:
                continue
            (h1, h2, h3), fits = [c["wall"] for c in group[:3]], []
            for k in range(len(radii) + 1):
                # s(h) = s0 + C h^p on the three finest sizes h1 < h2 < h3, or the reason why there is none: the means
                # not monotone in the size, or no order p in [0.05, 8]. With h3/h2 = h2/h1 = r the steps d1 = s2 - s1,
                # d2 = s3 - s2 must give d2/d1 = r^p: 1.41 for p = 1 and 2 for p = 2 at r = sqrt(2); d2/d1 < 1, the
                # finer step larger than the coarser one, has no positive order
                s1, s2, s3 = [means[c["folder"]][k] for c in group[:3]]
                if (s2 - s1) * (s3 - s2) <= 0.0:
                    fit = "not monotone in the size"
                else:
                    target = (s3 - s2) / (s2 - s1)
                    f = lambda p: (h3**p - h2**p) / (h2**p - h1**p) - target  # grows with p
                    low, high = 0.05, 8.0
                    if f(low) * f(high) > 0.0:
                        fit = (f"finer step too large for an order above {low:g} (d2/d1 = {target:.2f})" if f(low) > 0.0
                               else f"finer step too small for an order below {high:g} (d2/d1 = {target:.2f})")
                    else:
                        for _ in range(80):
                            mid = 0.5 * (low + high)
                            low, high = (low, mid) if f(low) * f(mid) <= 0.0 else (mid, high)
                        p = 0.5 * (low + high)
                        fit = p, s1 - (s2 - s1) * h1**p / (h2**p - h1**p)
                # not a trend if one of the two steps is within 2 standard errors of the mean of the pores of the class
                # (a different mesh moves every pore a little)
                pores_k = members(k)
                v = np.array([[value[c["folder"]][i] for i in pores_k] for c in group[:3]])
                steps = np.diff(v, axis=0) / v[0].mean()  # d1, d2 of every pore, relative to the mean of the finest
                se = steps.std(axis=1, ddof=1) / np.sqrt(len(pores_k)) if len(pores_k) > 1 else np.full(2, np.inf)
                if not np.all(np.abs(steps.mean(axis=1)) > 2.0 * se):
                    why = ("steps within the noise of the mean of its pores ("
                           + ", ".join(f"{100 * d:.2f} +/- {200 * e:.2f} %" for d, e in zip(steps.mean(axis=1), se)) + ")")
                    fit = f"{fit}; {why}" if isinstance(fit, str) else f"order {fit[0]:.2f} from {why}"
                fits.append(fit)
            ok = [not isinstance(fit, str) for fit in fits]
            if all(ok[k] for k in verdict):
                reference = np.array([fit[1] if good else m for fit, good, m in zip(fits, ok, means[finest["folder"]])])
                extrapolated = True
                how = (f"Richardson on the wall sizes {h1:g}, {h2:g}, {h3:g} um ({where}), observed order per class "
                       f"{', '.join(f'{fit[0]:.2f}' if good else '-' for fit, good in zip(fits[:-1], ok))}"
                       + (f", all pores {fits[-1][0]:.2f}" if ok[-1] else ", all pores -"))
                if not all(ok):
                    how += (f" (-: no fit, against {finest['folder']}: "
                            + "; ".join(f"{label(k)} {fits[k]}" for k in range(len(fits)) if not ok[k]) + ")")
                break
            failed = (where, len(group), [f"{label(k)}: {fits[k]}" for k in verdict if not ok[k]])
            print(f"{root}: Richardson on the {where} not possible for "
                  + "; ".join(f"{label(k)}: {fits[k]}" for k in range(len(fits)) if not ok[k]))
        coarse = family if extrapolated else family[1:]  # the meshes against the reference
        deviation = {c["folder"]: 100.0 * (means[c["folder"]] / reference - 1.0) for c in family}

        names = [f"{Rk:.3f}" for Rk in radii] + ["all"]
        h_small = lambda c: c["wall"] * (min(1.0, min(radii) / c["r_fine"]) if c["r_fine"] > 0.0 else 1.0)  # next to the smallest pore
        lines = [f"# 1. Is the mesh fine enough? s_hat of the {len(ids)} pores of side{side:g}_seed01 with {len(family)} meshes",
                 f"# - reference: {how}",
                 f"# - tolerance on the class means: {100 * EPS_MESH:g} %",
                 f"# - smallest radius {min(radii):.3f} um, smallest gap between two pores "
                 f"{min(gap_R[i] * R[i] for i in ids):.3f} um",
                 "# - h/Rmin: mesh size next to the smallest pore / its radius (wall size, or less if it scales with the "
                 "radius below r_fine)",
                 "# deviation of the class means from the reference (%), then per pore from the finest mesh: median "
                 "and largest |d| (%)",
                 f"# {'folder':<38s}{'wall':>7s}{'bulk':>7s}{'h/Rmin':>7s}" + "".join(f"{n:>8s}" for n in names)
                 + f"{'med|d|':>8s}{'max|d|':>8s}",
                 f"# {'share of the pore volume (%)':<59s}" + "".join(f"{100 * s:8.1f}" for s in shares) + f"{100.0:8.1f}"]
        for c in family:
            lines.append(f"  {c['folder']:<38s}{c['wall']:7.4g}{c['bulk']:7.4g}{h_small(c) / min(radii):7.2f}"
                         + "".join(f"{x:8.2f}" for x in deviation[c["folder"]])
                         + f"{np.median(np.abs(pore[c['folder']])):8.2f}{np.abs(pore[c['folder']]).max():8.2f}")

        # verdict: against the extrapolated value or, without it, between the two finest meshes
        small = [f"R = {radii[k]:.3g} um ({100 * shares[k]:.1f} % vol) changes by "
                 f"{max(abs(deviation[c['folder']][k]) for c in coarse):.0f} %"
                 for k in range(len(radii)) if not used[k]]
        worst = lambda c: np.abs(deviation[c["folder"]][used]).max() if used.any() else 0.0
        if extrapolated:
            if runs is None:
                summary = "extrapolated value available, but no folder without a variant (the mesh of the runs) to compare"
            else:
                mesh_status = int(worst(runs) > 100 * EPS_MESH)
                summary = (f"the mesh of the runs ({runs['wall']:g} um) is within {worst(runs):.2f} % of the extrapolated "
                           f"value on the classes of the verdict: {'converged' if not mesh_status else 'NOT converged'} at "
                           f"{100 * EPS_MESH:g} %")
        else:
            following = coarse[0]  # the mesh just coarser than the finest
            ratio = following["wall"] / finest["wall"]
            # s(h) = s0 + C h^p with p fixed: the finest mesh is off by d / (r^p - 1), the next one by d r^p / (r^p - 1)
            guess = lambda p, c: worst(following) * (ratio**p if c is following else 1.0) / (ratio**p - 1.0)
            orders = (f"; against h -> 0 (not measured) with an order 2 (1) {following['wall']:g} um would be off by "
                      f"{guess(2, following):.2f} % ({guess(1, following):.2f} %), {finest['wall']:g} um by "
                      f"{guess(2, finest):.2f} % ({guess(1, finest):.2f} %)" if ratio > 1.0 else "")
            summary = (f"the classes of the verdict change by at most {worst(following):.2f} % from {following['wall']:g} "
                       f"to {finest['wall']:g} um{orders}; "
                       + (f"Richardson not possible on the {failed[1]} {failed[0]}, for {'; '.join(failed[2])}" if failed
                          else "add a third mesh size of gmsh"))
            if runs is not None and runs is not finest:
                mesh_status = int(worst(runs) > 100 * EPS_MESH)
        lines += ["#", f"# {summary}"] + (["# not in the verdict: " + "; ".join(small)] if small else [])
        mesh_summary = f"# - mesh: {summary}"
        with open(os.path.join(root, "mesh_test.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")

        # figure: class means against the reference
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        width_bar = 0.8 / max(len(coarse), 1)
        big = max(abs(deviation[c["folder"]]).max() for c in coarse) > 20.0
        for j, c in enumerate(coarse):  # finest first: darkest to lightest blue
            x = np.arange(len(names)) - 0.4 + width_bar * (j + 0.5)
            d = deviation[c["folder"]]
            ax.bar(x, d, width=0.9 * width_bar, zorder=2,
                   color=SEQ_BLUE[len(SEQ_BLUE) - 1 - round(j * (len(SEQ_BLUE) - 1) / max(len(coarse) - 1, 1))],
                   label=f"{c['folder']} (wall {c['wall']:.3g}, bulk {c['bulk']:.3g} um)")
            for xk, dk in zip(x, d):
                ax.annotate(f"{dk:.1f}", (xk, dk), ha="center", va="bottom" if dk >= 0 else "top", fontsize=9,
                            xytext=(0, 2 if dk >= 0 else -2), textcoords="offset points")
        ax.axhspan(-100 * EPS_MESH, 100 * EPS_MESH, color=GRID, lw=0, zorder=0, label=rf"$\pm${100 * EPS_MESH:g} %")
        ax.axhline(0.0, color=REFERENCE_LINE, lw=0.8)
        ax.set_xticks(range(len(names)), class_ticks(radii, shares) + ["all\npores"])
        ax.set(xlabel="radius of the class (um) and its share of the pore volume",
               ylabel="class mean against the reference (%)" + (" (symlog)" if big else ""),
               title="Class means against " + ("the extrapolated value (Richardson)" if extrapolated else
                                               f"the finest mesh available ({finest['wall']:g} um)"))
        ax.margins(y=0.1)
        if big:  # symlog, linear within +/- 5 %
            ax.set_yscale("symlog", linthresh=5.0)
            low, high = ax.get_ylim()
            ticks = [t for t in (-200, -100, -50, -20, -10, -5, -2, 0, 2, 5, 10, 20, 50, 100, 200) if low <= t <= high]
            ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
        ax.grid(True, color=GRID)
        ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.25))  # under the axes
        fig.tight_layout()
        fig.savefig(os.path.join(root, "mesh_test.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("\n".join(lines))
        print(f"wrote mesh_test.txt and mesh_test.png in {root}")

    cubes = [c for c in cubes if c["variant"] is None]  # the size test: one cube per seed
    if not cubes:
        print(f"{root}: no cube of the size test solved")
        return mesh_status
    L, n = cubes[0]["side"], len(cubes)

    # %% 3. FGR of every cube and Kanit statistics of the critical pressure at every level of FGR_LEVELS -------------------
    print("\n== 3. FGR of every cube ==")
    xl = np.array([x_levels(c) for c in cubes])  # per cube, x_c at every level
    fgr = {lv: kanit(xl[:, k], L) for k, lv in enumerate(FGR_LEVELS)}
    sigma_r, rms = fit_sigma_r(cubes)
    one = np.array([fit_sigma_r([c])[0] for c in cubes])  # sigma_R fitted on one cube
    pk = any(c["load"] != "unit" for c in cubes)
    xc = np.concatenate([critical_pressures(c)[0] for c in cubes])
    xc = xc[np.isfinite(xc)]
    # per cube, x_c of every pore with Biot: s_hat p / p_ref = p / (2 p_ref) + xi
    biot = [1.0 / (0.5 * c["p"] / c["p_ref"] + c["xi"]) for c in cubes]
    xi = np.mean([c["xi"] for c in cubes])
    xb = np.concatenate(biot)
    lines = [f"# 3. FGR of every cube, as OperaHPC D5.2 Fig. 7 (s_hat; python3 plot_bubbles.py {root})",
             "# - a pore breaks when s_hat p > sigma_R; FGR(p) = volume of the broken pores / volume of all the pores",
             "# - x_c = p_ref / sigma_R at which the FGR of a cube reaches the level (mean over the cubes +/- 2 D / sqrt(n)); "
             "p_ref = sum(V p) / sum(V), the pressure of every pore when they all have the same",
             "# - sigma_R as in D5.2: least squares of the mean FGR of the cubes at x = p / sigma_R against the 11 points of "
             "the NFIR anneal (D5.2 Fig. 6 right: bubble pressure of D5.2, Kr-85 release compared directly); "
             "p_MPa = mean_x_c x sigma_R",
             f"# - eps_cube = 2 D / mean: error of x_c, i.e. of a sigma_R calibrated on the release, from one cube",
             f"# - cubes = (eps_cube / {100 * EPS_R:g} %)^2 cubes of this side; RVE_um = side x cubes^(1/3) (alpha = 1), "
             "with its 95 % interval",
             "# - first and last pore to break over all the cubes, and Biot (s_hat = 1/2 + xi p_ref / p, D5.2 footnote 2): "
             "their x_c; at one pressure s_hat = 1 / x_c, over 1/2 of the isolated pore",
             "#", f"# side {L:g} um, {n} cubes",
             f"# sigma_R = {sigma_r:.1f} MPa, rms {rms:.2f} points of FGR (D5.2, monodisperse cubes: {D52_SIGMA_R:g} MPa); "
             f"one cube: {one.min():.1f}-{one.max():.1f} MPa"
             + (f", 2 D = {2 * one.std(ddof=1):.1f} MPa ({200 * one.std(ddof=1) / one.mean():.1f} %)" if n > 1 else ""),
             f"# x_c: first pore {xc.min():.3f}, last pore {xc.max():.3f}, Biot "
             + (f"{xb.min():.3f}-{xb.max():.3f}" if pk else
                f"{xb.mean():.3f} (xi = {xi:.4f}); over the isolated pore (2 / x_c): first "
                f"{2 / xc.min():.2f}, last {2 / xc.max():.2f}, Biot {2 / xb.mean():.2f} (D5.2: about 3 and 1.2)"),
             f"#  {'FGR_%':>5s}{'mean_x_c':>10s}{'+/-':>8s}{'D':>10s}{'eps_cube_%':>11s}{'cubes':>7s}{'RVE_um':>8s}"
             f"{'[95 %]':>10s}{'p_MPa':>8s}"]
    for level, r in fgr.items():
        note = "" if level in IN_VERDICT else f"   (below {100 * VERDICT_LEVELS[0]:g} %: a tail, not in the verdict)"
        if np.isnan(r["cubes"]):
            lines.append(f"   {100 * level:5.3g}{r['mean']:10.4f}   fewer than 3 cubes{note}")
            continue
        lines.append(f"   {100 * level:5.3g}{r['mean']:10.4f}{r['half']:8.4f}{np.sqrt(r['D2']):10.2e}"
                     f"{100 * r['eps']:11.2f}{r['cubes']:7.1f}{r['rve']:8.1f}{r['interval']:>10s}"
                     f"{sigma_r * r['mean']:8.1f}{note}")
    lines.append(f"# smallest level of FGR from which one cube of {L:g} um is an RVE: "
                 f"{smallest_rve_level(fgr, L, 'rve')} (best estimate), "
                 f"{smallest_rve_level(fgr, L, 'rve_high')} (upper bound of the 95 % interval of the RVE side); "
                 "linear between the levels")
    with open(os.path.join(root, "rve_fgr.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # figure: FGR against the pressure at the fitted sigma_R with the NFIR anneal and a zoom on the onset (left, as D5.2
    # Fig. 7), the same against x = p / sigma_R with Biot (middle), the error of one cube per level (right)
    x = np.linspace(0.97 * xc.min(), 1.03 * max(xc.max(), xb.max()), 1000)
    fgr_biot = np.mean([(b[None, :] < x[:, None]) @ c["V"] / c["V"].sum() for c, b in zip(cubes, biot)], axis=0)
    fig, ax = plt.subplots(1, 3, figsize=(21.0, 5.8), gridspec_kw={"width_ratios": [1.4, 1.4, 1.0]})
    for a, scale in ((ax[0], sigma_r), (ax[1], 1.0)):
        draw_fgr(a, cubes, x, legend=a is ax[0], scale=scale)
        a.plot(scale * x, 100 * fgr_biot, "--", color=REFERENCE_LINE, lw=1.6,
               label=rf"Biot, {S_HAT} = 1/2 + $\xi$" + (r" $p_{ref} / p$" if pk else ""))
        a.plot(D52_P_MPA * scale / sigma_r, D52_FGR, "*-", color=D52_COLOR, ms=9, lw=1.2, label="NFIR anneal (D5.2 Fig. 6)")
        a.set(xlim=(scale * x[0], scale * x[-1]), ylim=(0.0, 100.0), ylabel="FGR (% of the pore volume)")
        a.grid(True, color=GRID)
    ax[0].set(xlabel=f"{'volume mean of the pore pressures' if pk else 'bubble pressure'} (MPa)",
              title=rf"FGR against the pressure, $\sigma_R$ = {sigma_r:.0f} MPa fitted on the NFIR anneal")
    ax[0].legend(loc="upper left", frameon=False, fontsize=9, ncol=2)
    ax[1].set(xlabel=(r"$p_{ref} / \sigma_R$ (volume mean of the pore pressures / rupture stress)" if pk
                      else r"$p / \sigma_R$ (pore pressure / rupture stress)"),
              title=r"FGR against the load $x$, and Biot")
    ax[1].legend(loc="upper left", frameon=False, fontsize=9)
    if not pk:
        ax[1].text(0.92, 0.04, f"{S_HAT} over the isolated pore (1/2):\nfirst pore {2 / xc.min():.1f}, last "
                   f"{2 / xc.max():.2f}, Biot {2 / xb.mean():.2f}\nD5.2: about 3 and 1.2", transform=ax[1].transAxes,
                   ha="right", va="bottom", fontsize=9, color=REFERENCE_LINE)
    zoom = ax[0].inset_axes([0.45, 0.08, 0.36, 0.40])
    first = np.mean([fgr_curve(c, x) for c in cubes], axis=0)
    draw_fgr(zoom, cubes, x, legend=False, scale=sigma_r)
    zoom.plot(D52_P_MPA, D52_FGR, "*-", color=D52_COLOR, ms=7, lw=1.0)
    zoom.set(xlim=(sigma_r * x[0], sigma_r * x[min(np.searchsorted(first, 0.30), len(x) - 1)]), ylim=(0.0, 30.0),
             title="zoom on the onset")
    zoom.title.set_fontsize(9)
    zoom.tick_params(labelsize=9)
    zoom.grid(True, color=GRID)

    ax[2].axhspan(0.0, 100 * EPS_R, color=GRID, alpha=0.8, lw=0, label=f"one cube is an RVE ({100 * EPS_R:g}%)")
    levels = [lv for lv in FGR_LEVELS if np.isfinite(fgr[lv]["eps"])]
    eps = np.array([100 * fgr[lv]["eps"] for lv in levels])
    ax[2].plot([100 * lv for lv in levels], eps, "-", color=SEQ_BLUE[3], lw=1.2, label=f"side {L:g} um ({n} cubes)")
    for i_lv, (lv, e) in enumerate(zip(levels, eps)):
        ax[2].plot(100 * lv, e, "o", ms=6, color=SEQ_BLUE[3], mfc=SEQ_BLUE[3] if lv in IN_VERDICT else "white")
        up = i_lv % 2 == 0  # close levels: labels above and below in turn, so that they do not overlap
        ax[2].annotate(f"{fgr[lv]['rve']:.0f} um", (100 * lv, e), xytext=(6 if up else -6, 7 if up else -7),
                       textcoords="offset points", ha="left" if up else "right", va="bottom" if up else "top",
                       fontsize=9, color=REFERENCE_LINE)
    ax[2].set(xlabel="level of FGR (%), log scale", ylabel=r"error of one cube on $x_c$, $2D$ / mean (%)", ylim=(0.0, None),
              xscale="log", title="Error of one cube; labels: side of the RVE")
    ax[2].set_xticks([1, 2, 5, 10, 20, 40, 80, 95], ["1", "2", "5", "10", "20", "40", "80 ", " 95"])
    ax[2].minorticks_off()
    ax[2].plot([], [], "o", color=REFERENCE_LINE, mfc="white",
               label=f"outside {100 * VERDICT_LEVELS[0]:g}-{100 * VERDICT_LEVELS[1]:g}%")
    ax[2].grid(True, color=GRID)
    ax[2].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "rve_fgr.png"), dpi=150)
    plt.close(fig)
    print("\n".join(lines))
    print(f"wrote rve_fgr.txt and rve_fgr.png in {root}")

    # %% 2. population and 4. size of the RVE: class means and verdict ----------------------------------------------------
    print("\n== 2. population and 4. size of the RVE ==")
    info = completeness(cubes)
    stats = class_stats(cubes, "s_hat", info)
    lines = [f"# Size of the RVE (python3 plot_bubbles.py {root}): {n} cubes of {L:g} um",
             f"# precision asked on a mean: {100 * EPS_R:g} %, quantity of the verdict: s_hat"]
    if info is not None:  # 2. classes expected and placed by Mérope
        lines += ["#", f"# 2. Does a cube of {L:g} um hold the whole population? (summary_*.txt of Mérope, {info['n']} cubes)",
                  f"#    porosity {info['porosity']:.4f} in the cubes against {info['porosity_asked']:.4f} asked: Mérope "
                  "places the integer part of the pores expected in every class",
                  f"#  {'R_um':>7s}{'vol_real_%':>11s}{'vol_cube_%':>11s}{'expected':>9s}{'placed':>8s}{'side_1_pore_um':>15s}"]
        for k, Rk in enumerate(info["R"]):
            lo, hi = info["placed_min"][k], info["placed_max"][k]
            placed = f"{lo:.0f}" if lo == hi else f"{lo:.0f}-{hi:.0f}"
            mark = "   <- not in the cubes" if k in info["missing"] else ""
            lines.append(f"   {Rk:7.4f}{100 * info['real'][k]:11.2f}{100 * info['cube'][k]:11.2f}{info['expected'][k]:9.1f}"
                         f"{placed:>8s}{side_for(info, k, 1.0):15.1f}{mark}")
        if info["missing"]:
            lines.append(f"#    {100 * sum(info['real'][k] for k in info['missing']):.1f} % of the pore volume of the "
                         f"distribution is in classes that no cube of {L:g} um holds: their FGR leaves it out, whatever "
                         "the number of seeds")
    lines += ["#", "# 4. Class means (the FGR is in rve_fgr.txt)",
              "# - Z = mean of the quantity over the pores of one class in one cube, n = cubes holding the class "
              "(* if some cubes miss it), pores = pores of the class per cube",
              "# - vol_cube / vol_real = share of the pore volume of the class in the cubes / in the measured distribution",
              "# - mean +/- 2 D / sqrt(n) = mean of Z over the cubes; D = standard deviation of Z between the cubes; "
              "S = standard deviation between single pores",
              "# - A_x_dens = D^2 x pores / S^2: 1 if the pores of a class are independent (then D^2 ~ 1/V, alpha = 1)",
              f"# - eps_cube = 2 D / mean: error of Z from one cube; cubes = (eps_cube / {100 * EPS_R:g} %)^2 cubes of this "
              "side; RVE_um = side x cubes^(1/3) (alpha = 1), with its 95 % interval"]
    for quantity, results in (("s_hat", stats), ("sigma", class_stats(cubes, "sigma", info))):
        lines += ["#", f"# {quantity}: {n} cubes of {L:g} um",
                  f"# {'class':>5s}{'R_um':>8s}{'n':>4s}{'pores':>7s}{'vol_cube':>9s}{'vol_real':>9s}{'mean':>8s}"
                  f"{'+/-':>8s}{'D':>9s}{'S':>8s}{'A_x_dens':>9s}{'eps_cube':>9s}{'cubes':>7s}{'RVE_um':>8s}{'[95 %]':>10s}"]
        for k, r in enumerate(results, 1):
            head = (f"{k:7d}{r['R']:8.4f}{str(r['n']) + ('*' if r['n'] < r['n_all'] else ''):>4s}{r['pores']:7.1f}"
                    f"{100 * r['vol_cube']:8.1f}%{100 * r['vol_real']:8.1f}%")
            if np.isnan(r["cubes"]):
                lines.append(f"{head}   the class is in fewer than 3 cubes")
                continue
            lines.append(f"{head}{r['mean']:8.4f}{r['half']:8.4f}{np.sqrt(r['D2']):9.2e}{r['S']:8.4f}"
                         f"{r['A_x_dens']:9.2f}{100 * r['eps']:8.1f}%{r['cubes']:7.1f}{r['rve']:8.1f}{r['interval']:>10s}")

    # verdict: side of the RVE for one cube (largest over the FGR levels and classes of the verdict), then the reserves
    lines += ["#", f"# 4. Verdict for one cube of {L:g} um, precision {100 * EPS_R:g} % on the mean, {n} cubes"]
    on_fgr = [(r["rve"], r, f"FGR {100 * lv:.0f} %") for lv, r in fgr.items() if lv in IN_VERDICT and np.isfinite(r["rve"])]
    on_classes = [(r["rve"], r, f"class R = {r['R']:.3g} um ({100 * r['share']:.0f} % of the pore volume)")
                  for r in stats if r["share"] >= MIN_VOLUME_SHARE and np.isfinite(r["rve"])]
    status = 0
    if not on_fgr and not on_classes:
        lines.append("# not determined: fewer than 3 cubes")
    else:
        rve, worst, why = max(on_fgr + on_classes, key=lambda t: t[0])
        status = int(rve > L)
        if not status:
            lines.append(f"# one cube of {L:g} um is an RVE: the largest side needed is {rve:.1f} um ({why})")
        else:
            lines += [f"# one cube of {L:g} um is NOT an RVE",
                      f"# side of the RVE: {rve:.0f} um (95 %: {worst['interval']} um), set by {why}; or the mean of "
                      f"{np.ceil(worst['cubes']):.0f} cubes of {L:g} um"]
        for what, cands in ((f"on the FGR (levels {100 * VERDICT_LEVELS[0]:g}-{100 * VERDICT_LEVELS[1]:g} %)", on_fgr),
                            ("on the classes of the verdict", on_classes)):
            if cands:
                s, r, w = max(cands, key=lambda t: t[0])
                lines.append(f"#   {what}: {s:.0f} um ({r['interval']}), {w}")
        small = [f"R = {r['R']:.3g} um ({100 * r['share']:.1f} %, {r['rve']:.0f} um)" for r in stats
                 if r["share"] < MIN_VOLUME_SHARE and np.isfinite(r["rve"])]
        if small:
            lines.append(f"#   classes below {100 * MIN_VOLUME_SHARE:g} % of the pore volume, not in the verdict: "
                         + ", ".join(small))
        lines += ["# Reserves", "# - one side only: alpha = 1 assumed (D^2 ~ 1/V)"]
        A = [r["A_x_dens"] for r in stats if np.isfinite(r["A_x_dens"])]
        if A:
            independent = 0.5 <= min(A) and max(A) <= 2.0
            lines.append(f"#   A x pore density of the classes is {min(A):.2f}-{max(A):.2f} (1 for independent pores): "
                         + ("the pores are close to independent, which supports alpha = 1" if independent else
                            "the pores are not independent, alpha may differ from 1"))
        if info is not None:
            needed = [r["cubes"] * r["pores"] for r in stats if np.isfinite(r["cubes"])]
            typical = np.median(needed) if needed else np.nan
            for k in [k for k in info["missing"] if info["real"][k] >= 0.001]:
                more = (f"; the {typical:.0f} pores that give the other classes {100 * EPS_R:g} % on their mean need "
                        f"{side_for(info, k, typical):.0f} um (more than the FGR needs, where the class weighs only its "
                        "share)" if typical >= 1.0 else "")
                lines.append(f"# - class R = {info['R'][k]:.2f} um ({percent(info['real'][k])} % of the pore volume) is in "
                             f"no cube: {info['expected'][k]:.1f} pores expected per cube; one pore needs a side of "
                             f"{side_for(info, k, 1.0):.1f} um{more}")
            negligible = [k for k in info["missing"] if info["real"][k] < 0.001]
            if negligible:
                lines.append("# - also in no cube, negligible: " + ", ".join(
                    f"R = {info['R'][k]:.2f} um ({100 * info['real'][k]:.2f} % of the pore volume)" for k in negligible))
            lines.append(f"# - porosity {info['porosity']:.4f} in the cubes against {info['porosity_asked']:.4f} asked")
        lines.append(mesh_summary)
    with open(os.path.join(root, "rve_size.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # figure: Z of Kanit per class, side of the RVE per class
    cubes = sorted(cubes, key=lambda c: c["seed"])
    radii, shares = [r["R"] for r in stats], [r["share"] for r in stats]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.4))
    spread = np.linspace(-0.32, 0.12, max(len(cubes), 2))
    for i, r in enumerate(stats):
        for j, c in enumerate(cubes):
            v = c["s_hat"][(c["R"] == r["R"]) & np.isfinite(c["s_hat"])]
            if len(v):
                ax[0].plot(i + spread[j], v.mean(), "o", ms=4.5, color=plt.get_cmap("tab10")(j % 10), alpha=0.85)
        if np.isfinite(r["D2"]):
            ax[0].errorbar(i + 0.28, r["mean"], yerr=2.0 * np.sqrt(r["D2"]), color=SEQ_BLUE[1], marker="s", ms=5, lw=0,
                           elinewidth=1.2, capsize=3, label=r"one cube, mean $\pm\,2D$" if i == 0 else None)
            ax[0].errorbar(i + 0.28, r["mean"], yerr=r["half"], color=SEQ_BLUE[4], marker="s", ms=5, lw=0,
                           elinewidth=2.6, capsize=0, label=rf"mean of the cubes $\pm\,2D/\sqrt{{n}}$" if i == 0 else None)
    ax[0].plot([], [], "o", color=REFERENCE_LINE, label="mean of the class in one cube (colour: seed)")
    ax[0].set_xticks(range(len(radii)), class_ticks(radii, shares))
    ax[0].set(xlabel="radius of the class (um) and its share of the pore volume", ylabel=f"class mean of {S_HAT}",
              title="Class mean of every cube")
    low, high = ax[0].get_ylim()
    ax[0].set_ylim(low, high + 0.28 * (high - low))  # room for the legend above the points

    for i, r in enumerate(stats):
        if not np.isfinite(r["rve"]):
            continue
        used = r["share"] >= MIN_VOLUME_SHARE
        color = SEQ_BLUE[3] if used else REFERENCE_LINE
        ax[1].errorbar(i, r["rve"], yerr=[[r["rve"] - r["rve_low"]], [r["rve_high"] - r["rve"]]], color=color, marker="o",
                       ms=7, mfc=color if used else "white", lw=0, elinewidth=1.4, capsize=4)
        ax[1].annotate(f"{r['rve']:.0f}", (i, r["rve"]), xytext=(8, 0), textcoords="offset points", va="center",
                       fontsize=9, color=color)
    ax[1].axhline(D52_SIDE, color=D52_COLOR, ls=":", lw=1.4, label=f"RVE of D5.2, {D52_SIDE:g} um")
    ax[1].plot([], [], "o", color=REFERENCE_LINE, mfc="white",
               label=f"below {100 * MIN_VOLUME_SHARE:g}% of the pore volume: not in the verdict")
    missing = [k for k in info["missing"] if info["real"][k] >= 0.001] if info is not None else []
    if missing:
        text = "not in the cubes: " + "; ".join(f"R = {info['R'][k]:.2f} um ({percent(info['real'][k])} % of the "
                                               f"pore volume)" for k in missing)
        ax[1].text(0.02, 0.02, text, transform=ax[1].transAxes, fontsize=9, color=D52_COLOR)
    highest = max([D52_SIDE] + [r["rve_high"] for r in stats if np.isfinite(r["rve_high"])])
    ax[1].set_xticks(range(len(radii)), class_ticks(radii, shares))
    ax[1].set(xlabel="radius of the class (um) and its share of the pore volume", ylim=(0.0, 1.35 * highest),
              xlim=(-0.5, len(radii) - 0.45), ylabel="side of one cube that is an RVE (um)",
              title=f"Side for {100 * EPS_R:g}% error on the class mean")
    for a in ax:
        a.grid(True, color=GRID)
        legend = a.legend(frameon=False, fontsize=9, loc="upper left")
        for handle in getattr(legend, "legend_handles", None) or legend.legendHandles:  # legendHandles: matplotlib < 3.7
            handle.set_alpha(1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "rve_size.png"), dpi=150)
    plt.close(fig)
    print("\n".join(lines))
    print(f"wrote rve_size.txt and rve_size.png in {root}")
    return max(mesh_status, status)


if __name__ == "__main__":
    sys.exit(main())
