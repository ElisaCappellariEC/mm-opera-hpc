"""
Analysis of the tests of the RVE written by test_rve.py, once MMM has solved the cubes: the size of the cube and the size of
the mesh, for the stress factor of every pore. Only numpy and matplotlib: no Mérope, it runs where the results are.

Folders of ROOT (test_rve.py writes them, MMM is run in each; its output may also be in a subfolder):
    side<L>_seed<NN>             size test, one cube per seed; several sides L are allowed (EXTRA_SIDES of test_rve.py)
    side<L>_seed01_<variant>     mesh test, the pores of side<L>_seed01 with another mesh (meshdouble, wall<h>_bulk<H>)
each with bubbles_<run>.txt and summary_<run>.txt of Mérope, bubbles_and_stresses_selected.txt and sig_hydro.txt of MMM.
Folders without the output of MMM are skipped. The mesh of a folder is read from its summary.

Stress factor of a pore, two quantities:
    sigma  sigma_I_max / p, the largest first principal stress near the pore (bubbles_and_stresses_selected.txt, read by
           the names of its columns), MMM: periodic cube at zero macroscopic strain
    s_hat  (sigma + h) / (1 - h), the same at zero macroscopic stress (rescaling of OperaHPC, exact for one pressure p in
           all the pores [?]), h p = xi p - (1 - xi) <sigma_h> the hydrostatic compression of the cube, <sigma_h> the
           mean over the matrix (sig_hydro.txt) and xi the porosity of the mesh. It takes out of sigma the constraint of
           the cube, which depends on its porosity and so differs between seeds. It is the quantity analysed when every
           cube has sig_hydro.txt; the tables give both.

Size of the cube, per pore class: Kanit et al., Int. J. Solids Struct. 40 (2003) 3647, Sections 5.2 and 6
    Z        mean of the quantity over the pores of the class in a cube; the n cubes where the class is: mean and
             variance D^2 of Z between them (* in the tables: the class is missing in some cubes)
    S^2      variance between the pores of the class (point variance); A = D^2 V / S^2 the integral range (Eq. 62),
             A x pore density of the class = 1 for independent pores
    eps      2 D / mean, the relative error of one cube (Eq. 68-69 with n = 1): one cube is an RVE if eps <= EPS_R
    cubes    (eps / EPS_R)^2 cubes of side L for the precision asked (Eq. 70); RVE = L cubes^(1/3), with its 95 % interval
             from the uncertainty of D^2 on n cubes (chi-square with n - 1 degrees of freedom: with 10 cubes D^2 is known
             within a factor of about 2)
    F90      fraction of the pores of a class in a cube above the 90 % quantile of the class (all the cubes together):
             what a rupture threshold sees, the pores that break; 2 D_F its absolute error in one cube
    with several sides: D^2 against V (Kanit Eq. 61, D^2 ~ V^-alpha, alpha = 1 in Eq. 62) and the mean against the side
    (a small cube can bias the mean, which more cubes do not correct)
Size of the mesh: Kanit Section 4.1.2, the class means and every pore of side<L>_seed01 with all its meshes, against the
finest; with three wall sizes or more at one bulk size, the observed order and the extrapolated value (Richardson), the
reference of every mesh then.

Usage
    python3 analyse_rve.py [ROOT]     default ROOT below, e.g. python3 analyse_rve.py test_rve_side10/run_NFIR
writes in ROOT rve_size.txt, rve_size.png, rve_cdf.png, rve_scaling.png (several sides), mesh_test.txt and mesh_test.png.
Exit 1 if one cube of the main side is not an RVE for a class, or if the mesh of the runs is more than EPS_MESH from the
reference of the mesh test.

@author E. Cappellari
@date   October 2026
"""

import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join("test_rve2", "run_NFIR")  # OUT/run_<RUN> of test_rve.py
EPS_R = 0.05      # relative precision asked on the mean of a class
EPS_MESH = 0.02   # tolerance on the class means between the mesh of the runs and the reference: well below EPS_R, so that
                  # the mesh does not use up the precision of the RVE
PREF = 1.0        # --pref of the MMM runs: the pressure of a pore is PREF x its pressure in the bubble file

STRESSES, HYDRO = "bubbles_and_stresses_selected.txt", "sig_hydro.txt"
CASE = re.compile(r"side(?P<side>\d+(?:\.\d+)?)_seed(?P<seed>\d+)(?:_(?P<variant>.+))?$")
LABEL = {"sigma": r"$\sigma_I / p$", "s_hat": r"$\hat s$"}
SEQ_BLUE, GRID = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"), "#e1e0d9"


# ---- Reading a solved cube --------------------------------------------------------------------------------------------

def read_csv(path):
    """Columns of a csv with a header, as float arrays ("nan" allowed); None if the file is missing or has no rows."""
    if path is None or not os.path.isfile(path) or os.path.getsize(path) == 0:
        return None
    with open(path) as f:
        rows = [line.strip().split(",") for line in f if line.strip()]
    if len(rows) < 2:
        return None
    data = np.array([[float(x) for x in row] for row in rows[1:]])
    return {name.strip(): data[:, k] for k, name in enumerate(rows[0])}


def output_of_mmm(folder):
    """Path of bubbles_and_stresses_selected.txt in folder or in its shallowest subfolder, None if MMM has not run."""
    hits = sorted(glob.glob(os.path.join(folder, "**", STRESSES), recursive=True), key=lambda p: (p.count(os.sep), p))
    if len(hits) > 1:
        print(f"{folder}: {len(hits)} {STRESSES}, reading {os.path.relpath(hits[0], folder)}")
    return hits[0] if hits else None


def mesh_of(folder):
    """(wall, bulk) mesh sizes (um) of a cube from its summary: the size field of test_rve.py, or the uniform size of
    Mérope."""
    summaries = glob.glob(os.path.join(folder, "summary_*.txt"))
    text = open(summaries[0]).read() if summaries else ""
    m = re.search(r"wall ([\d.eE+-]+) um, bulk ([\d.eE+-]+) um", text)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"mesh size ([\d.eE+-]+) um", text)
    return (float(m.group(1)),) * 2 if m else (np.nan, np.nan)


def nearest_gaps(centres, radii, side):
    """Distance from the wall of every pore to the wall of its nearest neighbour, in the periodic cube."""
    d = centres[:, None, :] - centres[None, :, :]
    d -= side * np.round(d / side)
    gap = np.linalg.norm(d, axis=2) - radii[:, None] - radii[None, :]
    np.fill_diagonal(gap, np.inf)
    return gap.min(axis=1)


def read_cube(folder):
    """Pores of a solved cube: dict with the folder, side, seed, variant, mesh, and per pore id, radius R, gap to the nearest
    pore, sigma and s_hat (None without sig_hydro.txt); None if MMM has not run in the folder."""
    case = CASE.match(os.path.basename(folder))
    side = float(case["side"])
    stresses = output_of_mmm(folder)
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
        raise SystemExit(f"{folder}: the pores of {STRESSES} ({len(ids)}) are not those of {os.path.basename(pores[0])} ({len(row)})")
    wall, bulk = mesh_of(folder)
    p = PREF * table["pressure"]
    sigma = table["sigma_I_max"] / p
    cube = {"folder": os.path.basename(folder), "side": side, "seed": int(case["seed"]), "variant": case["variant"],
            "wall": wall, "bulk": bulk, "id": ids, "R": np.round(table["radius"], 6), "sigma": sigma, "s_hat": None,
            "gap": nearest_gaps(bubbles[:, 1:4], bubbles[:, 4], side)[[row[i] for i in ids]]}
    if not np.isfinite(sigma).all():
        print(f"{folder}: {np.count_nonzero(~np.isfinite(sigma))} pores of {len(sigma)} without a stress (nan), left out")
    hydro = read_csv(os.path.join(os.path.dirname(stresses), HYDRO))
    if hydro is not None and {"Volume", "Average"} <= set(hydro):
        if np.ptp(p) > 0.0:
            print(f"{folder}: pores at different pressures, s_hat rescales with their mean pressure [?]")
        xi = 1.0 - hydro["Volume"][0] / side**3
        xi_pores = np.sum(4.0 / 3.0 * np.pi * bubbles[:, 4] ** 3) / side**3
        if abs(xi - xi_pores) > 0.01:
            print(f"{folder}: porosity {xi:.4f} from the volume of {HYDRO}, {xi_pores:.4f} from the pores: wrong side or "
                  "output of another run? s_hat of this cube is suspect")
        h = xi - (1.0 - xi) * hydro["Average"][0] / p.mean()
        cube.update(xi=xi, h=h, s_hat=(sigma + h) / (1.0 - h))
    return cube


# ---- Size of the cube ------------------------------------------------------------------------------------------------

def chi2_quantile(p, k):
    """Quantile p (0.025 or 0.975) of the chi-square law with k degrees of freedom (Wilson-Hilferty, within 1 % for k >= 3)."""
    z = {0.025: -1.959964, 0.975: 1.959964}[p]
    return k * (1.0 - 2.0 / (9.0 * k) + z * np.sqrt(2.0 / (9.0 * k))) ** 3


def class_values(cubes, q, R):
    """Values of the quantity q for the pores of class R, one array per cube where the class is."""
    groups = [c[q][(c["R"] == R) & np.isfinite(c[q])] for c in cubes]
    return [g for g in groups if len(g)]


def size_stats(cubes, q):
    """Statistics of Kanit per class from the cubes of one side, for the quantity q. A class is a radius."""
    side = cubes[0]["side"]
    results = []
    for R in np.unique(np.concatenate([c["R"] for c in cubes])):
        groups = class_values(cubes, q, R)
        pooled = np.concatenate(groups)
        r = {"R": R, "n": len(groups), "n_all": len(cubes), "pores": np.mean([len(g) for g in groups]),
             "S": pooled.std(ddof=1) if len(pooled) > 1 else np.nan, "cubes": np.nan}
        if len(groups) >= 3 and r["S"] > 0.0:
            Z = np.array([g.mean() for g in groups])
            n, mean, D2 = len(Z), Z.mean(), Z.var(ddof=1)
            eps = 2.0 * np.sqrt(D2) / mean                    # K03 Eq. 68-69 with n = 1
            cubes_needed = (eps / EPS_R) ** 2                  # K03 Eq. 70
            low, high = (n - 1) / chi2_quantile(0.975, n - 1), (n - 1) / chi2_quantile(0.025, n - 1)  # 95 % of D^2
            q90 = np.quantile(pooled, 0.9)
            F = np.array([np.mean(g > q90) for g in groups])
            r.update(mean=mean, D2=D2, A=D2 * side**3 / r["S"] ** 2, independent=D2 * r["pores"] / r["S"] ** 2, eps=eps,
                     cubes=cubes_needed, rve=side * np.cbrt(cubes_needed), rve_low=side * np.cbrt(cubes_needed * low),
                     rve_high=side * np.cbrt(cubes_needed * high), F90=F.mean(), dF90=2.0 * F.std(ddof=1))
        results.append(r)
    return results


def size_lines(side, cubes, q, results):
    """Table of the statistics of one side and one quantity."""
    lines = [f"# {q}: {len(cubes)} cubes of {side:g} um",
             f"# {'class':>5s}{'R_um':>8s}{'n':>4s}{'pores':>7s}{'mean':>8s}{'+/-':>8s}{'D':>9s}{'S':>8s}{'A_x_dens':>9s}"
             f"{'eps_cube':>9s}{'cubes':>7s}{'RVE_um':>8s}{'[95 %]':>14s}{'F90':>7s}{'+/-':>7s}"]
    for k, r in enumerate(results, 1):
        n = f"{r['n']}{'*' if r['n'] < r['n_all'] else ''}"
        if np.isnan(r["cubes"]):
            lines.append(f"{k:7d}{r['R']:8.4f}{n:>4s}{r['pores']:7.1f}   too few cubes with this class")
            continue
        interval = f"{r['rve_low']:.0f}-{r['rve_high']:.0f}"
        lines.append(f"{k:7d}{r['R']:8.4f}{n:>4s}{r['pores']:7.1f}{r['mean']:8.4f}{2.0 * np.sqrt(r['D2'] / r['n']):8.4f}"
                     f"{np.sqrt(r['D2']):9.2e}{r['S']:8.4f}{r['independent']:9.2f}{r['eps']:9.4f}{r['cubes']:7.1f}"
                     f"{r['rve']:8.1f}{interval:>14s}{r['F90']:7.3f}{r['dF90']:7.3f}")
    return lines


def scaling(by_side, q):
    """Per class tested at every side: sides, D^2, means and their half-widths 2 D / sqrt(n), and alpha of D^2 ~ V^-alpha."""
    sides = sorted(by_side)
    stats = {L: {r["R"]: r for r in size_stats(by_side[L], q) if not np.isnan(r["cubes"])} for L in sides}
    out = []
    for R in sorted(set.intersection(*(set(s) for s in stats.values()))):
        rows = [stats[L][R] for L in sides]
        D2 = np.array([r["D2"] for r in rows])
        alpha = -np.polyfit(np.log(np.array(sides) ** 3), np.log(D2), 1)[0]
        out.append({"R": R, "sides": np.array(sides), "D2": D2, "alpha": alpha, "mean": np.array([r["mean"] for r in rows]),
                    "half": np.array([2.0 * np.sqrt(r["D2"] / r["n"]) for r in rows])})
    return out


def scaling_lines(rows, q):
    """Table of the scaling with the side."""
    if not rows:
        return []
    lines = ["#", f"# {q}: scaling with the side (Kanit Eq. 61: D^2 ~ V^-alpha, alpha = 1 if the integral range holds; the mean",
             "# must not drift with the side beyond the half-widths 2 D / sqrt(n), otherwise the small cubes are biased)",
             f"# {'R_um':>6s}{'alpha':>7s}  mean +/- 2 D / sqrt(n) per side"]
    for r in rows:
        means = "  ".join(f"{L:g}: {m:.4f} +/- {h:.4f}" for L, m, h in zip(r["sides"], r["mean"], r["half"]))
        lines.append(f"{r['R']:8.4f}{r['alpha']:7.2f}  {means}")
    return lines


def size_figure(cubes, q, results, name):
    """Values per class and seed, the mean with the intervals of one cube and of the n cubes, the relative error of one
    cube against the precision asked, and the quantity against the gap to the nearest pore."""
    side = cubes[0]["side"]
    done = [r for r in results if not np.isnan(r["cubes"])]
    fig, ax = plt.subplots(2, 2, figsize=(14.0, 9.5))
    radii = [r["R"] for r in results]
    seeds = sorted(c["seed"] for c in cubes)
    width = 0.8 / max(len(seeds), 1)
    cmap = plt.get_cmap("tab10")
    for i, R in enumerate(radii):
        for j, c in enumerate(sorted(cubes, key=lambda c: c["seed"])):
            v = c[q][(c["R"] == R) & np.isfinite(c[q])]
            if len(v):
                bp = ax[0, 0].boxplot([v], positions=[i - 0.4 + width * (j + 0.5)], widths=0.85 * width, patch_artist=True,
                                      showfliers=False)
                bp["boxes"][0].set(facecolor=cmap(j % 10), alpha=0.7)
    ax[0, 0].set_xticks(range(len(radii)), [f"{R:.3g}" for R in radii])
    ax[0, 0].legend(handles=[plt.Line2D([0], [0], color=cmap(j % 10), lw=6, label=f"seed {s:02d}") for j, s in enumerate(seeds)],
                    fontsize=8, ncol=2, frameon=False)
    ax[0, 0].set(xlabel="radius of the class (um)", ylabel=LABEL[q], title=f"pores of every class and cube ({side:g} um)")
    R = np.array([r["R"] for r in done])
    mean, D = np.array([r["mean"] for r in done]), np.sqrt([r["D2"] for r in done])
    ax[0, 1].errorbar(R, mean, yerr=2.0 * D, color=SEQ_BLUE[1], marker="o", ms=5, lw=0, elinewidth=1.2, capsize=3,
                      label=r"one cube, $\pm\,2D$")
    ax[0, 1].errorbar(R, mean, yerr=2.0 * D / np.sqrt([r["n"] for r in done]), color=SEQ_BLUE[4], marker="o", ms=5, lw=0,
                      elinewidth=2.0, capsize=0, label=r"mean of the cubes, $\pm\,2D/\sqrt{n}$")
    ax[0, 1].set(xlabel="radius of the class (um)", ylabel=f"mean of {LABEL[q]} of the class", title="mean of a class")
    ax[1, 0].axhline(100 * EPS_R, color=GRID, lw=6, label=f"precision asked, {100 * EPS_R:g} %")
    ax[1, 0].plot(R, 100 * np.array([r["eps"] for r in done]), "o-", color=SEQ_BLUE[3], ms=5, lw=1.2)
    ax[1, 0].set(xlabel="radius of the class (um)", ylabel="relative error of one cube, 2 D / mean (%)", ylim=(0.0, None),
                 title="one cube is an RVE below the line")
    for k, Rk in enumerate(radii):
        gap = np.concatenate([c["gap"][c["R"] == Rk] for c in cubes])
        v = np.concatenate([c[q][c["R"] == Rk] for c in cubes])
        ax[1, 1].scatter(gap, v, s=10, color=plt.get_cmap("viridis")(k / max(len(radii) - 1, 1)), label=f"R = {Rk:.3g}")
    ax[1, 1].set(xscale="log", xlabel="gap to the nearest pore (um)", ylabel=LABEL[q],
                 title="the stress factor follows the nearest ligament")
    for a in ax.flat:
        a.grid(True, color=GRID)
    for a in (ax[0, 1], ax[1, 0], ax[1, 1]):
        a.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(name, dpi=150)
    plt.close(fig)


def cdf_figure(cubes, q, name):
    """Distribution of the quantity in every class: one line per cube, and all the cubes together."""
    radii = np.unique(np.concatenate([c["R"] for c in cubes]))
    cols = min(len(radii), 4)
    rows = int(np.ceil(len(radii) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.4 * rows), squeeze=False)
    for a, R in zip(ax.flat, radii):
        groups = class_values(cubes, q, R)
        for g in groups:
            a.step(np.sort(g), np.arange(1, len(g) + 1) / len(g), where="post", color=SEQ_BLUE[0], lw=0.8)
        pooled = np.sort(np.concatenate(groups))
        a.step(pooled, np.arange(1, len(pooled) + 1) / len(pooled), where="post", color=SEQ_BLUE[4], lw=2.0)
        a.set(title=f"R = {R:.3g} um, {len(pooled)} pores in {len(groups)} cubes", xlabel=LABEL[q], ylabel="fraction below")
        a.grid(True, color=GRID)
    for a in ax.flat[len(radii):]:
        a.set_visible(False)
    fig.tight_layout()
    fig.savefig(name, dpi=150)
    plt.close(fig)


def scaling_figure(rows, q, name):
    """D^2 against the volume per class, with the slope -1 of the integral range."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for k, r in enumerate(rows):
        V = r["sides"] ** 3
        color = plt.get_cmap("viridis")(k / max(len(rows) - 1, 1))
        ax.loglog(V, r["D2"], "o-", color=color, label=f"R = {r['R']:.3g}, alpha = {r['alpha']:.2f}")
        ax.loglog(V, r["D2"][0] * V[0] / V, ":", color=color, lw=0.8)
    ax.set(xlabel="volume of the cube (um$^3$)", ylabel=f"$D^2$ of the class mean of {LABEL[q]}",
           title="dotted: $D^2 \\sim 1/V$ (Kanit Eq. 62)")
    ax.grid(True, color=GRID, which="both")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(name, dpi=150)
    plt.close(fig)


def analyse_size(root, cubes, q):
    """Size test on the solved cubes without a mesh variant. Returns 1 if one cube of the main side (the side with the most
    cubes) is not an RVE for a class, 0 otherwise."""
    by_side = {}
    for c in cubes:
        by_side.setdefault(c["side"], []).append(c)
    if not by_side:
        print(f"{root}: no cube of the size test solved")
        return 0
    main = max(by_side, key=lambda L: (len(by_side[L]), -L))
    quantities = ["s_hat", "sigma"] if q == "s_hat" else ["sigma"]
    lines = [f"# size of the RVE (python3 analyse_rve.py {root}): cubes solved per side "
             + ", ".join(f"{L:g} um: {len(by_side[L])}" for L in sorted(by_side)),
             f"# precision asked on the mean of a class: {100 * EPS_R:g} %; quantity of the verdict and of the figures: {q}",
             "# Kanit 2003: D^2(V) = S^2 A / V (Eq. 62), eps = 2 D / sqrt(n) (Eq. 68), relative to the mean (Eq. 69), "
             "n = 4 D^2 / eps^2 (Eq. 70); RVE = the volume with n = 1 (Section 6)",
             "# per class: cubes n (* the class is missing in some), pores of the class per cube, mean with the half-width "
             "2 D / sqrt(n), D between the cubes, S between the pores,",
             "# A x pore density (1: independent pores), 2 D / mean of one cube, cubes of that side for the precision asked, "
             "side of the RVE with its 95 % interval, F90 with 2 D_F"]
    results = None
    for quantity in quantities:
        for L in sorted(by_side):
            stats = size_stats(by_side[L], quantity)
            lines += ["#"] + size_lines(L, by_side[L], quantity, stats)
            if quantity == q and L == main:
                results = stats
        if len(by_side) > 1:
            rows = scaling(by_side, quantity)
            lines += scaling_lines(rows, quantity)
            if quantity == q and rows:
                scaling_figure(rows, q, os.path.join(root, "rve_scaling.png"))
    more = [f"{k} ({np.ceil(r['cubes']):.0f} cubes)" for k, r in enumerate(results, 1) if r["cubes"] > 1.0]
    unknown = [str(k) for k, r in enumerate(results, 1) if np.isnan(r["cubes"])]
    lines.append("#")
    if more:
        lines.append(f"# {q}: one cube of {main:g} um is not an RVE at {100 * EPS_R:g} % for the classes {', '.join(more)}")
    if unknown:
        lines.append(f"# {q}: not determined, the class is in fewer than 3 cubes: classes {', '.join(unknown)}")
    if not more and not unknown:
        lines.append(f"# {q}: one cube of {main:g} um is an RVE at {100 * EPS_R:g} % for every class")
    with open(os.path.join(root, "rve_size.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    size_figure(by_side[main], q, results, os.path.join(root, "rve_size.png"))
    cdf_figure(by_side[main], q, os.path.join(root, "rve_cdf.png"))
    print("\n".join(lines))
    print(f"wrote rve_size.txt, rve_size.png, rve_cdf.png{', rve_scaling.png' if len(by_side) > 1 else ''} in {root}")
    return 1 if more else 0


# ---- Size of the mesh ------------------------------------------------------------------------------------------------

def richardson(h, s):
    """Observed order p and extrapolated value s0 of s(h) = s0 + C h^p from the three finest sizes (h ascending); None if s
    is not monotone in h or no order between 0.05 and 8 fits."""
    (h1, h2, h3), (s1, s2, s3) = h[:3], s[:3]
    if (s2 - s1) * (s3 - s2) <= 0.0:
        return None
    target = (s3 - s2) / (s2 - s1)
    f = lambda p: (h3**p - h2**p) / (h2**p - h1**p) - target
    low, high = 0.05, 8.0
    if f(low) * f(high) > 0.0:
        return None
    for _ in range(80):
        mid = 0.5 * (low + high)
        low, high = (low, mid) if f(low) * f(mid) <= 0.0 else (mid, high)
    p = 0.5 * (low + high)
    return p, s1 - (s2 - s1) * h1**p / (h2**p - h1**p)


def analyse_mesh(root, cubes, q):
    """Mesh test on side<L>_seed01 and its variants: class means and pores against the reference (the Richardson value
    when there are three wall sizes at one bulk size, otherwise the finest mesh). mesh_test.txt and mesh_test.png. Returns
    1 if the mesh of the runs (the folder without a variant) is more than EPS_MESH from the reference, 0 otherwise."""
    families = {}
    for c in cubes:
        if c["seed"] == 1:
            families.setdefault(c["side"], []).append(c)
    families = {L: f for L, f in families.items() if len(f) > 1 and any(c["variant"] for c in f)}
    if not families:
        print(f"{root}: no mesh variant of a seed01 solved, no mesh test")
        return 0
    side = max(families, key=lambda L: len(families[L]))
    family = sorted(families[side], key=lambda c: (c["wall"], c["bulk"]))
    finest, runs = family[0], next((c for c in family if c["variant"] is None), None)
    ids = sorted(set.intersection(*(set(c["id"][np.isfinite(c[q])]) for c in family)))
    value = {c["folder"]: dict(zip(c["id"], c[q])) for c in family}
    R = dict(zip(finest["id"], finest["R"]))
    gap = dict(zip(finest["id"], finest["gap"]))
    radii = sorted({R[i] for i in ids})
    means = {c["folder"]: np.array([np.mean([value[c["folder"]][i] for i in ids if R[i] == Rk]) for Rk in radii] +
                                   [np.mean([value[c["folder"]][i] for i in ids])]) for c in family}

    # reference: Richardson on the bulk size with three wall sizes or more, per class and for all the pores
    reference, how = means[finest["folder"]], f"the finest mesh, {finest['folder']}"
    for bulk in sorted({c["bulk"] for c in family}):
        group = [c for c in family if c["bulk"] == bulk]
        if len({c["wall"] for c in group}) >= 3:
            h = np.array([c["wall"] for c in group])
            fits = [richardson(h, np.array([means[c["folder"]][k] for c in group])) for k in range(len(radii) + 1)]
            if all(fit is not None for fit in fits):
                reference = np.array([fit[1] for fit in fits])
                how = (f"Richardson on the wall sizes {', '.join(f'{x:g}' for x in h[:3])} um at bulk {bulk:g} um, observed "
                       f"order per class {', '.join(f'{fit[0]:.2f}' for fit in fits[:-1])}, all pores {fits[-1][0]:.2f}")
                break
            print(f"{root}: Richardson at bulk {bulk:g} um not possible (class means not monotone in the wall size)")

    names = [f"{Rk:.3f}" for Rk in radii] + ["all"]
    lines = [f"# size of the mesh for {q} (python3 analyse_rve.py {root}): the {len(ids)} pores of side{side:g}_seed01 with "
             f"{len(family)} meshes",
             f"# reference: {how}",
             f"# tolerance on the class means of the mesh of the runs: {100 * EPS_MESH:g} %. Kanit 2003, Section 4.1.2: one "
             "geometry at several mesh densities",
             f"# smallest radius {min(radii):.3f} um, smallest gap between two pores {min(gap[i] for i in ids):.3f} um",
             "#",
             f"# deviation of the class means from the reference (%), then per pore from the finest mesh: median and largest |d| (%)",
             f"# {'folder':<34s}{'wall':>7s}{'bulk':>7s}{'w/Rmin':>7s}{'w/gap':>7s}" + "".join(f"{n:>8s}" for n in names)
             + f"{'med|d|':>8s}{'max|d|':>8s}"]
    for c in family:
        d = 100.0 * (means[c["folder"]] / reference - 1.0)
        pore = np.array([100.0 * (value[c["folder"]][i] / value[finest["folder"]][i] - 1.0) for i in ids])
        lines.append(f"  {c['folder']:<34s}{c['wall']:7g}{c['bulk']:7g}{c['wall'] / min(radii):7.2f}"
                     f"{c['wall'] / min(gap[i] for i in ids):7.2f}" + "".join(f"{x:8.2f}" for x in d)
                     + f"{np.median(np.abs(pore)):8.2f}{np.abs(pore).max():8.2f}")
    status = 0
    if runs is not None:
        worst = np.abs(means[runs["folder"]][:-1] / reference[:-1] - 1.0).max()
        status = int(worst > EPS_MESH)
        lines += ["#", f"# the mesh of the runs ({runs['folder']}, {runs['wall']:g} um) is within {100 * worst:.2f} % of the "
                       f"reference on every class: " + ("converged" if not status else "not converged") +
                       f" at {100 * EPS_MESH:g} %" + ("" if "Richardson" in how else
                       "; against the finest mesh only, which is not exact: add a third wall size (WALL_SIZES of test_rve.py)")]
    with open(os.path.join(root, "mesh_test.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    fig, ax = plt.subplots(1, 2, figsize=(13.0, 4.8))
    ax[0].axhspan(-100 * EPS_MESH, 100 * EPS_MESH, color=GRID, alpha=0.6, lw=0, label=rf"$\pm${100 * EPS_MESH:g} %")
    for k, n in enumerate(names):
        for bulk, marker in zip(sorted({c["bulk"] for c in family}), "osd^v"):
            group = [c for c in family if c["bulk"] == bulk]
            ax[0].plot([c["wall"] for c in group], [100.0 * (means[c["folder"]][k] / reference[k] - 1.0) for c in group],
                       marker=marker, ls="-", color="black" if n == "all" else plt.get_cmap("viridis")(k / max(len(radii) - 1, 1)),
                       label=(f"R = {n}" if n != "all" else "all pores") if marker == "o" else None)
    ax[0].set(xlabel="mesh size on the pore walls (um)", ylabel="class mean against the reference (%)",
              title="markers: bulk size (o the smallest)")
    for c in family[1:]:
        pore = [100.0 * (value[c["folder"]][i] / value[finest["folder"]][i] - 1.0) for i in ids]
        ax[1].scatter([gap[i] for i in ids], pore, s=12, label=f"{c['folder']} (wall {c['wall']:g}, bulk {c['bulk']:g})")
    ax[1].set(xscale="log", xlabel="gap to the nearest pore (um)", ylabel=f"{LABEL[q]} against the finest mesh (%)",
              title=f"every pore, against {finest['folder']}")
    for a in ax:
        a.grid(True, color=GRID)
        a.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "mesh_test.png"), dpi=150)
    plt.close(fig)
    print("\n".join(lines))
    print(f"wrote mesh_test.txt and mesh_test.png in {root}")
    return status


# ---- Main ---------------------------------------------------------------------------------------------------------

def main():
    """Size and mesh tests on the folders of ROOT (argument or default). Returns the worst status."""
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
    q = "s_hat" if all(c["s_hat"] is not None for c in cubes) else "sigma"
    if q == "sigma":
        print(f"{HYDRO} missing in some folders: the verdict is on sigma = sigma_I / p, at zero macroscopic strain")
    print(f"\n== size of the cube ==")
    status = analyse_size(root, [c for c in cubes if c["variant"] is None], q)
    print(f"\n== size of the mesh ==")
    return max(status, analyse_mesh(root, cubes, q))


if __name__ == "__main__":
    sys.exit(main())
