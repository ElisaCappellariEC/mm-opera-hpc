"""
The same cubes solved on two meshes of gmsh (e.g. a uniform mesh, and a mesh graded next to the pores by refine() of
merope_pore_pressure.py): do they give the same stress factors, and what does each one cost on the cluster?

REFERENCE and OTHER are two ROOT folders of plot_bubbles.py with the same cubes side<L>_seed<NN> (the same pores: their
bubbles_<run>.txt must be identical). Compared on s_hat (sigma = sigma_I / p if sig_hydro.txt is missing):
    per pore     d = (q_other - q_ref) / q_ref, pores matched by id
    per class    mean over the cubes of the class mean (Z of Kanit), against 2 D / sqrt(n) of the reference: the
                 uncertainty of that mean from the number of cubes
    FGR          x_c = p / sigma_R at the levels of FGR_LEVELS, mean over the cubes, and the FGR curves
    RVE          side of one cube that is an RVE, per class and on the FGR (Kanit, alpha = 1, as plot_bubbles.py)
    mesh size    the variants side<L>_seed01_<variant> of each ROOT (e.g. meshdouble) against its side<L>_seed01
Cost, from the logs of the cluster jobs in every cube (SLURM summary at the end of each .err): mesh of partition-mesh,
unknowns of MMM, core-hours (cores x elapsed) of partition-mesh, of the runs of MMM that solved and of those that
failed, cores and memory per process (MaxRSS) of the run that solved.

Writes in OTHER: mesh_compare.txt, mesh_compare.png
Usage: python3 mesh_compare.py REFERENCE OTHER
Exit 1 if OTHER moves a class of the verdict (>= 5 % of the pore volume) or a level of FGR of the verdict by more than
EPS_MESH from REFERENCE.

@author E. Cappellari
@date   November 2026
"""

import glob
import textwrap
import os
import re
import sys

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import plot_bubbles as pb

plt = pb.plt
COLORS = ("#2a78d6", "#eb6834")  # REFERENCE, OTHER (categorical slots 1-2, CVD-checked as a pair)
TINTS = ("#a9c8ef", "#f6b79c")   # partition-mesh in the cost bars
MUTED = "#52514e"
PROGRAMS = ("partition-mesh", "test-polybubble")
TIME = re.compile(r"^(?:(\d+)-)?(\d+):(\d\d):(\d\d)$")
RSS_GB = {"": 1024.0**-3, "K": 1024.0**-2, "M": 1024.0**-1, "G": 1.0, "T": 1024.0}


# ---- Reading -----------------------------------------------------------------------------------------------------------

def mesh_of(folder):
    """Mesh of gmsh in the first line of the summary of Mérope: (label, size next to the pores, size far from them)."""
    summaries = glob.glob(os.path.join(folder, "summary_*.txt"))
    head = open(summaries[0]).readline() if summaries else ""
    m = (re.search(r"mesh size ([\d.eE+-]+) um next to the pores, ([\d.eE+-]+) um after", head)
         or re.search(r"wall ([\d.eE+-]+) um, bulk ([\d.eE+-]+) um", head))
    if m:
        scaled = re.search(r"x R / ([\d.eE+-]+) um", head)  # size scaled with the radius below R_FINE (refine())
        return (f"graded {float(m[1]):g}-{float(m[2]):g} um" + (f", x R/{float(scaled[1]):g} below" if scaled else ""),
                float(m[1]), float(m[2]))
    m = re.search(r"mesh size ([\d.eE+-]+) um", head)
    if m:
        return f"uniform {float(m[1]):g} um", float(m[1]), float(m[1])
    return os.path.basename(folder), np.nan, np.nan


def pores_file(folder):
    """bubbles_<run>.txt of Mérope in a cube folder (not the output of MMM)."""
    files = [p for p in glob.glob(os.path.join(folder, "bubbles_*.txt")) if os.path.basename(p) != pb.STRESSES]
    return files[0] if len(files) == 1 else None


def load(root):
    """Solved cubes of a ROOT by folder name, and the folders not solved yet."""
    solved, unsolved = {}, []
    for folder in sorted(glob.glob(os.path.join(root, "side*_seed*"))):
        name = os.path.basename(folder)
        if not (os.path.isdir(folder) and pb.CASE.match(name)):
            continue
        cube = pb.read_cube(folder)
        if cube is None:
            unsolved.append(name)
            continue
        cube["path"] = folder
        cube["mesh_label"] = mesh_of(folder)[0]
        solved[name] = cube
    return solved, unsolved


def seconds(text):
    m = TIME.match(text)
    return int(m[1] or 0) * 86400 + int(m[2]) * 3600 + int(m[3]) * 60 + int(m[4])


def slurm(err):
    """Job of the cluster from the SLURM summary at the end of its .err: cores, nodes, elapsed (s), state and tasks of its
    program (partition-mesh or test-polybubble), MaxRSS of one process (GB). None without the summary."""
    text = open(err, errors="ignore").read()
    if "SLURM Accounting Summary" not in text:
        return None
    lines = text.split("SLURM Accounting Summary", 1)[1].splitlines()
    header = next((k for k, line in enumerate(lines) if "JobName" in line and "Elapsed" in line), None)
    if header is None:
        return None
    rows = []
    for line in lines[header + 2:]:
        if not line.strip():
            break
        rows.append(line.split())
    job = rows[0]
    step = next((r for r in rows[1:] if r[0] in PROGRAMS), None)
    elapsed = next(t for t in job if TIME.match(t))
    m = re.search(r"^\S+\.\d+\s+(?:%s)\s+(\d+(?:\.\d+)?)([KMGT]?)\s" % "|".join(map(re.escape, PROGRAMS)), text, re.M)
    state = step[-1] if step else job[-1]
    return {"cores": int(job[2]), "nodes": int(job[3]), "seconds": seconds(elapsed), "state": state,
            "ok": state.startswith("COMPLETED"), "tasks": int(step[4]) if step and len(step) >= 8 else 0,
            "rss": float(m[1]) * RSS_GB[m[2]] if m else np.nan, "out": err[:-4] + ".out"}


def cost(folder):
    """Jobs of the cluster in one cube (None without logs): core-hours of partition-mesh, of MMM solved and of MMM
    failed, mesh and unknowns, the run of MMM that solved and the states of those that failed."""
    jobs = {kind: [s for s in map(slurm, sorted(glob.glob(os.path.join(folder, f"{kind}_*_*.err")))) if s]
            for kind in ("split", "mmm")}
    if not jobs["mmm"]:
        return None
    read = lambda path: open(path, errors="ignore").read() if os.path.isfile(path) else ""
    core_h = lambda js: sum(j["cores"] * j["seconds"] for j in js) / 3600.0
    solved = [j for j in jobs["mmm"] if j["ok"]]
    failed = [j for j in jobs["mmm"] if not j["ok"]]
    c = {"split_ch": core_h(jobs["split"]), "solved_ch": core_h(solved), "failed_ch": core_h(failed),
         "failed": [j["state"] for j in failed], "run": solved[-1] if solved else None,
         "split": jobs["split"][-1] if jobs["split"] else None, "vertices": np.nan, "elements": np.nan,
         "unknowns": np.nan}
    for j in reversed(jobs["split"]):
        m = re.search(r"mesh: (\d+) vertices, (\d+) elements", read(j["out"]))
        if m:
            c["vertices"], c["elements"] = int(m[1]), int(m[2])
            break
    for j in reversed(jobs["mmm"]):
        m = re.search(r"finite element unknowns:\s*(\d+)", read(j["out"]))
        if m:
            c["unknowns"] = int(m[1])
            break
    return c


def total_cost(cubes):
    """Cost of a set of cubes: sums of core-hours, means of the mesh and of the run that solved; None without logs."""
    costs = [cost(c["path"]) for c in cubes]
    if any(c is None for c in costs):
        return None
    runs = [c["run"] for c in costs if c["run"]]
    splits = [c["split"] for c in costs if c["split"]]
    mean = lambda key: float(np.nanmean([c[key] for c in costs]))
    return {"n": len(costs), "split_ch": sum(c["split_ch"] for c in costs),
            "solved_ch": sum(c["solved_ch"] for c in costs), "failed_ch": sum(c["failed_ch"] for c in costs),
            "failed": [s for c in costs for s in c["failed"]], "solved": len(runs),
            "vertices": mean("vertices"), "elements": mean("elements"), "unknowns": mean("unknowns"),
            "cores": max((r["cores"] for r in runs), default=0), "nodes": max((r["nodes"] for r in runs), default=0),
            "tasks": max((r["tasks"] for r in runs), default=0),
            "rss": max((r["rss"] for r in runs), default=np.nan),
            "mmm_min": np.mean([r["seconds"] for r in runs]) / 60.0 if runs else np.nan,
            "split_min": np.mean([s["seconds"] for s in splits]) / 60.0 if splits else np.nan,
            "split_rss": max((s["rss"] for s in splits), default=np.nan)}


# ---- Comparison --------------------------------------------------------------------------------------------------------

def pair(ref, other, q):
    """Radius, q of REFERENCE and q of OTHER of the pores of one cube, matched by id (finite in both)."""
    where = {i: k for k, i in enumerate(other["id"])}
    k = np.array([where[i] for i in ref["id"]])
    a, b = ref[q], other[q][k]
    ok = np.isfinite(a) & np.isfinite(b)
    return ref["R"][ok], a[ok], b[ok]


def class_mean(cube, q, R):
    """Mean of q over the pores of class R of one cube (nan if the cube has none)."""
    v = cube[q][(cube["R"] == R) & np.isfinite(cube[q])]
    return v.mean() if len(v) else np.nan


def x_levels(cube, q):
    """x_c = p / sigma_R at which the FGR of a cube reaches every level of FGR_LEVELS (as plot_bubbles.analyse_fgr)."""
    xc, V = pb.critical_pressures(cube, q)
    order = np.argsort(xc)
    released = np.cumsum(V[order]) / V.sum()
    return np.array([xc[order][min(np.searchsorted(released, lv - 1e-12), len(xc) - 1)] for lv in pb.FGR_LEVELS])


def variants(solved, unsolved, q, classes):
    """Variants side<L>_seed01_<variant> of one ROOT against its side<L>_seed01: per class change of the mean (%)."""
    rows = []
    for name, cube in solved.items():
        if cube["variant"] is None or pb.REFINED.search(cube["variant"]):
            continue
        base = solved.get(f"side{cube['side']:g}_seed{cube['seed']:02d}")
        if base is None:
            continue
        d = [100.0 * (class_mean(cube, q, R) / class_mean(base, q, R) - 1.0) for R in classes]
        rows.append({"name": name, "label": cube["mesh_label"], "base": base["mesh_label"], "d": np.array(d)})
    missing = [name for name in unsolved if pb.CASE.match(name)["variant"]]
    return rows, missing


def main():
    args, opts = [], {"labels": None, "name": "mesh_compare", "cost": True}
    argv = sys.argv[1:]
    while argv:
        a = argv.pop(0)
        if a == "--labels" and len(argv) >= 2:
            opts["labels"] = (argv.pop(0), argv.pop(0))
        elif a == "--name" and argv:
            opts["name"] = argv.pop(0)
        elif a == "--no-cost":
            opts["cost"] = False
        else:
            args.append(a)
    if len(args) != 2:
        raise SystemExit("usage: python3 mesh_compare.py REFERENCE OTHER [--labels REF_LABEL OTHER_LABEL] [--name NAME] "
                         "[--no-cost]")
    roots = args
    loaded = [load(root) for root in roots]
    (ref_all, ref_unsolved), (oth_all, oth_unsolved) = loaded
    common = sorted(n for n in set(ref_all) & set(oth_all) if ref_all[n]["variant"] is None)
    if not common:
        raise SystemExit(f"no cube side<L>_seed<NN> solved in both {roots[0]} and {roots[1]}")
    sides = [ref_all[n]["side"] for n in common]
    side = max(set(sides), key=sides.count)
    common = [n for n in common if ref_all[n]["side"] == side]
    for n in common:
        a, b = pores_file(ref_all[n]["path"]), pores_file(oth_all[n]["path"])
        if a is None or b is None or not np.array_equal(np.loadtxt(a), np.loadtxt(b)):
            raise SystemExit(f"{n}: the pores differ between the two ROOTs, the meshes cannot be compared")
    refs, oths = [ref_all[n] for n in common], [oth_all[n] for n in common]
    q = "s_hat" if all(c["s_hat"] is not None for c in refs + oths) else "sigma"
    labels = refs[0]["mesh_label"], oths[0]["mesh_label"]
    if labels[0] == labels[1]:
        labels = tuple(f"{lab} ({os.path.basename(os.path.dirname(os.path.abspath(r)))})"
                       for lab, r in zip(labels, roots))
    if opts["labels"]:
        labels = opts["labels"]
    n = len(common)

    # per pore
    pairs = [pair(r, o, q) for r, o in zip(refs, oths)]
    R_all = np.concatenate([p[0] for p in pairs])
    a_all, b_all = np.concatenate([p[1] for p in pairs]), np.concatenate([p[2] for p in pairs])
    d_all = 100.0 * (b_all / a_all - 1.0)

    # per class: Kanit on both meshes, change of the mean and its range over the cubes
    info = pb.completeness(refs)
    stats = [pb.class_stats(cubes, q, pb.completeness(cubes)) for cubes in (refs, oths)]
    classes = np.array([s["R"] for s in stats[0]])
    share = np.array([s["share"] for s in stats[0]])
    verdict_class = share >= pb.MIN_VOLUME_SHARE
    Z = [np.array([[class_mean(c, q, R) for R in classes] for c in cubes]) for cubes in (refs, oths)]
    d_cube = 100.0 * (Z[1] / Z[0] - 1.0)
    d_mean = 100.0 * (np.nanmean(Z[1], axis=0) / np.nanmean(Z[0], axis=0) - 1.0)
    noise = np.array([100.0 * s["half"] / s["mean"] for s in stats[0]])  # 2 D / sqrt(n) / mean of REFERENCE
    by_class = [np.abs(d_all[R_all == R]) for R in classes]

    # FGR
    xl = [np.array([x_levels(c, q) for c in cubes]) for cubes in (refs, oths)]
    fgr = [{lv: pb.kanit(x[:, k], side) for k, lv in enumerate(pb.FGR_LEVELS)} for x in xl]
    d_fgr = np.array([100.0 * (fgr[1][lv]["mean"] / fgr[0][lv]["mean"] - 1.0) for lv in pb.FGR_LEVELS])
    in_verdict = np.array([lv in pb.IN_VERDICT for lv in pb.FGR_LEVELS])
    xc = np.concatenate([pb.critical_pressures(c, q)[0] for c in refs + oths])
    xc = xc[np.isfinite(xc)]
    x = np.linspace(0.97 * xc.min(), 1.02 * xc.max(), 800)
    curves = [np.array([pb.fgr_curve(c, q, x) for c in cubes]) for cubes in (refs, oths)]
    gap_fgr = 100.0 * np.abs(curves[1].mean(axis=0) - curves[0].mean(axis=0)).max()

    # mesh size: the variants of each ROOT
    var = [variants(all_, uns, q, classes) for all_, uns in ((ref_all, ref_unsolved), (oth_all, oth_unsolved))]

    # cost
    costs = [total_cost(cubes) for cubes in (refs, oths)]

    # ---- verdict -----------------------------------------------------------------------------------------------------
    worst_class = np.nanmax(np.abs(d_mean[verdict_class])) if verdict_class.any() else np.nan
    worst_fgr = np.nanmax(np.abs(d_fgr[in_verdict])) if in_verdict.any() else np.nan
    same = bool(worst_class <= 100 * pb.EPS_MESH and worst_fgr <= 100 * pb.EPS_MESH)
    below_noise = bool(np.all(np.abs(d_mean[verdict_class]) <= noise[verdict_class]))
    rve_class = [max((s["rve"] for s, v in zip(st, verdict_class) if v), default=np.nan) for st in stats]
    rve_fgr = [max((f[lv]["rve"] for lv in pb.IN_VERDICT), default=np.nan) for f in fgr]

    # ---- text --------------------------------------------------------------------------------------------------------
    ps = lambda v: f"{v:+7.2f}" if np.isfinite(v) else "      -"
    lines = [f"# The same {n} cubes of {side:g} um on two meshes (python3 mesh_compare.py {roots[0]} {roots[1]})",
             f"# reference: {labels[0]} ({roots[0]})",
             f"# other:     {labels[1]} ({roots[1]})",
             f"# quantity: {q}; d = other / reference - 1",
             "#",
             f"# 1. Per pore ({len(d_all)} pores): median |d| {np.median(np.abs(d_all)):.2f} %, 95 % of the pores within "
             f"{np.percentile(np.abs(d_all), 95):.2f} %, largest |d| {np.abs(d_all).max():.1f} % "
             f"(R = {R_all[np.argmax(np.abs(d_all))]:.3g} um)",
             "#",
             "# 2. Class means (Z of Kanit, mean over the cubes)",
             "# - d_mean = change of the mean of the cubes; d_cube = range of the change cube by cube",
             f"# - 2D/sqrt(n) = uncertainty of the mean of the {n} cubes of the reference (95 %), relative",
             "# - per pore: median and 95th percentile of |d| in the class",
             f"#     R_um  vol_%  {'mean_ref':>9s}{'mean_oth':>9s}  d_mean_%  d_cube_% range  2D/sqrt(n)_%"
             f"  med|d|  p95|d|  RVE_ref_um  RVE_oth_um"]
    for k, R in enumerate(classes):
        s0, s1 = stats[0][k], stats[1][k]
        lines.append(f"  {R:7.4f} {100 * share[k]:6.2f}  {s0['mean']:9.4f}{s1['mean']:9.4f}  {ps(d_mean[k])}  "
                     f"{np.nanmin(d_cube[:, k]):+6.2f} {np.nanmax(d_cube[:, k]):+6.2f}  {noise[k]:12.2f}"
                     f"  {np.median(by_class[k]):6.2f}  {np.percentile(by_class[k], 95):6.2f}"
                     f"  {s0['rve']:6.1f} {s0.get('interval', ''):>7s} {s1['rve']:6.1f} {s1.get('interval', ''):>7s}"
                     + ("" if verdict_class[k] else "   (below 5 % of the pore volume, not in the verdict)"))
    lines += ["#", "# 3. FGR: x_c = p / sigma_R at which the FGR reaches the level, mean over the cubes",
              f"#  FGR_%  {'x_c_ref':>8s}{'x_c_oth':>9s}  d_%     RVE_ref_um  RVE_oth_um"]
    for k, lv in enumerate(pb.FGR_LEVELS):
        f0, f1 = fgr[0][lv], fgr[1][lv]
        lines.append(f"  {100 * lv:5.3g}  {f0['mean']:8.4f}{f1['mean']:9.4f} {ps(d_fgr[k])}  {f0['rve']:6.1f} "
                     f"{f0.get('interval', ''):>7s} {f1['rve']:6.1f} {f1.get('interval', ''):>7s}"
                     + ("" if lv in pb.IN_VERDICT else "   (tail: not in the verdict)"))
    lines.append(f"# largest gap between the mean FGR curves: {gap_fgr:.2f} points of FGR")
    lines += ["#", "# 4. Size of the mesh: variants of side<L>_seed01 against side<L>_seed01 of the same ROOT, d of the "
                   "class means (%)",
              "#  " + " " * 52 + "".join(f"{R:8.3f}" for R in classes)]
    for (rows, missing), root in zip(var, roots):
        for r in rows:
            lines.append(f"  {r['label'] + ' against ' + r['base']:52s}" + "".join(ps(v) + " " for v in r["d"]))
        for name in missing:
            lines.append(f"  {name} ({os.path.basename(os.path.dirname(os.path.abspath(root)))}): not solved")
    lines += ["#", "# 5. Cost on the cluster (logs of the jobs; core-hours = cores x elapsed of the jobs)"]
    for lab, c in zip(labels, costs):
        if c is None:
            lines.append(f"  {lab}: no SLURM logs")
            continue
        states = ", ".join(f"{s} x{c['failed'].count(s)}" for s in sorted(set(c["failed"]))) or "none"
        lines += [f"  {lab}",
                  f"    mesh {c['vertices'] / 1e6:.2f} M vertices, {c['elements'] / 1e6:.2f} M elements, "
                  f"{c['unknowns'] / 1e6:.1f} M unknowns (mean of {c['n']} cubes)",
                  f"    partition-mesh {c['split_min']:.1f} min, {c['split_rss']:.1f} GB; MMM {c['mmm_min']:.1f} min on "
                  f"{c['cores']} cores ({c['nodes']} nodes, {c['tasks']} processes), {c['rss']:.2f} GB per process",
                  f"    core-hours of the {c['n']} cubes: partition {c['split_ch']:.1f}, MMM solved {c['solved_ch']:.1f}, "
                  f"MMM failed {c['failed_ch']:.1f} (runs that failed: {states}); total "
                  f"{c['split_ch'] + c['solved_ch'] + c['failed_ch']:.1f}"]
    total = [c["split_ch"] + c["solved_ch"] + c["failed_ch"] if c else np.nan for c in costs]
    useful = [c["split_ch"] + c["solved_ch"] if c else np.nan for c in costs]
    lines += ["#", "# Verdict",
              f"# classes of the verdict (>= {100 * pb.MIN_VOLUME_SHARE:g} % of the pore volume): largest |d_mean| "
              f"{worst_class:.2f} %" + (", within the uncertainty of the mean of the cubes" if below_noise else
                                        ", beyond the uncertainty of the mean of the cubes in some class"),
              f"# FGR levels {100 * pb.VERDICT_LEVELS[0]:g}-{100 * pb.VERDICT_LEVELS[1]:g} %: largest |d| "
              f"{worst_fgr:.2f} %",
              f"# side of the RVE (classes of the verdict): {rve_class[0]:.0f} um with the reference, {rve_class[1]:.0f}"
              f" um with the other; on the FGR: {rve_fgr[0]:.0f} um and {rve_fgr[1]:.0f} um",
              f"# {labels[1]} " + ("CAN" if same else "can NOT") + f" replace {labels[0]} (tolerance "
              f"{100 * pb.EPS_MESH:g} %)"]
    if opts["cost"] and all(np.isfinite(total)):
        lines.append(f"# cost: {total[1]:.0f} against {total[0]:.0f} core-hours ({total[0] / total[1]:.1f} x; "
                     f"{useful[0] / useful[1]:.1f} x without the runs that failed)")
    out = os.path.join(roots[1], opts["name"] + ".txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[-6:]))

    # ---- figure ------------------------------------------------------------------------------------------------------
    with_cost = opts["cost"] and all(cs is not None for cs in costs)
    # panels: every pore, the FGR and, with the logs of the cluster, the cost. The class means (a panel until
    # 11/11/2026) are in section 2 of the .txt: the scatter of every pore already shows which classes move
    if with_cost:
        fig, grid = plt.subplots(1, 3, figsize=(20.0, 6.4), layout="constrained",
                                 gridspec_kw={"width_ratios": [0.85, 1.0, 1.25]})
        ax = {(0, 0): grid[0], (1, 0): grid[1], (1, 1): grid[2]}
    else:
        fig, grid = plt.subplots(1, 2, figsize=(13.0, 6.2), layout="constrained",
                                 gridspec_kw={"width_ratios": [0.9, 1.0]})
        ax = {(0, 0): grid[0], (1, 0): grid[1]}
    nk = len(classes)
    wrap = lambda text, width=42: "\n".join(textwrap.wrap(text, width))  # long mesh labels must not squeeze the axes
    tag = {"s_hat": "s_hat", "sigma": "sigma_I / p"}[q]

    # (a) every pore
    a = ax[0, 0]
    for k in reversed(range(nk)):  # the small classes, the most scattered, on top
        m = R_all == classes[k]
        a.scatter(a_all[m], b_all[m], s=5, color=pb.class_color(k, nk), alpha=0.55, lw=0, rasterized=True,
                  label=f"R = {classes[k]:.3g} um ({pb.percent(share[k])}% vol)")
    lim = np.array([min(a_all.min(), b_all.min()), max(a_all.max(), b_all.max())]) * np.array([0.95, 1.03])
    a.plot(lim, lim, color=pb.REFERENCE_LINE, lw=1.0, zorder=0)
    for f in (0.95, 1.05):
        a.plot(lim, f * lim, color=pb.REFERENCE_LINE, lw=0.8, ls=":", zorder=0)
    a.set(xlim=lim, ylim=lim, xlabel=wrap(f"{pb.LABEL[q]}, {labels[0]}", 60), ylabel=wrap(f"{pb.LABEL[q]}, {labels[1]}", 60),
          title=f"Every pore of the {n} cubes: median |d| {np.median(np.abs(d_all)):.1f}%, "
                f"95% within {np.percentile(np.abs(d_all), 95):.1f}%")
    a.set_box_aspect(1)
    handles, names = a.get_legend_handles_labels()
    handles = handles[::-1] + [Line2D([], [], color=pb.REFERENCE_LINE, lw=0.8, ls=":")]
    a.legend(handles, names[::-1] + ["+/- 5 %"], loc="upper left", fontsize=9, markerscale=2.5, frameon=False)

    # (c) FGR
    c = ax[1, 0]
    m0, D0 = curves[0].mean(axis=0), curves[0].std(axis=0, ddof=1) if n > 1 else 0.0 * x
    c.fill_between(x, 100 * np.clip(m0 - 2 * D0, 0, 1), 100 * np.clip(m0 + 2 * D0, 0, 1), color=COLORS[0], alpha=0.15,
                   lw=0, label=wrap(f"{labels[0]}: mean +/- 2D (spread of one cube)", 48))
    c.plot(x, 100 * m0, color=COLORS[0], lw=2.2, label=wrap(f"{labels[0]}: mean of {n} cubes", 48))
    c.plot(x, 100 * curves[1].mean(axis=0), color=COLORS[1], lw=2.0, ls="--", label=wrap(f"{labels[1]}: mean of {n} cubes", 48))
    c.set(xlabel=r"$p / \sigma_R$", ylabel="FGR (%)", ylim=(0, 100), xlim=(x[0], x[-1]),
          title=f"FGR of the cubes")
    c.legend(loc="upper left", fontsize=9, frameon=False)

    # (d) cost
    if with_cost:
        d = ax[1, 1]
        ys = [1.0, 0.0]
        for i, (cs, y) in enumerate(zip(costs, ys)):
            parts = [(cs["split_ch"], dict(color=TINTS[i])), (cs["solved_ch"], dict(color=COLORS[i])),
                     (cs["failed_ch"], dict(facecolor="white", edgecolor=COLORS[i], hatch="///"))]
            left = 0.0
            for value, style in parts:
                if value <= 0.0:
                    continue
                d.barh(y, value, left=left, height=0.42, lw=0.8, **style)
                if value > 0.05 * max(total):
                    d.text(left + value / 2, y, f"{value:.0f}", ha="center", va="center", fontsize=9,
                           color="white" if style.get("color") == COLORS[i] else "black")
                left += value
            d.text(left + 0.01 * max(total), y, f"{left:.0f} core-h", va="center", fontsize=9, fontweight="bold")
            fails = len(cs["failed"])
            d.text(0.0, y - 0.29, f"{cs['elements'] / 1e6:.1f} M elements, {cs['unknowns'] / 1e6:.0f} M unknowns;  "
                                  f"MMM on {cs['cores']} cores ({cs['nodes']} nodes), {cs['rss']:.1f} GB per process"
                                  + (f";  {fails} failed runs" if fails else ""),
                   va="center", fontsize=9, color=MUTED)
        d.set_yticks(ys, [wrap(lab, 24) for lab in labels])
        d.set(xlim=(0, 1.22 * max(total)), ylim=(-0.55, 1.4),
              xlabel="core-hours of the jobs (cores x elapsed time)",
              title=wrap(f"Cost of the {n} cubes on the cluster: {max(total) / min(total):.1f} x "
                         + ("less" if total[1] < total[0] else "more") + f" with {labels[1]}", 70))
        grey = ("#c8c7c1", "#8a8a85")
        d.legend([Patch(color=grey[0]), Patch(color=grey[1]), Patch(facecolor="white", edgecolor=grey[1], hatch="///")],
                 ["partition-mesh", "MMM, solved", "MMM, failed (out of memory or cancelled)"], loc="upper right",
                 fontsize=9, frameon=False, ncol=3, bbox_to_anchor=(1.0, 1.0))
        d.grid(axis="x", color=pb.GRID, lw=0.6)
        d.set_axisbelow(True)
        for spine in ("top", "right"):
            d.spines[spine].set_visible(False)

    for a in (ax[0, 0], ax[1, 0]):
        a.grid(color=pb.GRID, lw=0.6)
        a.set_axisbelow(True)
    png = os.path.join(roots[1], opts["name"] + ".png")
    fig.savefig(png, dpi=150)
    print(f"wrote {out} and {png}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
