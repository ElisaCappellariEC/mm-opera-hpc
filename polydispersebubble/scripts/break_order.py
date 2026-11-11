"""
Which pores break first in the cubes of a run of MMM, and why: the unit runs (same pressure in every pore, test_rve.py,
merope_pore_pressure.py) or the pk cubes (pressure of its class in every pore, set_pressures.py).

Folders of ROOT as for plot_bubbles.py: the size cubes side<L>_seed<NN> (the mesh variants are left out). Per pore:
    s_hat, x_c = p_ref / (s_hat p) stress factor at zero macroscopic stress (plot_bubbles.py) and the load x = p_ref /
                                   sigma_R at which the pore breaks (1 / s_hat at one pressure; p_ref = sum(V p) / sum(V)):
                                   the pores break in the order of x_c, FGR = volume of the broken pores / volume of all
    neighbourhood                  gap (wall to wall, periodic cube) and radius of the first and second nearest pores;
                                   close pores, gap < CLOSE x distMin; F = sum_j (R_j / d_ij)^3, since the stress of a
                                   pressurised sphere decays as (R / d)^3
    point of the maximum           distance from the own wall, cosine with the direction of the nearest pore, distance
                                   from the wall of the nearest other pore (the shell r < R + delta of MMM can reach it)

Writes in ROOT
    pores.csv          one row per pore and cube, read by sciantix-official/utilities/HBFF/mmm_surrogate.py
    break_order.txt    s_hat per class (log-normal and normal fits), the first pores to break, correlations with the
                       neighbourhood, share of the scatter it explains, point of the maximum, clusters of broken pores
    break_order.png
Usage: python3 break_order.py [ROOT]

@author E. Cappellari
@date   October 2026
"""

import glob
import os
import re
import sys

import numpy as np
from scipy import stats

import plot_bubbles as pb

plt = pb.plt
CLOSE = 2.0                                  # close pores, and pores of one cluster: gap < CLOSE x distMin
                                             # (also read by npoint.py)
LEVELS = (0.01, 0.05, 0.10, 0.25, 0.50)      # levels of FGR of the table of the first pores to break
FEATURES = ("R", "gap1", "R_nn1", "gap2", "R_nn2", "close", "F")  # neighbourhood seen by the regression


def dist_min(folder):
    """distMin of Mérope (um) from summary_<run>.txt of the cube; nan if it is not written."""
    for path in glob.glob(os.path.join(folder, "summary_*.txt")):
        m = re.search(r"distMin ([\d.eE+-]+) um", open(path).read())
        if m:
            return float(m.group(1))
    return np.nan


def neighbourhood(c, d_min):
    """Per pore of the cube c (read_cube): the features of its neighbourhood, the point of its maximum, and the close
    pairs (gap < CLOSE x d_min) as a list of neighbours."""
    X, R, L = c["centre"], c["R"], c["side"]
    n, i = len(R), np.arange(len(R))
    d = X[:, None, :] - X[None, :, :]
    d -= L * np.round(d / L)
    dist = np.linalg.norm(d, axis=2)
    np.fill_diagonal(dist, np.inf)
    gaps = dist - R[:, None] - R[None, :]
    first2 = np.argpartition(gaps, 1, axis=1)[:, :2]
    swap = gaps[i, first2[:, 0]] > gaps[i, first2[:, 1]]
    first2[swap] = first2[swap][:, ::-1]
    nn1, nn2 = first2[:, 0], first2[:, 1]
    close = gaps < CLOSE * d_min
    f = {"gap1": gaps[i, nn1], "R_nn1": R[nn1], "gap2": gaps[i, nn2], "R_nn2": R[nn2], "close": close.sum(axis=1),
         "F": np.sum((R[None, :] / dist) ** 3, axis=1)}
    f["max_wall"] = f["max_cos"] = f["max_other"] = np.full(n, np.nan)
    if c["max_point"] is not None:
        m = c["max_point"] - X
        m -= L * np.round(m / L)
        to_nn = -d[i, nn1]
        f["max_wall"] = np.linalg.norm(m, axis=1) - R
        f["max_cos"] = np.sum(m * to_nn, axis=1) / (np.linalg.norm(m, axis=1) * np.linalg.norm(to_nn, axis=1))
        e = c["max_point"][:, None, :] - X[None, :, :]
        e -= L * np.round(e / L)
        walls = np.linalg.norm(e, axis=2) - R[None, :]
        walls[i, i] = np.inf
        f["max_other"] = walls.min(axis=1)
    return f, [np.nonzero(row)[0] for row in close]


def break_sequence(xc, V, neighbours):
    """The pores broken one by one in the order of x_c: rank of every pore (1 = first), FGR after every break and the
    volume share of the largest cluster of broken pores (linked by close pairs) after every break."""
    order = np.argsort(xc, kind="stable")
    parent, size = np.arange(len(xc)), V.copy()
    broken = np.zeros(len(xc), dtype=bool)

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    largest, best = np.empty(len(xc)), 0.0
    for k, a in enumerate(order):
        broken[a] = True
        for b in neighbours[a]:
            if broken[b]:
                ra, rb = root(a), root(b)
                if ra != rb:
                    parent[rb] = ra
                    size[ra] += size[rb]
        best = max(best, size[root(a)])
        largest[k] = best
    rank = np.empty(len(xc), dtype=int)
    rank[order] = np.arange(1, len(xc) + 1)
    return rank, np.cumsum(V[order]) / V.sum(), largest / V.sum()


def read_run(root):
    """Size cubes of root with s_hat, as dicts of per-pore arrays (cube, features, breaking order)."""
    folders = sorted(f for f in glob.glob(os.path.join(root, "side*_seed*"))
                     if os.path.isdir(f) and pb.CASE.match(os.path.basename(f)) and not pb.CASE.match(os.path.basename(f))["variant"])
    cubes = []
    for folder in folders:
        c = pb.read_cube(folder)
        if c is None:
            continue
        q = "s_hat" if c["s_hat"] is not None else "sigma"
        d_min = dist_min(folder)
        if not np.isfinite(d_min):
            d_min = np.nanmin(c["gap"])
            print(f"{folder}: no distMin in the summary, the smallest gap {d_min:.3f} um is used")
        f, neighbours = neighbourhood(c, d_min)
        s = c[q]
        xc = pb.critical_x(c, s)
        rank, fgr, largest = break_sequence(xc, c["V"], neighbours)
        c.update(q=q, d_min=d_min, s=s, xc=xc, rank=rank, fgr_steps=fgr, largest=largest, fgr_at_break=fgr[rank - 1],
                 w=c["p"] / c["p_ref"], **f)
        cubes.append(c)
    if not cubes:
        raise SystemExit(f"no solved side<L>_seed<NN> folder in {root}")
    if len({c["load"] for c in cubes}) > 1:
        raise SystemExit(f"{root}: cubes with different loads ({', '.join(sorted({c['load'] for c in cubes}))})")
    return cubes


def pooled(cubes, key):
    return np.concatenate([np.asarray(c[key], dtype=float) for c in cubes])


def write_pores(root, cubes, radii):
    """pores.csv: one row per pore and cube."""
    names = ["cube", "seed", "side_um", "pore_id", "class", "R_um", "xi", "h", "sigma", "s_hat", "x_c", "gap1_um",
             "R_nn1_um", "gap2_um", "R_nn2_um", "close", "F", "max_wall_um", "max_cos_nn", "max_other_wall_um",
             "break_rank", "fgr_at_break", "p_MPa", "p_ref_MPa", "s_hat_1", "max_shift_um"]
    path = os.path.join(root, "pores.csv")
    load = cubes[0]["load"]
    with open(path, "w") as out:
        out.write(f"# pores of {root} (python3 break_order.py), load {load}: "
                  + ("s_hat = (sigma + h) / (1 - h), x_c = 1 / s_hat" if load == "unit" else
                     "s_hat = sigma + h p_ref (1 + s_hat_1) / p, s_hat_1 of the unit run of the same cube, "
                     "x_c = p_ref / (s_hat p)")
                  + "; p_ref = sum(V p) / sum(V); max_shift = distance between the points of the maxima of the run and "
                  "of its unit run (pk cubes); "
                  f"class 0 = smallest radius; close = pores with a gap < {CLOSE:g} distMin; "
                  "F = sum_j (R_j / d_ij)^3; break_rank 1 = first pore to break in its cube\n")
        out.write(",".join(names) + "\n")
        for c in cubes:
            k = np.searchsorted(radii, c["R"])
            nan = np.full(len(c["R"]), np.nan)
            s_hat = c["s_hat"] if c["s_hat"] is not None else nan
            s_hat_1 = c["s_hat_1"] if c["s_hat_1"] is not None else (s_hat if c["load"] == "unit" else nan)
            shift = c["max_shift"] if c["max_shift"] is not None else nan
            for j in range(len(c["R"])):
                row = [c["folder"], c["seed"], f"{c['side']:g}", c["id"][j], k[j], f"{c['R'][j]:.6f}", f"{c.get('xi', np.nan):.6f}",
                       f"{c.get('h', np.nan):.6f}", f"{c['sigma'][j]:.6f}", f"{s_hat[j]:.6f}", f"{c['xc'][j]:.6f}",
                       f"{c['gap1'][j]:.4f}", f"{c['R_nn1'][j]:.6f}", f"{c['gap2'][j]:.4f}", f"{c['R_nn2'][j]:.6f}",
                       c["close"][j], f"{c['F'][j]:.5f}", f"{c['max_wall'][j]:.4f}", f"{c['max_cos'][j]:.4f}",
                       f"{c['max_other'][j]:.4f}", c["rank"][j], f"{c['fgr_at_break'][j]:.6f}", f"{c['p'][j]:.6g}",
                       f"{c['p_ref']:.6g}", f"{s_hat_1[j]:.6f}", f"{shift[j]:.4f}"]
                out.write(",".join(str(v) for v in row) + "\n")
    return path


def class_table(s, cls, radii, V):
    """Lines of the statistics of s_hat per class, with the Kolmogorov-Smirnov distance of a normal and a log-normal fit."""
    lines = ["# 1. s_hat per class, all the cubes together; KS = Kolmogorov-Smirnov distance of the fit (smaller is better)",
             f"# {'R_um':>6s}{'vol_%':>7s}{'pores':>7s}{'mean':>8s}{'std':>7s}{'skew':>6s}{'q05':>7s}{'q50':>7s}{'q95':>7s}"
             f"{'mu_ln':>8s}{'sig_ln':>8s}{'KS_norm':>9s}{'KS_logn':>9s}  better"]
    for k, R in enumerate(radii):
        x = s[cls == k]
        x = x[np.isfinite(x) & (x > 0)]
        mu, sig = np.log(x).mean(), np.log(x).std(ddof=1)
        ks_n = stats.kstest(x, "norm", args=(x.mean(), x.std(ddof=1))).statistic
        ks_l = stats.kstest(np.log(x), "norm", args=(mu, sig)).statistic
        share = V[cls == k].sum() / V.sum()
        q05, q50, q95 = np.quantile(x, (0.05, 0.5, 0.95))
        lines.append(f"  {R:6.3f}{100 * share:7.2f}{len(x):7d}{x.mean():8.4f}{x.std(ddof=1):7.4f}{stats.skew(x):6.2f}"
                     f"{q05:7.3f}{q50:7.3f}{q95:7.3f}{mu:8.4f}{sig:8.4f}{ks_n:9.4f}{ks_l:9.4f}  "
                     f"{'log-normal' if ks_l < ks_n else 'normal'}")
    return lines


def first_to_break(cubes, cls, radii):
    """Lines of the first pores to break, all the cubes together (one common x = p / sigma_R per level)."""
    s, xc, V = pooled(cubes, "s"), pooled(cubes, "xc"), pooled(cubes, "V")
    gap, close = pooled(cubes, "gap1"), pooled(cubes, "close")
    unit = all(c["load"] == "unit" for c in cubes)
    order = np.argsort(xc, kind="stable")
    released = np.cumsum(V[order]) / V.sum()
    head = "".join(f"{R:>7.2f}" for R in radii)
    lines = ["#", "# 2. First pores to break, all the cubes together: at the x = p_ref / sigma_R where the FGR reaches the level",
             "# - pores by class (top line) and share of the broken volume by class (%, bottom line)",
             "# - gap = median gap of the broken pores to their nearest pore (all the pores in the last row); close = median "
             f"number of pores with a gap < {CLOSE:g} distMin",
             f"# - cluster = largest cluster of broken pores (linked by gaps < {CLOSE:g} distMin), % of the broken volume, "
             "mean over the cubes",
             "# - s_hat>= = 1 / x, the stress factor of the pores broken at x" + ("" if unit else
                 " (class pressures: s_hat p / p_ref>=, the pores of class k break when s_hat >= 1 / (x p_k / p_ref))"),
             f"# {'FGR_%':>5s}{'x':>7s}{'s_hat>=':>8s}{'pores':>7s}{head}{'gap_um':>8s}{'close':>6s}{'cluster_%':>10s}"]
    for lv in LEVELS:
        k = min(np.searchsorted(released, lv - 1e-12), len(order) - 1)
        x = xc[order][k]
        b = order[: k + 1]
        counts = [np.count_nonzero(cls[b] == j) for j in range(len(radii))]
        shares = [V[b][cls[b] == j].sum() / V[b].sum() for j in range(len(radii))]
        cluster = []
        for c in cubes:
            n = np.count_nonzero(c["xc"] <= x)
            if n:
                cluster.append(c["largest"][n - 1] / c["fgr_steps"][n - 1])
        lines.append(f"  {100 * lv:5.3g}{x:7.4f}{1.0 / x:8.3f}{len(b):7d}" + "".join(f"{v:7d}" for v in counts)
                     + f"{np.median(gap[b]):8.3f}{np.median(close[b]):6.0f}{100 * np.mean(cluster):10.1f}")
        lines.append(f"  {'':5s}{'':7s}{'':8s}{'':7s}" + "".join(f"{100 * v:7.1f}" for v in shares))
    lines.append(f"  {'all':>5s}{'':7s}{'':8s}{len(s):7d}" + "".join(f"{np.count_nonzero(cls == j):7d}" for j in range(len(radii)))
                 + f"{np.median(gap):8.3f}{np.median(close):6.0f}")
    return lines


def correlations(cubes, cls, radii):
    """Lines of the Spearman correlation of s_hat with the neighbourhood, per class and for all the pores."""
    s, R = pooled(cubes, "s"), pooled(cubes, "R")
    feats = {"gap1": pooled(cubes, "gap1"), "gap1/R": pooled(cubes, "gap1") / R, "R_nn1/R": pooled(cubes, "R_nn1") / R,
             "gap2": pooled(cubes, "gap2"), "close": pooled(cubes, "close"), "F": pooled(cubes, "F")}
    ok = np.isfinite(s)
    lines = ["#", "# 3. Spearman correlation of s_hat with the neighbourhood (gap1, gap2 = gaps to the first and second "
             "nearest pores; R_nn1 = radius of the nearest pore)",
             f"# {'feature':>8s}" + "".join(f"{R_:>8.3f}" for R_ in radii) + f"{'all':>8s}"]
    for name, v in feats.items():
        row = [stats.spearmanr(s[ok & (cls == k)], v[ok & (cls == k)]).statistic for k in range(len(radii))]
        row.append(stats.spearmanr(s[ok], v[ok]).statistic)
        lines.append(f"  {name:>8s}" + "".join(f"{r:8.2f}" for r in row))
    return lines


def explained(cubes, cls, radii):
    """Lines of the share of the variance of s_hat explained out of sample (one cube left out at a time) by the class
    alone and by the neighbourhood (gradient boosting on FEATURES)."""
    lines = ["#", "# 4. Share of the variance of s_hat explained by the neighbourhood (R^2 out of sample: every cube is "
             "predicted by a model fitted on the other cubes)"]
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.model_selection import GroupKFold
    except ImportError:
        return lines + ["#   sklearn missing: not computed"]
    s = pooled(cubes, "s")
    X = np.column_stack([pooled(cubes, f) for f in FEATURES])
    group = np.concatenate([np.full(len(c["R"]), c["seed"]) for c in cubes])
    ok = np.isfinite(s) & np.all(np.isfinite(X), axis=1)
    s, X, group, k = s[ok], X[ok], group[ok], cls[ok]
    folds = min(len(cubes), 10)
    if folds < 3:
        return lines + ["#   fewer than 3 cubes: not computed"]
    by_class, by_model = np.empty_like(s), np.empty_like(s)
    for train, test in GroupKFold(n_splits=folds).split(X, s, group):
        means = {j: s[train][k[train] == j].mean() for j in np.unique(k)}
        by_class[test] = [means[j] for j in k[test]]
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=1).fit(X[train], s[train])
        by_model[test] = model.predict(X[test])

    def r2(pred, mask):
        return 1.0 - np.sum((s[mask] - pred[mask]) ** 2) / np.sum((s[mask] - s[mask].mean()) ** 2)

    all_ = np.ones(len(s), dtype=bool)
    lines += [f"#   features: {', '.join(FEATURES)}; {folds} folds by cube",
              f"#   class alone: R^2 = {r2(by_class, all_):.3f};  neighbourhood: R^2 = {r2(by_model, all_):.3f}",
              f"# {'R_um':>6s}{'R2_within_class':>17s}"]
    lines += [f"  {R:6.3f}{r2(by_model, k == j):17.3f}" for j, R in enumerate(radii)]
    return lines


def maximum_point(cubes):
    """Lines of where the maximum of sigma_I is, for all the pores and for the first 5 % of the volume to break."""
    wall, cos, other = pooled(cubes, "max_wall"), pooled(cubes, "max_cos"), pooled(cubes, "max_other")
    if not np.isfinite(wall).any():
        return ["#", "# 5. Point of the maximum: no x, y, z in the outputs of MMM"]
    fgr = pooled(cubes, "fgr_at_break")
    h = np.nanmedian([c["wall"] for c in cubes])
    first = fgr <= 0.05
    lines = ["#", f"# 5. Point of the maximum of sigma_I (mesh size at the walls {h:g} um)",
             f"# {'pores':>14s}{'n':>7s}{'wall_med_um':>12s}{'wall>h_%':>10s}{'cos_med':>8s}{'cos<0.7_%':>10s}"
             f"{'on_other_wall_%':>16s}"]
    for name, m in (("all", np.isfinite(wall)), ("first 5 % vol", first & np.isfinite(wall))):
        lines.append(f"  {name:>14s}{np.count_nonzero(m):7d}{np.median(wall[m]):12.3f}{100 * np.mean(wall[m] > h):10.1f}"
                     f"{np.median(cos[m]):8.2f}{100 * np.mean(cos[m] < 0.7):10.1f}{100 * np.mean(other[m] < h / 2):16.1f}")
    lines.append(f"#   wall = distance of the point from the own wall; cos = cosine between the point and the nearest pore, "
                 f"seen from the centre; on_other_wall = point within {h / 2:g} um of the wall of another pore")
    return lines


def figure(root, cubes, cls, radii):
    """break_order.png: s_hat against the gap; broken share of every class against x. (The FGR stacked by class and the
    largest cluster against the FGR, the bottom row until 11/11/2026, were dropped: the shares by class and the clusters
    are in sections 2 of break_order.txt.)"""
    s, xc, gap = pooled(cubes, "s"), pooled(cubes, "xc"), pooled(cubes, "gap1")
    w = pooled(cubes, "w")  # p / p_ref of every pore (1 at one pressure)
    n = len(radii)
    d_min = np.nanmedian([c["d_min"] for c in cubes])
    fig, ax = plt.subplots(1, 2, figsize=(13.0, 5.0))

    a = ax[0]
    for k in range(n):
        m = cls == k
        a.scatter(gap[m], s[m], s=3, color=pb.class_color(k, n), alpha=0.35, lw=0, label=f"R = {radii[k]:.3g} um")
    bins = np.geomspace(np.nanmin(gap), np.nanmax(gap), 25)
    mid, med = np.sqrt(bins[1:] * bins[:-1]), []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (gap >= lo) & (gap < hi) & np.isfinite(s)
        med.append(np.median(s[m]) if np.count_nonzero(m) >= 20 else np.nan)
    a.plot(mid, med, color="black", lw=2.0, label="median of all the pores")
    a.axvline(d_min, color=pb.REFERENCE_LINE, ls="--", lw=1.0)
    a.annotate("distMin", (d_min, 0.98), xycoords=("data", "axes fraction"), ha="left", va="top", fontsize=9,
               color=pb.REFERENCE_LINE, xytext=(3, 0), textcoords="offset points")
    a.set(xscale="log", xlabel="gap to the nearest pore, wall to wall (um)", ylabel=pb.LABEL[cubes[0]["q"]],
          title="Stress factor of every pore against its gap")
    ticks = [t for t in (0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0) if np.nanmin(gap) * 0.95 <= t <= np.nanmax(gap)]
    a.set_xticks(ticks, [f"{t:g}" for t in ticks])
    a.minorticks_off()
    legend = a.legend(frameon=False, fontsize=9, markerscale=3, loc="upper right")
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)

    finite = np.isfinite(xc)
    x = np.linspace(0.97 * xc[finite].min(), min(1.02 * xc[finite].max(), 1.6), 600)
    a = ax[1]
    for k in range(n):
        m = (cls == k) & np.isfinite(s) & (s > 0)
        sk, xk, lw = s[m], xc[m], np.log(w[m]).mean()
        a.plot(x, 100 * np.mean(xk[None, :] < x[:, None], axis=1), color=pb.class_color(k, n), lw=1.6,
               label=f"R = {radii[k]:.3g} um" + ("" if abs(lw) < 1e-9 else f" (p / p_ref = {np.exp(lw):.2f})"))
        mu, sig = np.log(sk).mean(), np.log(sk).std(ddof=1)
        a.plot(x, 100 * stats.norm.sf((-np.log(x) - lw - mu) / sig), color=pb.class_color(k, n), lw=1.0, ls="--")
    a.plot([], [], color="black", ls="--", lw=1.0, label="log-normal fit of the class")
    a.set(xlabel=r"$p / \sigma_R$" if np.allclose(w, 1.0) else r"$p_{ref} / \sigma_R$", ylabel="broken pores of the class (%)", ylim=(0, 100),
          title="Broken pores of every class")
    a.legend(frameon=False, fontsize=9)

    for a in ax.ravel():
        a.grid(True, color=pb.GRID)
    fig.tight_layout()
    path = os.path.join(root, "break_order.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else pb.ROOT
    cubes = read_run(root)
    q = cubes[0]["q"]
    radii = np.unique(pooled(cubes, "R"))
    cls = np.searchsorted(radii, pooled(cubes, "R"))
    s, V = pooled(cubes, "s"), pooled(cubes, "V")
    lines = [f"# Which pores break first (python3 break_order.py {root}): {len(cubes)} cubes, {len(s)} pores, "
             f"quantity {q}, distMin {np.nanmedian([c['d_min'] for c in cubes]):g} um",
             f"# load {cubes[0]['load']}: a pore breaks when s_hat p > sigma_R, i.e. at x = p_ref / sigma_R = p_ref / (s_hat p) "
             "(1 / s_hat at one pressure, p_ref = sum(V p) / sum(V)): the pores break in the order of x"]
    lines += class_table(s, cls, radii, V)
    lines += first_to_break(cubes, cls, radii)
    lines += correlations(cubes, cls, radii)
    lines += explained(cubes, cls, radii)
    lines += maximum_point(cubes)
    with open(os.path.join(root, "break_order.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {write_pores(root, cubes, radii)}, break_order.txt and {figure(root, cubes, cls, radii)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
