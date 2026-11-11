"""
n-point probabilities of the pores and of the broken pores in the cubes of a run of MMM, and the intact stretches of a
line between broken pores as a measure of the fragment size (same pressure in every pore, as break_order.py).

Phases of a cube (periodic), sampled on random lines (RAYS per cube, isotropic, uniform origins, H = side / 2 on both
sides of the origin):
    pore     inside a pore. S1 = porosity, S2(r) = both ends of a segment of length r in a pore. For spheres that do not
             overlap, S2 = same-pore term (a sphere and itself shifted by r: the radii only) + different-pore term (the
             arrangement); S2(0) = S1, S2(inf) = S1^2, S2'(0) = -s / 4 (s = pore surface per volume)
    intact   outside the cracked zones of the broken pores: spheres of radius (1 + k) R around the pores with
             x_c <= x (break_order.py), k in CRACK [?] (k = 0: the broken pore alone)
S_n(z): n points evenly spaced on a segment of length z, all in the phase. S_n -> L(z) for n -> inf (lineal path,
B. Lu, S. Torquato, Phys. Rev. A 45 (1992) 922): 1 - L(z) = probability that a segment of length z is broken (it
enters a cracked zone). The intact chords of a line are the line intercepts of the fragments:
    number-weighted mean   l_n = 4 L(0) / s_c for any arrangement (s_c = surface of the cracked zones per volume)
    length-weighted mean   l_w = <l^2> / <l>, the intercept a random point sees: it depends on the arrangement
SCIANTIX option 4 (HighBurnupStructureFragmentation.C): d = 2 R_WS / D, D the number fraction of broken pores; a cube
of side d has the mean intercept 2 d / 3.
"random": the same number of broken pores per class and cube taken at random within the class (PERMUTATIONS), the
assumption of SCIANTIX (broken share per class, no neighbourhood).
Pairs of pores: probability that both pores of a pair with the gap g are broken, over the same for broken pores taken
at random within their class (1 = no grouping of the broken pores).

Writes in ROOT: npoint.txt, npoint.png
Usage: python3 npoint.py [ROOT] [RAYS]

@author E. Cappellari
@date   November 2026
"""

import os
import sys

import numpy as np
from scipy.spatial import cKDTree

import break_order as bo

pb = bo.pb
plt = pb.plt

RAYS = 100_000                                       # random lines per cube
SEED = 2026                                          # of the lines and of the random choices, + seed of the cube
STEP = 0.05                                          # um: points of a line for S_n, from the origin to H
LEVELS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0)   # FGR (broken pore volume / pore volume) of the broken sets
CRACK = (0.0, 0.5, 1.0, 1.5)                         # k: cracked zone of a broken pore = sphere of radius (1 + k) R
N_POINTS = (2, 3, 5, 10)                             # n of S_n(z)
PERMUTATIONS = 10                                    # random choices of the broken pores, per cube and level
GAP_MAX, GAP_BINS = 6.0, 12                          # um, log bins of the gap of the pairs of pores from distMin
CHUNK = 400                                          # lines per block in the search of the pores met by the lines
CAPPED = 0.05                                        # share of chords reaching an end of the line: mean drawn hollow
Z_SHOWN = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 14.0)  # um: rows of the S_n table


def add(acc, key, value, w=1.0):
    acc[key] = acc.get(key, 0.0) + w * np.asarray(value, dtype=float)


def levels_x(cubes):
    """x = p / sigma_R at which the FGR of all the cubes together reaches every level (as break_order.py)."""
    xc, V = bo.pooled(cubes, "xc"), bo.pooled(cubes, "V")
    order = np.argsort(xc, kind="stable")
    released = np.cumsum(V[order]) / V.sum()
    return [xc[order][min(np.searchsorted(released, lv - 1e-12), len(order) - 1)] for lv in LEVELS]


def candidates(c, o, u, H, grow):
    """Pores that the lines o + t u, |t| <= H, can meet with a zone of radius grow R: per (line, pore) the abscissa t of
    the centre and its squared distance d2 from the line. The line is cut in pieces short enough that one image of a
    pore at most meets a piece: the image nearest to the middle of the piece."""
    X, R, side = c["centre"], c["R"], c["side"]
    reach = grow * R
    piece = side - 2.0 * reach.max()
    if piece <= 0.0:
        raise SystemExit(f"{c['folder']}: cracked zones of {reach.max():.2f} um do not fit the side {side:g} um")
    n = int(np.ceil(H / (0.999 * piece)))
    edges = np.linspace(-H, H, 2 * n + 1)
    out = {key: [] for key in ("line", "pore", "t", "d2", "piece")}
    for start in range(0, len(o), CHUNK):
        oc, uc = o[start:start + CHUNK], u[start:start + CHUNK]
        for p, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            mid = 0.5 * (a + b)
            dv = X[None, :, :] - (oc + mid * uc)[:, None, :]
            dv -= side * np.round(dv / side)
            dv += mid * uc[:, None, :]                   # from the origin of the line to the image
            t = np.einsum("ijk,ik->ij", dv, uc)
            d2 = np.einsum("ijk,ijk->ij", dv, dv) - t * t
            i, j = np.nonzero((d2 < reach**2) & (t + reach > a) & (t - reach < b))
            out["line"].append((i + start).astype(np.int32))
            out["pore"].append(j.astype(np.int32))
            out["t"].append(t[i, j])
            out["d2"].append(d2[i, j])
            out["piece"].append(np.full(len(i), p, dtype=np.int16))
    cand = {key: np.concatenate(v) for key, v in out.items()}
    cand["edges"] = edges
    return cand


def zones(cand, R, k):
    """Zones of radius (1 + k) R met by the lines: line, pore, entry and exit abscissae, ends of the piece."""
    r2 = ((1.0 + k) * R[cand["pore"]]) ** 2
    m = cand["d2"] < r2
    w = np.sqrt(r2[m] - cand["d2"][m])
    t, p = cand["t"][m], cand["piece"][m]
    return {"line": cand["line"][m], "pore": cand["pore"][m], "t_in": t - w, "t_out": t + w,
            "a": cand["edges"][p], "b": cand["edges"][p + 1]}


def hits(z, broken, H, lines):
    """Per line: origin in the zone of a broken pore; distance to the first zone forward and backward (H at most)."""
    m = broken[z["pore"]]
    line, t_in, t_out = z["line"][m], z["t_in"][m], z["t_out"][m]
    inside = np.zeros(lines, dtype=bool)
    inside[line[(t_in <= 0.0) & (t_out > 0.0)]] = True
    fwd, bwd = np.full(lines, H), np.full(lines, H)
    f = t_in > 0.0
    np.minimum.at(fwd, line[f], t_in[f])
    g = t_out <= 0.0
    np.minimum.at(bwd, line[g], -t_out[g])
    return inside, fwd, bwd


def intact(inside, fwd, bwd, H, z):
    """Lineal path L(z) of the intact phase and the sums of its chords through the origins (length-biased samples)."""
    out = ~inside
    f = np.sort(fwd[out])
    chord = fwd[out] + bwd[out]
    return {"L": (len(f) - np.searchsorted(f, z, side="right")) / len(inside), "n": len(chord),
            "inv": np.sum(1.0 / chord), "sum": np.sum(chord), "capped": np.count_nonzero((fwd[out] >= H) | (bwd[out] >= H))}


def paint(z, broken, lines, T, label=False):
    """Zones of the broken pores on the points t = j STEP (j < T) of every line: their number, or the pore id + 1 with
    label (zones that do not overlap, k = 0). Each zone is cut to its piece, so that a zone met by two pieces counts once."""
    m = broken[z["pore"]]
    t0 = np.maximum(np.maximum(z["t_in"][m], z["a"][m]), 0.0)
    t1 = np.minimum(z["t_out"][m], z["b"][m])
    i0 = np.ceil(t0 / STEP).astype(np.int64)
    i1 = np.minimum(np.ceil(t1 / STEP).astype(np.int64), T)
    ok = i0 < i1
    line = z["line"][m][ok].astype(np.int64)
    w = z["pore"][m][ok] + 1.0 if label else np.ones(np.count_nonzero(ok))
    flat = np.bincount(np.concatenate([line * (T + 1) + i0[ok], line * (T + 1) + i1[ok]]),
                       weights=np.concatenate([w, -w]), minlength=lines * (T + 1))
    return np.cumsum(flat.reshape(lines, T + 1)[:, :T], axis=1)


def n_point(phase):
    """S_n(z_j), z_j = j STEP: the n points 0, z/(n-1), ..., z of a line all in the phase (lines x T booleans)."""
    T = phase.shape[1]
    out = {1: np.full(T, phase[:, 0].mean())}
    for n in N_POINTS:
        pos = np.rint(np.outer(np.arange(T), np.arange(n)) / (n - 1)).astype(int)
        out[n] = np.array([phase[:, p].all(axis=1).mean() for p in pos])
    return out


def same_pore(R, V, r):
    """Same-pore term of S2: volume shared by every pore and itself shifted by r, over the volume of the cube."""
    x = r[None, :] / R[:, None]
    lens = 4.0 / 3.0 * np.pi * R[:, None] ** 3 * (1.0 - 0.75 * x + x**3 / 16.0)
    return np.sum(np.where(x < 2.0, lens, 0.0), axis=0) / V


def pairs(c):
    """Pairs of pores with a gap below GAP_MAX in the periodic cube: i, j, gap (wall to wall)."""
    side, R = c["side"], c["R"]
    X = np.mod(c["centre"], side)
    ij = cKDTree(X, boxsize=side).query_pairs(GAP_MAX + 2.0 * R.max(), output_type="ndarray")
    d = X[ij[:, 0]] - X[ij[:, 1]]
    d -= side * np.round(d / side)
    gap = np.linalg.norm(d, axis=1) - R[ij[:, 0]] - R[ij[:, 1]]
    keep = gap < GAP_MAX
    return ij[keep, 0], ij[keep, 1], gap[keep]


def pair_counts(i, j, kbin, nbins, broken, cls, nclass):
    """Per gap bin: pairs with both pores broken, and their expected number with the broken pores of every class taken
    at random within the class (without replacement)."""
    n_c = np.bincount(cls, minlength=nclass).astype(float)
    b_c = np.bincount(cls[broken], minlength=nclass).astype(float)
    p = np.divide(b_c, n_c, out=np.zeros(nclass), where=n_c > 0)
    q = np.divide(b_c * (b_c - 1.0), n_c * (n_c - 1.0), out=np.zeros(nclass), where=n_c > 1)
    expected = np.where(cls[i] == cls[j], q[cls[i]], p[cls[i]] * p[cls[j]])
    observed = (broken[i] & broken[j]).astype(float)
    return np.bincount(kbin, weights=observed, minlength=nbins), np.bincount(kbin, weights=expected, minlength=nbins)


def random_broken(rng, cls, broken):
    """The same number of broken pores per class, taken at random within the class."""
    out = np.zeros_like(broken)
    for k in np.unique(cls):
        idx = np.nonzero(cls == k)[0]
        out[rng.choice(idx, np.count_nonzero(broken[idx]), replace=False)] = True
    return out


def analyse_cube(c, radii, xs, bins, rays, acc):
    """Adds to acc the sums of one cube."""
    rng = np.random.default_rng(SEED + c["seed"])
    side, R = c["side"], c["R"]
    H, V, T = 0.5 * side, side**3, int(round(0.5 * side / STEP))
    z = np.arange(T) * STEP
    o = rng.uniform(0.0, side, size=(rays, 3))
    u = rng.normal(size=(rays, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    cand = candidates(c, o, u, H, 1.0 + max(CRACK))
    zs = {k: zones(cand, R, k) for k in CRACK}
    z0 = zs[0.0] if 0.0 in zs else zones(cand, R, 0.0)
    every = np.ones(len(R), dtype=bool)

    # pores: S2 and its same-pore term on the lines, S_n of the matrix
    label = np.rint(paint(z0, every, rays, T, label=True)).astype(np.int32)
    pore = label > 0
    add(acc, "S2", (pore[:, :1] & pore).mean(axis=0))
    add(acc, "S2_same", (pore[:, :1] & (label == label[:, :1])).mean(axis=0))
    add(acc, "S2_same_exact", same_pore(R, V, z))
    add(acc, "phi_lines", pore[:, 0].mean())
    for n, s_n in n_point(~pore).items():
        add(acc, ("Pn", n), s_n)
    del label, pore
    add(acc, "phi", np.sum(4.0 / 3.0 * np.pi * R**3) / V)
    add(acc, "s", np.sum(4.0 * np.pi * R**2) / V)
    add(acc, "pores", len(R))
    add(acc, "volume", V)

    # pairs of pores by gap
    cls = np.searchsorted(radii, R)
    i, j, gap = pairs(c)
    kbin = np.digitize(gap, bins) - 1
    m = (kbin >= 0) & (kbin < len(bins) - 1)
    i, j, kbin = i[m], j[m], kbin[m]
    add(acc, "pairs", np.bincount(kbin, minlength=len(bins) - 1))

    for lv, x in zip(LEVELS, xs):
        broken = c["xc"] <= x
        add(acc, ("broken", lv), np.count_nonzero(broken))
        add(acc, ("phi_b", lv), np.sum(4.0 / 3.0 * np.pi * R[broken] ** 3) / V)
        add(acc, ("s_b", lv), np.sum(4.0 * np.pi * R[broken] ** 2) / V)
        obs, exp = pair_counts(i, j, kbin, len(bins) - 1, broken, cls, len(radii))
        add(acc, ("obs", lv), obs)
        add(acc, ("exp", lv), exp)
        randoms = ([random_broken(rng, cls, broken) for _ in range(PERMUTATIONS)]
                   if 0 < np.count_nonzero(broken) < len(R) else [broken])
        for k in CRACK:
            for key, value in intact(*hits(zs[k], broken, H, rays), H, z).items():
                add(acc, (lv, k, "actual", key), value)
            for b in randoms:
                for key, value in intact(*hits(zs[k], b, H, rays), H, z).items():
                    add(acc, (lv, k, "random", key), value, 1.0 / len(randoms))
    add(acc, "cubes", 1)
    return z


def summary(acc):
    """Per level: D, FGR-related fractions, SCIANTIX d, and the chord means of every k (actual, random)."""
    n = float(acc["cubes"])
    N_p = acc["pores"] / acc["volume"]
    r_ws = (3.0 / (4.0 * np.pi * N_p)) ** (1.0 / 3.0)
    rows = []
    for lv in LEVELS:
        D = float(acc[("broken", lv)] / acc["pores"])
        phi_b, s_b = float(acc[("phi_b", lv)] / n), float(acc[("s_b", lv)] / n)
        row = {"level": lv, "D": D, "phi_b": phi_b, "s_b": s_b, "d_sc": 2.0 * r_ws / D if D > 0 else np.inf,
               "exact": 4.0 * (1.0 - phi_b) / s_b if s_b > 0 else np.inf}
        for k in CRACK:
            for kind in ("actual", "random"):
                cnt = acc[(lv, k, kind, "n")]
                row[(k, kind)] = {"l_n": float(cnt / acc[(lv, k, kind, "inv")]), "l_w": float(acc[(lv, k, kind, "sum")] / cnt),
                                  "capped": float(acc[(lv, k, kind, "capped")] / cnt), "L": acc[(lv, k, kind, "L")] / n}
        rows.append(row)
    return rows, r_ws, N_p


def report(root, acc, rows, r_ws, N_p, z, xs, bins, rays, d_min):
    n = float(acc["cubes"])
    phi, s = float(acc["phi"] / n), float(acc["s"] / n)
    S2, same, same_mc = acc["S2"] / n, acc["S2_same_exact"] / n, acc["S2_same"] / n
    far = z >= 10.0
    matrix = rows[-1][(0.0, "actual")] if LEVELS[-1] == 1.0 else None
    lines = [f"# n-point probabilities and intact stretches (python3 npoint.py {root}): {int(n)} cubes, {rays} lines per "
             f"cube, step {STEP:g} um, distMin {d_min:g} um",
             "#", "# 1. Pores, mean of the cubes",
             f"#   S1 = porosity: lines {float(acc['phi_lines'] / n):.5f}, exact {phi:.5f}",
             f"#   S2(z >= 10 um) = {S2[far].mean():.5f}, S1^2 = {phi**2:.5f}",
             f"#   slope at 0: lines {(S2[1] - S2[0]) / STEP:.5f} /um, -s/4 = {-s / 4:.5f} /um (s = {s:.4f} /um)"]
    if matrix is not None:
        lines.append(f"#   matrix chords between pores: number-weighted {matrix['l_n']:.3f} um (4 (1 - S1) / s = "
                     f"{4 * (1 - phi) / s:.3f} um), length-weighted {matrix['l_w']:.3f} um, capped {100 * matrix['capped']:.2f} %")
    lines.append(f"#   SCIANTIX: N_p = {N_p:.5f} /um3, R_WS = {r_ws:.4f} um, d(D = 1) = 2 R_WS = {2 * r_ws:.3f} um, "
                 f"mean intercept 2 d / 3 = {4 * r_ws / 3:.3f} um")
    head = "".join(f"{'P' + str(k) + '_m':>9s}" for k in (1,) + N_POINTS)
    lines.append(f"# {'z_um':>6s}{'S2':>10s}{'same':>10s}{'same_lin':>10s}{'diff':>10s}{head}{'L_m':>9s}")
    L_m = matrix["L"] if matrix is not None else np.full(len(z), np.nan)
    for zz in Z_SHOWN:
        j = min(int(round(zz / STEP)), len(z) - 1)
        lines.append(f"  {z[j]:6.2f}{S2[j]:10.5f}{same[j]:10.5f}{same_mc[j]:10.5f}{S2[j] - same_mc[j]:10.5f}"
                     + "".join(f"{acc[('Pn', k)][j] / n:9.5f}" for k in (1,) + N_POINTS) + f"{L_m[j]:9.5f}")

    lines += ["#", "# 2. Broken pores and intact chords, all the cubes together, at the x = p / sigma_R where the FGR reaches the level",
              "# - D = number fraction of broken pores; phi_b = their volume fraction (S1 of the broken phase); s_b = their "
              "surface per volume",
              "# - d_SC = SCIANTIX 2 R_WS / D, its mean intercept 2 d / 3; exact = 4 (1 - phi_b) / s_b, number-weighted mean "
              "chord at k = 0",
              "# - per k (cracked zone (1 + k) R): l_n = number-weighted, l_w = length-weighted mean intact chord (um), "
              "actual and random within class; capped = % of chords reaching an end of the line (means are lower bounds)",
              f"# {'FGR_%':>5s}{'x':>8s}{'D_%':>7s}{'phi_b_%':>8s}{'s_b':>8s}{'d_SC':>8s}{'2d/3':>8s}{'exact':>8s}{'k':>5s}"
              f"{'l_n':>8s}{'l_n_rnd':>8s}{'l_w':>8s}{'l_w_rnd':>8s}{'capped':>7s}"]
    for row, x in zip(rows, xs):
        for k in CRACK:
            a, r = row[(k, "actual")], row[(k, "random")]
            lines.append(f"  {100 * row['level']:5.3g}{x:8.4f}{100 * row['D']:7.2f}{100 * row['phi_b']:8.3f}{row['s_b']:8.4f}"
                         f"{row['d_sc']:8.2f}{2 * row['d_sc'] / 3:8.2f}{row['exact']:8.2f}{k:5.2g}"
                         f"{a['l_n']:8.2f}{r['l_n']:8.2f}{a['l_w']:8.2f}{r['l_w']:8.2f}{100 * a['capped']:7.1f}")

    mid = np.sqrt(bins[1:] * bins[:-1])
    shown = [lv for lv in LEVELS if lv < 1.0]
    lines += ["#", "# 3. Pairs of pores by gap: P(both broken) / the same with the broken pores taken at random within their class",
              "#    (1 = no grouping; > 1 = the broken pores come in close pairs); pairs = pairs of the bin, all the cubes",
              f"# {'gap_um':>7s}{'pairs':>8s}" + "".join(f"{'FGR' + format(100 * lv, 'g') + '%':>10s}" for lv in shown)]
    for b in range(len(mid)):
        cells = []
        for lv in shown:
            e = acc[("exp", lv)][b]
            cells.append(f"{acc[('obs', lv)][b] / e:10.3f}" if e >= 5.0 else f"{'-':>10s}")
        lines.append(f"  {mid[b]:7.3f}{int(acc['pairs'][b]):8d}" + "".join(cells))
    return lines


def figure(root, acc, rows, r_ws, z, bins, d_min):
    """npoint.png: S2 of the pores; S_n of the matrix up to the lineal path; pairs of broken pores; L(z) of the intact
    phase; number- and length-weighted intact chords against D, with SCIANTIX."""
    n = float(acc["cubes"])
    phi, s = float(acc["phi"] / n), float(acc["s"] / n)
    level_color = {lv: plt.get_cmap("Blues")(0.35 + 0.6 * i / max(len(LEVELS) - 1, 1)) for i, lv in enumerate(LEVELS)}
    crack_color = {k: plt.get_cmap("Oranges")(0.4 + 0.55 * i / max(len(CRACK) - 1, 1)) for i, k in enumerate(CRACK)}
    fig, ax = plt.subplots(2, 3, figsize=(18.0, 10.5))

    a = ax[0, 0]
    S2, same, same_mc = acc["S2"] / n, acc["S2_same_exact"] / n, acc["S2_same"] / n
    m = z <= 8.0
    a.plot(z[m], S2[m], color=pb.SEQ_BLUE[4], lw=2.0, label=r"$S_2(r)$: both ends in a pore")
    a.plot(z[m], same[m], color=pb.SEQ_BLUE[1], lw=1.6, label="same pore (exact, radii only)")
    a.plot(z[m], same_mc[m], color=pb.SEQ_BLUE[1], lw=0, marker="o", ms=3, markevery=6, label="same pore (lines)")
    a.plot(z[m], (S2 - same_mc)[m], color=pb.D52_COLOR, lw=1.6, label="different pores (lines: arrangement)")
    t = z <= 4.0 * phi / s * 0.5
    a.plot(z[t], phi - s * z[t] / 4.0, color=pb.REFERENCE_LINE, ls=":", lw=1.2, label=r"slope $-s/4$")
    for v, text in ((phi, r"$S_1$ = porosity"), (phi**2, r"$S_1^2$")):
        a.axhline(v, color=pb.REFERENCE_LINE, ls="--", lw=0.9)
        a.annotate(text, (1.0, v), xycoords=("axes fraction", "data"), ha="right", va="bottom", fontsize=8,
                   color=pb.REFERENCE_LINE, xytext=(-3, 2), textcoords="offset points")
    a.set(xlabel="distance r between the two points (um)", ylabel="probability", xlim=(0, 8), ylim=(0, None),
          title="Two-point probability of the pores")
    a.legend(frameon=False, fontsize=8, loc="center right")

    a = ax[0, 1]
    ns = (1,) + N_POINTS
    for i, k in enumerate(ns):
        a.plot(z, acc[("Pn", k)] / n, color=plt.get_cmap("Blues")(0.3 + 0.6 * i / max(len(ns) - 1, 1)), lw=1.6,
               label=f"n = {k}")
    if LEVELS[-1] == 1.0:
        a.plot(z, rows[-1][(0.0, "actual")]["L"], color="black", lw=2.2, label=r"n $\to\infty$: lineal path L(z)")
    a.set(ylim=(0, 1), xlim=(0, z[-1]), xlabel="segment length z (um)",
          ylabel="probability that the n points are all in the matrix",
          title="n points on a segment in the matrix: n → ∞ is the intact segment")
    a.legend(frameon=False, fontsize=8, loc="lower left")

    a = ax[0, 2]
    mid = np.sqrt(bins[1:] * bins[:-1])
    for lv in LEVELS:
        if lv >= 1.0:
            continue
        e = acc[("exp", lv)]
        ok = e >= 5.0
        a.plot(mid[ok], acc[("obs", lv)][ok] / e[ok], color=level_color[lv], lw=1.6, marker="o", ms=4,
               label=f"FGR {100 * lv:g} %")
    a.axhline(1.0, color=pb.REFERENCE_LINE, lw=0.9)
    for v, text in ((d_min, "distMin"), (bo.CLOSE * d_min, f"{bo.CLOSE:g} distMin")):
        a.axvline(v, color=pb.REFERENCE_LINE, ls="--", lw=0.9)
        a.annotate(text, (v, 0.98), xycoords=("data", "axes fraction"), ha="left", va="top", fontsize=8,
                   color=pb.REFERENCE_LINE, xytext=(3, 0), textcoords="offset points")
    a.set(xscale="log", xlabel="gap between the two pores, wall to wall (um)",
          ylabel="P(both broken) / same at random within the class",
          title="Two-point probability of the broken pores (pairs)")
    ticks = [t for t in (0.3, 0.5, 1.0, 2.0, 4.0, 6.0) if bins[0] * 0.95 <= t <= bins[-1]]
    a.set_xticks(ticks, [f"{t:g}" for t in ticks])
    a.minorticks_off()
    a.legend(frameon=False, fontsize=8, loc="upper right")

    a = ax[1, 0]
    for row in rows:
        lv = row["level"]
        a.plot(z, row[(0.0, "actual")]["L"], color=level_color[lv], lw=1.6, label=f"FGR {100 * lv:g} %")
        if lv < 1.0:
            a.plot(z, row[(0.0, "random")]["L"], color=level_color[lv], lw=1.0, ls="--")
    a.plot([], [], color=pb.REFERENCE_LINE, ls="--", lw=1.0, label="broken pores at random in the class")
    a.set(ylim=(0, 1), xlim=(0, z[-1]), xlabel="segment length z (um)",
          ylabel="L(z) = P(segment intact); 1 - L = P(segment broken)",
          title="Lineal path of the intact phase (k = 0: cut by broken pores)")
    a.legend(frameon=False, fontsize=8, loc="lower left")

    D = np.array([row["D"] for row in rows])
    line = 2.0 * len(z) * STEP
    Dg = np.geomspace(D.min(), 1.0, 100)
    for a, key, title in ((ax[1, 1], "l_n", "Number-weighted mean intact chord (line intercept)"),
                          (ax[1, 2], "l_w", "Length-weighted mean intact chord (seen by a random point)")):
        for k in CRACK:
            for kind, ls, lw in (("actual", "-", 1.6), ("random", "--", 1.0)):
                y = np.array([row[(k, kind)][key] for row in rows])
                capped = np.array([row[(k, kind)]["capped"] for row in rows]) > CAPPED
                a.plot(100 * D, y, color=crack_color[k], ls=ls, lw=lw,
                       label=f"k = {k:g}: zone (1 + k) R" if kind == "actual" else None)
                if kind == "actual":
                    a.scatter(100 * D[~capped], y[~capped], color=crack_color[k], s=22, zorder=3)
                    a.scatter(100 * D[capped], y[capped], facecolor="white", edgecolor=crack_color[k], s=22, zorder=3)
        if key == "l_n":
            a.plot(100 * Dg, 4.0 * r_ws / (3.0 * Dg), color="black", lw=1.6, label=r"SCIANTIX: $2d/3$ (intercept of its cubes)")
            a.plot(100 * Dg, 2.0 * r_ws / Dg, color="black", lw=1.0, ls="--", label=r"SCIANTIX: $d = 2R_{WS}/D$")
            a.scatter(100 * D, [row["exact"] for row in rows], marker="x", color=pb.REFERENCE_LINE, s=30, zorder=4,
                      label=r"$4(1-\phi_b)/s_b$ (k = 0, exact)")
        a.axhline(line, color=pb.REFERENCE_LINE, ls=":", lw=1.0)
        a.annotate(f"line length 2H = {line:g} um", (1.0, line), xycoords=("axes fraction", "data"), ha="right",
                   va="bottom", fontsize=8, color=pb.REFERENCE_LINE, xytext=(-3, 2), textcoords="offset points")
        a.plot([], [], color=pb.REFERENCE_LINE, ls="--", lw=1.0, label="broken pores at random in the class")
        a.scatter([], [], facecolor="white", edgecolor=pb.REFERENCE_LINE, s=22,
                  label=f"> {100 * CAPPED:g} % chords reach the line end")
        a.set(xscale="log", yscale="log", xlabel="D = broken pores (% of the number)", ylabel="mean intact chord (um)",
              title=title)
        a.legend(frameon=False, fontsize=7.5, loc="lower left")

    for a in ax.ravel():
        a.grid(True, color=pb.GRID)
    fig.tight_layout()
    path = os.path.join(root, "npoint.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else pb.ROOT
    rays = int(float(sys.argv[2])) if len(sys.argv) > 2 else RAYS
    cubes = bo.read_run(root)
    if len({c["side"] for c in cubes}) != 1:
        raise SystemExit(f"{root}: cubes of different sides")
    radii = np.unique(bo.pooled(cubes, "R"))
    xs = levels_x(cubes)
    d_min = float(np.nanmedian([c["d_min"] for c in cubes]))
    bins = np.geomspace(0.999 * d_min, GAP_MAX, GAP_BINS + 1)
    acc = {}
    for c in cubes:
        print(f"{c['folder']}: {len(c['R'])} pores", flush=True)
        z = analyse_cube(c, radii, xs, bins, rays, acc)
    rows, r_ws, N_p = summary(acc)
    lines = report(root, acc, rows, r_ws, N_p, z, xs, bins, rays, d_min)
    with open(os.path.join(root, "npoint.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote npoint.txt and {figure(root, acc, rows, r_ws, z, bins, d_min)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
