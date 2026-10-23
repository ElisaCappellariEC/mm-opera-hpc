#!/usr/bin/env pvpython
"""

cd ~/mmm/build/polydispersebubble
pvpython --force-offscreen-rendering plot_max_stress_slice.py     # slices with normal x, y, z through the max
pvpython --force-offscreen-rendering plot_max_stress_slice.py --normal x --zoom 4   # close-up
paraview --state=run_unit_NFIR/figures/max_stress_slice.pvsm      # open the same scene in the GUI

Slice the stress intensification field through the point where it is maximum.

The field is FirstEigenStress (first principal stress sigma_I) written by
test-polybubble in <run>/Eigenstress-pp. With --pref 1 and bubble pressure 1,
it is directly the stress intensification factor sigma_I / p.

The script:
  1. finds the point with the maximum value (see "Which maximum" below),
  2. cuts the domain with planes (normal x, y, z) through that point,
  3. saves one PNG per plane + a 3D overview PNG,
  4. saves a ParaView state file (.pvsm) to open the same scene in the GUI.

Which maximum:
  default        max of the exported nodal field (what ParaView shows; it is an
                 L2 projection, so peaks are smoothed)
  --from-table   max sigma_I_max of bubbles_and_stresses_selected.txt (raw
                 quadrature values, usually a bit higher, may be another bubble)
  --point X Y Z  any point you choose

Usage (from build/polydispersebubble):
    pvpython plot_max_stress_slice.py
    pvpython plot_max_stress_slice.py --normal x --zoom 4
    pvpython plot_max_stress_slice.py --from-table run_unit_NFIR/bubbles_and_stresses_selected.txt
    pvpython plot_max_stress_slice.py --exclude-outer 0.3   # ignore max on outer faces
    paraview --state=run_unit_NFIR/figures/max_stress_slice.pvsm
"""
import csv
import argparse
import os

import numpy as np
from paraview import servermanager
from paraview.simple import *  # noqa: F401,F403
from vtk.numpy_interface import dataset_adapter as dsa

HERE = os.path.dirname(os.path.abspath(__file__))
AXES = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}
VIEW_UP = {"x": [0.0, 0.0, 1.0], "y": [0.0, 0.0, 1.0], "z": [0.0, 1.0, 0.0]}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pvd", default=os.path.join(
        HERE, "run_unit_NFIR", "Eigenstress-pp", "Eigenstress-pp.pvd"))
    p.add_argument("--field", default="FirstEigenStress")
    p.add_argument("--normal", default="all", choices=["x", "y", "z", "all"])
    p.add_argument("--out-dir", default=None,
                   help="default: <run>/figures next to the .pvd folder")
    p.add_argument("--range", nargs=2, type=float, default=None,
                   metavar=("MIN", "MAX"), help="color range (default: data range)")
    p.add_argument("--exclude-outer", type=float, default=0.0, metavar="D",
                   help="ignore points closer than D to the bounding box faces "
                        "when looking for the maximum")
    where = p.add_mutually_exclusive_group()
    where.add_argument("--from-table", metavar="CSV",
                       help="slice at the max sigma_I_max of this bubble table")
    where.add_argument("--point", nargs=3, type=float, metavar=("X", "Y", "Z"),
                       help="slice at this point instead of the field maximum")
    p.add_argument("--zoom", type=float, default=1.0,
                   help="zoom factor of the 2D images, centred on the point "
                        "(1 = whole domain)")
    p.add_argument("--resolution", nargs=2, type=int, default=[1600, 1300])
    return p.parse_args()


def max_from_table(path):
    rows = [r for r in csv.DictReader(open(path)) if r["sigma_I_max"] != "nan"]
    best = max(rows, key=lambda r: float(r["sigma_I_max"]))
    print(f"table max: bubble {best['bubble_id']} (R={best['radius']}), "
          f"sigma_I_max = {float(best['sigma_I_max']):.6g}")
    return np.array([float(best[c]) for c in "xyz"]), float(best["sigma_I_max"])


def find_max(source, field, exclude_outer):
    data = dsa.WrapDataObject(servermanager.Fetch(MergeBlocks(Input=source)))
    pts = np.asarray(data.Points)
    val = np.asarray(data.PointData[field])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    mask = np.ones(len(val), dtype=bool)
    if exclude_outer > 0:
        mask = np.all(pts - lo >= exclude_outer, axis=1) & \
               np.all(hi - pts >= exclude_outer, axis=1)
    idx = np.flatnonzero(mask)[np.argmax(val[mask])]
    return pts[idx], val[idx], (val.min(), val.max()), lo, hi


def white_background(view):
    try:  # ParaView >= 5.10 uses color palettes by default
        view.UseColorPaletteForBackground = 0
    except AttributeError:
        pass
    view.Background = [1.0, 1.0, 1.0]
    view.OrientationAxesVisibility = 1
    view.OrientationAxesLabelColor = [0.0, 0.0, 0.0]


def add_text(view, text, location):
    src = Text(Text=text)
    disp = Show(src, view)
    disp.WindowLocation = location
    disp.FontSize = 18
    disp.Color = [0.0, 0.0, 0.0]
    return src


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(args.pvd))), "figures")
    os.makedirs(out_dir, exist_ok=True)

    reader = PVDReader(FileName=args.pvd)
    reader.UpdatePipeline()

    pmax, vmax, (vmin_all, vmax_all), lo, hi = find_max(
        reader, args.field, args.exclude_outer)
    print(f"field max {args.field} = {vmax:.6g} at "
          f"x={pmax[0]:.6g}, y={pmax[1]:.6g}, z={pmax[2]:.6g}")
    label = "max sigma_I / p (nodal field)"
    if args.from_table:
        pmax, vmax = max_from_table(args.from_table)
        label = "max sigma_I / p (bubble table)"
    elif args.point:
        pmax, vmax = np.array(args.point), None
        label = "slice point"
    print(f"slicing at x={pmax[0]:.6g}, y={pmax[1]:.6g}, z={pmax[2]:.6g}")
    print(f"field range = [{vmin_all:.6g}, {vmax_all:.6g}]")
    print(f"domain = [{lo[0]:.4g}, {hi[0]:.4g}] x [{lo[1]:.4g}, {hi[1]:.4g}] "
          f"x [{lo[2]:.4g}, {hi[2]:.4g}]")

    crange = args.range or [vmin_all, vmax_all]
    diag = float(np.linalg.norm(hi - lo))
    center = 0.5 * (lo + hi)

    view = GetActiveViewOrCreate("RenderView")
    view.ViewSize = args.resolution
    white_background(view)

    outline_disp = Show(Outline(Input=reader), view)
    outline_disp.AmbientColor = outline_disp.DiffuseColor = [0.0, 0.0, 0.0]

    marker = Sphere(Center=list(pmax), Radius=0.004 * diag,
                    ThetaResolution=32, PhiResolution=32)
    marker_disp = Show(marker, view)
    marker_disp.AmbientColor = marker_disp.DiffuseColor = [1.0, 0.0, 1.0]

    lut = GetColorTransferFunction(args.field)
    if not lut.ApplyPreset("Turbo", True):
        lut.ApplyPreset("Viridis (matplotlib)", True)
    lut.AutomaticRescaleRangeMode = "Never"
    lut.RescaleTransferFunction(*crange)

    value = f" = {vmax:.4f}" if vmax is not None else ""
    add_text(view, f"{label}{value}\n"
                   f"at ({pmax[0]:.3f}, {pmax[1]:.3f}, {pmax[2]:.3f})",
             "Upper Left Corner")

    normals = ["x", "y", "z"] if args.normal == "all" else [args.normal]
    slices = {}
    for name in normals:
        sl = Slice(Input=reader)
        sl.SliceType = "Plane"
        sl.SliceType.Origin = list(pmax)
        sl.SliceType.Normal = AXES[name]
        RenameSource(f"Slice_{name}_through_max", sl)
        disp = Show(sl, view)
        ColorBy(disp, ("POINTS", args.field))
        disp.SetScalarBarVisibility(view, True)
        slices[name] = (sl, disp)

    bar = GetScalarBar(lut, view)
    bar.Title = "sigma_I / p"
    bar.ComponentTitle = ""
    bar.TitleColor = bar.LabelColor = [0.0, 0.0, 0.0]
    bar.TitleFontSize = bar.LabelFontSize = 18
    bar.WindowLocation = "Any Location"
    bar.Position = [0.88, 0.1]
    bar.ScalarBarLength = 0.8

    # One 2D image per plane, looking along its normal.
    view.CameraParallelProjection = 1
    marker_radius = marker.Radius
    marker.Radius = marker_radius / max(args.zoom, 1.0)
    for name in normals:
        for other, (_, d) in slices.items():
            d.Visibility = int(other == name)
        n = np.array(AXES[name])
        focal = center - np.dot(center - pmax, n) * n
        view.CameraFocalPoint = list(focal)
        view.CameraPosition = list(focal + 2.0 * diag * n)
        view.CameraViewUp = VIEW_UP[name]
        view.ResetCamera(False)
        suffix = ""
        if args.zoom > 1.0:
            view.CameraFocalPoint = list(pmax)
            view.CameraPosition = list(pmax + 2.0 * diag * n)
            view.CameraParallelScale /= args.zoom
            suffix = f"_zoom{args.zoom:g}"
        png = os.path.join(out_dir, f"max_stress_slice_{name}{suffix}.png")
        SaveScreenshot(png, view, ImageResolution=args.resolution)
        print(f"saved {png}")

    # 3D overview with all planes, then the state file for the GUI.
    marker.Radius = marker_radius
    for _, d in slices.values():
        d.Visibility = 1
    view.CameraParallelProjection = 0
    view.CameraFocalPoint = list(center)
    view.CameraPosition = list(center + diag * np.array([1.2, -1.6, 1.0]))
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.ResetCamera(False)
    png = os.path.join(out_dir, "max_stress_slice_3d.png")
    SaveScreenshot(png, view, ImageResolution=args.resolution)
    print(f"saved {png}")

    state = os.path.join(out_dir, "max_stress_slice.pvsm")
    SaveState(state)
    print(f"saved {state}\nopen it with: paraview --state={state}")


if __name__ == "__main__":
    main()
