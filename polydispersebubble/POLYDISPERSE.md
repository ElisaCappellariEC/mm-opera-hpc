# Polydisperse bubbles with one pressure per bubble

Copy of `bubble/` for populations of bubbles of different radii, with one pressure per bubble (6th column of the bubble file).
---

## 1. What changes with respect to `bubble/`

| What | `bubble/` | `polydispersebubble/` |
|---|---|---|
| Bubble file | 5 columns `id x y z r` | 5 **or 6** columns `id x y z r [p]` (p = 1 if missing) |
| Pressure | `-pr`, the same for all | bubble i: `pref × p_i` (`-pr` = multiplier, default 1) |
| Search radius of σ_I max | fixed, `dmin` = 0.95 µm | with `-dl δ > 0`: `r_i + δ` |
| Bubbles with a value | only those near the points with σ_I > 0.9·max | with `-sf 0`: **all** |
| Search in MPI | all the points copied to every process | local points, then the maximum per bubble on rank 0 |
| CSV and `sig_hydro.txt` | only with `-p 1` | **always**; `-p` controls only ParaView |
| CSV columns | `id, x, y, z, stress` | `bubble_id, radius, pressure, sigma_I_max, x, y, z` (`nan` if no point) |
| Linear solver | PCG + `HypreDiagScale` | PCG + `-pc` (default `HypreBoomerAMG`), `-li` max iterations (default 5000) |

```bash
mpirun -n <N> ./test-polybubble -m mesh.msh -f bubbles.txt -dl 0.2 -sf 0 -p 0 -n my_case
```

With `-dl 0 -sf 0.9` (the defaults) the computation is the original one; only the format of the CSV changes.
Line-by-line differences: `diff -ru bubble polydispersebubble`.

---

## 2. How `test-polybubble` works

Stack: MFEM (finite elements, MPI) + hypre (solvers) + MFront/MGIS (law `Elasticity.mfront` → `src/libBehaviour.so`)
+ mfem-mgis (problems, boundary conditions, post-processing).

**Mesh tags.** `Physical Volume(1)` = matrix; `Physical Surface(k)` = surface of bubble k, and k must be
column 1 of the bubble file (see `mesh/single_sphere.geo` + `mesh/single_bubble.txt`). Mérope meshes: volume 1,
bubbles 2…N+1, periodic envelope N+2 (no condition imposed).

**`main` in `src/Bubble.cxx`, in order:**

1. options (`fill_parameters`) and bubble file (`src/BubbleDescription.cxx`);
2. mesh: `-pm 0` the whole Gmsh mesh read by every process, `-pm 1` one MFEM file per process (§5); H1 order 2, `-r k` uniform refinements;
3. `PeriodicNonLinearEvolutionProblem` with **zero macroscopic gradient**: the RVE does not expand, so the matrix is
   in compression on average (`sig_hydro.txt`);
4. one `UniformImposedPressureBoundaryCondition` per bubble, value `pref × p_i`;
5. elastic law on material 1 (E = 150e-3 N/µm², ν = 0.3), PCG, a single iteration (linear problem);
6. σ_I at every Gauss point (`src/Utilities.cxx`); every process keeps its points above the threshold (all of them with `-sf 0`);
7. for every bubble: points sorted by x, binary search of the slice |x − x_c| < r_s, filter on y, z and distance, maximum;
8. `MPI_Gather` on rank 0 of the maximum per bubble of every process (every Gauss point belongs to one process only → exact maximum);
9. writes `bubbles_and_stresses_selected.txt` and `sig_hydro.txt`; with `-p 1` also `<case name>/` and `Eigenstress-pp/` for ParaView.

**Units.** Lengths in µm. With pressure loads only and a homogeneous material, σ does not depend on E and is linear in p:
pressures in MPa → `sigma_I_max` in MPa. With p = 1, `sigma_I_max` = σ_I/p.


---

## 3. Scripts (`scripts/`)

CMake copies them into `~/mmm/build/polydispersebubble/` and copies them again at every build when they change.

| Script | What it does |
|---|---|
| `run_mmm.sh` | from `build/polydispersebubble`: copies the cubes from `~/mmm/Merope/mmm_files/<folder>` and runs `test-polybubble` (`-p 1 -sf 0 -dl 0.5`, `-r k` for the `_refinement<k>` folders) on those not solved yet |
| `plot_bubbles.py` | analysis of all the cubes: mesh convergence, size of the RVE (Kanit), FGR across the seeds (`mesh_test`, `rve_size`, `rve_fgr` .txt/.png); exits with 1 if the cube is not an RVE or the mesh is not fine enough |

The Mérope scripts (`test_rve.py`, `merope_pore_pressure.py`, `run_gmsh.sh`, `inverted_elements.py`) are in
`~/mmm/Merope/mmm_files`, not in this repository.
---

## 4. Build and check (EC, not possible on other devices)

It needs **mfem-mgis@master** (with 1.0.3 the repository does not build: `CatchTimeSection(ctx, …)`, `getLocation`,
`isInvalid`, `MGIS/Function`).

```bash
source ~/mmm/env.sh                               # or the environment of the cluster
cmake -S mm-opera-hpc -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target lib-elasticity-poly  # without -j: MFront fails with "invalid --jobserver-auth"
cmake --build build -j 8
ctest --test-dir build -R PolyBubbleTest --output-on-failure
```

Regression, from `build/polydispersebubble`:

```bash
mpirun -n 4 ./test-polybubble -m mesh/mesh_sphere.msh -f mesh/single_bubble_ci.txt -p 0 && cp bubbles_and_stresses_selected.txt ref.csv
printf '2 0 0 0 0.400 1.0\n' > p1.txt   # 6 columns, p = 1 → identical to ref.csv
printf '2 0 0 0 0.400 2.0\n' > p2.txt   # p = 2 → sigma_I_max and sig_hydro exactly doubled
mpirun -n 4 ./test-polybubble -m mesh/mesh_sphere.msh -f p1.txt -p 0 && diff ref.csv bubbles_and_stresses_selected.txt
mpirun -n 4 ./test-polybubble -m mesh/mesh_sphere.msh -f p2.txt -p 0 && cat bubbles_and_stresses_selected.txt
mpirun -n 6 ./test-polybubble && python3 mmm_vs_analytical.py   # "Check PASS." (~20 s)
# sigma_I_max = 0.507159400619, sig_hydro Average = -0.00043534
```

With `-dl 0 -sf 0.9` the values match `bubble/` to the last digit (exactly with `-pc HypreDiagScale`).
