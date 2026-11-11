# Polydisperse bubbles with one pressure per bubble: the model

`polydispersebubble/` is a copy of `bubble/` for populations of pores of different radii, each with its own pressure.
It builds `test-polybubble` (the solver) and `partition-mesh` (the mesh splitter for clusters).

E. Cappellari (POLIMI-CEA), 2026; original `bubble/` by T. Helfer et al. State of 10/11/2026.

---

## 1. Problem solved

A periodic cube of side $L$ of a homogeneous, isotropic, linear elastic matrix with $N$ spherical pores. The pores are
not meshed: each one is a boundary of the matrix, loaded by a uniform pressure.

$$\nabla\cdot\boldsymbol\sigma=0,\qquad \boldsymbol\sigma=\lambda\,\mathrm{tr}(\boldsymbol\varepsilon)\,\mathbf I+2\mu\,\boldsymbol\varepsilon,\qquad
\boldsymbol\varepsilon=\bar{\boldsymbol\varepsilon}+\nabla^s\tilde{\mathbf u}\quad\text{in the matrix}$$

$$\boldsymbol\sigma\,\mathbf n=-p_i\,\mathbf n\ \text{on the wall of pore } i\ (\mathbf n\ \text{out of the matrix}),\qquad
\tilde{\mathbf u}\ \text{periodic},\qquad \bar{\boldsymbol\varepsilon}=\mathbf 0$$

- **Zero macroscopic strain** ($\bar{\boldsymbol\varepsilon}=0$, `setMacroscopicGradientsEvolution` with six zeros):
  the cube cannot expand, so the pressurised pores put the matrix in compression on average. The mean hydrostatic
  stress of the matrix, $\langle\sigma_h\rangle<0$, is an output (`sig_hydro.txt`) and is what brings the result to
  zero macroscopic *stress* afterwards (section 5).
- **One load per pore**: $p_i=$ `pref` × (6th column of the bubble file); `pref` is the option `-pr` (default 1) and
  the 6th column is 1 when missing. In `bubble/` one pressure is given to all the pores.
- **Material**: `Elasticity.mfront` (MFront, generic interface → `src/libBehaviour.so`), Young modulus
  $E$ = 150 GPa and Poisson ratio ν = 0.3 (UO₂), set in `src/Bubble.cxx`.
- **Units**: lengths in µm; $E$ is written 150e-3 N/µm² so that displacements are consistent. With pressure loads
  only and one material, **σ does not depend on $E$** and is linear in the pressures: pressures in MPa give stresses
  in MPa, and with $p=1$ everywhere the output is σ_I/p. ν does matter.
- The problem is linear: one Newton iteration, one linear solve. `broken` exists in the code (a broken pore would
  get zero pressure) but is never set: there is no cascade.

**Quantity extracted.** For each pore $i$, the largest first principal stress of the matrix within a sphere of
radius $R_i+\delta$ around its centre:

$$\sigma_{I,i}^{max}=\max_{\mathbf x_g:\ |\mathbf x_g-\mathbf c_i|<R_i+\delta}\ \sigma_I(\mathbf x_g),\qquad \mathbf x_g\ \text{the Gauss points}$$

A pore is taken as broken when this stress exceeds a rupture stress $\sigma_R$ (criterion of OperaHPC D5.2); the
solver only returns $\sigma_{I,i}^{max}$, the criterion is applied by the analysis scripts and by SCIANTIX.

---

## 2. Discretisation

Stack: MFEM (finite elements, MPI) + hypre (linear solvers) + MFront/MGIS (behaviour law) + mfem-mgis (periodic
problem, boundary conditions, post-processing). It needs **mfem-mgis@master** (1.0.3 does not build this folder).

**Mesh.** Tetrahedra from Gmsh, format MSH 2.2 with its `$Periodic` section. Tags:

| entity | tag | meaning |
|---|---|---|
| `Physical Volume(1)` | 1 | the matrix (material 1) |
| `Physical Surface(k)` | k | the wall of pore k; **k is column 1 of the bubble file** |
| Mérope meshes | 2 … N+1 pores, N+2 the periodic envelope | no condition is put on the envelope: periodicity comes from the mesh |

**Bubble file** (`-f`): one line per pore, `id x y z r [p]` (5 or 6 columns), `#` comments allowed
(`src/BubbleDescription.cxx`). Examples: `mesh/single_bubble.txt`, `mesh/single_bubble_ci.txt`.

**Elements.** Displacement in H1, order 2 (`-o`, default 2), 3 unknowns per node, on geometry of order 2 (elements
curved on the spheres when the mesh is of order 2, as Mérope + gmsh write it). Stresses live at the Gauss points of
the integration rule of mfem-mgis for the behaviour integrator. `-r k` refines every element uniformly $k$ times in
MFEM (each level ×8 elements), only with the whole mesh (section 6).

---

## 3. Linear solver and preconditioner

| | choice | where |
|---|---|---|
| nonlinear loop | 1 iteration (`MaximumNumberOfIterations` 1), relative tolerance 1e-8 | `setSolverParameters` |
| linear solver | `HyprePCG` (conjugate gradient: the matrix is symmetric positive definite on the periodic space), relative tolerance **1e-10**, at most `-li` iterations (default 5000; hypre's default is 1000) | `setLinearSolver` |
| preconditioner | `-pc`, default **`HypreBoomerAMG`** (algebraic multigrid); `HypreDiagScale` is the one of `bubble/` | `linear_solver_preconditioner` |

**Why BoomerAMG.** With diagonal scaling the number of CG iterations grows with the number of unknowns and with the
contrast of element sizes (graded meshes): 1355 iterations on the default single-bubble mesh. With BoomerAMG the
count is nearly independent of the size:

| case | unknowns | PCG iterations |
|---|---|---|
| side-10 cubes, 0.3 µm (66 pores) | ≈ 1 M | 18–29 |
| side-30 cube, graded 0.3 µm (1867 pores) | 8.8 M | 31 |
| side-30 cube, graded 0.212 µm | 22.7 M | 51 |
| side-30 cubes, graded 0.15 µm, 10 seeds | 56.9–57.1 M | 27–54 (eight cubes 27–36; one 54) |

Final relative residual of the PCG ≈ 9e-11, Newton residual after the solve ≈ 3.5e-10. In mfem-mgis master BoomerAMG
on a vector field is set up in its "systems" variant (the three displacement components are treated as a system),
which is what makes it effective for elasticity. A direct solver (MUMPS) does not fit in memory at these sizes.
Time of the linear solve on 512 processes for 57 M unknowns: ≈ 40–50 s.

"the non linear solver did not converge" = the PCG did not reach 1e-10 within `-li` iterations. Look at the mesh
first; then `-v 1` (prints `PCG Iterations`) and a larger `-li`.

The L2 projection of σ_I to the nodes for ParaView (`-p 1` only) uses a separate `HyprePCG` + `HypreILU`.

---

## 4. Post-processing: from the stress field to one value per pore

`main` of `src/Bubble.cxx`, after the solve:

1. **σ_I at every Gauss point** of material 1: eigenvalues of the stress tensor (`tfel::math`), largest one
   (`src/Utilities.cxx`). The global maximum and its place are found first.
2. **Points kept**: with `-sf f > 0` only the points with σ_I > f × (global maximum) — the original behaviour,
   f = 0.9, which gives a value only to the pores near the most loaded points; with **`-sf 0` all the points**, so that
   every pore gets a value. Each MPI process keeps **its own** points (`getLocalPointsAndStressAboveStressThreshold`);
   the original code copied all the points to every process, which is impossible with 10⁸ points.
3. **Search per pore**: the points of the process are sorted by x; for pore $i$ a binary search gives the slice
   $|x-x_c|<r_s$, then a filter on y, z and on the distance, and the maximum is kept with its position. Search radius
   $r_s=R_i+\delta$ with **`-dl δ > 0`**; with `-dl 0` (default) the fixed radius `-dm` (0.95 µm) of `bubble/`, which
   only makes sense for pores of one size. Distances are **not periodic**: a pore cut by a face of the cube sees only
   the points on the side of its centre, not the part of its wall that re-enters from the opposite face (12 % of
   the pores of a 30 µm cube; no significant effect on the class means, −3.6 % ± 6 % on the largest class).
4. **Reduction**: every process sends (σ, x, y, z) of each pore to rank 0 (`MPI_Gather`), which keeps the largest.
   Every Gauss point belongs to one process only, so the result is the exact maximum over the search sphere.
5. **Hydrostatic stress**: tr(σ)/3 at the Gauss points, integrated over the matrix with the weights of the
   behaviour integrator (`MPI_Reduce`).
6. A pore with no point in its sphere is written as `nan` and a warning is printed (it happens with `-sf > 0`).

**Choice of δ.** The shell $R<r<R+\delta$ must contain the points next to the pore's own wall and nothing that
belongs to a neighbour. The true maxima sit at the first Gauss points from the wall (median 0.006 µm, 99th percentile
0.013 µm on the 0.15 µm meshes). If δ reaches the wall of another pore, the larger pore of a close pair takes the
stress of the smaller one: with δ equal to the minimum gap between pores (0.3 µm) this happened to 5 % of the pores
and raised the mean of the largest class by 20 %. Rule: **δ ≤ (minimum gap)/2**; the runs use `-dl 0.15`.

**Outputs** (always written, in the current folder):

| file | content |
|---|---|
| `bubbles_and_stresses_selected.txt` | CSV, one line per pore in the order of the bubble file: `bubble_id, radius, pressure, sigma_I_max, x, y, z` (pressure as read, before `pref`; position of the maximum; 12 digits) |
| `sig_hydro.txt` | `Volume, Integral, Average`: volume of the matrix, integral and mean of the hydrostatic stress |
| `<case name>/`, `Eigenstress-pp/` | only with `-p 1`: stress at the nodes and projected σ_I for ParaView (about 5 GB for a cube of 18 M Gmsh nodes; +5 % memory) |

---

## 5. From σ_I to the stress factor ŝ

Done by `scripts/plot_bubbles.py` (and used by `break_order.py`), not by the solver.

With one pressure $p$ in all the pores, per unit pressure:

$$\sigma_i=\frac{\sigma_{I,i}^{max}}{p},\qquad h=\xi-(1-\xi)\frac{\langle\sigma_h\rangle}{p},\qquad
\hat s_i=\frac{\sigma_i+h}{1-h}$$

ξ is the porosity of the cube. $-h\,p$ is the macroscopic stress of the constrained cube (pores at $-p$, matrix at
$\langle\sigma_h\rangle$). Adding a uniform tension $h\,p$ to the whole cube brings it to zero macroscopic stress;
this is the same as lowering the pore pressure to $p(1-h)$ and adding $h\,p$ to every stress, hence the formula (the
rescaling of OperaHPC D5.2). Example: `side30_seed01` of the report, ξ = 0.1131, $\langle\sigma_h\rangle$ = −0.1576,
$h$ = 0.2529.

With a pressure per class (`pk` cubes of `set_pressures.py`), $p_{ref}=\sum V_ip_i/\sum V_i$ and the correction needs
the field of the *unit* run of the same cube, $\hat s_{1,i}$:
$\hat s_i=\sigma_i+h\,p_{ref}(1+\hat s_{1,i})/p_i$. It is exact for equal pressures; otherwise it adds two maxima
that may not be at the same point (the script prints their distance). Not yet used on a solved cube.

A pore breaks when $\hat s_i\,p_i>\sigma_R$, i.e. at the load $x=p_{ref}/\sigma_R=p_{ref}/(\hat s_ip_i)$; the released
fraction is the volume of the broken pores over the volume of all the pores.

---

## 6. Reading the mesh: whole (`-pm 0`) or split (`-pm 1`)

- **`-pm 0`** (mfem-mgis `MeshReadMode = FromScratch`): every MPI process reads the **whole** Gmsh file, the mesh is
  partitioned with METIS, each process keeps its part. Memory of this phase ≈ 0.37 kB per Gmsh node *per process*:
  fine on a workstation, impossible with 128 processes per node on a cluster.
- **`-pm 1`** (`MeshReadMode = Restart`): each process reads its own file `<prefix>` + rank on 6 digits
  (`mesh_par.000000`, …) in the MFEM parallel mesh format. The number of files must be the number of processes.

**`partition-mesh`** (`tools/PartitionMesh.cxx`) writes those files:

```bash
./partition-mesh -m mesh.msh -np 512 -o mesh_par512/mesh_par.      # the folder must exist; ONE process, no mpirun
mpirun -n 512 ./test-polybubble -pm 1 -m mesh_par512/mesh_par. -f bubbles.txt -sf 0 -dl 0.15 -p 0
```

It reads the mesh once with the same arguments as mfem-mgis (`mfem::Mesh(file, 0, 1, true)`), partitions it with
`mfem::MeshPartitioner` (METIS, default method, the same partition `mfem::ParMesh` would make), and writes the parts
one by one; `ExtractPart` leaves out the nodal grid function, which carries the order-2 geometry and the
periodicity, so it is extracted and attached to every part. It runs on one process because it needs the whole mesh
in memory once (14–15 GB for 2.47 M vertices / 13.4 M tetrahedra, 11 minutes, 512 files), instead of once per
process. Meshes split by Gmsh itself (`gmsh -part N`) are another format and cannot be read by `-pm 1`.

Verified: `-pm 1` gives the same `sigma_I_max` as `-pm 0` (relative difference 3e-10, the solver tolerance) and the
same `sig_hydro.txt` on the CI mesh and on side-10 cubes; `-p 1` with `-pm 1` gives the same two text files as `-p 0`.

**Known defect: `-r k` with `-pm 1` is wrong.** On a split mesh refined once in MFEM, maxima appear inside the
matrix, 0.15–0.3 µm from the walls, up to +77 % (pores 9, 10, 20 of `side10_seed01_meshdouble_refinement`: 0.643,
1.322, 0.714), while `-pm 0 -r 1` gives the values of the Gmsh mesh of the same size (0.464, 0.747, 0.502 against
0.458, 0.770, 0.499). The other pores differ by 1e-4–1e-3, so the two refined meshes are not the same; the cause is
not found. Use `-r` only with `-pm 0`.

**Memory of a run** (`-sf 0 -p 0`): about 21 kB per Gmsh node in total, independent of the number of processes
(side-10 cube, 336 k nodes: 7 GB). Side-30 cube with the graded 0.15 µm mesh: 532 GB over 512 processes, 1.02 GB per
process at most. With 40 % of margin on nodes of 1867 MB per core, one core per process:
NP ≥ (Gmsh nodes × 21 kB)/(0.6 × 1867 MB) — about 14 M nodes with 256 processes, 27 M with 512, 55 M with 1024.
The number of Gmsh nodes is the line after `$Nodes` in the `.msh`.

---

## 7. Options of `test-polybubble`

| option | default | meaning |
|---|---|---|
| `-m`, `--mesh` | `mesh/single_sphere.msh` | Gmsh 2.2 file (`-pm 0`) or prefix of the split files (`-pm 1`) |
| `-f`, `--bubble-file` | `mesh/single_bubble.txt` | `id x y z r [p]` |
| `-pm`, `--parallel-mesh` | 0 | 1 = one MFEM file per process |
| `-l`, `--library` | `src/libBehaviour.so` | behaviour law; give the absolute path when running outside the build folder |
| `-o`, `--order` | 2 | order of the displacement |
| `-r`, `--refinement` | 0 | uniform refinements in MFEM (not with `-pm 1`) |
| `-pr`, `--pref` | 1 | multiplier of the pressures of the bubble file |
| `-dl`, `--delta` | 0 | δ > 0: search radius $R_i+\delta$; 0: fixed radius `-dm` |
| `-dm`, `--dmin` | 0.950 | fixed search radius (µm) of `bubble/` |
| `-sf`, `--scale-factor-vp` | 0.9 | only points with σ_I > sf × max; ≤ 0: all the points |
| `-pc`, `--preconditioner` | `HypreBoomerAMG` | or `HypreDiagScale` |
| `-li`, `--linear-solver-iterations` | 5000 | maximum PCG iterations |
| `-p`, `--post-processing` | 1 | 1 = also the ParaView fields |
| `-v`, `--verbosity-level` | 0 | 1 = PCG iterations and one block per pore in the log |
| `-n`, `--name-case` | `TestCaseBubble` | name of the ParaView folder |

Options of the runs of the report: `-sf 0 -dl 0.15 -v 1` (+ `-pm 1` on the cluster). With `-dl 0 -sf 0.9` (the
defaults) the computation is the one of `bubble/`.

**Differences from `bubble/`** (`diff -ru bubble polydispersebubble`):

| | `bubble/` | `polydispersebubble/` |
|---|---|---|
| bubble file | `id x y z r` | `id x y z r [p]` |
| pressure | `-pr`, the same for all | pore i: `pref × p_i` |
| search radius | fixed, `dmin` | `r_i + δ` with `-dl` |
| pores with a value | those near the points above 0.9 × max | all, with `-sf 0` |
| search in MPI | all the points copied to every process | local points, per-pore maximum gathered on rank 0 |
| CSV and `sig_hydro.txt` | only with `-p 1` | always; `-p` controls ParaView only |
| CSV columns | `id, x, y, z, stress` | `bubble_id, radius, pressure, sigma_I_max, x, y, z` |
| preconditioner | `HypreDiagScale` | `-pc`, default `HypreBoomerAMG`; `-li` |
| initial value of the per-pore maximum | 0 | "no value", so that negative maxima are kept with `-sf 0` |
| mesh splitter | — | `partition-mesh` |

---

## 8. Build and verification

```bash
source ~/mmm/env.sh                               # or the environment of the cluster (spack load mfem-mgis)
cmake -S mm-opera-hpc -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target lib-elasticity-poly  # without -j: MFront fails with "invalid --jobserver-auth"
cmake --build build -j 8                          # test-polybubble, partition-mesh; copies scripts/ into build/polydispersebubble
ctest --test-dir build -R PolyBubbleTest --output-on-failure      # single bubble, 5 s
```

CMake copies `scripts/*.py`, `scripts/*.sh`, `verification/mmm_vs_analytical.py`, `mesh/` and `Elasticity.mfront`
into `build/polydispersebubble/` at every build: edit the sources, run the copies. Results do not belong in the build
tree (the run scripts write to `~/mmm/results` or to the folder given).

Checks, from `build/polydispersebubble`:

```bash
# 1. linearity in the pressure and the 6th column
mpirun -n 4 ./test-polybubble -m mesh/mesh_sphere.msh -f mesh/single_bubble_ci.txt -p 0 && cp bubbles_and_stresses_selected.txt ref.csv
printf '2 0 0 0 0.400 1.0\n' > p1.txt   # 6 columns, p = 1 → identical to ref.csv
printf '2 0 0 0 0.400 2.0\n' > p2.txt   # p = 2 → sigma_I_max and sig_hydro exactly doubled
mpirun -n 4 ./test-polybubble -m mesh/mesh_sphere.msh -f p1.txt -p 0 && diff ref.csv bubbles_and_stresses_selected.txt
mpirun -n 4 ./test-polybubble -m mesh/mesh_sphere.msh -f p2.txt -p 0 && cat bubbles_and_stresses_selected.txt
# 2. against the analytical solution of one pressurised sphere
mpirun -n 6 ./test-polybubble && python3 mmm_vs_analytical.py   # "Check PASS." (~20 s)
```

| check | result |
|---|---|
| CI mesh, defaults (`-sf 0.9 -dl 0`) | `sigma_I_max` = 0.507159400619, `sig_hydro` average = −0.00043534; identical to `bubble/` to the last digit (exactly with `-pc HypreDiagScale`) |
| CI mesh, `-sf 0 -dl 0.15` | 0.58156378511 (the maximum next to the wall, not within the fixed `dmin`) |
| analytical | hoop stress of an isolated sphere, $\sigma_{\theta\theta}=\tfrac12p\,(R/r)^3$, along a line from the wall (R = 0.4 µm, cube of 10 µm): within 10 % + 0.01 at every sampled point (`mmm_vs_analytical.py`); at the wall the isolated pore gives σ_I/p = 0.5 |
| `-pm 1` against `-pm 0` | 3e-10 (CI mesh, `side10_seed01`, `side10_seed02`) |
| 14 side-10 cubes, 8 processes, split then run | split 111 s, runs 436 s in all, 18–23 PCG iterations, 7.4–7.6 GB per cube |

---

## 9. Scripts (`scripts/`)

Described with their inputs and outputs in the report (section 2). In one line each:

| script | role |
|---|---|
| `run_mmm.sh CUBES [RESULTS]` | workstation: copies a tree of cubes to the results folder and solves the unsolved cases (`-sf 0 -dl 0.15`, `_refinement<k>` folders with `-r k`) |
| (cluster) | one batch job per case for `partition-mesh`, then one for `test-polybubble -pm 1`; the batch script depends on the machine and is not in the repository |
| `set_pressures.py` | cubes at unit pressure → the same cubes with the pressure of the class of each pore (same mesh: `mesh_case.txt`) |
| `plot_bubbles.py` | ŝ per pore; mesh test, size of the RVE (Kanit 2003), released fraction of every cube |
| `break_order.py` | order of rupture, neighbourhood of each pore, `pores.csv` (interface with SCIANTIX) |
| `mesh_compare.py` | the same cubes on two meshes, with the cost from the job logs |
| `npoint.py` | n-point probabilities and intact chords between broken pores |

A case is a folder with one `bubbles_<name>.txt` (the output `bubbles_and_stresses_selected.txt` apart) and its
`mesh_<name>.msh`, or a `mesh_case.txt` whose first line is the folder (relative) that holds the mesh.

---

## 10. Limits of the model

- Linear elastic, isotropic, homogeneous matrix: no plasticity, no grain boundaries, no pre-existing cracks. The
  absolute stress next to a pore is therefore an elastic concentration, and $\sigma_R$ is a calibrated number.
- Perfect spheres, not meshed; no surface tension (the capillary term is in the SCIANTIX criterion, not here).
- **No cascade**: the pressure of a broken pore is not removed and the stress is not redistributed.
- The quantity is a **maximum over Gauss points**. It sits ≈ 0.04 h from the wall, so it converges slowly with the
  mesh (observed order 0.7–1.4 between 0.3 and 0.15 µm) and decreases with refinement; it has noise per pore
  (≈ 1 % median, 4 % at the 95th percentile between two meshes of the same size). Use class statistics, not single pores.
- The search sphere is not periodic (pores cut by a face are searched on one side only) and can touch a neighbour
  if δ is too large (section 4).
- Zero macroscopic strain; the zero-stress result is obtained afterwards by superposition (section 5), exactly only
  for equal pressures.
- `-r` unusable with `-pm 1` (section 6). No check of inverted elements in the solver (section 2).
