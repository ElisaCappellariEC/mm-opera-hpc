# Bolle polidisperse con pressione per bolla

Guida alle modifiche del caso `bubble` per una popolazione di bolle di raggi diversi
(es. `side30_seed01`: RVE periodico 30 µm, 1867 bolle in 7 classi di raggio 0.22–1.70 µm)
e con una pressione per bolla (6ª colonna del file bolle).

Questa cartella (`polydispersebubble/`) è una copia di `bubble/` con le modifiche; `bubble/` resta l'originale.
Nomi propri per non collidere con `bubble` in CMake: eseguibile `test-polybubble`, legge di comportamento
`lib-elasticity-poly`, test `PolyBubbleTest`. La cartella è agganciata al build in `mm-opera-hpc/CMakeLists.txt`
(`add_subdirectory(polydispersebubble)`).

---

## 1. In breve

| Cosa | Prima | Ora |
|---|---|---|
| File bolle | esattamente 5 colonne `id x y z r` | 5 **o 6** colonne `id x y z r [p]` (p = 1 se manca) |
| Pressione | `-pr` uguale per tutte | bolla i: `pref × p_i` (`-pr` diventa un moltiplicatore, default 1) |
| Ricerca di σ_I max | sfera di raggio fisso `dmin` = 0.95 µm | con `-dl δ > 0`: sfera di raggio `r_i + δ` |
| Bolle considerate | solo quelle vicine ai punti con σ_I > 0.9·max | con `-sf 0`: **tutte** |
| Memoria della ricerca | tutti i punti copiati su ogni processo MPI | ogni processo usa i suoi punti, poi riduzione su rank 0 |
| CSV / `sig_hydro.txt` | scritti solo con `-p 1` (insieme agli export ParaView) | **sempre**; `-p` controlla solo ParaView |
| CSV | `id, x, y, z, stress` | `bubble_id, radius, pressure, sigma_I_max, x, y, z` (`nan` se nessun punto trovato) |
| Mesh grandi | ogni processo legge la mesh intera | nuovo strumento `partition-mesh` + `test-polybubble -pm 1` |

Comando tipo per una popolazione:

```bash
mpirun -n <N> ./test-polybubble -m mesh.msh -f bubbles.txt -dl 0.2 -sf 0 -p 0 -n mio_caso
```

Con `-dl 0 -sf 0.9` (i default) il calcolo è quello originale (cambia solo il formato del CSV). Il solutore lineare
usa ora BoomerAMG invece del precondizionatore diagonale (§3): stessi risultati, molte meno iterazioni.

---

## 2. Come funziona il codice

### Lo stack software

| Pezzo | Cosa fa | Dove lo vedi nel codice |
|---|---|---|
| **MFEM** | libreria elementi finiti: mesh, spazi H1, assemblaggio, MPI | `mfem::forall`, `ParaViewDataCollection`, `ParMesh` |
| **hypre** | solutori lineari paralleli | `"HyprePCG"`, `"HypreDiagScale"` |
| **MFront / TFEL** | genera la legge di comportamento da `Elasticity.mfront` → `src/libBehaviour.so` | `CMakeLists.txt` (`mfront --obuild`) |
| **MGIS** | chiama la legge MFront in ogni punto di Gauss | `setMaterialProperty`, `getThermodynamicForce` |
| **mfem-mgis ("MMM")** | colla tra MFEM e MGIS: problemi, condizioni al contorno, post-processing | `PeriodicNonLinearEvolutionProblem`, `UniformImposedPressureBoundaryCondition` |

### File da leggere, in quest'ordine

1. `README.md` (questa cartella) e la sezione *Over-pressurized bubbles* del `README.md` principale: fisica e formula FGR.
2. `mesh/single_sphere.geo` + `mesh/single_bubble.txt`: convenzione dei tag.
   `Physical Volume(1)` = matrice (materiale 1), `Physical Surface(k)` = superficie della bolla k.
   La colonna 1 del file bolle **deve** essere il tag della superficie della bolla nella mesh.
   Nella mesh Merope di `side30_seed01`: volume = 1, bolle = 2…1868, involucro periodico = 1869 → già compatibile.
3. `include/OperaHPC/BubbleDescription.hxx`, `src/BubbleDescription.cxx`: struttura `BubbleDescription`
   (id, centro, raggio, pressione) e lettura del file bolle.
4. `src/Bubble.cxx`: il programma `test-polybubble`. Il `main`, nell'ordine:
   - `fill_parameters`: opzioni da riga di comando;
   - lettura delle bolle;
   - `FiniteElementDiscretization`: legge la mesh (`-pm 0`: mesh intera su ogni processo, `-pm 1`: un file per processo),
     elementi H1 di ordine 2, 3 incognite (spostamento);
   - `PeriodicNonLinearEvolutionProblem` con **gradiente macroscopico nullo**: il RVE periodico non può dilatarsi,
     quindi la pressione nelle bolle mette la matrice in compressione media (vedi `sig_hydro.txt`);
   - un `UniformImposedPressureBoundaryCondition` **per bolla**, sul tag della bolla, valore `pref × p_i`;
   - solutore PCG + legge elastica sul materiale 1, E = 150e-3 N/µm² (= 150 GPa), ν = 0.3, poi `problem.solve`
     (problema lineare: una sola iterazione);
   - `findFirstPrincipalStressValueAndLocation`: σ_I massimo su tutto il RVE (serve per la soglia `sf`);
   - `getLocalPointsAndStressAboveStressThreshold`: punti di Gauss di questo processo con σ_I sopra soglia;
   - ciclo sulle bolle (`mfem::forall`): punti ordinati per x, ricerca binaria della fetta |x − x_c| < r_s, poi filtro
     su y, z e distanza < r_s; tiene il massimo;
   - riduzione `MPI_Gather` sul rank 0: per ogni bolla il massimo fra i processi;
   - scrittura di `bubbles_and_stresses_selected.txt` e `sig_hydro.txt`; con `-p 1` export ParaView
     (`<nome caso>/`, `Eigenstress-pp/`).
5. `src/Utilities.cxx`: calcolo di σ_I (autovalori del tensore degli sforzi) in ogni punto di Gauss, raccolta dei punti.
6. `verification/mmm_vs_analytical.py`: confronto con la soluzione analitica σ_θθ = p R³ / (2 r³) per una bolla isolata.

### Concetti utili per leggere i risultati

- **Unità.** Lunghezze in µm, E in N/µm². Con carichi di sola pressione e materiale omogeneo le tensioni non dipendono
  da E (solo da ν) e sono lineari in p: se le pressioni del file sono in MPa, `sigma_I_max` è in MPa.
- **Compressione media.** `sig_hydro.txt` contiene `Volume, Integral, Average` con `Average` = media di tr(σ)/3 sul RVE
  (negativa). Il README principale spiega come passare al caso senza confinamento:
  σ(p − p_hyd, 0) = σ(p, p_hyd) + p_hyd I. Il codice **non** applica questa correzione; lo script FGR la offre
  (`--shift-hydro`). Con pressioni diverse fra le bolle la correzione è solo approssimata (vedi §7).
- **Ogni punto di Gauss appartiene a un solo processo MPI**: massimo locale per bolla + massimo fra processi = massimo globale esatto.

---

## 3. Le modifiche, file per file

Tutte le modifiche sono commentate nel codice (commenti in inglese, come il resto del repository).

| File | Modifica | Perché |
|---|---|---|
| `include/OperaHPC/BubbleDescription.hxx` | nuovo campo `pressure = 1` | pressione per bolla; 1 = comportamento originale |
| `src/BubbleDescription.cxx` | accetta 5 o 6 colonne; la 6ª è la pressione | il file Merope ha 6 colonne e prima veniva rifiutato ("ill-formed line") |
| `src/Bubble.cxx` – condizioni al contorno | valore `pref * b.pressure` | pressione diversa per ogni bolla |
| `src/Bubble.cxx` – opzione `-dl/--delta` | raggio di ricerca `r_i + δ` se δ > 0 | con `dmin` fisso = 0.95 µm le 124 bolle con r ≥ 1.19 µm non trovavano **nessun** punto e le 452 con r = 0.85 µm solo un guscio di 0.1 µm (< 1 elemento): ≈ 72 % del volume dei pori; le bolle piccole invece "vedevano" quelle vicine |
| `src/Bubble.cxx` – soglia | `-sf <= 0` → tutti i punti | per la FGR serve σ_I max di **ogni** bolla, non solo di quelle vicine al massimo globale |
| `src/Utilities.cxx/.hxx` | `getLocalPointsAndStressAboveStressThreshold` (flag `gather` sull'implementazione esistente) | con `-sf 0` la versione originale copierebbe centinaia di milioni di punti su ogni processo |
| `src/Bubble.cxx` – riduzione | `MPI_Gather` di `[σ, x, y, z] × n_bolle` sul rank 0 | combina i massimi locali (poche decine di KB) |
| `src/Bubble.cxx` – output | CSV con raggio e pressione, `nan` se nessun punto, avviso se ci sono bolle senza valore; intestazioni senza virgola finale; dettaglio per bolla solo con `-v 1` | post-processing diretto (pandas/numpy), log leggibile con 1867 bolle |
| `src/Bubble.cxx` – post-processing | CSV e `sig_hydro.txt` sempre scritti, `-p` controlla solo ParaView | sulla mesh grande gli export ParaView pesano decine di GB e richiedono una proiezione L2 in più |
| `src/Bubble.cxx` – solutore lineare | precondizionatore `-pc` (default `HypreBoomerAMG`, prima `HypreDiagScale` fisso) e massimo di iterazioni `-li` (default 5000, prima 1000 di hypre) | sulla mesh di default il PCG diagonale richiede 1355 iterazioni: con MFEM 4.10 + mfem-mgis develop il superamento delle 1000 è un errore (`solve: the non linear solver did not converge`). BoomerAMG: 18 iterazioni, risultati identici (10 cifre) |
| `src/Bubble.cxx` – controlli con `-p 1` | `\| or_die` su export ParaView, σ_I, proiezione L2 e tensione idrostatica | l'originale ignorava gli errori: se la proiezione L2 falliva, il programma andava in segfault senza messaggio |
| `tools/PartitionMesh.cxx` (nuovo) | programma `partition-mesh` | vedi §5 |
| `CMakeLists.txt` | target rinominati (`test-polybubble`, `lib-elasticity-poly`, test `PolyBubbleTest`); sorgenti limitati a `src/`; target `partition-mesh` presente ma commentato | nomi unici nel progetto CMake; il glob ricorsivo avrebbe incluso il secondo `main` di `tools/` |
| `scripts/fgr_population.py` (nuovo) | FGR(σ_R) dal CSV | formula FGR del README principale |
| `README.md` | tabella opzioni aggiornata + rimando a questa guida | |

---

## 4. Come rivedere le modifiche

```bash
cd mm-opera-hpc
diff -rq bubble polydispersebubble          # elenco dei file diversi o nuovi
diff -ru bubble polydispersebubble | less   # tutte le differenze riga per riga
diff -u bubble/src/Bubble.cxx polydispersebubble/src/Bubble.cxx   # un file alla volta
```

Checklist:

- lettura del file: 5 colonne → pressione 1; 6 colonne → pressione letta; altro → errore (verificato qui, anche sul file reale:
  1867 bolle, id 2…1868);
- condizione al contorno: `pref * b.pressure`; la lambda cattura `b` per riferimento a un vettore che non viene più ridimensionato;
- ricerca: `rs` usato in **tutti e tre** i filtri (fetta in x, test y/z, distanza);
- MPI: la ricerca usa solo punti locali, la riduzione avviene **dopo** il ciclo, solo il rank 0 scrive;
- con `-dl 0 -sf 0.9` i valori devono essere identici all'originale (all'ultima cifra con il default BoomerAMG,
  esattamente con `-pc HypreDiagScale`).

Per togliere tutto: cancellare `polydispersebubble/` e la riga `add_subdirectory(polydispersebubble)` in `mm-opera-hpc/CMakeLists.txt`.

---

## 5. La mesh

### Controlli

- La mesh deve essere **Gmsh 2.2** (`$MeshFormat` = `2.2 0 8`), ordine 2, con la sezione `$Periodic`.
- Deve essere completa: `tail -c 100 mesh.msh` deve finire con `$EndPeriodic` e `grep -c '^\$EndElements' mesh.msh` deve dare 1.
  La copia locale di `side30_seed01/mesh_unit_NFIR.msh` è **troncata** (349 MB, si ferma al nodo 5,6M su 49M, manca `$Elements`).
- I tag: volume 1, bolle = colonna 1 del file bolle, involucro 1869 (nessuna condizione imposta sull'involucro: è periodico).

### Mesh grandi: come vengono lette

Verificato su MFEM 4.7 e sul sorgente di mfem-mgis (`MeshDiscretization.cxx`):

- `-pm 0` (`MeshReadMode = FromScratch`, default): **ogni processo MPI legge la mesh seriale intera**, poi MFEM la partiziona
  (METIS). Per 49M nodi P2 servono indicativamente 10–20 GB **per processo** solo per questa fase → pochi processi per nodo.
- `-pm 1` (`MeshReadMode = Restart`): ogni processo legge **un suo file** di nome `<MeshFileName>` + rank su 6 cifre
  (es. `-m mesh_par.` → `mesh_par.000000`, `mesh_par.000001`, …) nel formato parallelo nativo di MFEM
  (quello scritto da `ParMesh::ParPrint`). Numero di processi = numero di file.

### Se hai una mesh Gmsh 4.1 divisa in 50 file

Il lettore Gmsh di MFEM legge **solo il formato 2.2**; i file partizionati da Gmsh (`-part 50`,
`Mesh.PartitionSplitMeshFiles`) non sono nel formato di `-pm 1`. Quindi non si possono usare direttamente.

1. **Ottieni una mesh unica in formato 2.2.**
   - Preferibile: rigenerarla dal `.geo` (che contiene già `Mesh.MshFileVersion = 2.2` e nessuna partizione):
     `gmsh -3 mesh_unit_NFIR.geo -o mesh_unit_NFIR.msh` (circa 50 min secondo il log di Gmsh).
   - In alternativa, ricucire i pezzi con l'API Python di Gmsh:
     ```python
     import glob, gmsh
     gmsh.initialize()
     for f in sorted(glob.glob("mesh_*.msh")):
         gmsh.merge(f)
     gmsh.model.mesh.unpartition()
     gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
     gmsh.write("mesh_unica.msh")
     gmsh.finalize()
     ```
     Poi verificare: presenza di `$Periodic`, tag fisici esattamente {1, 2…1868, 1869} (nessun gruppo creato per le
     partizioni), stesso numero di nodi della mesh originale. Se `$Periodic` manca la mesh non va bene → rigenerare.
2. **Se il nodo ha abbastanza memoria** per (mesh intera × processi per nodo): lancia direttamente con `-pm 0`.
3. **Altrimenti partiziona con MFEM**, una volta sola, con `partition-mesh`. Il target è commentato in
   `polydispersebubble/CMakeLists.txt`: decommenta le due righe `add_executable(partition-mesh ...)` e
   `target_link_libraries(partition-mesh ...)`, poi `cmake --build build --target partition-mesh`.
   ```bash
   mpirun -n 50 ./partition-mesh -m mesh_unica.msh -o mesh_par.
   mpirun -n 50 ./test-polybubble -pm 1 -m mesh_par. -f bubbles.txt -dl 0.2 -sf 0 -p 0
   ```
   `partition-mesh` fa esattamente la lettura `FromScratch` di mfem-mgis (stessa partizione METIS) e salva ogni pezzo.
   È ancora il passaggio con il picco di memoria (ogni processo legge la mesh intera), ma si fa una volta sola:
   distribuisci i processi su più nodi (pochi per nodo). Il numero 50 non è obbligatorio: scegli N in base alle risorse
   (la simulazione deve poi girare con lo stesso N).
   Provato qui sulla mesh CI (`mesh/mesh_sphere.msh`, 4 processi): i file si rileggono con lo stesso numero di elementi
   (6899), di elementi di bordo (1080), gli stessi attributi e i nodi P2 periodici.

---

## 6. Piano di lancio (sull'altro computer)

### 0. Portare il codice

`git clone` copia solo ciò che è committato (e pushato, se cloni da GitHub). Se la cartella non è committata,
copiala a mano (senza `build/`), ad esempio `tar czf mm-opera-hpc-poly.tgz --exclude=build mm-opera-hpc`.

### 1. Ambiente e compilazione

Serve **mfem-mgis@master** (README principale): con la 1.0.3 installata in `~/mmm/spack` il repository non compila
nemmeno senza queste modifiche (API cambiate: `CatchTimeSection(ctx, …)`, `getLocation`, `isInvalid`, `MGIS/Function`).

```bash
spack load mfem-mgis      # o l'ambiente equivalente del cluster
cmake -S mm-opera-hpc -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target lib-elasticity-poly      # senza -j (vedi sotto)
cmake --build build --target test-polybubble -j 8     # solo questo caso (oppure tutto: cmake --build build -j 8)
ctest --test-dir build -R PolyBubbleTest --output-on-failure   # test CI (file a 5 colonne)
```

Se `mfront` fallisce con `invalid --jobserver-auth`, compila prima `lib-elasticity-poly` senza `-j` come sopra (successo qui).

### 2. Regressione su una bolla (minuti)

```bash
cd build/polydispersebubble
mpirun -n 4 ./test-polybubble -m mesh/mesh_sphere.msh -f mesh/single_bubble_ci.txt -p 0
cp bubbles_and_stresses_selected.txt ref.csv
printf '2 0 0 0 0.400 1.0\n' > p1.txt   # 6 colonne, p = 1 → stesso risultato di ref.csv
printf '2 0 0 0 0.400 2.0\n' > p2.txt   # p = 2 → sigma_I_max e sig_hydro esattamente doppi
mpirun -n 4 ./test-polybubble -m mesh/mesh_sphere.msh -f p1.txt -p 0 && diff ref.csv bubbles_and_stresses_selected.txt
mpirun -n 4 ./test-polybubble -m mesh/mesh_sphere.msh -f p2.txt -p 0 && cat bubbles_and_stresses_selected.txt
# verifica analitica (mesh fine di default, serve -p 1):
mpirun -n 6 ./test-polybubble && python3 mmm_vs_analytical.py      # → "Check PASS." (~20 s, provato qui)
# sigma_I_max = 0.507159400619, sig_hydro Average = -0.00043534 (stessi valori con -pc HypreDiagScale -li 20000)
# partition-mesh (solo se abilitato in CMakeLists.txt): stessi risultati con -pm 1
mpirun -n 4 ./partition-mesh -m mesh/mesh_sphere.msh -o ci_par.
mpirun -n 4 ./test-polybubble -pm 1 -m ci_par. -f mesh/single_bubble_ci.txt -p 0 && cat bubbles_and_stresses_selected.txt
# → stessi valori di ref.csv (a meno di differenze all'ultima cifra) e stesso sig_hydro.txt
```

**Se compare `solve: the non linear solver did not converge`** (poi `signal 6 (Aborted)` su tutti i processi):
il solutore lineare ha raggiunto il massimo di iterazioni. Era il caso della mesh di default con il precondizionatore
originale (`HypreDiagScale`: 1355 iterazioni > 1000). Rilancia con `-v 1` per vedere `PCG Iterations` e, se serve,
aumenta `-li` o cambia `-pc`. Sulle mesh grandi tieni `HypreBoomerAMG`: con il precondizionatore diagonale le
iterazioni crescono con la dimensione della mesh.

### 3. Test polidisperso piccolo (minuti, anche su un PC)

Genera con lo stesso script Merope di `side30_seed01` un cubo piccolo (lato 8–10 µm, mesh 0.3 µm), poi:

```bash
mpirun -n 8 ./test-polybubble -m piccolo.msh -f bolle_piccolo.txt -dl 0.2 -sf 0 -p 1 -n piccolo
```

Controlli:
- nessun avviso "have no value" e nessun `nan` nel CSV;
- per le bolle isolate `sigma_I_max / pressure` ≈ 0.5 (σ_θθ alla parete = p/2), un po' meno per la compressione media;
  valori più alti nei legamenti fra bolle vicine;
- cambia la pressione di **una** bolla (es. 2.0): il suo valore deve circa raddoppiare, le vicine cambiare poco;
- prova δ = 0.15, 0.2, 0.3: i risultati devono essere stabili (se cambiano molto, il massimo è lontano dalla parete).
- apri `piccolo/` e `Eigenstress-pp/` in ParaView per vedere dove si concentra σ_I.

### 4. Preparazione della mesh grande

Vedi §5: mesh completa in formato 2.2; se serve, `partition-mesh` con N processi.

### 5. Calcolo completo

Ordine di grandezza: 49M nodi P2 ≈ 150M gradi di libertà → centinaia di GB di RAM in totale.
Con `-sf 0` ogni processo tiene in memoria tutti i suoi punti di Gauss per la ricerca (32 byte per punto, qualche
centinaio di MB per processo con 50 processi).

Esempio SLURM (da adattare al cluster):

```bash
#!/bin/bash
#SBATCH --job-name=bubbles-side30
#SBATCH --ntasks=50
#SBATCH --ntasks-per-node=10        # in base alla memoria per nodo
#SBATCH --time=04:00:00
spack load mfem-mgis
srun ./test-polybubble -pm 1 -m mesh_par. -f bubbles_unit_NFIR.txt -dl 0.2 -sf 0 -p 0 -n side30_seed01
```

Usa `-p 1` solo se ti servono i campi in ParaView (decine di GB in più e una proiezione L2 in più).

### 6. Risultati e FGR

File prodotti nella cartella di lancio:
- `bubbles_and_stresses_selected.txt`: `bubble_id, radius, pressure, sigma_I_max, x, y, z`;
- `sig_hydro.txt`: `Volume, Integral, Average` (tensione idrostatica media del RVE);
- con `-p 1`: `<nome caso>/` (tensore degli sforzi ai nodi) e `Eigenstress-pp/` (σ_I proiettata ai nodi).

```bash
python3 scripts/fgr_population.py bubbles_and_stresses_selected.txt                 # FGR(σ_R) → fgr.csv, fgr.png
python3 scripts/fgr_population.py bubbles_and_stresses_selected.txt --shift-hydro   # senza confinamento (approssimato)
```

---

## 7. Limiti noti e punti da discutere

- **Condizione al contorno macroscopica.** Il calcolo impone deformazione macroscopica nulla (come l'originale).
  La correzione del README (+p_hyd I) dà il caso senza confinamento ma con pressioni `p_i − p_hyd`: esatta solo con
  pressioni tutte uguali (riscalando). Per pressioni diverse e tensione macroscopica nulla esatta servirebbe un secondo
  calcolo (deformazione isotropa unitaria, pressioni nulle) e una combinazione lineare dei due campi.
- **Scelta di δ.** Il guscio `r_i + δ` può includere punti del legame con una bolla vicina (distanza minima fra bolle
  0.3 µm): il massimo può venire dalla concentrazione fra le due bolle, che è fisicamente il punto dove partirebbe la cricca.
  Con δ ≤ 0.3 µm il guscio non supera la distanza minima.
- **Bordo periodico.** La ricerca usa le coordinate dei punti senza immagini periodiche: se il guscio di una bolla
  attraversa l'involucro periodico, la parte oltre l'involucro (che nella mesh sta sul lato opposto, traslata di 30 µm)
  non viene vista. L'involucro di Merope non taglia le bolle, quindi la parete è sempre inclusa; l'effetto è limitato.
- `Bubble::broken` non viene mai messo a `true` (residuo di una versione iterativa): la pressione è sempre applicata.
- `test-polybubble` è stato compilato e provato su PC-EC (mfem-mgis develop, MFEM 4.10, OpenMPI 5): regressione CI
  e verifica analitica (`Check PASS.`) del passo 2. Il caso polidisperso (passo 3) e `-pm 1` non sono ancora stati provati
  con questa versione.
