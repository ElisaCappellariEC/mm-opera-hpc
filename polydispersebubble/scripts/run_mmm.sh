#!/bin/bash
# Runs test-polybubble on every case of a tree of cubes not solved yet (workstation, whole mesh).
#
#   bash run_mmm.sh CUBES [RESULTS]
#
# CUBES    tree of cubes written by Mérope and meshed by gmsh (e.g. ~/mmm/Merope/mmm_files/<folder>)
# RESULTS  where MMM runs and the results stay: default ~/mmm/results/<name of CUBES>. NOT in the build tree
#          (until 10/11/2026 the results were in build/polydispersebubble: deleting the build deleted them).
#          The new or newer files of CUBES are copied there; the cases already solved are left as they are.
#          RESULTS = CUBES solves the cubes where they are.
# A case is a folder with one bubbles_<name>.txt of Mérope (bubbles_and_stresses_selected.txt of MMM apart). Its mesh is
# its mesh_<name>.msh or, if it has mesh_case.txt (the cubes with the class pressures written by set_pressures.py), the
# mesh of the folder named on the first line of that file, relative to the case.
# Run the copy of the build folder (build/polydispersebubble): it looks for test-polybubble and src/libBehaviour.so
# next to itself. Environment: source ~/mmm/env.sh in the same shell.
# Settings: NPROC=8 POST=0 bash run_mmm.sh ...
# author: E.Cappellari, POLIMI-CEA 2026

set -o pipefail

NPROC=${NPROC:-4}    # MPI processes (the memory does not depend on it: ~21 kB per Gmsh node in all)
POST=${POST:-1}      # -p: 1 also writes the fields for ParaView
OPTIONS=(-sf 0 -dl 0.15)   # -dl = distMin / 2 of Mérope (0.3 um)

if [[ $# -lt 1 || $# -gt 2 || ! -d "$1" ]]; then
    echo "Usage: bash run_mmm.sh CUBES [RESULTS]   (CUBES: tree of cubes of Mérope; RESULTS: default ~/mmm/results/<name of CUBES>)"
    exit 1
fi
DIRECTORYMSH=$(cd "$1" && pwd)
FOLDER=${2:-$HOME/mmm/results/$(basename "$DIRECTORYMSH")}
mkdir -p "$FOLDER" || exit 1
FOLDER=$(cd "$FOLDER" && pwd)

BUILD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXE="$BUILD/test-polybubble"
LIBRARY="$BUILD/src/libBehaviour.so"
for f in "$EXE" "$LIBRARY"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: $f not found: run the copy of this script in the build folder (cmake --build ~/mmm/build -j 8)"
        exit 1
    fi
done

OUTPUTS=(bubbles_and_stresses_selected.txt sig_hydro.txt)
if [[ "$POST" == 1 ]]; then
    OUTPUTS+=(TestCaseBubble Eigenstress-pp)
fi

# true if the case folder $1 has all the OUTPUTS
solved() {
    local output
    for output in "${OUTPUTS[@]}"; do
        [[ -e "$1/$output" ]] || return 1
    done
}

# Copy the new or newer files of DIRECTORYMSH into FOLDER; solved cases are left as they are.
if [[ "$DIRECTORYMSH" != "$FOLDER" ]]; then
    while IFS= read -r -d '' src; do
        dest="$FOLDER${src#"$DIRECTORYMSH"}"
        if solved "$dest"; then
            echo "Kept as it is (already solved): $dest"
            continue
        fi
        mkdir -p "$dest" || exit 1
        find "$src" -maxdepth 1 -type f -exec cp -u -t "$dest/" {} + || exit 1
    done < <(find "$DIRECTORYMSH" -type d -print0)
fi

mapfile -t CASES < <(find "$FOLDER" -type f -name "bubbles_*.txt" ! -name "bubbles_and_stresses_selected.txt" \
                         -printf '%h\n' | sort -u)
TOTAL=${#CASES[@]}
echo "Found $TOTAL cases."
if [[ "$TOTAL" -eq 0 ]]; then
    echo "ERROR: No bubbles files found in $FOLDER"
    exit 1
fi

COUNT=0
for DIR in "${CASES[@]}"; do
    COUNT=$((COUNT + 1))
    echo "=================================="
    echo "CASE $COUNT/$TOTAL: $DIR"
    date

    if solved "$DIR"; then
        echo "Skipping $DIR: all outputs already exist."
        continue
    fi
    mapfile -t FILES < <(find "$DIR" -maxdepth 1 -type f -name "bubbles_*.txt" ! -name "bubbles_and_stresses_selected.txt")
    if [[ ${#FILES[@]} -ne 1 ]]; then
        echo "ERROR: ${#FILES[@]} bubbles_*.txt in $DIR (one expected)"
        continue
    fi
    BUBBLES="${FILES[0]}"
    MESHDIR="$DIR"
    if [[ -f "$DIR/mesh_case.txt" ]]; then
        if ! MESHDIR=$(cd "$DIR/$(head -n 1 "$DIR/mesh_case.txt")" 2>/dev/null && pwd); then
            echo "ERROR: the folder of the mesh named in $DIR/mesh_case.txt does not exist"
            continue
        fi
    fi
    mapfile -t MSH < <(find "$MESHDIR" -maxdepth 1 -type f -name "mesh_*.msh")
    if [[ ${#MSH[@]} -ne 1 ]]; then
        echo "ERROR: ${#MSH[@]} mesh_*.msh in $MESHDIR (one expected)"
        continue
    fi
    file="${MSH[0]}"
    echo "Bubbles $BUBBLES, mesh $file"

    echo "Starting mpirun..."
    # folders ending in _refinement<k>: k uniform refinements in MMM (1 if k is missing)
    EXTRA_ARGS=()
    if [[ "$(basename "$DIR")" =~ _refinement([0-9]*)$ ]]; then
        EXTRA_ARGS=(-r "${BASH_REMATCH[1]:-1}")
        echo "Refinement enabled for $DIR: ${EXTRA_ARGS[*]}"
    fi

    # the run is done in a temporary folder of the case: the outputs (created empty at the start) are moved into the
    # case folder only when test-polybubble ends without error
    WORK="$DIR/mmm_tmp"
    rm -rf "$WORK" && mkdir -p "$WORK" || exit 1
    ( cd "$WORK" && mpirun -n "$NPROC" "$EXE" -l "$LIBRARY" -m "$file" -f "$BUBBLES" -p "$POST" "${OPTIONS[@]}" "${EXTRA_ARGS[@]}" )
    STATUS=$?
    if [[ $STATUS -ne 0 ]]; then
        echo "ERROR: mpirun failed with status $STATUS for $file (run folder kept: $WORK)"
        continue
    fi
    echo "mpirun completed successfully."

    for output in "${OUTPUTS[@]}"; do
        if [[ ! -e "$WORK/$output" ]]; then
            echo "Warning: output not found: $output"
        else
            rm -rf -- "$DIR/$output" && mv -- "$WORK/$output" "$DIR/" || echo "ERROR moving $output"
        fi
    done
    rmdir "$WORK" 2>/dev/null
    echo "CASE $COUNT/$TOTAL completed."
done

echo "Finished: processed $COUNT cases out of $TOTAL. Results in $FOLDER"
