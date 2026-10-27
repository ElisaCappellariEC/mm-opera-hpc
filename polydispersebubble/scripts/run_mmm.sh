#!/bin/bash
# Runs test-polybubble on every case of FOLDER not solved yet. Run it from build/polydispersebubble.
# author: E.Cappellari, POLIMI-CEA 2026

set -o pipefail

DIRECTORYMSH="./../../Merope/mmm_files/test_rve_side10_mesh0.3"   # cubes of Mérope
FOLDER="test_rve_side10_mesh0.3"                                   # their copy, where MMM runs
OUTPUTS=(bubbles_and_stresses_selected.txt sig_hydro.txt TestCaseBubble Eigenstress-pp)

# true if the case folder $1 has all the OUTPUTS
solved() {
    local output
    for output in "${OUTPUTS[@]}"; do
        [[ -e "$1/$output" ]] || return 1
    done
}

# Copy the new or newer files of DIRECTORYMSH into FOLDER; solved cases are left as they are.
if [[ ! -d "$DIRECTORYMSH" ]]; then
    echo "ERROR: input folder not found: $DIRECTORYMSH"
    exit 1
fi
while IFS= read -r -d '' src; do
    dest="$FOLDER${src#"$DIRECTORYMSH"}"
    if solved "$dest"; then
        echo "Kept as it is (already solved): $dest"
        continue
    fi
    mkdir -p "$dest" || exit 1
    find "$src" -maxdepth 1 -type f -exec cp -u -t "$dest/" {} + || exit 1
done < <(find "$DIRECTORYMSH" -type d -print0)

mapfile -t MESHES < <(find "$FOLDER" -type f -name "mesh_unit_NFIR.msh" -print | sort)
TOTAL=${#MESHES[@]}
echo "Found $TOTAL meshes."
if [[ "$TOTAL" -eq 0 ]]; then
    echo "ERROR: No mesh files found in $FOLDER"
    exit 1
fi

COUNT=0
for file in "${MESHES[@]}"; do
    COUNT=$((COUNT + 1))
    DIR=$(dirname "$file")
    BUBBLES="$DIR/bubbles_unit_NFIR.txt"
    echo "=================================="
    echo "CASE $COUNT/$TOTAL: $file"
    date

    if solved "$DIR"; then
        echo "Skipping $DIR: all outputs already exist."
        continue
    fi
    if [[ ! -f "$BUBBLES" ]]; then
        echo "ERROR: Missing bubbles file: $BUBBLES"
        continue
    fi

    echo "Starting mpirun..."
    # folders ending in _refinement<k>: k uniform refinements in MMM (1 if k is missing)
    EXTRA_ARGS=()
    if [[ "$(basename "$DIR")" =~ _refinement([0-9]*)$ ]]; then
        EXTRA_ARGS=(-r "${BASH_REMATCH[1]:-1}")
        echo "Refinement enabled for $DIR: ${EXTRA_ARGS[*]}"
    fi

    mpirun -n 4 ./test-polybubble -m "$file" -f "$BUBBLES" -p 1 -sf 0 -dl 0.5 "${EXTRA_ARGS[@]}"
    STATUS=$?
    if [[ $STATUS -ne 0 ]]; then
        echo "ERROR: mpirun failed with status $STATUS for $file"
        continue
    fi
    echo "mpirun completed successfully."

    # move the outputs into the case folder (folders merged with those already there)
    for output in "${OUTPUTS[@]}"; do
        if [[ ! -e "$output" ]]; then
            echo "Warning: output not found: $output"
        elif [[ -d "$output" ]]; then
            mkdir -p "$DIR/$output" && cp -a "$output/." "$DIR/$output/" && rm -rf -- "$output" \
                || echo "ERROR copying $output"
        else
            mv -f -- "$output" "$DIR/$output" || echo "ERROR moving $output"
        fi
    done
    echo "CASE $COUNT/$TOTAL completed."
done

echo "Finished: processed $COUNT mesh entries out of $TOTAL."
