#!/bin/bash
# author: E.Cappellari, POLIMI-CEA 2026

set -o pipefail

DIRECTORYMSH="./../../Merope/mmm_files/test_rve_side10_mesh0.3"
FOLDER="test_rve_side10_mesh0.3"

OUTPUTS=(
    "bubbles_and_stresses_selected.txt"
    "sig_hydro.txt"
    "TestCaseBubble"
    "Eigenstress-pp"
)

# True if the case folder $1 has all the OUTPUTS (already solved)
solved() {
    local output
    for output in "${OUTPUTS[@]}"; do
        [[ -e "$1/$output" ]] || return 1
    done
}

# Copy the inputs of DIRECTORYMSH into FOLDER, also when FOLDER is already there:
#  - files and folders missing in FOLDER are added (e.g. the meshes made after the first copy);
#  - in a case not solved yet, a file newer in DIRECTORYMSH replaces the old one (e.g. a mesh made again);
#  - a case already solved (all OUTPUTS there) is left as it is, mesh included.
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
    # only the files of this folder: its subfolders come in their own turn of the loop
    find "$src" -maxdepth 1 -type f -exec cp -u -t "$dest/" {} + || exit 1
done < <(find "$DIRECTORYMSH" -type d -print0)

# Collect all meshes before starting the loop
mapfile -t MESHES < <(
    find "$FOLDER" -type f -name "mesh_unit_NFIR.msh" -print | sort
)

TOTAL=${#MESHES[@]}
echo "Found $TOTAL meshes."

if [[ "$TOTAL" -eq 0 ]]; then
    echo "ERROR: No mesh files found in $FOLDER"
    exit 1
fi

COUNT=0

for file in "${MESHES[@]}"; do
    COUNT=$((COUNT + 1))

    echo "=================================="
    echo "CASE $COUNT/$TOTAL: $file"
    date

    DIR=$(dirname "$file")

    # Skip cases that already have all expected outputs
    SKIP_CASE=true

    for output in "${OUTPUTS[@]}"; do
        if [[ ! -e "$DIR/$output" ]]; then
            SKIP_CASE=false
            break
        fi
    done

    if [[ "$SKIP_CASE" == true ]]; then
        echo "Skipping $DIR: all outputs already exist."
        continue
    fi

    BUBBLES="$DIR/bubbles_unit_NFIR.txt"

    if [[ ! -f "$BUBBLES" ]]; then
        echo "ERROR: Missing bubbles file: $BUBBLES"
        continue
    fi

    echo "Starting mpirun..."

    # Uniform refinement in MMM only for folders ending in "_refinement<k>": k levels (-r k), 1 when k is missing.
    # Each level halves the element size of the mesh read from the folder (plot_bubbles.py reads k the same way).
    EXTRA_ARGS=()
    if [[ "$(basename "$DIR")" =~ _refinement([0-9]*)$ ]]; then
        LEVELS="${BASH_REMATCH[1]:-1}"
        EXTRA_ARGS=(-r "$LEVELS")
        echo "Refinement enabled for $DIR: -r $LEVELS"
    fi

    if mpirun -n 4 ./test-polybubble \
        -m "$file" -f "$BUBBLES" -p 1 -sf 0 -dl 0.5 \
        "${EXTRA_ARGS[@]}"; then

        echo "mpirun completed successfully."

        for output in "${OUTPUTS[@]}"; do
            if [[ -e "$output" ]]; then
                DEST="$DIR/$output"

                if [[ -d "$output" ]]; then
                    # Merge directory contents into the case directory
                    mkdir -p "$DEST"
                    cp -a "$output/." "$DEST/" || {
                        echo "ERROR copying $output"
                        continue
                    }
                    rm -rf -- "$output"
                else
                    mv -f -- "$output" "$DEST" || {
                        echo "ERROR moving $output"
                        continue
                    }
                fi
            else
                echo "Warning: output not found: $output"
            fi
        done
    else
        STATUS=$?
        echo "ERROR: mpirun failed with status $STATUS for $file"
        continue
    fi

    echo "CASE $COUNT/$TOTAL completed."
done

echo "Finished: processed $COUNT mesh entries out of $TOTAL."
