
#!/bin/bash
# author: E.Cappellari, POLIMI-CEA 2026

set -o pipefail

DIRECTORYMSH="./../../Merope/mmm_files/test_rve_side10"
FOLDER="test_rve_side10"

# # Copy input directory only if it does not already exist
# if [[ ! -d "$FOLDER" ]]; then
#     cp -r "$DIRECTORYMSH" . || exit 1
# fi

OUTPUTS=(
    "bubbles_and_stresses_selected.txt"
    "sig_hydro.txt"
    "TestCaseBubble"
    "Eigenstress-pp"
)

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

    # Activate refinement only for folders containing "_refinement"
    EXTRA_ARGS=()
    if [[ "$DIR" == *"_refinement"* ]]; then
        EXTRA_ARGS=(-r 0)
        echo "Refinement enabled for $DIR"
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
