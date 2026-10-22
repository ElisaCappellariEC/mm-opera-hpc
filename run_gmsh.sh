#!/bin/bash
# author: E.Cappellari, POLIMI-CEA 2026

# =============================================================================
# 3D mesh of every mesh_*.geo written by Mérope in the selected folder: 
#
# Outputs:
#   mesh_<name>.msh
#   mesh_<name>.log
#
# Usage, from the directory where the Python scripts were run
#   ./run_gmsh.sh
# =============================================================================

set -uo pipefail

# =====CONFIGURATION============================================================
GMSH="./../../gmsh-4.15.2-Linux64/bin/gmsh"
FOLDER="test_rve2"
MSH_FORMAT="2.2"
case "$MSH_FORMAT" in
    2.2) GMSH_FORMAT="msh2" ;;
    4.1) GMSH_FORMAT="msh41" ;;
    *)
        echo "ERROR: Unsupported MSH format version: $MSH_FORMAT" >&2
        exit 1
        ;;
esac
# Optional Gmsh partitioning arguments.
GMSH_EXTRA_ARGS=(
    # -setnumber Mesh.MetisAlgorithm 2
    # -part 50
    # -part_split
)

# =====ERRORS==================================================================
if [[ ! -x "$GMSH" ]]; then
    echo "ERROR: Gmsh executable not found or not executable: $GMSH" >&2
    exit 1
fi

if [[ ! -d "$FOLDER" ]]; then
    echo "ERROR: Input folder not found: $FOLDER" >&2
    exit 1
fi

# =====MESH GENERATION=========================================================
total=0
success=0
failed=0

while IFS= read -r -d '' geo; do
    total=$((total + 1))

    out="${geo%.geo}"
    msh="${out}.msh"
    log="${out}.log"

    echo "============================================================"
    echo "[$total] Processing: $geo"
    echo "    Format: MSH $MSH_FORMAT"

    {
        echo "============================================================"
        echo "Gmsh mesh generation log"
        echo "============================================================"
        echo "Input geometry : $geo"
        echo "Output mesh    : $msh"
        echo "Requested format: MSH $MSH_FORMAT"
        echo "Gmsh executable: $GMSH"
        echo "Start time     : $(date '+%Y-%m-%d %H:%M:%S')"
        echo
    } > "$log"

    # Detect and override Mesh.MshFileVersion in the .geo file.

    if grep -Eq '^[[:space:]]*Mesh[.]MshFileVersion[[:space:]]*=' "$geo"; then
        old_format=$(
            sed -nE \
                's/^[[:space:]]*Mesh[.]MshFileVersion[[:space:]]*=[[:space:]]*([0-9]+([.][0-9]+)?).*/\1/p' \
                "$geo" | head -n 1
        )

        if [[ -n "$old_format" && "$old_format" != "$MSH_FORMAT" ]]; then
            echo "    Override: $old_format -> $MSH_FORMAT"

            {
                echo "Geometry format override:"
                echo "  Original: $old_format"
                echo "  Requested: $MSH_FORMAT"
                echo
            } >> "$log"

            # Replace the numeric version in the assignment.
            if ! sed -i -E \
                "s/^([[:space:]]*Mesh[.]MshFileVersion[[:space:]]*=[[:space:]]*)[0-9]+([.][0-9]+)?/\1${MSH_FORMAT}/" \
                "$geo"; then
                echo "ERROR: Could not update format in $geo" | tee -a "$log"
                failed=$((failed + 1))
                continue
            fi

        elif [[ -n "$old_format" ]]; then
            echo "    Geometry format already matches: $old_format"
            echo "Geometry format: $old_format (already matches)" >> "$log"
        else
            echo "    WARNING: Format assignment found but not recognized."
            echo "WARNING: Could not parse Mesh.MshFileVersion assignment." >> "$log"
        fi
    else
        echo "    No Mesh.MshFileVersion assignment found in .geo."
        echo "Geometry format: no explicit assignment found" >> "$log"
    fi

    # Run Gmsh and capture ALL output.

    start_seconds=$(date +%s)

    echo "    Running Gmsh..."

    {
        echo
        echo "Gmsh output:"
        echo "------------------------------------------------------------"
    } >> "$log"

    set +e
    "$GMSH" "$geo" \
        -3 \
        -format "$GMSH_FORMAT" \
        "${GMSH_EXTRA_ARGS[@]}" \
        -o "$msh" \
        >> "$log" 2>&1
    status=$?
    set -e

    end_seconds=$(date +%s)
    elapsed=$((end_seconds - start_seconds))

    # Show only relevant information in the terminal.
    if [[ $status -eq 0 && -s "$msh" ]]; then
        echo "    SUCCESS | $(basename "$msh") | ${elapsed}s"
    else
        echo "    FAILED  | $(basename "$geo") | exit code $status"

        # Show warnings and errors immediately when a mesh fails.
        grep -Ei 'warning|error|fatal' "$log" | tail -n 20 || true
    fi

    {
        echo
        echo "------------------------------------------------------------"
        echo "Exit code    : $status"
        echo "Elapsed time : ${elapsed} seconds"
        echo "End time     : $(date '+%Y-%m-%d %H:%M:%S')"
    } >> "$log"

    if [[ $status -eq 0 && -s "$msh" ]]; then
        echo "Status: SUCCESS" >> "$log"
        success=$((success + 1))
    else
        echo "Status: FAILED" >> "$log"
        failed=$((failed + 1))
    fi

done < <(find "$FOLDER" -type f -name "mesh_*.geo" -print0 | sort -z)

# =====SUMMARY=================================================================

echo
echo "============================================================"
echo "MESH GENERATION SUMMARY"
echo "============================================================"
echo "Folder          : $FOLDER"
echo "Requested format: MSH $MSH_FORMAT"
echo "Total           : $total"
echo "Successful      : $success"
echo "Failed          : $failed"
echo "============================================================"

if [[ $failed -gt 0 ]]; then
    exit 1
fi

exit 0