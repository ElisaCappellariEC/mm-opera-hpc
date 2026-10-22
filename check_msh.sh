# =============================================================================
#!/usr/bin/env bash
# Checks existent .msh
# Usage, from the directory where the Python scripts were run
#   ./check_gmsh.sh
# =============================================================================

set -u

ROOT="test_rve/run_NFIR"
GMSH="./../../gmsh-4.15.2-Linux64/bin/gmsh"
REPORT="mesh_quality_report_$(date +%Y%m%d_%H%M%S).csv"
DETAILS="mesh_quality_details_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$DETAILS"

if [[ ! -d "$ROOT" ]]; then
    echo "ERRORE: directory non trovata: $ROOT"
    exit 1
fi

if [[ ! -x "$GMSH" ]]; then
    echo "ERRORE: eseguibile Gmsh non trovato: $GMSH"
    echo "Correggi il percorso GMSH nello script."
    exit 1
fi

printf '"mesh","size_bytes","log_present","generation_warning",'
printf '"generation_error","generation_completed","quality_check"\n' \
    > "$REPORT"

find "$ROOT" -type f -name '*.msh' -print0 |
while IFS= read -r -d '' msh; do

    log="${msh%.msh}.log"
    detail="$DETAILS/$(basename "${msh%.msh}").check.log"

    size=$(stat -c '%s' "$msh")
    log_present="NO"
    warning="UNKNOWN"
    error="UNKNOWN"
    completed="UNKNOWN"
    quality="NOT_RUN"

    if [[ -f "$log" ]]; then
        log_present="YES"

        if grep -qi 'Warning' "$log"; then
            warning="YES"
        else
            warning="NO"
        fi

        if grep -qiE 'Error|Fatal' "$log"; then
            error="YES"
        else
            error="NO"
        fi

        if grep -qi 'Done meshing order 2' "$log"; then
            completed="YES"
        elif grep -qi 'Done meshing' "$log"; then
            completed="YES"
        else
            completed="UNKNOWN"
        fi
    fi

    if (( size == 0 )); then
        quality="EMPTY_FILE"
    else
        echo "Checking: $msh"

        # Analizza il file esistente; non specifica un file di output.
        if "$GMSH" "$msh" -check -v 3 -nopopup \
             > "$detail" 2>&1; then
            quality="CHECK_EXIT_0"
        else
            quality="CHECK_NONZERO_EXIT"
        fi
    fi

    # CSV con escaping dei nomi dei file.
    printf '"%s",%s,%s,%s,%s,%s,%s\n' \
        "$msh" "$size" "$log_present" "$warning" \
        "$error" "$completed" "$quality" >> "$REPORT"

done

echo
echo "Controllo terminato."
echo "Report: $REPORT"
echo "Log dettagliati: $DETAILS/"
