#!/bin/bash
# Analysis of the runs of a Latin hypercube solved by MMM: break_order.py on every run folder, then the tables of
# s_hat for SCIANTIX (mmm_surrogate.py) on the unit runs.
#
#   bash analyse_lhs.sh [LHS]
#
# LHS  tree of the hypercube, with the results of MMM in its cubes:
#          LHS/unit/run_<id>/side<L>_seed<NN>/   bubbles_unit_<id>.txt, summary_unit_<id>.txt (Mérope),
#                                                bubbles_and_stresses_selected.txt, sig_hydro.txt (MMM)
#          LHS/pk/run_<id>/side<L>_seed<NN>/     the same with the class pressures, and mesh_case.txt (set_pressures.py)
#      default: the only lhs_* folder of the build folder (build/polydispersebubble), next to this script or, if the
#      script is run from scripts/, in BUILD_DIR (default ~/mmm/build/polydispersebubble).
# Per run folder (unit first, then pk; a pk run needs the unit run of the same cube, solved):
#     break_order.py     pores.csv, break_order.txt/png; the output of the script goes to break_order.log
#     plot_bubbles.py    only with PLOT=1: rve_size.txt, rve_fgr.txt/png (plot_bubbles.log). Its exit 1 also means
#                        "not an RVE" or "mesh not fine", which says little with one cube per run: read the log
# A run with no cube solved is skipped; a run already analysed (pores.csv newer than the results of MMM) is skipped
# unless FORCE=1.
# Then, on the unit runs with pores.csv: mmm_surrogate.py (SURROGATE, default
# ~/mmm/sciantix-official/utilities/HBFF/mmm_surrogate.py; skipped if the file is not there), which writes
# mmm_outputs/shat_<id>.txt/png next to itself.
# Settings: PLOT=1 FORCE=1 bash analyse_lhs.sh ...
# author: E.Cappellari, POLIMI-CEA 2026

set -o pipefail

PLOT=${PLOT:-0}      # 1: also plot_bubbles.py on every run
FORCE=${FORCE:-0}    # 1: redo the runs already analysed
BUILD_DIR=${BUILD_DIR:-$HOME/mmm/build/polydispersebubble}
SURROGATE=${SURROGATE:-$HOME/mmm/sciantix-official/utilities/HBFF/mmm_surrogate.py}
PYTHON=${PYTHON:-python3}
STRESSES="bubbles_and_stresses_selected.txt"

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
for f in break_order.py plot_bubbles.py; do
    if [[ ! -f "$HERE/$f" ]]; then
        echo "ERROR: $HERE/$f not found: this script runs next to the Python scripts"
        exit 1
    fi
done

if [[ $# -gt 1 ]]; then
    echo "Usage: bash analyse_lhs.sh [LHS]   (LHS: tree of the hypercube; default: the lhs_* folder of the build folder)"
    exit 1
fi
if [[ $# -eq 1 ]]; then
    LHS=$1
else
    for where in "$HERE" "$BUILD_DIR"; do
        mapfile -t FOUND < <(find "$where" -maxdepth 1 -type d -name "lhs_*" 2>/dev/null | sort)
        [[ ${#FOUND[@]} -gt 0 ]] && break
    done
    if [[ ${#FOUND[@]} -ne 1 ]]; then
        echo "ERROR: ${#FOUND[@]} lhs_* folders in $HERE and $BUILD_DIR (one expected): give the folder, bash analyse_lhs.sh LHS"
        [[ ${#FOUND[@]} -gt 0 ]] && printf '    %s\n' "${FOUND[@]}"
        exit 1
    fi
    LHS=${FOUND[0]}
fi
if [[ ! -d "$LHS" ]]; then
    echo "ERROR: $LHS is not a folder"
    exit 1
fi
LHS=$(cd "$LHS" && pwd)

# the run folders: unit first, then the others (pk)
mapfile -t UNIT < <(find "$LHS" -type d -name "run_*" -path "*/unit/*" | sort)
mapfile -t OTHER < <(find "$LHS" -type d -name "run_*" ! -path "*/unit/*" | sort)
RUNS=("${UNIT[@]}" "${OTHER[@]}")
echo "Found ${#RUNS[@]} run folders in $LHS (${#UNIT[@]} unit)."

# Runs the script $1 on the run folder $2, output in $2/<name of the script>.log; prints its last line if it fails
analyse() {
    local script=$1 run=$2 log
    log="$run/${script%.py}.log"
    if "$PYTHON" "$HERE/$script" "$run" > "$log" 2>&1; then
        return 0
    fi
    echo "    $script: exit $? ($(tail -n 1 "$log")), see $log"
    return 1
}

DONE=0
SKIPPED=0
NOT_SOLVED=0
FAILED=()
for RUN in "${RUNS[@]}"; do
    NAME=${RUN#"$LHS"/}
    mapfile -t CUBES < <(find "$RUN" -mindepth 1 -maxdepth 1 -type d -name "side*_seed*" | sort)
    SOLVED=()
    for CUBE in "${CUBES[@]}"; do
        [[ -s "$CUBE/$STRESSES" ]] && SOLVED+=("$CUBE")
    done
    if [[ ${#SOLVED[@]} -eq 0 ]]; then
        echo "Not solved yet (${#CUBES[@]} cubes): $NAME"
        NOT_SOLVED=$((NOT_SOLVED + 1))
        continue
    fi
    NOTE=""
    if [[ ${#SOLVED[@]} -lt ${#CUBES[@]} ]]; then
        NOTE=" (only ${#SOLVED[@]} of ${#CUBES[@]} cubes solved)"
    fi
    # already analysed: pores.csv newer than the results of every solved cube
    NEW=1
    if [[ "$FORCE" != 1 && -s "$RUN/pores.csv" ]]; then
        NEW=0
        for CUBE in "${SOLVED[@]}"; do
            [[ "$CUBE/$STRESSES" -nt "$RUN/pores.csv" ]] && NEW=1
        done
    fi
    if [[ "$NEW" == 0 ]]; then
        echo "Already analysed$NOTE: $NAME"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    echo "Analysing$NOTE: $NAME"
    if analyse break_order.py "$RUN"; then
        DONE=$((DONE + 1))
    else
        rm -f "$RUN/pores.csv"   # not left half written for the next step
        FAILED+=("$NAME")
        continue
    fi
    if [[ "$PLOT" == 1 ]]; then
        analyse plot_bubbles.py "$RUN"
    fi
done

echo "break_order.py: $DONE runs analysed, $SKIPPED already analysed, $NOT_SOLVED not solved yet, ${#FAILED[@]} failed."
if [[ ${#FAILED[@]} -gt 0 ]]; then
    printf '    FAILED: %s\n' "${FAILED[@]}"
fi

# Tables of s_hat per class for SCIANTIX: the unit runs only (mmm_surrogate.py is written for one pressure)
READY=()
for RUN in "${UNIT[@]}"; do
    [[ -s "$RUN/pores.csv" ]] && READY+=("$RUN")
done
if [[ ${#READY[@]} -eq 0 ]]; then
    echo "No unit run with pores.csv: mmm_surrogate.py not run."
elif [[ ! -f "$SURROGATE" ]]; then
    echo "mmm_surrogate.py not found ($SURROGATE): set SURROGATE=<path> to write the tables of s_hat."
else
    echo "mmm_surrogate.py on ${#READY[@]} unit runs (log: $LHS/mmm_surrogate.log)"
    if "$PYTHON" "$SURROGATE" "${READY[@]}" > "$LHS/mmm_surrogate.log" 2>&1; then
        grep "^wrote" "$LHS/mmm_surrogate.log" | sed 's/^/    /'
    else
        echo "    mmm_surrogate.py: exit $? ($(tail -n 1 "$LHS/mmm_surrogate.log")), see $LHS/mmm_surrogate.log"
        FAILED+=("mmm_surrogate.py")
    fi
fi

[[ ${#FAILED[@]} -eq 0 ]]
