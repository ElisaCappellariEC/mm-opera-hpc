#!/bin/bash
# Analysis of the runs of a Latin hypercube solved by MMM: break_order.py on every run folder, then the surrogates of
# s_hat over the runs for SCIANTIX (mmm_surrogate.py --fit).
#
#   bash analyse_lhs.sh LHS
#
# LHS  tree of the hypercube, with the results of MMM in its cubes:
#          LHS/unit/run_<id>/side<L>_seed<NN>/   bubbles_unit_<id>.txt, summary_unit_<id>.txt (Mérope),
#                                                bubbles_and_stresses_selected.txt, sig_hydro.txt (MMM)
#          LHS/pk/run_<id>/side<L>_seed<NN>/     the same with the class pressures, and mesh_case.txt (set_pressures.py)
# Per run folder (unit first, then pk; a pk run needs the unit run of the same cube, solved):
#     break_order.py     pores.csv, break_order.txt/png; the output of the script goes to break_order.log
#     plot_bubbles.py    if PLOT=1: rve_size.txt, rve_fgr.txt/png (plot_bubbles.log). Its exit 1 also means
#                        "not an RVE" or "mesh not fine", which says little with one cube per run: read the log
# A run with no cube solved is skipped; a run already analysed (pores.csv newer than the results of MMM) is skipped
# unless FORCE=1.
# Then mmm_surrogate.py --fit (SURROGATE; skipped if the file is not there) on the pores.csv of the runs and of NFIR_RUN
# (the reference run, kept as the test of the surrogates; left out if it has no pores.csv): it writes next to itself
# mmm_outputs/surrogate_unit.txt/png, surrogate_pk.txt/png and surrogate_<load>_sciantix.txt (input_hbs_shat.txt).
# author: E.Cappellari, POLIMI-CEA 2026

set -o pipefail

PLOT=0      # 1: also plot_bubbles.py on every run
FORCE=0     # 1: redo the runs already analysed
SURROGATE=$HOME/mmm/sciantix-official/utilities/HBFF/mmm_surrogate.py
NFIR_RUN=$HOME/mmm/results/test_rve_side30_mesh0.15_graded_rfine0.45_delta0.15/run_NFIR
PYTHON=python3
STRESSES="bubbles_and_stresses_selected.txt"

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
for f in break_order.py plot_bubbles.py; do
    if [[ ! -f "$HERE/$f" ]]; then
        echo "ERROR: $HERE/$f not found: this script runs next to the Python scripts"
        exit 1
    fi
done
if [[ $# -ne 1 || ! -d "$1" ]]; then
    echo "Usage: bash analyse_lhs.sh LHS   (LHS: tree of the hypercube, with unit/ and pk/)"
    exit 1
fi
LHS=$(cd "$1" && pwd)

# the run folders: unit first, then the others (pk)
mapfile -t UNIT < <(find "$LHS" -type d -name "run_*" -path "*/unit/*" | sort)
mapfile -t OTHER < <(find "$LHS" -type d -name "run_*" ! -path "*/unit/*" | sort)
RUNS=("${UNIT[@]}" "${OTHER[@]}")
echo "Found ${#RUNS[@]} run folders in $LHS (${#UNIT[@]} unit)."

# Runs the script $1 on the run folder $2, output in $2/<name of the script>.log; prints its last line if it fails
analyse() {
    local script=$1 run=$2 log status
    log="$run/${script%.py}.log"
    "$PYTHON" "$HERE/$script" "$run" > "$log" 2>&1
    status=$?
    if [[ $status -ne 0 ]]; then
        echo "    $script: exit $status ($(tail -n 1 "$log")), see $log"
    fi
    return $status
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

# Surrogates of s_hat over the runs (--fit reads pores.csv of every run folder below LHS and of NFIR_RUN; the script
# tells the load of a run from the pressures of its pores.csv)
READY=()
for RUN in "${RUNS[@]}"; do
    [[ -s "$RUN/pores.csv" ]] && READY+=("$RUN")
done
if [[ ${#READY[@]} -eq 0 ]]; then
    echo "No run with pores.csv: mmm_surrogate.py not run."
elif [[ ! -f "$SURROGATE" ]]; then
    echo "mmm_surrogate.py not found ($SURROGATE): set SURROGATE at the top of analyse_lhs.sh to fit the surrogates."
else
    FOLDERS=("$LHS")
    [[ -s "$NFIR_RUN/pores.csv" ]] && FOLDERS+=("$NFIR_RUN")
    echo "mmm_surrogate.py --fit on ${#READY[@]} runs and $((${#FOLDERS[@]} - 1)) reference run (log: $LHS/mmm_surrogate_fit.log)"
    "$PYTHON" "$SURROGATE" --fit "${FOLDERS[@]}" > "$LHS/mmm_surrogate_fit.log" 2>&1
    STATUS=$?
    if [[ $STATUS -eq 0 ]]; then
        grep "^wrote" "$LHS/mmm_surrogate_fit.log" | sed 's/^/    /'
    else
        echo "    mmm_surrogate.py --fit: exit $STATUS ($(tail -n 1 "$LHS/mmm_surrogate_fit.log")), see $LHS/mmm_surrogate_fit.log"
        FAILED+=("mmm_surrogate.py --fit")
    fi
fi

[[ ${#FAILED[@]} -eq 0 ]]
