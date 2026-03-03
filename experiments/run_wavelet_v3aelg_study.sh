#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# WaveletV3AELG Successive-Halving Study Orchestrator
#
# Phases:
#   1  TrendAELG search   — 3-round successive halving (m4, tourism, traffic, weather)
#   2  TrendAE search     — same with TrendAE trend blocks
#   3  TrendAELG cross    — cross-dataset benchmark from round-3 results
#   4  TrendAE cross      — same for TrendAE
#   5  Analysis           — execute Jupyter notebook, emit markdown report
#
# Usage:
#   ./experiments/run_wavelet_v3aelg_study.sh                          # all phases
#   ./experiments/run_wavelet_v3aelg_study.sh --phase 1                # single phase
#   ./experiments/run_wavelet_v3aelg_study.sh --phase 1,2              # multiple
#   ./experiments/run_wavelet_v3aelg_study.sh --phase search           # alias for 1,2
#   ./experiments/run_wavelet_v3aelg_study.sh --phase cross            # alias for 3,4
#   ./experiments/run_wavelet_v3aelg_study.sh --phase analysis         # alias for 5
#   ./experiments/run_wavelet_v3aelg_study.sh --dry-run                # print commands
#   ./experiments/run_wavelet_v3aelg_study.sh --accelerator cuda --no-wandb
# =============================================================================

# ---------------------------------------------------------------------------
# 1. Path resolution
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIGS_DIR="$SCRIPT_DIR/configs"
RESULTS_DIR="$SCRIPT_DIR/results"
RUNNER="$SCRIPT_DIR/run_wavelet_v3ae_study.py"
NOTEBOOK="$SCRIPT_DIR/analysis/wavelet_v3aelg_study_analysis.ipynb"

# Python / Jupyter resolution
if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo "ERROR: No Python interpreter found." >&2
    exit 1
fi

if [[ -x "$PROJECT_ROOT/.venv/bin/jupyter" ]]; then
    JUPYTER="$PROJECT_ROOT/.venv/bin/jupyter"
elif command -v jupyter &>/dev/null; then
    JUPYTER="jupyter"
else
    JUPYTER=""
fi

# Sanity checks
if [[ ! -f "$RUNNER" ]]; then
    echo "ERROR: Python runner not found at $RUNNER" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Config arrays
# ---------------------------------------------------------------------------
TRENDAELG_CONFIGS=(
    "$CONFIGS_DIR/wavelet_v3aelg_trendaelg_m4.yaml"
    "$CONFIGS_DIR/wavelet_v3aelg_trendaelg_tourism.yaml"
    "$CONFIGS_DIR/wavelet_v3aelg_trendaelg_traffic.yaml"
    "$CONFIGS_DIR/wavelet_v3aelg_trendaelg_weather.yaml"
)
TRENDAE_CONFIGS=(
    "$CONFIGS_DIR/wavelet_v3aelg_trendae_m4.yaml"
    "$CONFIGS_DIR/wavelet_v3aelg_trendae_tourism.yaml"
    "$CONFIGS_DIR/wavelet_v3aelg_trendae_traffic.yaml"
    "$CONFIGS_DIR/wavelet_v3aelg_trendae_weather.yaml"
)

# ---------------------------------------------------------------------------
# 3. Argument parsing
# ---------------------------------------------------------------------------
declare -A PHASES
DRY_RUN=0
PASSTHROUGH_ARGS=()
PHASE_SPECIFIED=0

usage() {
    cat <<'USAGE'
Usage: run_wavelet_v3aelg_study.sh [OPTIONS]

Options:
  --phase SPEC        Phase(s) to run. Comma-separated numbers (1-5) or aliases:
                        search   = 1,2   (successive-halving search)
                        cross    = 3,4   (cross-dataset benchmark)
                        analysis = 5     (Jupyter notebook)
                        all      = 1-5   (default)
  --dry-run           Print commands without executing
  --help              Show this help message

Pass-through flags (forwarded to the Python runner):
  --accelerator VALUE   auto|cuda|mps|cpu
  --n-gpus N            Number of GPUs
  --batch-size N        Override batch size
  --num-workers N       DataLoader workers
  --wandb               Enable W&B logging
  --no-wandb            Disable W&B logging
  --wandb-project NAME  W&B project name
USAGE
    exit 0
}

enable_phases() {
    local spec="$1"
    for p in ${spec//,/ }; do
        PHASES[$p]=1
    done
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)
            PHASE_SPECIFIED=1
            shift
            case "$1" in
                search)   enable_phases "1,2" ;;
                cross)    enable_phases "3,4" ;;
                analysis) enable_phases "5" ;;
                all)      enable_phases "1,2,3,4,5" ;;
                *)        enable_phases "$1" ;;
            esac
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            usage
            ;;
        --accelerator|--n-gpus|--batch-size|--num-workers|--wandb-project)
            PASSTHROUGH_ARGS+=("$1" "$2")
            shift 2
            ;;
        --wandb|--no-wandb)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
        *)
            echo "WARNING: Unknown argument '$1' — passing through to runner" >&2
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

# Default: all phases
if [[ $PHASE_SPECIFIED -eq 0 ]]; then
    enable_phases "1,2,3,4,5"
fi

# ---------------------------------------------------------------------------
# 4. Logging setup
# ---------------------------------------------------------------------------
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$RESULTS_DIR"
LOGFILE="$RESULTS_DIR/wavelet_v3aelg_study_${TIMESTAMP}.log"
exec > >(tee -a "$LOGFILE") 2>&1

# ---------------------------------------------------------------------------
# 5. Utility functions
# ---------------------------------------------------------------------------
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_cmd() {
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] $*"
    else
        log "Running: $*"
        "$@"
    fi
}

phase_enabled() {
    [[ -v "PHASES[$1]" ]]
}

# ---------------------------------------------------------------------------
# 6. Phase functions
# ---------------------------------------------------------------------------

# run_search_phase <group_label> <nameref to config array>
# Returns the number of failed datasets.
run_search_phase() {
    local group_label="$1"
    local -n configs_ref="$2"
    local failures=0

    log "=== Search phase: $group_label ==="

    for cfg in "${configs_ref[@]}"; do
        local dataset_name
        dataset_name="$(basename "$cfg" .yaml | sed 's/wavelet_v3aelg_[a-z]*_//')"
        log "--- $group_label / $dataset_name ---"

        if ! run_cmd "$PYTHON" "$RUNNER" search --config "$cfg" --no-tmp-log "${PASSTHROUGH_ARGS[@]}"; then
            log "ERROR: $group_label / $dataset_name failed (continuing)"
            ((failures++)) || true
        fi
    done

    log "=== Search phase $group_label complete ($failures failure(s)) ==="
    return "$failures"
}

# run_cross_phase <group_label> <nameref to config array>
run_cross_phase() {
    local group_label="$1"
    local -n configs_ref="$2"

    log "=== Cross-dataset phase: $group_label ==="
    run_cmd "$PYTHON" "$RUNNER" cross --configs "${configs_ref[@]}" --no-tmp-log "${PASSTHROUGH_ARGS[@]}"
    log "=== Cross-dataset phase $group_label complete ==="
}

# run_analysis
run_analysis() {
    log "=== Analysis phase ==="

    if [[ -z "$JUPYTER" ]]; then
        log "ERROR: jupyter not found — skipping analysis"
        return 1
    fi
    if [[ ! -f "$NOTEBOOK" ]]; then
        log "ERROR: Notebook not found at $NOTEBOOK — skipping analysis"
        return 1
    fi

    run_cmd "$JUPYTER" nbconvert \
        --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=600 \
        "$NOTEBOOK"

    log "Analysis notebook executed: $NOTEBOOK"
    log "=== Analysis phase complete ==="
}

# ---------------------------------------------------------------------------
# 7. Main flow
# ---------------------------------------------------------------------------
TOTAL_ERRORS=0

log "============================================================"
log "WaveletV3AELG Study Orchestrator"
log "Enabled phases: ${!PHASES[*]}"
log "Log file: $LOGFILE"
log "Python: $PYTHON"
log "Pass-through args: ${PASSTHROUGH_ARGS[*]:-<none>}"
if [[ $DRY_RUN -eq 1 ]]; then
    log "*** DRY-RUN MODE — no commands will be executed ***"
fi
log "============================================================"

# Phase 1: TrendAELG search
if phase_enabled 1; then
    if ! run_search_phase "TrendAELG" TRENDAELG_CONFIGS; then
        ((TOTAL_ERRORS += $?)) || true
    fi
fi

# Phase 2: TrendAE search
if phase_enabled 2; then
    if ! run_search_phase "TrendAE" TRENDAE_CONFIGS; then
        ((TOTAL_ERRORS += $?)) || true
    fi
fi

# Phase 3: TrendAELG cross-dataset
if phase_enabled 3; then
    if ! run_cross_phase "TrendAELG" TRENDAELG_CONFIGS; then
        ((TOTAL_ERRORS++)) || true
    fi
fi

# Phase 4: TrendAE cross-dataset
if phase_enabled 4; then
    if ! run_cross_phase "TrendAE" TRENDAE_CONFIGS; then
        ((TOTAL_ERRORS++)) || true
    fi
fi

# Phase 5: Analysis
if phase_enabled 5; then
    if ! run_analysis; then
        ((TOTAL_ERRORS++)) || true
    fi
fi

# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------
log "============================================================"
if [[ $TOTAL_ERRORS -eq 0 ]]; then
    log "All phases completed successfully."
else
    log "Completed with $TOTAL_ERRORS error(s). Check log: $LOGFILE"
fi
log "============================================================"

exit "$TOTAL_ERRORS"
