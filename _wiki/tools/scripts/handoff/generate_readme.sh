#!/usr/bin/env bash
# ==============================================================================
# generate_readme.sh - Generates INCIDENT_OVERVIEW.md, JOB_README.md, or README.md
#
# Supports dynamic bucket parameterization via --bucket=<bucket_name>.
# The ONLY data source is --findings=<troubleshoot_findings.json>.
# ==============================================================================
set -euo pipefail

MODE="job"
FINDINGS=""
OUTPUT_FILE=""
BUCKET_NAME="${HANDOFF_BUCKET:-}"
UPLOAD=false
DRY_RUN=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --mode=*) MODE="${1#*=}"; shift ;;
        --mode) MODE="$2"; shift 2 ;;
        --findings=*) FINDINGS="${1#*=}"; shift ;;
        --findings) FINDINGS="$2"; shift 2 ;;
        --bucket=*)
            BUCKET_NAME="${1#*=}"
            BUCKET_NAME="${BUCKET_NAME#gs://}"
            BUCKET_NAME="${BUCKET_NAME%/}"
            shift ;;
        --bucket)
            BUCKET_NAME="$2"
            BUCKET_NAME="${BUCKET_NAME#gs://}"
            BUCKET_NAME="${BUCKET_NAME%/}"
            shift 2 ;;
        --output_file=*) OUTPUT_FILE="${1#*=}"; shift ;;
        --output_file) OUTPUT_FILE="$2"; shift 2 ;;
        --upload) UPLOAD=true; shift ;;
        --dry_run) DRY_RUN=true; shift ;;
        --help|-h)
            echo "Usage: ./generate_readme.sh --mode=[incident|job|readme] --findings=<troubleshoot_findings.json> [--bucket=<name>] [--output_file=<path>] [--upload] [--dry_run]"
            exit 0
            ;;
        *) echo "[ERROR] Unknown parameter: $1"; exit 1 ;;
    esac
done

if [[ -z "$FINDINGS" ]]; then
    echo "[ERROR] Missing required parameter --findings"
    exit 1
fi
if [[ ! -f "$FINDINGS" ]]; then
    echo "[ERROR] Findings file not found: ${FINDINGS}"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH_DIR="${TMPDIR:-/tmp}"
FIO="${SCRIPT_DIR}/findings_io.py"

STORAGE_KEY="$(python3 "$FIO" get --findings="$FINDINGS" --key=identity.storage_key)"
WORKLOAD="$(python3 "$FIO" get --findings="$FINDINGS" --key=identity.workload)"
RESOLVED_BUCKET="$(python3 "$FIO" bucket --findings="$FINDINGS" --bucket="$BUCKET_NAME")"
if [[ "$RESOLVED_BUCKET" == "NOT PROVIDED" ]]; then
    echo "[ERROR] No destination bucket. Pass --bucket=<name>, set identity.bucket, or export HANDOFF_BUCKET."
    exit 1
fi
KEY_SLUG="${STORAGE_KEY// /_}"
WORKLOAD_SLUG="${WORKLOAD// /_}"

if [[ "$MODE" == "incident" ]]; then
    DEFAULT_OUT="${SCRATCH_DIR}/INCIDENT_OVERVIEW_${KEY_SLUG}.md"
    GCS_REL="INCIDENT_OVERVIEW.md"
elif [[ "$MODE" == "readme" || "$MODE" == "root" ]]; then
    MODE="readme"
    DEFAULT_OUT="${SCRATCH_DIR}/README_${KEY_SLUG}.md"
    GCS_REL="README.md"
else
    MODE="job"
    DEFAULT_OUT="${SCRATCH_DIR}/JOB_README_${KEY_SLUG}_${WORKLOAD_SLUG}.md"
    GCS_REL="jobs/${WORKLOAD}/JOB_README.md"
fi

DEST_FILE="${OUTPUT_FILE:-$DEFAULT_OUT}"
mkdir -p "$(dirname "$DEST_FILE")"

python3 "${SCRIPT_DIR}/render_readme.py" \
  --findings="$FINDINGS" \
  --mode="$MODE" \
  --bucket="$RESOLVED_BUCKET" \
  --output_file="$DEST_FILE"

echo "[OK] Generated ${MODE} overview at ${DEST_FILE} (bucket: gs://${RESOLVED_BUCKET})"

if [[ "$UPLOAD" == true ]]; then
    if [[ "$STORAGE_KEY" == "NOT PROVIDED" ]]; then
        echo "[ERROR] identity.storage_key is missing in ${FINDINGS}; refusing to upload."
        exit 1
    fi
    if [[ "$MODE" == "job" && "$WORKLOAD" == "NOT PROVIDED" ]]; then
        echo "[ERROR] identity.workload is missing in ${FINDINGS}; refusing to upload a job README."
        exit 1
    fi
    GCS_TARGET="gs://${RESOLVED_BUCKET}/${STORAGE_KEY}/${GCS_REL}"
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] gcloud storage cp ${DEST_FILE} ${GCS_TARGET}"
    else
        echo "[INFO] Uploading to ${GCS_TARGET}..."
        gcloud storage cp "$DEST_FILE" "$GCS_TARGET"
        echo "[OK] Successfully uploaded to ${GCS_TARGET}"
    fi
fi
