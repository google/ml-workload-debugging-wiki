#!/usr/bin/env bash
# ==============================================================================
# scaffold_bucket.sh - Provisions multi-job diagnostic incident tree on GCS
#
# Supports dynamic bucket parameterization via --bucket=<bucket_name>.
# The positional key is the STORAGE KEY (normally the support case number, i.e.
# identity.storage_key in troubleshoot_findings.json).
#
# Only BUCKET_STRUCTURE.md and link files are written.
# Empty directories and folder markers (.keep) are never created: directories
# in GCS are created strictly on-demand as real files are uploaded by
# package-handoff-artifacts and upload-handoff-package.
# Manifests, READMEs and failure tables are never pre-created: an empty
# placeholder would sit in the bucket until something overwrote it. They are
# published by upload-handoff-package once they have real content.
# ==============================================================================
set -euo pipefail

BUG_ID=""
JOB_NAME=""
BUCKET_NAME="${HANDOFF_BUCKET:-}"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bucket=*)
            BUCKET_NAME="${1#*=}"
            BUCKET_NAME="${BUCKET_NAME#gs://}"
            BUCKET_NAME="${BUCKET_NAME%/}"
            shift
            ;;
        -b|--bucket)
            BUCKET_NAME="$2"
            BUCKET_NAME="${BUCKET_NAME#gs://}"
            BUCKET_NAME="${BUCKET_NAME%/}"
            shift 2
            ;;
        --storage_key=*)
            BUG_ID="${1#*=}"
            shift
            ;;
        -k|--storage_key)
            BUG_ID="$2"
            shift 2
            ;;
        --job=*)
            JOB_NAME="${1#*=}"
            shift
            ;;
        -j|--job)
            JOB_NAME="$2"
            shift 2
            ;;
        --dry_run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: ./scaffold_bucket.sh <StorageKey> [--bucket=<bucket_name>] [--job=<job_name>] [--dry_run]"
            echo "Examples:"
            echo "  ./scaffold_bucket.sh <StorageKey> --bucket=<name>  # Scaffolds in customer-specified bucket"
            echo "  ./scaffold_bucket.sh <StorageKey> --job=<job_name> # Logs isolated job workspace path (created on-demand on upload)"
            echo "  ./scaffold_bucket.sh <StorageKey> --dry_run        # Tests scaffolding without writing to GCS"
            exit 0
            ;;
        *)
            if [[ -z "$BUG_ID" && "$1" != -* ]]; then
                BUG_ID="$1"
            fi
            shift
            ;;
    esac
done

if [[ -z "$BUG_ID" ]]; then
    echo "[ERROR] Missing or invalid storage key. A storage key (support case number or key) is required."
    echo "Usage: ./scaffold_bucket.sh <StorageKey> [--bucket=<bucket_name>] [--job=<job_name>] [--dry_run]"
    exit 1
fi

if [[ -z "$BUCKET_NAME" ]]; then
    echo "[ERROR] No destination bucket. Pass --bucket=<name> or export HANDOFF_BUCKET; ask the operator for the customer project's bucket."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INCIDENT_PATH="gs://${BUCKET_NAME}/${BUG_ID}"

TMP_DIR="$(TMPDIR="${TMPDIR:-/tmp}" mktemp -d -t workload_scaffold_XXXXXX)"
trap 'rm -rf "${TMP_DIR}"' EXIT

# Generate BUCKET_STRUCTURE.md rendered for this incident
STRUCTURE_TEMPLATE="${SCRIPT_DIR}/templates/BUCKET_STRUCTURE.template.md"
if [[ ! -s "${STRUCTURE_TEMPLATE}" ]]; then
    echo "[ERROR] Missing or empty template: ${STRUCTURE_TEMPLATE}"
    exit 1
fi
sed -e "s/{{BUCKET_NAME}}/${BUCKET_NAME}/g" -e "s/{{STORAGE_KEY}}/${BUG_ID}/g" "${STRUCTURE_TEMPLATE}" > "${TMP_DIR}/BUCKET_STRUCTURE.md"

gcs_upload_file() {
    local src="$1"
    local target="$2"
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] gcloud storage cp ${src} ${target}"
    else
        gcloud storage cp "${src}" "${target}"
    fi
}

echo "[INFO] Setting up multi-job incident repository for storage key ${BUG_ID} in gs://${BUCKET_NAME}..."
echo "[INFO] Target root: ${INCIDENT_PATH}"

echo "[INFO] Scaffolding incident structure dictionary..."
gcs_upload_file "${TMP_DIR}/BUCKET_STRUCTURE.md" "${INCIDENT_PATH}/BUCKET_STRUCTURE.md"

echo "[INFO] Scaffolding shared links and deep-linked shortcut pointers..."
# The storage key is the support case; the partner issue id and the GCP project
# are different identifiers, so no links are guessed for them here.
echo "https://console.cloud.google.com/support/cases/detail/${BUG_ID}" > "${TMP_DIR}/support_case.url"
gcs_upload_file "${TMP_DIR}/support_case.url" "${INCIDENT_PATH}/links/support_case.url"

echo "[OK] Multi-job incident repository successfully scaffolded at ${INCIDENT_PATH}"

if [[ -n "$JOB_NAME" ]]; then
    JOB_PATH="${INCIDENT_PATH}/jobs/${JOB_NAME}"
    echo "[INFO] Adding isolated job workspace for '${JOB_NAME}' under storage key ${BUG_ID} in gs://${BUCKET_NAME}..."
    echo "[INFO] Job root: ${JOB_PATH}"
    echo "[INFO] Directories will be created on-demand as real files are uploaded."
    echo "[OK] Job workspace successfully scaffolded at ${JOB_PATH}"
else
    echo "[INFO] Job-specific subdirectories under ${INCIDENT_PATH}/jobs/<job_name>/ are created on-demand by package-handoff-artifacts and upload-handoff-package."
fi
