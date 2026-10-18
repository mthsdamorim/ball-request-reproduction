#!/usr/bin/env sh
# Download the pinned public inputs (about 1.4 GB) into ./data and verify each
# file by size and SHA-256. Safe to re-run: verified files are skipped and
# interrupted downloads resume.
#
#   scripts/download_data.sh                 # both matches
#   scripts/download_data.sh --matches 1996435   # only the examined match
set -eu
cd "$(dirname "$0")/.."
exec uv run ball-request-lab "$@" download
