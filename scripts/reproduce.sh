#!/usr/bin/env sh
# One command from a fresh clone to a verified reproduction of the article.
#
#   scripts/reproduce.sh              # everything, including the PyTorch stages
#   scripts/reproduce.sh --skip-ml    # without the synthetic and real-window model stages
#
# Needs uv (https://docs.astral.sh/uv/), network access and about 4 GB of disk.
set -eu
cd "$(dirname "$0")/.."
uv sync --locked
exec uv run ball-request-lab all "$@"
