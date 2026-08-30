#!/usr/bin/env bash
# Candidate acceptance report (Umiusi_sim#3): eval battery (caps 0.2/0.25/0.4) + FF transfer
# check, condensed to the acceptance table. Usage:
#   tools/candidate_report.sh models/av_mode10 [episodes]
# Writes the full logs to <run>/eval_$(date +%Y%m%d).log and prints the summary.
set -o pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$1"; EP="${2:-20}"
[ -d "$S/$RUN" ] || { echo "no such run dir: $RUN"; exit 1; }
L="$S/$RUN/eval_$(date -u +%Y%m%d-%H%M).log"
: > "$L"
for md in 0.2 0.25 0.4; do
  echo "===== max_duty=$md =====" >> "$L"
  (cd "$S" && uv run python -m umiusi_rl.eval --model "$RUN/final.zip" --episodes "$EP" --max-duty "$md") >> "$L" 2>&1
done
echo "===== ff_transfer =====" >> "$L"
(cd "$S" && uv run python -m tools.ff_transfer --model "$RUN/final.zip" --max-duty 0.25) >> "$L" 2>&1
echo "report: $L"
grep -a "=====\|cruise vs\|null share (pw)\|ori err\|roll authority\|mode rate\|cmd 0\|monotonic\|spread\|along cmd" "$L" | grep -v "^ep"
