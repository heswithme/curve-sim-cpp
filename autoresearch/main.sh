#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  autoresearch/main.sh

Environment overrides:
  CODEX_MODEL=gpt-5.4
  CODEX_REASONING_EFFORT=xhigh
  CODEX_SANDBOX=workspace-write
  MAX_ITERATIONS=0              # 0 means loop forever
  SLEEP_SECONDS=0
  ALLOW_DIRTY_TRACKED=0         # set to 1 to bypass tracked-worktree cleanliness check
  RUN_DIR=autoresearch/codex-loop

Behavior:
  - runs one codex-cli research iteration at a time
  - uses autoresearch/program.md as the main prompt body
  - stores per-iteration prompts, transcripts, and last messages under RUN_DIR
  - stops on the first nonzero codex exit code
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PROGRAM_PATH="${PROGRAM_PATH:-${SCRIPT_DIR}/program.md}"
OBSERVATIONS_PATH="${OBSERVATIONS_PATH:-${SCRIPT_DIR}/observations.md}"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/codex-loop}"
CODEX_MODEL="${CODEX_MODEL:-gpt-5.4}"
CODEX_REASONING_EFFORT="${CODEX_REASONING_EFFORT:-xhigh}"
CODEX_SANDBOX="${CODEX_SANDBOX:-workspace-write}"
MAX_ITERATIONS="${MAX_ITERATIONS:-0}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0}"
ALLOW_DIRTY_TRACKED="${ALLOW_DIRTY_TRACKED:-0}"

if [[ ! -f "${PROGRAM_PATH}" ]]; then
  echo "program.md not found at ${PROGRAM_PATH}" >&2
  exit 1
fi

if [[ ! -f "${OBSERVATIONS_PATH}" ]]; then
  echo "observations.md not found at ${OBSERVATIONS_PATH}" >&2
  exit 1
fi

if ! command -v codex >/dev/null 2>&1; then
  echo "codex CLI not found on PATH" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}"

cd "${REPO_ROOT}"

if [[ "${ALLOW_DIRTY_TRACKED}" != "1" ]]; then
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Tracked worktree is dirty. Commit/stash/revert tracked changes or set ALLOW_DIRTY_TRACKED=1." >&2
    git status --short >&2
    exit 1
  fi
fi

loop_log="${RUN_DIR}/loop.tsv"
if [[ ! -f "${loop_log}" ]]; then
  printf "timestamp\titeration\texit_code\ttranscript\tlast_message\n" > "${loop_log}"
fi

build_iteration_prompt() {
  cat "${PROGRAM_PATH}"
  cat <<'EOF'

Operational override for this codex exec invocation:

- This invocation is exactly one iteration of the outer bash loop.
- Perform at most one baseline-or-candidate experiment iteration, then stop.
- Do not start a second experiment in this same Codex session.
- The outer bash loop will invoke you again if needed.

Execution requirements for this one iteration:

- Follow program.md as the main policy.
- Inspect `autoresearch/fee_autoresearch_runs.jsonl` if it exists.
- Inspect the current best baseline/candidate result files if they exist.
- Inspect `autoresearch/observations.md` and update it when an iteration yields useful qualitative insight.
- Use git only, not jj.
- If the tracked worktree is dirty for reasons unrelated to the experiment you are about to run, do not proceed.
- At the end, print a short human-readable summary with:
  - decision
  - idea name
  - edited files
  - benchmark command
  - result path
  - best_seen.loss if available
  - apy_net if available
  - avg_rel_price_diff if available
EOF
}

iteration=0
trap 'echo; echo "Interrupted. Loop stopped after iteration ${iteration}."' INT TERM

while :; do
  iteration=$((iteration + 1))
  if [[ "${MAX_ITERATIONS}" -gt 0 && "${iteration}" -gt "${MAX_ITERATIONS}" ]]; then
    echo "Reached MAX_ITERATIONS=${MAX_ITERATIONS}. Stopping."
    break
  fi

  ts="$(date -u +"%Y%m%dT%H%M%SZ")"
  iter_dir="${RUN_DIR}/${ts}-iter$(printf "%06d" "${iteration}")"
  mkdir -p "${iter_dir}"

  prompt_file="${iter_dir}/prompt.md"
  transcript_file="${iter_dir}/transcript.log"
  last_message_file="${iter_dir}/last-message.md"

  build_iteration_prompt > "${prompt_file}"

  echo "================================================================"
  echo "Iteration ${iteration} | ${ts}"
  echo "Prompt:       ${prompt_file}"
  echo "Transcript:   ${transcript_file}"
  echo "Last message: ${last_message_file}"
  echo "Model:        ${CODEX_MODEL}"
  echo "Reasoning:    ${CODEX_REASONING_EFFORT}"
  echo "Sandbox:      ${CODEX_SANDBOX}"
  echo "================================================================"

  set +e
  codex exec \
    --model "${CODEX_MODEL}" \
    -c "model_reasoning_effort=\"${CODEX_REASONING_EFFORT}\"" \
    --sandbox "${CODEX_SANDBOX}" \
    --full-auto \
    --color never \
    --output-last-message "${last_message_file}" \
    -C "${REPO_ROOT}" \
    - < "${prompt_file}" | tee "${transcript_file}"
  exit_code=${PIPESTATUS[0]}
  set -e

  printf "%s\t%s\t%s\t%s\t%s\n" \
    "${ts}" \
    "${iteration}" \
    "${exit_code}" \
    "${transcript_file}" \
    "${last_message_file}" >> "${loop_log}"

  if [[ "${exit_code}" -ne 0 ]]; then
    echo "codex exec exited with code ${exit_code}. Stopping loop." >&2
    exit "${exit_code}"
  fi

  if [[ "${SLEEP_SECONDS}" -gt 0 ]]; then
    sleep "${SLEEP_SECONDS}"
  fi
done
