#!/usr/bin/env bash
set -euo pipefail

llama_server="${LLAMA_SERVER:-${HOME}/llama.cpp/build/bin/llama-server}"
model_dir="${TIEL_MODEL_DIR:-${HOME}/models/tiel-coder-mtp}"
model_file="${model_dir}/Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf"
mmproj_file="${model_dir}/mmproj-BF16.gguf"
host="${TIEL_HOST:-0.0.0.0}"
port="${TIEL_PORT:-30000}"
parallel="${TIEL_PARALLEL:-4}"
context="${TIEL_CONTEXT:-262144}"
draft_n="${TIEL_DRAFT_N:-1}"
draft_p_min="${TIEL_DRAFT_P_MIN:-0}"

if [[ ! -x "${llama_server}" ]]; then
  echo "llama-server is not executable: ${llama_server}" >&2
  exit 1
fi

for required_file in "${model_file}" "${mmproj_file}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "required model file is missing: ${required_file}" >&2
    exit 1
  fi
done

for numeric_value in "${port}" "${parallel}" "${context}" "${draft_n}"; do
  if [[ ! "${numeric_value}" =~ ^[0-9]+$ ]]; then
    echo "expected a non-negative integer, got: ${numeric_value}" >&2
    exit 1
  fi
done

exec "${llama_server}" \
  --model "${model_file}" \
  --mmproj "${mmproj_file}" \
  --host "${host}" \
  --port "${port}" \
  --n-gpu-layers 99 \
  --jinja \
  --parallel "${parallel}" \
  --kv-unified-per-slot "${context}" \
  --cache-prompt \
  --chat-template-kwargs '{"enable_thinking":false,"preserve_thinking":false}' \
  --spec-type draft-mtp \
  --spec-draft-n-max "${draft_n}" \
  --spec-draft-p-min "${draft_p_min}"
