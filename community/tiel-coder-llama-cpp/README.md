# Run Tiel-Coder 35B-A3B with native MTP on one DGX Spark

This recipe adapts NVIDIA's [llama.cpp DGX Spark playbook](../../nvidia/llama-cpp/)
for [`peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP`](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP).
It serves the recommended 23.3 GB `UD-Q4_K_XL` quant on a single GB10, enables
the retained MTP/NextN head, disables thinking by default, and exposes four
concurrent llama.cpp slots.

This is a candidate DGX Spark configuration derived from the NVIDIA playbook
and the model author's settings. It has not yet been performance-qualified on
DGX Spark. Benchmark it before replacing an existing service.

## Configuration

| Setting | Value | Reason |
|---|---:|---|
| Quant | `UD-Q4_K_XL` | Model author's recommended starting tier |
| GPU offload | all layers (`-ngl 99`) | Keep generation on GB10 |
| Context | 262,144 tokens per slot | Native long-context target |
| Parallel slots | 4 | First concurrency target for a clean benchmark |
| Thinking | off | Avoid hidden reasoning-token overhead for agent workloads |
| Speculation | native MTP, one-token draft | Fastest setting reported by the model author |
| Port | `30000` | Matches the NVIDIA llama.cpp playbook |

The model card says one 262K BF16 KV cache uses less than 5 GB. Four full-depth
slots therefore need substantially more memory than a single-slot run. If the
server cannot allocate the requested KV pool, start with `TIEL_CONTEXT=131072`
or `TIEL_PARALLEL=1` and increase one dimension at a time.

## 1. Build llama.cpp for GB10

```bash
sudo apt update
sudo apt install -y git clang cmake libcurl4-openssl-dev libssl-dev

git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cd ~/llama.cpp
cmake -B build \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA=ON \
  -DGGML_CURL=ON \
  -DGGML_RPC=ON \
  -DCMAKE_CUDA_ARCHITECTURES=121a-real
cmake --build build --config Release --target llama-server -j
```

Use a current llama.cpp checkout. The recipe depends on `draft-mtp`,
`--kv-unified-per-slot`, and JSON chat-template arguments.

## 2. Download the model and vision projector

```bash
mkdir -p ~/models/tiel-coder-mtp
hf download peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP \
  Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf \
  mmproj-BF16.gguf \
  --local-dir ~/models/tiel-coder-mtp
```

The MTP checkpoint is intentional. Do not use it without `--spec-type
draft-mtp`; otherwise the retained 0.9 GB prediction head is loaded but unused.

## 3. Start the server

The supplied launcher checks the required files and arguments, then replaces
itself with `llama-server`:

```bash
cd community/tiel-coder-llama-cpp
./start-server.sh
```

Equivalent command:

```bash
~/llama.cpp/build/bin/llama-server \
  --model ~/models/tiel-coder-mtp/Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf \
  --mmproj ~/models/tiel-coder-mtp/mmproj-BF16.gguf \
  --host 0.0.0.0 \
  --port 30000 \
  --n-gpu-layers 99 \
  --jinja \
  --parallel 4 \
  --kv-unified-per-slot 262144 \
  --cache-prompt \
  --chat-template-kwargs '{"enable_thinking":false,"preserve_thinking":false}' \
  --spec-type draft-mtp \
  --spec-draft-n-max 1 \
  --spec-draft-p-min 0
```

`--kv-unified-per-slot` is used instead of dividing a single `--ctx-size`
across four server slots. The server must print `server is listening` before a
client or agent is resumed.

### Overrides

The launcher accepts environment overrides without editing the file:

```bash
TIEL_PORT=30001 TIEL_PARALLEL=1 TIEL_CONTEXT=262144 ./start-server.sh
```

Available overrides are `LLAMA_SERVER`, `TIEL_MODEL_DIR`, `TIEL_HOST`,
`TIEL_PORT`, `TIEL_PARALLEL`, `TIEL_CONTEXT`, `TIEL_DRAFT_N`, and
`TIEL_DRAFT_P_MIN`.

## 4. Verify readiness and generation

```bash
timeout 900 bash -c \
  'until curl -sf http://127.0.0.1:30000/health >/dev/null; do sleep 5; done'

curl -sS http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tiel-coder",
    "messages": [{"role": "user", "content": "Write a Python function that atomically replaces a file."}],
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "max_tokens": 256
  }' | jq
```

Check the server log for the speculative-decoding context and accepted draft
tokens. Compare end-to-end throughput against a run with `--spec-type none`;
acceptance rate alone is not the success metric.

## 5. Benchmark before cutover

Run a clean baseline in this order:

1. One slot, MTP off.
2. One slot, MTP with `n=1`, `p-min=0`.
3. Four slots, MTP with `n=1`, `p-min=0`.
4. Only then test the production concurrency and representative prompt sizes.

Record prompt tok/s, generation tok/s per stream and aggregate, TTFT, MTP draft
acceptance, memory use, and failures. Do not resume every long-context client at
once; stagger them after the server is ready to avoid a prefill storm.

## Claude Code compatibility

`llama-server` exposes an OpenAI-compatible `/v1/chat/completions` API. Claude
Code clients that call Anthropic's `/v1/messages` cannot use it directly. Keep
the existing Anthropic-to-OpenAI compatibility adapter in front of this server,
or add and validate one before moving any Claude Code slot. This recipe does
not alter client launchers or perform a fleet cutover.

## Rollback

Stop the `llama-server` process and restart the prior inference service. The
model directory and llama.cpp build are self-contained and can remain on disk
for later testing.
