# spectalo-whisper-coreml

ANE-optimized custom **Whisper CoreML** models for on-device ASR in **Spectalo** (Whiplay) and **SpectaLing** (SpectaloWhisper).

Built with [Argmax `whisperkittools`](https://github.com/argmaxinc/whisperkittools). Model **weights are hosted on Hugging Face**; this repo holds the **build recipe, verification, and docs** (model binaries are intentionally git-ignored).

## Why this exists

Argmax's `argmaxinc/whisperkit-coreml` has **no ANE-compressed `medium` (24-layer) model**. Its ANE lineup jumps from `small_216MB` (12 decoder layers) to `large-v3_947MB` (32 layers) — nothing in between. This repo fills that gap.

**Key finding (Spectalo, 2026-07): hallucination is governed by _decoder depth_, not model size.**

| model | decoder layers | hallucination |
|---|---|---|
| `large-v3-turbo` / `v20240930` | **4** | many |
| `small` | **12** | few (verified) |
| **`medium`** | **24** | fewer + more accurate |
| `large-v2` / `large-v3` (2023) | 32 | fewest |

> Naming trap: `v20240930` = OpenAI's 4-layer turbo. Argmax's `_turbo` suffix = a streaming optimization that **keeps** the layer count. Always confirm via `config.json` → `decoder_layers`.

## Target spec — v1 (`whisper-medium`, convert-only)

| | |
|---|---|
| Base | `openai/whisper-medium` — multilingual, **24 decoder layers**, 769M |
| Goals | ≥12 & ≤24 layers · minimize hallucination · **better transcription than `small_216MB`** |
| Compression | Mixed-bit palettization → **~450–500MB** (fits 8GB devices; vs `large-v3_947MB`) |
| Recipe | whisperkittools `--generate-quantized-variants --allowed-nbits 4 6 8` — the **recipe search auto-assigns per-layer bits for WER**, so the hallucination-sensitive TextDecoder keeps more bits automatically (no manual per-component config; do not pass `--force-recipe-nbits`) |
| ANE | fused `layer_norm` + palettized weights → requires **A14+/M1+** |
| Output | `openai_whisper-medium_XXXMB` (WhisperKit folder layout) |

Deeper decoder (24 vs small's 12) is the whole point: **lower hallucination _and_ higher accuracy** without retraining.

## Build (macOS Apple Silicon only)

CoreML/ANE conversion requires macOS. A CUDA GPU is **not** used (coremltools targets the Neural Engine, not CUDA). See `convert_medium.sh`.

```bash
# Python 3.11, macOS Apple Silicon. whisperkittools is NOT on PyPI — install from source.
conda create -n whisperkit python=3.11 -y && conda activate whisperkit
git clone https://github.com/argmaxinc/whisperkittools.git
pip install -e ./whisperkittools
./convert_medium.sh
```

## Verify

```bash
./verify_model.sh models/openai_whisper-medium
```
Checks `decoder_layers` (should be 24) + ANE indicators (fused `layer_norm`, palettization) in `model.mil`.

## Integrate into the apps

1. Upload the produced model folder to a Hugging Face repo (e.g. `calidalab/spectalo-whisper-coreml`).
2. Add a `WhisperModel` case (rawValue = variant folder name); the app's generic downloader fetches it from HF.
3. Allow it in `maxAllowed` for 8GB devices; set `recommended` only after on-device validation.

## Ship criteria (must pass, else re-tune)

Vs `small_216MB` on the **same CJK clip**: WER **↓** **and** hallucination **≤** small **and** loads on an 8GB device without jetsam (with translation running concurrently).

## Roadmap
- **v1** — convert `whisper-medium` (24-layer) + mixed-bit palettization (this repo).
- v2 (only if 24-layer is too heavy/slow on 8GB) — distill decoder to ~16–20 layers (requires CUDA training + data).
- v3 (optional) — CJK fine-tune for higher Japanese/Korean accuracy.

## ✅ SOLVED (2026-07-03) — palettized 24-layer medium DOES run on ANE

Two fixes together produce a working ANE-resident palettized medium: **`openai_whisper-medium_467MB`** (446 MB, 24 decoder layers, `palettize_lut`>0 on both encoder+decoder).

**Fix 1 — generate with `CPU_ONLY`** so the palettizer's correctness `predict()` (argmaxtools `test_utils.py:308`, default `CPU_AND_NE`) never invokes the ANE compiler *during conversion*. Driver (`.wkt/convert_cpu.py`):
```python
import sys, coremltools as ct
from argmaxtools import test_utils as at
at.TEST_COMPUTE_UNIT = ct.ComputeUnit.CPU_ONLY   # skip ANE AOT compile during generation
from scripts.generate_model import cli
sys.exit(cli())
```

**Fix 2 — AudioEncoder SDPA = `Cat`** (not the default `SplitHeadsQ`). The default `SplitHeadsQ` encoder graph makes the ANE AOT compiler **hang (>200 s)**; `Cat` (the SDPA the TextDecoder already uses, which always compiled fine) compiles cleanly. **This — not model size/layer-count — was the "medium ANE edge case" Argmax triaged away.**

Working command (from the `.wkt` clone):
```bash
WANDB_MODE=disabled python convert_cpu.py \
  --model-version openai/whisper-medium --output-dir ../models \
  --generate-quantized-variants --allowed-nbits 4 --allowed-nbits 6 --allowed-nbits 8 \
  --audio-encoder-sdpa-implementation Cat
```

**Measured ANE compile times** (Mac M1 Max, `CompiledMLModel(path, ct.ComputeUnit.CPU_AND_NE)`, first load; cached after):
- TextDecoder **12.5 s** · AudioEncoder(`Cat`) **81 s** · (`SplitHeadsQ` AudioEncoder = >200 s hang)

**Still to validate before shipping:** on-device (iPhone/iPad A14+) first-load compile time; transcription WER/quality vs `small_216MB`; memory on 8 GB devices.

---

## Build environment — the hang, diagnosed (superseded by SOLVED above)

Base conversion of `whisper-medium` **works** (validated: 24 decoder layers + fused `layer_norm` = ANE-ready). But the **quantized/palettization step hangs at final assembly** with bleeding-edge deps:

- Installed by default: **coremltools 9.0**, **scikit-learn 1.9.0** — both too new. coremltools prints `scikit-learn 1.9.0 is not supported ... Disabling` at startup, and the mixed-bit recipe **finishes** (both components get `recipe_results.json`) but the final palettized-model assembly then **sleeps at 0% CPU indefinitely** (repro'd twice, even with `--disable-default-tests` + `WANDB_MODE=disabled`).
- The argmax `whisperkit-coreml` models were built with **coremltools 8.x**.

**Version pin (done, did NOT fix the hang):**
```bash
pip install "coremltools>=8.1,<9" "scikit-learn<=1.5.1"   # -> coremltools 8.3.0, sklearn 1.5.1
```
The recipe search finishes and is cached under `models/openai_whisper-medium/compression_artifacts/`.

**Real root cause (stack-sampled the stuck process, 2026-07-03):** the hang is the **Apple Neural Engine AOT compiler**, not coremltools. Main-thread stack:
```
+[MLModel modelWithContentsOfURL:...]  ->  MLE5Engine loadModelFromCompiledArchive
->  Espresso e5rt_e5_compiler_compile_from_ir_program  ->  MILCompilerForANE::CompileUsingANEF
->  -[_ANEClient compileModel:...]  ->  -[_ANEDaemonConnection compileModel:...withReply:]   <-- blocked here
```
Loading/compiling the **palettized 24-layer medium** on ANE is pathologically slow (the `aned` daemon sits at 0% while the sync `withReply:` XPC blocks for many minutes; the python process oscillates idle<->100% CPU = very slow, not a clean deadlock). The **fp16** base compiles on ANE fine — only the **palettized** model triggers it. This step is reached when the tool loads the model for `.mlcomputeplan.json` / prefill generation.

**➡️ CHOSEN NEXT STEP (decided 2026-07-03): option 2 — uniform 6-bit, no prefill.** Ready to run:
```bash
conda activate whisperkit   # env has coremltools 8.3 + sklearn 1.5.1 pinned
cd <this-repo>
WANDB_MODE=disabled ~/miniconda3/envs/whisperkit/bin/whisperkit-generate-model \
  --model-version openai/whisper-medium --output-dir ./models \
  --generate-quantized-variants --allowed-nbits 6 --force-recipe-nbits \
  --disable-default-tests
#   - uniform 6-bit (--force-recipe-nbits) = simpler graph, may compile on ANE faster than mixed-bit
#   - dropped --generate-decoder-context-prefill-data (that model-load-on-ANE is where it hung)
# then:  ./verify_model.sh models/<variant>   # expect palettize_lut>0, ~24 layers, ~500MB
# WATCH: if it still stalls at 0% CPU with aned idle -> same ANE-compile bottleneck -> go to option 4.
```

**All options (in order):**
1. Retry **without** `--generate-decoder-context-prefill-data` (removes model-load-on-ANE during generation).
2. Try a single uniform low bit-width (`--allowed-nbits 6 --force-recipe-nbits`) — simpler palettized graph, may compile faster on ANE.
3. Try alternate SDPA impls (`--text-decoder-sdpa-implementation`, `--audio-encoder-sdpa-implementation`) that may lower to ANE better.
4. If ANE compile stays pathological, palettized medium may not be ANE-friendly at 24 layers -> reconsider (small_216MB stays the shipping model).
> ⚠️ If ANE **compile** hangs during generation, first on-device load could be similarly slow — validate compile time on a target device before shipping.

## Credits / licenses
- OpenAI Whisper — MIT
- Argmax WhisperKit / whisperkittools — MIT
