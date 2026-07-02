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

## Credits / licenses
- OpenAI Whisper — MIT
- Argmax WhisperKit / whisperkittools — MIT
