#!/bin/bash
# @file        convert_medium.sh
# @description Convert openai/whisper-medium -> ANE-optimized, mixed-bit palettized CoreML (WhisperKit) via whisperkittools.
# @author      Kennt Kim
# @company     Calida Lab
# @created     2026-07-02
# @lastUpdated 2026-07-02
#
# Requirements: macOS (Apple Silicon), Python 3.11, whisperkittools (installed from source: pip install -e .wkt).
# CoreML/ANE conversion is Apple-only; a CUDA GPU is NOT used.
#
# Flags confirmed from whisperkittools scripts/generate_model.py (not guessed):
#   --generate-quantized-variants  : run the palettizer to emit compressed variants
#   --allowed-nbits N (repeatable)  : bit widths the mixed-bit RECIPE may pick per-layer.
#                                     The recipe search auto-optimizes per-layer bits for WER -> the
#                                     hallucination-sensitive TextDecoder naturally keeps more bits.
#                                     (Do NOT pass --force-recipe-nbits: that forces uniform bits, killing the search.)
#   --generate-decoder-context-prefill-data : KV-cache prefill (Argmax "_turbo" streaming opt; layer-preserving).
#   MODEL_REPO_ID + --upload-results : optional HF upload target.
set -euo pipefail

MODEL_VERSION="openai/whisper-medium"   # multilingual, 24 decoder layers
OUTPUT_DIR="./models"
# HF upload target (only used with --upload-results). Change to your repo before uploading.
export MODEL_REPO_ID="${MODEL_REPO_ID:-calidalab/spectalo-whisper-coreml}"

echo "== Converting ${MODEL_VERSION} -> mixed-bit palettized CoreML (WhisperKit) =="
echo "   output: ${OUTPUT_DIR}   (models/ is git-ignored; weights go to HF, not GitHub)"

whisperkit-generate-model \
  --model-version "${MODEL_VERSION}" \
  --output-dir "${OUTPUT_DIR}" \
  --generate-quantized-variants \
  --allowed-nbits 4 --allowed-nbits 6 --allowed-nbits 8 \
  --generate-decoder-context-prefill-data

# The recipe emits several mixed-bit variants at different size/WER points (like argmax's _547MB vs _626MB).
# Pick the one nearest ~450-500MB that passes verify_model.sh + on-device WER/hallucination checks.
echo ""
echo "Generated variants under ${OUTPUT_DIR}. Verify each candidate:"
echo "  ./verify_model.sh ${OUTPUT_DIR}/<variant-folder>"
echo "To publish the chosen one:  MODEL_REPO_ID=<your-hf-repo> whisperkit-generate-model ... --upload-results"
