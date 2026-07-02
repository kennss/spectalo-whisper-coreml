#!/bin/bash
# @file        convert_medium.sh
# @description Convert openai/whisper-medium -> ANE-optimized palettized CoreML (WhisperKit) via whisperkittools.
# @author      Kennt Kim
# @company     Calida Lab
# @created     2026-07-02
# @lastUpdated 2026-07-02
#
# Requirements: macOS (Apple Silicon), Python 3.11, whisperkittools (pulls coremltools, ane_transformers, openai-whisper).
# CoreML/ANE conversion is Apple-only; a CUDA GPU is NOT used.
set -euo pipefail

MODEL_VERSION="openai/whisper-medium"   # multilingual, 24 decoder layers
OUTPUT_DIR="./models"

echo "== Converting ${MODEL_VERSION} -> CoreML (WhisperKit) =="
echo "   output: ${OUTPUT_DIR}"

# Base conversion — produces the WhisperKit CoreML folder (MelSpectrogram/AudioEncoder/TextDecoder .mlmodelc).
whisperkit-generate-model \
  --model-version "${MODEL_VERSION}" \
  --output-dir "${OUTPUT_DIR}"

# TODO(recipe): mixed-bit palettization — decoder 6-8 bit / encoder 4 bit, target ~480MB.
#   whisperkittools exposes optimization/quantization args; confirm the EXACT flags first:
#       whisperkit-generate-model -h
#   then re-run with the chosen --quantize / palettization recipe. Do NOT guess the flags — verify from -h.

echo ""
echo "Done. Verify layers + ANE indicators:"
echo "  ./verify_model.sh ${OUTPUT_DIR}/openai_whisper-medium"
