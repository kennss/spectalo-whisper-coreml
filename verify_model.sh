#!/bin/bash
# @file        verify_model.sh
# @description Verify a WhisperKit CoreML model — decoder layer count (hallucination indicator) + ANE residency
#              (fused layer_norm not decomposed, palettized weights). Works on the downloaded/converted folder.
# @author      Kennt Kim
# @company     Calida Lab
# @created     2026-07-02
# @lastUpdated 2026-07-02
#
# Usage: ./verify_model.sh <model-dir>
set -euo pipefail
DIR="${1:?usage: verify_model.sh <model-dir>}"

echo "== decoder layers (hallucination indicator: deeper = fewer hallucinations) =="
python3 -c "import json; c=json.load(open('${DIR}/config.json')); print('  decoder_layers =', c.get('decoder_layers'), '| encoder_layers =', c.get('encoder_layers'), '| vocab =', c.get('vocab_size'))"

echo "== ANE indicators (model.mil) =="
for m in TextDecoder AudioEncoder; do
  mil="${DIR}/${m}.mlmodelc/model.mil"
  if [ ! -f "${mil}" ]; then echo "  ${m}: (no model.mil)"; continue; fi
  ln=$(grep -o 'layer_norm' "${mil}" | wc -l | tr -d ' ')
  rm=$(grep -o 'reduce_mean' "${mil}" | wc -l | tr -d ' ')
  lut=$(grep -o 'constexpr_lut_to_dense' "${mil}" | wc -l | tr -d ' ')
  printf "  %-14s fused_layer_norm=%s  reduce_mean=%s  palettize_lut=%s\n" "${m}" "${ln}" "${rm}" "${lut}"
done
echo "  -> ANE-resident when: fused layer_norm present (NOT decomposed into many reduce_mean) AND palettize_lut > 0"
