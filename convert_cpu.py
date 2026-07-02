#!/usr/bin/env python
# @file        convert_cpu.py
# @description CPU_ONLY driver for whisperkit-generate-model. Forces argmaxtools TEST_COMPUTE_UNIT=CPU_ONLY
#              so the palettizer correctness predict() does NOT invoke the ANE AOT compiler DURING generation
#              (that ANE compile hangs on the palettized medium). The shipped .mlmodelc stays fully
#              ANE-capable — ANE compiles lazily at device load time.
# @author      Kennt Kim
# @company     Calida Lab
# @created     2026-07-03
# @lastUpdated 2026-07-03
#
# USAGE: copy this file INTO the whisperkittools clone (next to its `scripts/` dir), then run it in place
#   of `whisperkit-generate-model`. For the working ANE-resident medium, pass Cat SDPA for the encoder:
#     python convert_cpu.py --model-version openai/whisper-medium --output-dir ../models \
#       --generate-quantized-variants --allowed-nbits 4 --allowed-nbits 6 --allowed-nbits 8 \
#       --audio-encoder-sdpa-implementation Cat
import sys
import coremltools as ct
from argmaxtools import test_utils as at

at.TEST_COMPUTE_UNIT = ct.ComputeUnit.CPU_ONLY   # avoid ANE AOT compiler (_ANEDaemonConnection hang) during generation

from scripts.generate_model import cli  # noqa: E402  (import after the CPU_ONLY override)

sys.exit(cli())
