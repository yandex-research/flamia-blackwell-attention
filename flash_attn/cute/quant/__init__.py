"""The low-precision quantisers (CuTe DSL): fp8.py (block-scaled and per-tensor FP8), fp4.py (NVFP4 / MXFP4),
fp4_diagonal.py (the causal diagonal routes' one-launch operand builder), _rotations.py (the register-resident Hadamard /
RoPE helpers). Their eager PyTorch references, written from the format definitions, are tests/cute/quant_reference.py."""
