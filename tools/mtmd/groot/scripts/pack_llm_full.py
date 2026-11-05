#!/usr/bin/env python3
"""Pack the LIBERO LLM checkpoint into a complete 16-layer GGUF for llama.cpp.

The deployment GGUF (`llm_libero_fused.gguf`) is a feature-extractor *header*:
token table plus metadata, with the layers in the private fused container. This
script writes the ordinary form, every `blk.N.*` tensor included, so the same
model can be evaluated with llama.cpp's native ggml graph — the baseline for
deciding whether a custom fused prefill kernel is worth keeping.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

LAYERS = {
    "attn_norm": "input_layernorm",
    "ffn_norm": "post_attention_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "attn_q_norm": "self_attn.q_norm",
    "attn_k_norm": "self_attn.k_norm",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}

# Keys the fused header carries that must not leak into a runnable model.
DROP_KEYS = ("groot.fused_prefill_only",)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llama", type=Path, required=True, help="llama.cpp checkout (for gguf-py)")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--template", type=Path, required=True, help="GGUF to copy tokenizer/metadata from")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--f16", action="store_true", help="keep weights fp16 instead of Q8_0")
    a = p.parse_args()

    sys.path.insert(0, str(a.llama / "gguf-py"))
    import gguf

    reader = gguf.GGUFReader(a.template)
    writer = gguf.GGUFWriter(a.output, "qwen3vl", use_temp_file=True)
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key in DROP_KEYS:
            continue
        if key in ("general.architecture", "general.file_type", "general.name"):
            continue
        writer.add_key_value(key, field.contents(), field.types[0],
                             field.types[1] if len(field.types) > 1 else None)
    writer.add_name("GR00T-N1.7-LIBERO-16L")
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16 if a.f16 else gguf.LlamaFileType.MOSTLY_Q8_0)

    index = json.loads((a.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "backbone.model.model.language_model."
    mapping = {"token_embd.weight": "embed_tokens.weight", "output_norm.weight": "norm.weight"}
    for i in range(16):
        for dest, source in LAYERS.items():
            mapping[f"blk.{i}.{dest}.weight"] = f"layers.{i}.{source}.weight"

    for dest, source in mapping.items():
        key = prefix + source
        with safe_open(a.checkpoint / index[key], framework="pt") as f:
            w = f.get_tensor(key).float().numpy()
        if w.ndim == 1:
            writer.add_tensor(dest, w.astype(np.float32))
        elif a.f16 or dest == "token_embd.weight":
            writer.add_tensor(dest, w.astype(np.float16))
        else:
            writer.add_tensor(dest, gguf.quants.quantize(w, gguf.GGMLQuantizationType.Q8_0),
                              raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        print(dest, flush=True)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"wrote {a.output} ({a.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
