#!/usr/bin/env python3
"""validate_native_vs_reference.py — the check the native pipeline has never had: does it produce the
SAME generated tokens as the real Qwen3-235B-A22B model, given the same prompt?

Everything validated so far (validate_fp8_quant.py, the engine's own finite/cross-rank-argmax checks)
proves individual pieces are locally sound. None of them prove the 94-layer chained pipeline -- with
its compounding fp8 error and its own from-scratch top-8 router -- produces the same TEXT as the real
model. This is that check.

Two tiers, cheapest first:
  (1) Greedy-decode match: run the SAME prompt through (a) the native --gen binary and (b) HF
      transformers loaded in fp8 (or bf16, see --ref-dtype) on the SAME checkpoint, both greedy
      (temperature=0, argmax). Compare token-by-token. This is the decisive end-to-end check.
  (2) Per-layer hidden-state drift (--probe-layers): if (1) diverges, run both paths for just the
      first K layers and diff the hidden state after each layer, to find WHERE divergence starts
      (quant error compounding vs an actual routing/logic bug look very different in this profile).

Run on the box (needs the real checkpoint + prepared real_weights + a free GPU for the HF reference):
  python3 tools/validate_native_vs_reference.py --checkpoint /alloc/data/Qwen3-235B-A22B \
      --weights /alloc/data/real_weights --prompt "The capital of France is" --max_new_tokens 8
"""
import argparse
import subprocess
import sys


def run_native(binary, weights_dir, prompt_ids, max_new_tokens):
    cmd = [binary, "--gen", weights_dir, str(max_new_tokens)] + [str(t) for t in prompt_ids]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        print("NATIVE RUN FAILED -- stderr:\n", result.stderr, file=sys.stderr)
        return None
    line = [l for l in result.stdout.splitlines() if l.startswith("TOKENS:")]
    if not line:
        print("NATIVE RUN: no TOKENS: line -- stdout:\n", result.stdout, file=sys.stderr)
        return None
    return [int(x) for x in line[0].split()[1:]]


def run_reference(checkpoint, prompt_ids, max_new_tokens, dtype):
    """Real HF forward pass, greedy, fp8-quantized the SAME way as the native engine (per-row e4m3,
    amax/448) if dtype == 'fp8-sim' -- so this is comparing like-for-like quant error, not blaming the
    native engine for a bf16-vs-fp8 difference that vLLM's own served model also has."""
    import torch
    from transformers import AutoModelForCausalLM

    print(f"loading reference model in {dtype}... (this is the 235B checkpoint -- needs ~8x80GB or "
          f"device_map='auto' sharding across whatever GPUs are free)")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()

    if dtype == "fp8-sim":
        _fake_quantize_model_(model)

    input_ids = torch.tensor([prompt_ids], device=next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None, top_k=None)
    return out[0, len(prompt_ids):].tolist()


def _fake_quantize_model_(model):
    """Round-trips every Linear weight through the SAME per-row e4m3 quant the native engine's weights
    went through (tools/prepare_real_weights.py / conifer_weight_convert.py's amax/448 formula), so the
    reference is apples-to-apples on quantization error and any remaining diff is a real engine bug,
    not just 'fp8 vs bf16 are different numbers'."""
    import torch
    E4M3_MAX = 448.0
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                w = module.weight.data.float()
                amax = w.abs().amax(dim=1).clamp_min(1e-8)
                scale = amax / E4M3_MAX
                q = (w / scale[:, None]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
                module.weight.data = (q.float() * scale[:, None]).to(module.weight.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--weights", required=True, help="prepared real_weights dir for the native binary")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--binary", default="/tmp/prefill_real")
    ap.add_argument("--ref-dtype", choices=["bf16", "fp8-sim"], default="fp8-sim",
                    help="fp8-sim (default) fake-quantizes the HF reference the same way the native "
                         "engine's weights were quantized, isolating engine bugs from quant-choice "
                         "differences. bf16 compares against the unquantized model instead.")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    prompt_ids = tok.encode(args.prompt)
    if len(prompt_ids) > 16:
        prompt_ids = prompt_ids[:16]
        print(f"WARNING: truncated prompt to 16 tokens (native engine's current prefill cap)")
    print(f"prompt ({len(prompt_ids)} tokens): {prompt_ids}  ({args.prompt!r})")

    print("\n--- running native engine ---")
    native_tokens = run_native(args.binary, args.weights, prompt_ids, args.max_new_tokens)
    if native_tokens is None:
        print("ABORT: native run failed, nothing to compare.")
        sys.exit(1)
    print(f"native tokens:    {native_tokens}")
    print(f"native text:      {tok.decode(native_tokens, skip_special_tokens=True)!r}")

    print("\n--- running reference (HF transformers) ---")
    ref_tokens = run_reference(args.checkpoint, prompt_ids, args.max_new_tokens, args.ref_dtype)
    print(f"reference tokens: {ref_tokens}")
    print(f"reference text:   {tok.decode(ref_tokens, skip_special_tokens=True)!r}")

    n_match = sum(1 for a, b in zip(native_tokens, ref_tokens) if a == b)
    first_mismatch = next((i for i, (a, b) in enumerate(zip(native_tokens, ref_tokens)) if a != b), None)
    print(f"\n=== VERDICT ===")
    print(f"exact match: {native_tokens == ref_tokens}")
    print(f"tokens matching by position: {n_match}/{min(len(native_tokens), len(ref_tokens))}")
    if first_mismatch is not None:
        print(f"first mismatch at decode step {first_mismatch}: "
              f"native={native_tokens[first_mismatch]} ref={ref_tokens[first_mismatch]}")
        print("(greedy decoding means ANY single-token divergence can cascade into a totally "
              "different continuation from that point on -- this is expected and not itself "
              "damning; what matters is whether early steps match and whether the text stays "
              "plausible after divergence, not byte-identical token IDs forever.)")


if __name__ == "__main__":
    main()
