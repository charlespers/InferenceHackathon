#!/usr/bin/env python3
"""run_e2e.py — the simplest possible end-to-end pipeline for the native engine: tokenize a real
prompt with the real Qwen3 tokenizer, run kernels/prefill_step_tp8.cu's --gen mode (real weights,
real per-token MoE routing, real embedding-chained decode), detokenize the result, print the text.

No server, no concurrency, no multi-chunk prefill -- this is a one-shot CLI script, the simplest path
to "type text in, get text out" given everything already built. Prompt is capped at 16 tokens (the
GEMM panels' proven width); longer prompts need the chunking loop that's a separate, stated TODO.

USAGE:
  python3 tools/run_e2e.py --checkpoint /alloc/data/Qwen3-235B-A22B --weights /alloc/data/real_weights \
      --prompt "The capital of France is" --max_new_tokens 20
"""
import argparse
import subprocess

from transformers import AutoTokenizer

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="real Qwen3 checkpoint dir, for the tokenizer")
    ap.add_argument("--weights", required=True, help="prepared real_weights dir (tools/prepare_real_weights.py output)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=20)
    ap.add_argument("--binary", default="/tmp/prefill_real")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    prompt_ids = tok.encode(args.prompt)
    if len(prompt_ids) > 16:
        print(f"WARNING: prompt is {len(prompt_ids)} tokens, truncating to 16 "
              f"(multi-chunk prefill isn't implemented yet -- see prefill_step_tp8.cu's file header)")
        prompt_ids = prompt_ids[:16]
    print(f"prompt tokens ({len(prompt_ids)}): {prompt_ids}")

    cmd = [args.binary, "--gen", args.weights, str(args.max_new_tokens)] + [str(t) for t in prompt_ids]
    print("running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("FAILED -- stderr:")
        print(result.stderr)
        raise SystemExit(result.returncode)

    gen_line = [l for l in result.stdout.splitlines() if l.startswith("TOKENS:")]
    if not gen_line:
        print("FATAL: binary didn't print a TOKENS: line -- see stdout above.")
        raise SystemExit(1)
    gen_ids = [int(x) for x in gen_line[0].split()[1:]]
    text = tok.decode(gen_ids, skip_special_tokens=True)

    print("\n=== generated tokens ===")
    print(gen_ids)
    print("\n=== generated text ===")
    print(text)
