#!/usr/bin/env python3
# ngram_drafter_tau.py — REAL measured speculative-decoding acceptance length (tau) for a
# prompt-lookup / n-gram drafter, over REAL text, tokenized with the REAL Qwen3 tokenizer.
# ============================================================================================
# WHY this exists (path-to-1000 honesty):
#   The spec-decode loop needs a MEASURED acceptance length tau, not a projected one. An EAGLE3
#   head requires the model on the box (and its target-variant pinning); the empirical vLLM run
#   gave an anomalous tau=1.27 from a target mismatch. A *prompt-lookup n-gram drafter* needs NO
#   model: at each step it proposes k tokens by finding the most recent earlier occurrence of the
#   last n context tokens and copying what followed. On real text this yields a GENUINE acceptance
#   distribution — a real, reproducible tau for THIS workload and THIS tokenizer.
#
# PROTOCOL (greedy, lossless — exactly accept.rs's greedy special case):
#   For each forward position t over the real token stream:
#     1. DRAFT: take last n tokens ctx[t-n:t]; find most recent j<t with toks[j-n:j]==ctx[t-n:t];
#        propose draft = toks[j:j+k] (the k tokens that followed that earlier match). If no match,
#        propose nothing (0 accepted, the forward still emits 1 token).
#     2. VERIFY+ACCEPT: the "target" is the GROUND-TRUTH real text. Accept the longest prefix of
#        the draft that equals the real continuation toks[t:t+k]. (Greedy verify == argmax == the
#        real next token, since the real text IS the greedy target output for this measurement.)
#     3. EMIT: accepted_len matched tokens + 1 bonus token (the target's own next token, always
#        correct) = accepted_len+1 tokens advanced this forward.
#   tau = mean over forwards of (accepted_len + 1).  This is E[tokens per verify forward].
#
# This is the SAME accept rule as engine/src/spec/accept.rs in its greedy limit (accept iff
# draft==target argmax; +1 bonus token). It is lossless: output == target text exactly.
#
# Output: per-corpus tau, per-position acceptance, and a combined tau, all from REAL token streams.
# Usage:  python3 ngram_drafter_tau.py <tokenizer_dir> <k> <n> <corpus1> [corpus2 ...]
# ============================================================================================
import sys, os, json

def load_tokenizer(tok_dir):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)

def ngram_speculate(toks, k, n):
    """Run prompt-lookup n-gram speculation over a token stream; return acceptance stats.
    Returns: forwards, accepted_total, per-position accept counts, draft-issued count."""
    N = len(toks)
    # last-occurrence index for fast n-gram lookup: map ngram-tuple -> most recent end index j
    # We scan left-to-right; at forward position t we want the latest j<t with toks[j-n:j]==ctx.
    forwards = 0
    accepted_total = 0          # sum of matched draft tokens (excludes the always-correct bonus)
    emitted_total = 0           # sum of (accepted_len + 1) over forwards  -> tau = emitted/forwards
    pos_accept = [0]*k          # pos_accept[i] = #forwards where draft position i was accepted
    pos_drafted = [0]*k         # pos_drafted[i] = #forwards where a draft existed at position i
    drafts_issued = 0           # forwards where the n-gram lookup found a candidate

    # last_end[ngram] = end index j of the most-recent EARLIER occurrence of that n-gram, i.e.
    #   toks[j-n:j] == ngram. The tokens that FOLLOWED it are toks[j:j+...] -> the draft proposal.
    # Invariant: when we look up the context ending at the current cursor t, last_end contains only
    # occurrences with end index < t (we insert t AFTER the lookup), so a hit is a strictly earlier
    # match. We index EVERY position the cursor sweeps over (t advances by step>=1 each forward, and
    # we backfill the skipped interior positions) so no earlier n-gram is missed.
    last_end = {}
    t = n                       # need n tokens of context before the first speculation
    # seed index with the n-grams strictly before the first speculation cursor (end indices n..t-1)
    for j in range(n, t):
        last_end[tuple(toks[j-n:j])] = j

    while t < N:
        forwards += 1
        ctx = tuple(toks[t-n:t])
        j = last_end.get(ctx, None)   # earlier end index with toks[j-n:j]==ctx, or None
        accepted_len = 0
        if j is not None:
            # propose toks[j:j+k] (what followed the earlier match); accept longest prefix == real
            kk = min(k, N - t)
            for i in range(kk):
                if j+i >= N:
                    break
                pos_drafted[i] += 1
                if toks[j+i] == toks[t+i]:
                    pos_accept[i] += 1
                    accepted_len += 1
                else:
                    break
            drafts_issued += 1
        # emit accepted_len matched + 1 bonus (the target's real next token)
        step = accepted_len + 1
        emitted_total += step
        accepted_total += accepted_len
        # advance the cursor by step; record the current n-gram (end index t) and backfill every
        # interior n-gram end position the cursor jumps past, so they're available for later lookups.
        new_t = min(t + step, N)
        for jj in range(t, new_t):
            if jj >= n:
                last_end[tuple(toks[jj-n:jj])] = jj
        t = new_t

    tau = emitted_total / forwards if forwards else 0.0
    return {
        "tokens": N, "forwards": forwards, "accepted_total": accepted_total,
        "emitted_total": emitted_total, "tau": tau, "drafts_issued": drafts_issued,
        "pos_accept": pos_accept, "pos_drafted": pos_drafted,
    }

def main():
    if len(sys.argv) < 5:
        print("usage: ngram_drafter_tau.py <tokenizer_dir> <k> <n> <corpus1> [corpus2 ...]")
        sys.exit(1)
    tok_dir = sys.argv[1]
    k = int(sys.argv[2])
    n = int(sys.argv[3])
    corpora = sys.argv[4:]
    tok = load_tokenizer(tok_dir)

    print("="*92)
    print(" ngram_drafter_tau.py — REAL prompt-lookup n-gram tau on REAL text (Qwen3 tokenizer)")
    print(f" tokenizer: {tok_dir}   draft k={k}   n-gram n={n}")
    print(" accept rule = greedy lossless (accept longest draft prefix == real continuation; +1 bonus)")
    print("="*92)

    all_emitted = 0; all_forwards = 0; all_accepted = 0
    agg_pos_accept = [0]*k; agg_pos_drafted = [0]*k
    for path in corpora:
        with open(path, "r", errors="replace") as f:
            text = f.read()
        ids = tok(text, add_special_tokens=False)["input_ids"]
        r = ngram_speculate(ids, k, n)
        all_emitted += r["emitted_total"]; all_forwards += r["forwards"]; all_accepted += r["accepted_total"]
        for i in range(k):
            agg_pos_accept[i] += r["pos_accept"][i]; agg_pos_drafted[i] += r["pos_drafted"][i]
        name = os.path.basename(path)
        print(f"\n--- corpus: {name}  ({r['tokens']} tokens, {r['forwards']} forwards) ---")
        print(f"  MEASURED tau (mean tokens/forward) = {r['tau']:.3f}")
        print(f"  accepted draft tokens = {r['accepted_total']} ; +1 bonus/forward always correct")
        print(f"  forwards with a draft candidate = {r['drafts_issued']}/{r['forwards']} "
              f"({100.0*r['drafts_issued']/r['forwards']:.1f}%)")
        pp = []
        for i in range(k):
            rate = (r['pos_accept'][i]/r['pos_drafted'][i]) if r['pos_drafted'][i] else 0.0
            pp.append(f"p{i}={rate:.3f}")
        print("  per-draft-position accept rate (of forwards where that position was drafted): " + " ".join(pp))

    tau_all = all_emitted/all_forwards if all_forwards else 0.0
    print("\n" + "="*92)
    print(f"COMBINED MEASURED tau over all corpora = {tau_all:.3f}   "
          f"({all_forwards} forwards, {all_emitted} tokens emitted)")
    pp = []
    for i in range(k):
        rate = (agg_pos_accept[i]/agg_pos_drafted[i]) if agg_pos_drafted[i] else 0.0
        pp.append(f"p{i}={rate:.3f}")
    print("combined per-position accept rate: " + " ".join(pp))
    print("="*92)
    # machine-readable line for the .cu loop to ingest
    print(f"NGRAM_TAU_RESULT k={k} n={n} tau={tau_all:.4f} forwards={all_forwards} emitted={all_emitted}")

if __name__ == "__main__":
    main()
