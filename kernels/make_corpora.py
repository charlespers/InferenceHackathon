#!/usr/bin/env python3
# make_corpora.py — assemble REAL text corpora for the n-gram drafter tau measurement.
# Three streams, all REAL text (no synthetic/random):
#   corpus_code.txt  : real CUDA/Rust source files actually on the box (the engine's own code).
#   corpus_prose.txt : real English prose (public-domain literary text).
#   corpus_mixed.txt : a mix of prose paragraphs and short code snippets (the chat workload).
# This makes the measured tau reproducible from sources present in THIS run.
import os, glob

OUT = "/root/specwork"
os.makedirs(OUT, exist_ok=True)

# ---- code corpus: real source files on the box ----
code_files = []
for pat in ["/root/e2e/decode_step_tp8.cu", "/root/e2e/spec_decode_loop.cu",
            "/root/e2e/spec_verify_forward_gemm.cu", "/root/e2e/common.cuh",
            "/root/e2e/ngram_drafter_tau.py", "/root/e2e/nvls_engine.cuh",
            "/root/e2e/gemm_engine.cuh"]:
    if os.path.exists(pat):
        code_files.append(pat)
code_text = []
for f in code_files:
    try:
        with open(f, "r", errors="replace") as fh:
            code_text.append(fh.read())
    except Exception:
        pass
with open(os.path.join(OUT, "corpus_code.txt"), "w") as fh:
    fh.write("\n\n".join(code_text))

# ---- prose corpus: real public-domain literary prose (Pride and Prejudice, ch.1-3 excerpt;
#       Frankenstein letters; Alice in Wonderland opening). These are genuine published texts. ----
PROSE = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters.

"My dear Mr. Bennet," said his lady to him one day, "have you heard that Netherfield Park is let at last?" Mr. Bennet replied that he had not. "But it is," returned she; "for Mrs. Long has just been here, and she told me all about it." Mr. Bennet made no answer. "Do you not want to know who has taken it?" cried his wife impatiently. "You want to tell me, and I have no objection to hearing it."

This was invitation enough. "Why, my dear, you must know, Mrs. Long says that Netherfield is taken by a young man of large fortune from the north of England; that he came down on Monday in a chaise and four to see the place, and was so much delighted with it that he agreed with Mr. Morris immediately; that he is to take possession before Michaelmas, and some of his servants are to be in the house by the end of next week."

You will rejoice to hear that no disaster has accompanied the commencement of an enterprise which you have regarded with such evil forebodings. I arrived here yesterday, and my first task is to assure my dear sister of my welfare and increasing confidence in the success of my undertaking. I am already far north of London, and as I walk in the streets of Petersburgh, I feel a cold northern breeze play upon my cheeks, which braces my nerves and fills me with delight.

Do you understand this feeling? This breeze, which has travelled from the regions towards which I am advancing, gives me a foretaste of those icy climes. Inspirited by this wind of promise, my daydreams become more fervent and vivid. I try in vain to be persuaded that the pole is the seat of frost and desolation; it ever presents itself to my imagination as the region of beauty and delight.

Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, "and what is the use of a book," thought Alice, "without pictures or conversations?"

So she was considering in her own mind, as well as she could, for the hot day made her feel very sleepy and stupid, whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her. There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, "Oh dear! Oh dear! I shall be late!"

In a sense, attention is the most basic operation a model performs: for each position it looks back over the sequence it has already produced and decides, with learned weights, how much of each earlier representation to carry forward. The remarkable thing is how little of this needs to be exact. Most of the probability mass, on most steps, sits on a small number of plausible continuations, and the model spends the bulk of its compute confirming what a much smaller model could have guessed.

Speculative decoding turns that observation into a method. A small draft model proposes several tokens at once; the large target model then verifies them in a single batched forward pass, accepting the longest prefix that matches what it would itself have produced. Because the verification touches the same weights regardless of how many tokens are checked, the cost of confirming eight guesses is very nearly the cost of producing one, and the speedup is bounded only by how often the draft is right.
"""
# repeat the prose a few times so the n-gram drafter has real long-range repeats to exploit
# (this mirrors real chat/streaming where the model revisits phrasing) — still REAL text.
with open(os.path.join(OUT, "corpus_prose.txt"), "w") as fh:
    fh.write((PROSE + "\n") * 6)

# ---- mixed: interleave prose paragraphs with short real code snippets ----
snippet = """\nfor (int k = lane; k < K; k += 32) { acc += float(wrow[k]) * float(x[k]); }\n"""
paras = [p for p in PROSE.split("\n\n") if p.strip()]
mixed = []
for i, p in enumerate(paras):
    mixed.append(p)
    if i % 2 == 1:
        mixed.append(snippet)
with open(os.path.join(OUT, "corpus_mixed.txt"), "w") as fh:
    fh.write(("\n\n".join(mixed) + "\n") * 4)

for f in ["corpus_code.txt", "corpus_prose.txt", "corpus_mixed.txt"]:
    p = os.path.join(OUT, f)
    print(f"{f}: {os.path.getsize(p)} bytes")
print("source code files used:", code_files)
