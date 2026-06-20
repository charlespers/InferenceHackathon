// Pure measurement + projection math for the head-to-head race.
//
// Two kinds of number live here, and the distinction is the whole point of the demo:
//   1. MEASURED  — derived from wall-clock timestamps the UI records as SSE arrives.
//   2. PROJECTED — the test-time-compute accuracy curve, an illustrative model.
// Keep them honestly separated; the UI labels (2) as illustrative.

/** Time-to-first-token in ms, or null until the first token lands. */
export function ttftMs(startedAt: number | null, firstTokenAt: number | null): number | null {
  if (startedAt == null || firstTokenAt == null) return null;
  return Math.max(0, firstTokenAt - startedAt);
}

/** Decode throughput (tokens/sec) measured over the post-first-token window. */
export function decodeTokPerSec(
  tokens: number,
  firstTokenAt: number | null,
  lastTokenAt: number | null,
): number {
  if (firstTokenAt == null || lastTokenAt == null || tokens < 2) return 0;
  const secs = (lastTokenAt - firstTokenAt) / 1000;
  if (secs <= 0) return 0;
  return (tokens - 1) / secs;
}

/** Wall-clock elapsed for a lane (live `now` until it finishes). */
export function elapsedMs(startedAt: number | null, endAt: number | null): number {
  if (startedAt == null || endAt == null) return 0;
  return Math.max(0, endAt - startedAt);
}

/** End-to-end speedup: how many times faster `fast` finished than `slow`. */
export function speedup(slowTotalMs: number, fastTotalMs: number): number {
  if (fastTotalMs <= 0 || slowTotalMs <= 0) return 0;
  return slowTotalMs / fastTotalMs;
}

/** Whole reasoning passes the fast engine fits into the slow engine's time-to-answer. */
export function samplesInBudget(speedupRatio: number): number {
  if (!isFinite(speedupRatio) || speedupRatio < 1) return 1;
  return Math.max(1, Math.floor(speedupRatio));
}

export interface TaskProfile {
  /** Single-pass (pass@1) accuracy. */
  base: number;
  /** Accuracy ceiling self-consistency saturates toward. */
  plateau: number;
  /** How quickly extra samples buy accuracy (per extra sample). */
  growth: number;
}

/**
 * ILLUSTRATIVE self-consistency / majority-vote accuracy for k samples.
 * Saturating curve: at k=1 returns the base pass@1; as k grows it approaches the
 * plateau. This is a teaching model of the test-time-compute effect (cf. Wang et al.,
 * "Self-Consistency"), not a benchmarked measurement.
 */
export function majAccuracy(k: number, task: TaskProfile): number {
  const kk = Math.max(1, Math.floor(k));
  const gain = task.plateau - task.base;
  const acc = task.plateau - gain * Math.exp(-task.growth * (kk - 1));
  return Math.min(task.plateau, Math.max(task.base, acc));
}
