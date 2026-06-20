import { describe, it, expect } from "vitest";
import {
  ttftMs,
  decodeTokPerSec,
  elapsedMs,
  speedup,
  samplesInBudget,
  majAccuracy,
} from "./raceMath";

describe("raceMath measurements", () => {
  it("ttft is the gap to the first token, null until it lands", () => {
    expect(ttftMs(1000, 1180)).toBe(180);
    expect(ttftMs(1000, null)).toBeNull();
    expect(ttftMs(null, 1180)).toBeNull();
  });

  it("decode tok/s measures the post-first-token window", () => {
    // 11 tokens, 10 inter-token gaps over 1s => 10 tok/s
    expect(decodeTokPerSec(11, 1000, 2000)).toBeCloseTo(10, 5);
    expect(decodeTokPerSec(1, 1000, 2000)).toBe(0);
    expect(decodeTokPerSec(11, 1000, 1000)).toBe(0);
  });

  it("elapsed is clamped and zero when unfinished", () => {
    expect(elapsedMs(1000, 4400)).toBe(3400);
    expect(elapsedMs(1000, null)).toBe(0);
  });

  it("speedup is slow/fast and guards bad input", () => {
    expect(speedup(3400, 880)).toBeCloseTo(3.86, 2);
    expect(speedup(3400, 0)).toBe(0);
    expect(speedup(0, 880)).toBe(0);
  });
});

describe("test-time-compute projection", () => {
  it("samples-in-budget floors the speedup, min 1", () => {
    expect(samplesInBudget(3.86)).toBe(3);
    expect(samplesInBudget(1.0)).toBe(1);
    expect(samplesInBudget(0.5)).toBe(1);
    expect(samplesInBudget(Infinity)).toBe(1);
  });

  it("maj@k starts at pass@1 and saturates toward the plateau", () => {
    const task = { base: 0.6, plateau: 0.85, growth: 0.6 };
    expect(majAccuracy(1, task)).toBeCloseTo(0.6, 5);
    const k3 = majAccuracy(3, task);
    expect(k3).toBeGreaterThan(0.6);
    expect(k3).toBeLessThan(0.85);
    // monotonic non-decreasing and bounded by the plateau
    expect(majAccuracy(10, task)).toBeGreaterThan(k3);
    expect(majAccuracy(1000, task)).toBeLessThanOrEqual(0.85);
  });
});
