import { describe, it, expect } from "vitest";
import { liveStats } from "./latency";

const tel = (t: number, acc: number, prop: number) => ({
  token_index: 0, t_ms: t, experts: [], spec: { accepted: acc, proposed: prop },
});

describe("liveStats", () => {
  it("returns zeros for empty input", () => {
    expect(liveStats([])).toEqual({ tokens: 0, avgInterMs: 0, tokPerSec: 0, specAccept: 0 });
  });
  it("computes averages and acceptance", () => {
    const s = liveStats([tel(10, 3, 4), tel(10, 1, 4)] as any);
    expect(s.tokens).toBe(2);
    expect(s.avgInterMs).toBe(10);
    expect(s.tokPerSec).toBe(100);
    expect(s.specAccept).toBeCloseTo(0.5);
  });
});
