import { describe, it, expect } from "vitest";
import { gpuHits, recentGpus } from "./gpuLoad";

const t = (gpus: number[]) => ({
  token_index: 0, t_ms: 8, spec: { proposed: 4, accepted: 3 },
  experts: gpus.map((g) => ({ layer: 0, expert_id: 0, gpu: g })),
});

describe("gpuLoad", () => {
  it("counts hits per gpu", () => {
    expect(gpuHits([t([0, 1]), t([1, 1])] as any, 4)).toEqual([1, 3, 0, 0]);
  });
  it("returns gpus from last token only", () => {
    expect([...recentGpus([t([0]), t([2, 3])] as any)].sort()).toEqual([2, 3]);
  });
});
