import type { Telemetry } from "../types";

export function gpuHits(telemetry: Telemetry[], numGpus: number): number[] {
  const hits = new Array(numGpus).fill(0);
  for (const t of telemetry)
    for (const e of t.experts)
      if (e.gpu >= 0 && e.gpu < numGpus) hits[e.gpu]++;
  return hits;
}

export function recentGpus(telemetry: Telemetry[], window = 1): Set<number> {
  const set = new Set<number>();
  for (const t of telemetry.slice(-window))
    for (const e of t.experts) set.add(e.gpu);
  return set;
}
