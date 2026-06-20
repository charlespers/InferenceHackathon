import type { Telemetry } from "../types";

export function liveStats(telemetry: Telemetry[]) {
  const tokens = telemetry.length;
  if (tokens === 0) return { tokens: 0, avgInterMs: 0, tokPerSec: 0, specAccept: 0 };

  // Compute inter-token gaps from cumulative timestamps
  let avgInterMs = 0;
  if (tokens >= 2) {
    let gapSum = 0;
    for (let i = 1; i < tokens; i++) gapSum += telemetry[i].t_ms - telemetry[i - 1].t_ms;
    avgInterMs = gapSum / (tokens - 1);
  } else {
    avgInterMs = telemetry[0].t_ms; // single token: use TTFT as proxy
  }

  const proposed = telemetry.reduce((a, t) => a + t.spec.proposed, 0);
  const accepted = telemetry.reduce((a, t) => a + t.spec.accepted, 0);
  return {
    tokens,
    avgInterMs,
    tokPerSec: avgInterMs > 0 ? Math.round(1000 / avgInterMs) : 0,
    specAccept: proposed > 0 ? accepted / proposed : 0,
  };
}
