import type { Telemetry } from "../types";

export function liveStats(telemetry: Telemetry[]) {
  const tokens = telemetry.length;
  if (tokens === 0) return { tokens: 0, avgInterMs: 0, tokPerSec: 0, specAccept: 0 };
  const avgInterMs = telemetry.reduce((a, t) => a + t.t_ms, 0) / tokens;
  const proposed = telemetry.reduce((a, t) => a + t.spec.proposed, 0);
  const accepted = telemetry.reduce((a, t) => a + t.spec.accepted, 0);
  return {
    tokens,
    avgInterMs,
    tokPerSec: avgInterMs > 0 ? Math.round(1000 / avgInterMs) : 0,
    specAccept: proposed > 0 ? accepted / proposed : 0,
  };
}
