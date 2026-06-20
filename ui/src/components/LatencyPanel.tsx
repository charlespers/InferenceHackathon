import { liveStats } from "../lib/latency";
import type { Telemetry, Summary } from "../types";

function Sparkline({ values }: { values: number[] }) {
  if (values.length < 2) return <svg className="w-full h-10" />;
  const max = Math.max(...values, 1);
  const pts = values.map((v, i) =>
    `${(i / (values.length - 1)) * 100},${30 - (v / max) * 28}`).join(" ");
  return (
    <svg viewBox="0 0 100 30" preserveAspectRatio="none" className="w-full h-10">
      <polyline points={pts} fill="none" stroke="currentColor" strokeWidth="1"
                className="text-emerald-400" />
    </svg>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-neutral-900 rounded p-2">
      <div className="text-[10px] uppercase tracking-wide text-neutral-500">{label}</div>
      <div className="text-lg font-mono text-emerald-300">{value}</div>
    </div>
  );
}

export function LatencyPanel({ telemetry, summary }: { telemetry: Telemetry[]; summary: Summary | null }) {
  const s = liveStats(telemetry);
  const ttft = summary ? `${summary.ttft_ms.toFixed(0)} ms` : "—";
  return (
    <div className="border border-neutral-800 rounded-lg p-3">
      <div className="text-xs text-neutral-400 mb-2">Latency · B=1</div>
      <div className="grid grid-cols-2 gap-2">
        <Stat label="TTFT" value={ttft} />
        <Stat label="tok/s" value={String(s.tokPerSec)} />
        <Stat label="tokens" value={String(s.tokens)} />
        <Stat label="spec accept" value={`${Math.round(s.specAccept * 100)}%`} />
      </div>
      <div className="mt-2 text-neutral-500">
        <div className="text-[10px] uppercase tracking-wide mb-1">inter-token ms</div>
        <Sparkline values={telemetry.map((t) => t.t_ms)} />
      </div>
    </div>
  );
}
