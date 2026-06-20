import { liveStats } from "../lib/latency";
import type { Telemetry, Summary } from "../types";

function Sparkline({ values }: { values: number[] }) {
  if (values.length < 2) return <svg className="w-full h-10" />;
  const max = Math.max(...values, 1);
  const pts = values.map((v, i) =>
    `${(i / (values.length - 1)) * 100},${30 - (v / max) * 28}`).join(" ");
  return (
    <svg viewBox="0 0 100 30" preserveAspectRatio="none" className="w-full h-10">
      <polyline points={pts} fill="none" stroke="var(--conifer)" strokeWidth="1" />
    </svg>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-sunk border hair p-2">
      <div className="micro">{label}</div>
      <div className="text-lg metric-num mt-0.5" style={{ color: "var(--conifer)" }}>{value}</div>
    </div>
  );
}

export function LatencyPanel({ telemetry, summary }: { telemetry: Telemetry[]; summary: Summary | null }) {
  const s = liveStats(telemetry);
  const ttft = summary ? `${summary.ttft_ms.toFixed(0)} ms` : "—";
  return (
    <div className="panel p-3">
      <div className="micro mb-2">Latency · B=1</div>
      <div className="grid grid-cols-2 gap-2">
        <Stat label="TTFT" value={ttft} />
        <Stat label="tok/s" value={String(s.tokPerSec)} />
        <Stat label="tokens" value={String(s.tokens)} />
        <Stat label="spec accept" value={`${Math.round(s.specAccept * 100)}%`} />
      </div>
      <div className="mt-3 text-ink-mute">
        <div className="micro mb-1">inter-token ms</div>
        <Sparkline values={telemetry.map((t) => t.t_ms)} />
      </div>
    </div>
  );
}
