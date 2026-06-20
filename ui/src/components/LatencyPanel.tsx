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

// "Where the time goes" — the floor breakdown (docs/overhead-attribution.md) made visible. Only renders when
// the backend supplies optimization telemetry (server/optimization_telemetry.py); harmless otherwise.
function FloorBar({ summary }: { summary: Summary }) {
  const fb = summary.floor_breakdown_ms;
  if (!fb) return null;
  const total = fb.weight + fb.comms + fb.kv + fb.overhead || 1;
  const segs = [
    { label: "overhead", ms: fb.overhead, color: "var(--base)" },   // the dominant floor
    { label: "comms", ms: fb.comms, color: "var(--roof)" },
    { label: "weight", ms: fb.weight, color: "var(--conifer)" },     // the irreducible bytes
    { label: "kv", ms: fb.kv, color: "var(--conifer2)" },
  ];
  return (
    <div className="mt-3">
      <div className="micro mb-1">
        where the time goes{summary.regime ? ` · ${summary.regime}` : ""}
        {summary.pct_of_ceiling != null ? ` · ${summary.pct_of_ceiling.toFixed(1)}% of ceiling` : ""}
      </div>
      <div className="flex h-4 w-full overflow-hidden border hair">
        {segs.filter((s) => (s.ms / total) * 100 > 0.5).map((s) => (
          <div key={s.label} style={{ width: `${(s.ms / total) * 100}%`, background: s.color }}
               title={`${s.label} ${s.ms.toFixed(2)} ms`} />
        ))}
      </div>
      <div className="micro mt-1 text-ink-mute">
        overhead {fb.overhead.toFixed(1)} · comms {fb.comms.toFixed(1)} · weight {fb.weight.toFixed(1)} ms
        {summary.next_lever ? ` → ${summary.next_lever}` : ""}
      </div>
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
      {summary && <FloorBar summary={summary} />}
    </div>
  );
}
