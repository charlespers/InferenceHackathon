import { liveStats } from "../lib/latency";
import type { Telemetry, Summary } from "../types";

function Sparkline({ values }: { values: number[] }) {
  if (values.length < 2) return (
    <svg className="w-full h-8" viewBox="0 0 100 20" preserveAspectRatio="none">
      <line x1="0" y1="10" x2="100" y2="10" stroke="currentColor" strokeWidth="0.5"
            strokeDasharray="2,3" className="text-neutral-400 dark:text-neutral-700" />
    </svg>
  );
  const max = Math.max(...values, 1);
  const pts = values.map((v, i) =>
    `${(i / (values.length - 1)) * 100},${18 - (v / max) * 16}`).join(" ");
  return (
    <svg viewBox="0 0 100 20" preserveAspectRatio="none" className="w-full h-8">
      <polyline points={pts} fill="none" stroke="currentColor" strokeWidth="0.9"
                className="text-emerald-500 dark:text-emerald-600" />
    </svg>
  );
}

function Stat({ label, value, dim }: { label: string; value: string; dim?: boolean }) {
  return (
    <div>
      <div className="text-[9px] uppercase tracking-widest mb-0.5
                      text-neutral-400 dark:text-neutral-700">{label}</div>
      <div className={`text-xl font-mono ${
        dim ? "text-neutral-400 dark:text-neutral-600" : "text-neutral-700 dark:text-neutral-200"
      }`}>{value}</div>
    </div>
  );
}

export function LatencyPanel({ telemetry, summary }: { telemetry: Telemetry[]; summary: Summary | null }) {
  const s = liveStats(telemetry);
  const ttft = summary ? `${summary.ttft_ms.toFixed(0)} ms` : "—";

  return (
    <div className="rounded-xl p-4
                    border border-black/10 dark:border-white/[0.05]
                    bg-white/30 dark:bg-white/[0.02]
                    backdrop-blur-sm">
      <div className="text-[9px] uppercase tracking-widest mb-3
                      text-neutral-400 dark:text-neutral-700">
        latency · b=1
      </div>
      <div className="grid grid-cols-4 gap-4 mb-3">
        <Stat label="ttft" value={ttft} />
        <Stat label="tok/s" value={s.tokPerSec > 0 ? String(s.tokPerSec) : "—"} />
        <Stat label="tokens" value={s.tokens > 0 ? String(s.tokens) : "—"} dim />
        <Stat label="spec" value={`${Math.round(s.specAccept * 100)}%`} dim />
      </div>
      <div>
        <div className="text-[9px] uppercase tracking-widest mb-1
                        text-neutral-400 dark:text-neutral-700">inter-token ms</div>
        <Sparkline values={telemetry.slice(1).map((t, i) => t.t_ms - telemetry[i].t_ms)} />
      </div>
    </div>
  );
}
