import { liveStats } from "../lib/latency";
import type { Telemetry, Summary } from "../types";

function Stat({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div>
      <div className="text-[9px] uppercase tracking-widest mb-0.5
                      text-neutral-400 dark:text-neutral-700">{label}</div>
      <div className={`text-xl font-mono ${
        accent ? "text-emerald-600 dark:text-emerald-400" : "text-neutral-700 dark:text-neutral-200"
      }`}>{value}</div>
    </div>
  );
}

// Stacked floor-breakdown bar: weight (irreducible) | comms (NVLS) | kv | overhead (the target).
function FloorBar({ bd, regime }: { bd: NonNullable<Summary["floor_breakdown_ms"]>; regime?: string }) {
  const total = bd.weight + bd.comms + bd.kv + bd.overhead || 1;
  const seg = [
    { k: "weight", ms: bd.weight, cls: "bg-emerald-500/80", lbl: "weight" },
    { k: "comms",  ms: bd.comms,  cls: "bg-amber-500/80",   lbl: "comms · NVLS" },
    { k: "kv",     ms: bd.kv,     cls: "bg-sky-500/70",     lbl: "kv" },
    { k: "over",   ms: bd.overhead, cls: "bg-neutral-400/60 dark:bg-neutral-600/60", lbl: "overhead" },
  ];
  return (
    <div className="mt-3">
      <div className="flex items-center justify-between mb-1">
        <div className="text-[9px] uppercase tracking-widest text-neutral-400 dark:text-neutral-700">
          per-forward floor · {total.toFixed(2)} ms
        </div>
        {regime && (
          <div className="text-[9px] uppercase tracking-widest text-neutral-500 dark:text-neutral-500">
            {regime}
          </div>
        )}
      </div>
      <div className="flex h-3 w-full overflow-hidden rounded-sm border border-black/10 dark:border-white/[0.06]">
        {seg.map(s => s.ms > 0 && (
          <div key={s.k} className={s.cls} style={{ width: `${(s.ms / total) * 100}%` }} title={`${s.lbl} ${s.ms.toFixed(2)}ms`} />
        ))}
      </div>
      <div className="mt-1.5 grid grid-cols-2 gap-x-4 gap-y-0.5 text-[10px] font-mono
                      text-neutral-500 dark:text-neutral-500">
        {seg.map(s => (
          <div key={s.k} className="flex items-center justify-between">
            <span className="flex items-center gap-1">
              <span className={`inline-block h-2 w-2 rounded-[1px] ${s.cls}`} />{s.lbl}
            </span>
            <span>{s.ms.toFixed(2)} ms</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function Sparkline({ values }: { values: number[] }) {
  if (values.length < 2) return (
    <svg className="w-full h-7" viewBox="0 0 100 20" preserveAspectRatio="none">
      <line x1="0" y1="10" x2="100" y2="10" stroke="currentColor" strokeWidth="0.5"
            strokeDasharray="2,3" className="text-neutral-400 dark:text-neutral-700" />
    </svg>
  );
  const max = Math.max(...values, 1);
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * 100},${18 - (v / max) * 16}`).join(" ");
  return (
    <svg viewBox="0 0 100 20" preserveAspectRatio="none" className="w-full h-7">
      <polyline points={pts} fill="none" stroke="currentColor" strokeWidth="0.9"
                className="text-emerald-500 dark:text-emerald-600" />
    </svg>
  );
}

export function LatencyPanel({ telemetry, summary }: { telemetry: Telemetry[]; summary: Summary | null }) {
  const s = liveStats(telemetry);
  // Prefer the engine's authoritative decode_tok_per_s (the mock emits a constant per-token t_ms, so the
  // live gap-based rate is 0); fall back to the live estimate only when no summary has arrived yet.
  const tokPerSec = summary?.decode_tok_per_s ?? (s.tokPerSec || 0);
  const ttft = summary ? `${summary.ttft_ms.toFixed(0)} ms` : "—";
  const mbu = summary?.pct_of_roofline != null ? `${summary.pct_of_roofline.toFixed(0)}%` : "—";
  const specPct = summary ? Math.round(summary.spec_accept_rate * 100) : Math.round(s.specAccept * 100);

  return (
    <div className="rounded-xl p-4
                    border border-black/10 dark:border-white/[0.05]
                    bg-white/30 dark:bg-white/[0.02]
                    backdrop-blur-sm">
      <div className="text-[9px] uppercase tracking-widest mb-3
                      text-neutral-400 dark:text-neutral-700">
        latency · b=1
      </div>
      <div className="grid grid-cols-4 gap-4">
        <Stat label="tok/s" value={tokPerSec > 0 ? tokPerSec.toFixed(1) : "—"} accent />
        <Stat label="ttft" value={ttft} />
        <Stat label="mbu" value={mbu} />
        <Stat label="spec acc" value={summary || s.tokens ? `${specPct}%` : "—"} />
      </div>
      {summary?.floor_breakdown_ms && <FloorBar bd={summary.floor_breakdown_ms} regime={summary.regime} />}
      <div className="mt-3">
        <div className="text-[9px] uppercase tracking-widest mb-1
                        text-neutral-400 dark:text-neutral-700">inter-token ms</div>
        <Sparkline values={telemetry.slice(1).map((t, i) => Math.abs(t.t_ms - telemetry[i].t_ms))} />
      </div>
    </div>
  );
}
