import { gpuHits, recentGpus } from "../lib/gpuLoad";
import type { Telemetry, Topology } from "../types";

export function GpuExpertViz({ telemetry, topology }: { telemetry: Telemetry[]; topology: Topology | null }) {
  const numGpus = topology?.gpus.length ?? 8;
  const names = topology?.gpus.map((g) => g.name) ?? Array.from({ length: 8 }, (_, i) => `H100-${i}`);
  const hits = gpuHits(telemetry, numGpus);
  const max = Math.max(...hits, 1);
  const hot = recentGpus(telemetry);
  const last = telemetry[telemetry.length - 1];

  return (
    <div className="border border-neutral-800 rounded-lg p-3 flex-1 min-h-0 flex flex-col">
      <div className="flex justify-between text-xs text-neutral-400 mb-2">
        <span>GPU / expert routing</span>
        {!topology && <span className="text-amber-500/70">topology unavailable — fallback</span>}
      </div>
      <div className="grid grid-cols-4 gap-2">
        {Array.from({ length: numGpus }, (_, i) => (
          <div key={i}
               className={`rounded p-2 bg-neutral-900 border transition-colors ${
                 hot.has(i) ? "border-emerald-400 shadow-[0_0_12px] shadow-emerald-500/40" : "border-neutral-800"}`}>
            <div className="text-[10px] text-neutral-400">{names[i]}</div>
            <div className="h-1.5 mt-1 rounded bg-neutral-800 overflow-hidden">
              <div className="h-full bg-emerald-500" style={{ width: `${(hits[i] / max) * 100}%` }} />
            </div>
            <div className="text-[10px] font-mono text-neutral-500 mt-1">{hits[i]}</div>
          </div>
        ))}
      </div>
      <div className="mt-3 text-[10px] text-neutral-500">
        <div className="uppercase tracking-wide mb-1">last token experts</div>
        <div className="flex flex-wrap gap-1">
          {last?.experts.map((e, i) => (
            <span key={i} className="px-1.5 py-0.5 rounded bg-neutral-800 font-mono text-emerald-300">
              L{e.layer}·E{e.expert_id}→GPU{e.gpu}
            </span>
          )) ?? <span className="text-neutral-600">idle</span>}
        </div>
      </div>
    </div>
  );
}
