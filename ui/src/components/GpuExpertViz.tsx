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
    <div className="panel p-3 flex-1 min-h-0 flex flex-col">
      <div className="flex justify-between micro mb-2">
        <span>GPU / expert routing</span>
        {!topology && <span style={{ color: "var(--studio-warn)" }}>topology unavailable — fallback</span>}
      </div>
      <div className="grid grid-cols-4 gap-2">
        {Array.from({ length: numGpus }, (_, i) => (
          <div key={i}
               className="p-2 bg-sunk border stripe transition-colors"
               style={{
                 borderColor: hot.has(i) ? "var(--conifer)" : "var(--studio-rule)",
                 borderInlineStartColor: hot.has(i) ? "var(--conifer)" : "var(--studio-rule)",
                 background: hot.has(i) ? "color-mix(in oklch, var(--conifer) 8%, var(--studio-sunk))" : undefined,
               }}>
            <div className="text-[10px] text-ink-mute metric-num">{names[i]}</div>
            <div className="h-1.5 mt-1 bg-paper border hair overflow-hidden">
              <div className="h-full" style={{ width: `${(hits[i] / max) * 100}%`, background: "var(--conifer)" }} />
            </div>
            <div className="text-[10px] metric-num text-ink-mute mt-1">{hits[i]}</div>
          </div>
        ))}
      </div>
      <div className="mt-3 text-[10px] text-ink-mute">
        <div className="micro mb-1">last token experts</div>
        <div className="flex flex-wrap gap-1">
          {last?.experts.map((e, i) => (
            <span key={i} className="px-1.5 py-0.5 bg-sunk border hair metric-num" style={{ color: "var(--conifer)" }}>
              L{e.layer}·E{e.expert_id}→GPU{e.gpu}
            </span>
          )) ?? <span className="text-ink-faint">idle</span>}
        </div>
      </div>
    </div>
  );
}
