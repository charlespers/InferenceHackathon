import { gpuHits, recentGpus } from "../lib/gpuLoad";
import type { GpuInfo, Telemetry, Topology } from "../types";

function Bar({ pct, accent }: { pct: number; accent?: boolean }) {
  return (
    <div className="h-1 rounded-full overflow-hidden"
         style={{ backgroundColor: "var(--bar-track)" }}>
      <div
        className="h-full transition-all duration-300"
        style={{
          width: `${Math.min(Math.max(pct, 0), 100)}%`,
          backgroundColor: accent ? "var(--bar-fill-accent)" : "var(--bar-fill)",
        }}
      />
    </div>
  );
}

function GpuCard({ gpu, hits, maxHits, isHot }: {
  gpu: GpuInfo; hits: number; maxHits: number; isHot: boolean;
}) {
  const memPct = gpu.mem_total_mb > 0 ? (gpu.mem_used_mb / gpu.mem_total_mb) * 100 : 0;
  const memUsedGb = (gpu.mem_used_mb / 1024).toFixed(1);
  const memTotalGb = (gpu.mem_total_mb / 1024).toFixed(0);

  return (
    <div className={`rounded-lg p-2.5 transition-all duration-200 border ${
      isHot
        ? "border-emerald-500/40 dark:border-emerald-700/40 bg-emerald-50/40 dark:bg-emerald-950/20"
        : "border-black/10 dark:border-white/[0.05] bg-white/30 dark:bg-white/[0.02]"
    }`}>
      <div className="text-[9px] uppercase tracking-widest mb-2
                      text-neutral-400 dark:text-neutral-700">gpu {gpu.id}</div>
      <div className="space-y-1.5">
        <div>
          <div className="flex justify-between text-[9px] text-neutral-400 dark:text-neutral-700 mb-0.5">
            <span>mem</span><span>{memUsedGb}/{memTotalGb}g</span>
          </div>
          <Bar pct={memPct} />
        </div>
        <div>
          <div className="flex justify-between text-[9px] text-neutral-400 dark:text-neutral-700 mb-0.5">
            <span>sm</span><span>{gpu.utilization_pct}%</span>
          </div>
          <Bar pct={gpu.utilization_pct} accent={gpu.utilization_pct > 20} />
        </div>
        <div>
          <div className="flex justify-between text-[9px] text-neutral-400 dark:text-neutral-700 mb-0.5">
            <span>route</span><span>{hits}</span>
          </div>
          <Bar pct={maxHits > 0 ? (hits / maxHits) * 100 : 0} accent />
        </div>
      </div>
      {gpu.temp_c > 0 && (
        <div className={`text-[9px] mt-2 ${
          gpu.temp_c >= 80 ? "text-red-400" : gpu.temp_c >= 60 ? "text-amber-400"
          : "text-neutral-400 dark:text-neutral-700"
        }`}>{gpu.temp_c}°c</div>
      )}
    </div>
  );
}

export function GpuExpertViz({ telemetry, topology }: { telemetry: Telemetry[]; topology: Topology | null }) {
  const numGpus = topology?.gpus.length ?? 8;
  const hits = gpuHits(telemetry, numGpus);
  const maxHits = Math.max(...hits, 1);
  const hot = recentGpus(telemetry);
  const last = telemetry[telemetry.length - 1];

  const fallbackGpus: GpuInfo[] = Array.from({ length: numGpus }, (_, i) => ({
    id: i, name: `H100-${i}`, mem_total_mb: 81920, mem_used_mb: 0,
    utilization_pct: 0, temp_c: 0,
  }));
  const gpus = topology?.gpus ?? fallbackGpus;

  return (
    <div className="rounded-xl p-4 flex-1 min-h-0 flex flex-col
                    border border-black/10 dark:border-white/[0.05]
                    bg-white/30 dark:bg-white/[0.02]
                    backdrop-blur-sm">
      <div className="flex justify-between mb-3">
        <span className="text-[9px] uppercase tracking-widest
                         text-neutral-400 dark:text-neutral-700">gpu · expert routing</span>
        {!topology && (
          <span className="text-[9px] text-amber-500/60">fallback</span>
        )}
      </div>

      <div className="grid grid-cols-4 gap-2">
        {gpus.map((gpu) => (
          <GpuCard key={gpu.id} gpu={gpu}
                   hits={hits[gpu.id] ?? 0} maxHits={maxHits}
                   isHot={hot.has(gpu.id)} />
        ))}
      </div>

      <div className="mt-3">
        <div className="text-[9px] uppercase tracking-widest mb-1.5
                        text-neutral-400 dark:text-neutral-700">last token experts</div>
        <div className="flex flex-wrap gap-1">
          {last?.experts.map((e, i) => (
            <span key={i} className="px-1.5 py-0.5 rounded text-[9px] font-mono
                                     bg-black/[0.06] dark:bg-white/[0.06]
                                     border border-black/10 dark:border-white/[0.08]
                                     text-neutral-500 dark:text-emerald-600">
              L{e.layer}·E{e.expert_id}→{e.gpu}
            </span>
          )) ?? <span className="text-[9px] text-neutral-400 dark:text-neutral-700">idle</span>}
        </div>
      </div>
    </div>
  );
}
