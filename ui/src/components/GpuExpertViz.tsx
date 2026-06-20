import { gpuHits, recentGpus } from "../lib/gpuLoad";
import type { GpuInfo, Telemetry, Topology } from "../types";

function GpuCard({
  gpu,
  hits,
  maxHits,
  isHot,
}: {
  gpu: GpuInfo;
  hits: number;
  maxHits: number;
  isHot: boolean;
}) {
  const memPct = gpu.mem_total_mb > 0 ? (gpu.mem_used_mb / gpu.mem_total_mb) * 100 : 0;
  const memUsedGb = (gpu.mem_used_mb / 1024).toFixed(1);
  const memTotalGb = (gpu.mem_total_mb / 1024).toFixed(0);
  const tempColor =
    gpu.temp_c >= 80 ? "text-red-400" : gpu.temp_c >= 60 ? "text-amber-400" : "text-neutral-500";
  const utilColor =
    gpu.utilization_pct >= 90
      ? "bg-emerald-400"
      : gpu.utilization_pct >= 50
      ? "bg-emerald-600"
      : "bg-neutral-700";

  return (
    <div
      className={`rounded p-2 bg-neutral-900 border transition-colors ${
        isHot ? "border-emerald-400 shadow-[0_0_12px] shadow-emerald-500/40" : "border-neutral-800"
      }`}
    >
      {/* GPU name */}
      <div className="text-[10px] text-neutral-400 truncate">GPU {gpu.id}</div>

      {/* HBM usage */}
      <div className="mt-1">
        <div className="flex justify-between text-[9px] text-neutral-500 mb-0.5">
          <span>MEM</span>
          <span>{memUsedGb}/{memTotalGb}G</span>
        </div>
        <div className="h-1 rounded bg-neutral-800 overflow-hidden">
          <div
            className="h-full bg-blue-500 transition-all"
            style={{ width: `${memPct}%` }}
          />
        </div>
      </div>

      {/* SM utilization */}
      <div className="mt-1">
        <div className="flex justify-between text-[9px] text-neutral-500 mb-0.5">
          <span>SM</span>
          <span>{gpu.utilization_pct}%</span>
        </div>
        <div className="h-1 rounded bg-neutral-800 overflow-hidden">
          <div
            className={`h-full transition-all ${utilColor}`}
            style={{ width: `${gpu.utilization_pct}%` }}
          />
        </div>
      </div>

      {/* Expert routing hits bar */}
      <div className="mt-1">
        <div className="flex justify-between text-[9px] text-neutral-500 mb-0.5">
          <span>ROUTE</span>
          <span>{hits}</span>
        </div>
        <div className="h-1 rounded bg-neutral-800 overflow-hidden">
          <div
            className="h-full bg-emerald-500 transition-all"
            style={{ width: `${maxHits > 0 ? (hits / maxHits) * 100 : 0}%` }}
          />
        </div>
      </div>

      {/* Temp */}
      <div className={`text-[9px] font-mono mt-1 ${tempColor}`}>
        {gpu.temp_c > 0 ? `${gpu.temp_c}°C` : "—"}
      </div>
    </div>
  );
}

export function GpuExpertViz({
  telemetry,
  topology,
}: {
  telemetry: Telemetry[];
  topology: Topology | null;
}) {
  const numGpus = topology?.gpus.length ?? 8;
  const hits = gpuHits(telemetry, numGpus);
  const maxHits = Math.max(...hits, 1);
  const hot = recentGpus(telemetry);
  const last = telemetry[telemetry.length - 1];

  const fallbackGpus: GpuInfo[] = Array.from({ length: numGpus }, (_, i) => ({
    id: i,
    name: `H100-${i}`,
    mem_total_mb: 81920,
    mem_used_mb: 0,
    utilization_pct: 0,
    temp_c: 0,
  }));
  const gpus = topology?.gpus ?? fallbackGpus;

  return (
    <div className="border border-neutral-800 rounded-lg p-3 flex-1 min-h-0 flex flex-col">
      <div className="flex justify-between text-xs text-neutral-400 mb-2">
        <span>GPU status · expert routing</span>
        {!topology && (
          <span className="text-amber-500/70">topology unavailable — fallback</span>
        )}
      </div>

      <div className="grid grid-cols-4 gap-2">
        {gpus.map((gpu) => (
          <GpuCard
            key={gpu.id}
            gpu={gpu}
            hits={hits[gpu.id] ?? 0}
            maxHits={maxHits}
            isHot={hot.has(gpu.id)}
          />
        ))}
      </div>

      <div className="mt-3 text-[10px] text-neutral-500">
        <div className="uppercase tracking-wide mb-1">last token experts</div>
        <div className="flex flex-wrap gap-1">
          {last?.experts.map((e, i) => (
            <span
              key={i}
              className="px-1.5 py-0.5 rounded bg-neutral-800 font-mono text-emerald-300"
            >
              L{e.layer}·E{e.expert_id}→GPU{e.gpu}
            </span>
          )) ?? <span className="text-neutral-600">idle</span>}
        </div>
      </div>
    </div>
  );
}
