import { useState } from "react";
import { majAccuracy, samplesInBudget } from "../lib/raceMath";
import { TASK_PRESETS, type TaskPreset } from "../config";
import { CountUp } from "./primitives";

const CONIFER = "var(--conifer)";
const BASELINE = "var(--baseline)";

function AccuracyBar({
  label, sub, value, accent, max,
}: { label: string; sub: string; value: number; accent: string; max: number }) {
  const pct = (value / max) * 100;
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1.5">
        <span className="text-[12px] text-ink-soft">
          {label} <span className="text-ink-faint metric-num">· {sub}</span>
        </span>
        <span className="metric-num text-sm" style={{ color: accent }}>
          <CountUp value={value * 100} decimals={1} suffix="%" />
        </span>
      </div>
      <div className="h-2.5 bg-sunk border border-rule overflow-hidden">
        <div
          className="h-full transition-[width] duration-700 ease-out"
          style={{ width: `${pct}%`, background: accent }}
        />
      </div>
    </div>
  );
}

export function TestTimeCompute({
  speedupRatio, budgetSeconds,
}: { speedupRatio: number; budgetSeconds: number }) {
  const [task, setTask] = useState<TaskPreset>(TASK_PRESETS[1]);
  const ready = speedupRatio > 1 && budgetSeconds > 0;
  const k = ready ? samplesInBudget(speedupRatio) : 1;
  const p1 = task.base;
  const pk = ready ? majAccuracy(k, task) : task.base;
  const max = Math.max(task.plateau + 0.05, 0.4);

  return (
    <div className="panel p-4 flex flex-col gap-4">
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <div>
          <div className="micro text-ink-mute">Test-time compute · what the speed buys</div>
          <div className="font-display text-xl text-ink mt-1.5 leading-tight font-light">
            {ready ? (
              <>
                Same <span className="metric-num" style={{ color: CONIFER }}>{budgetSeconds.toFixed(1)}s</span> budget —{" "}
                <span style={{ color: CONIFER }}>{k}×</span> the reasoning
              </>
            ) : (
              <span className="text-ink-mute">Run a race to project quality at equal latency</span>
            )}
          </div>
        </div>
        <div className="flex border hair">
          {TASK_PRESETS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTask(t)}
              className={`text-[11px] px-3 py-1.5 transition-colors border-r last:border-r-0 hair ${
                t.id === task.id ? "bg-sunk text-ink" : "text-ink-mute hover:text-ink"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      <div className="grid gap-3">
        <AccuracyBar label="vLLM" sub="1 pass · pass@1" value={p1} accent={BASELINE} max={max} />
        <AccuracyBar
          label="Conifer" sub={`${k} passes · maj@${k} self-consistency`} value={pk} accent={CONIFER} max={max}
        />
      </div>

      <div className="flex items-center justify-between gap-3 text-[11px] text-ink-mute border-t hair pt-3">
        <span>
          In vLLM's time-to-answer, Conifer fits <span className="metric-num text-ink">{k}</span> reasoning
          passes on {task.label.toLowerCase()}.
        </span>
        {ready && pk > p1 && (
          <span
            className="metric-num shrink-0 stripe px-2 py-1"
            style={{ color: CONIFER, background: "color-mix(in oklch, var(--conifer) 12%, transparent)", borderInlineStartColor: CONIFER }}
          >
            +{((pk - p1) * 100).toFixed(1)} pts
          </span>
        )}
      </div>
      <div className="text-[10px] text-ink-faint -mt-1 metric-num">
        Speedup is measured live; the accuracy curve is an illustrative self-consistency model.
      </div>
    </div>
  );
}
