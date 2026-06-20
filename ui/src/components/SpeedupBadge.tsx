import { decodeTokPerSec, elapsedMs, speedup } from "../lib/raceMath";
import type { LaneState } from "../lib/useRace";
import { CountUp } from "./primitives";

/** Center gauge: live tok/s ratio while racing, locked end-to-end speedup when done. */
export function SpeedupBadge({ conifer, vllm }: { conifer: LaneState; vllm: LaneState }) {
  const bothDone = conifer.status === "done" && vllm.status === "done";
  const anyRunning = conifer.status === "streaming" || vllm.status === "streaming";

  let ratio = 0;
  let caption = "awaiting run";
  let sub = "";

  if (bothDone) {
    const cTotal = elapsedMs(conifer.startedAt, conifer.doneAt);
    const vTotal = elapsedMs(vllm.startedAt, vllm.doneAt);
    ratio = speedup(vTotal, cTotal);
    caption = "faster · end-to-end";
    const sooner = (vTotal - cTotal) / 1000;
    sub = sooner > 0 ? `${sooner.toFixed(1)}s sooner` : "";
  } else if (anyRunning) {
    const cTps = decodeTokPerSec(conifer.tokens, conifer.firstTokenAt, conifer.lastTokenAt);
    const vTps = decodeTokPerSec(vllm.tokens, vllm.firstTokenAt, vllm.lastTokenAt);
    ratio = vTps > 0 ? cTps / vTps : 0;
    caption = "live tok/s ratio";
    sub = "racing…";
  }

  const show = ratio > 0;

  return (
    <div className="flex flex-col items-center justify-center text-center select-none px-2 py-4">
      <div className="micro text-ink-faint mb-3">vs</div>

      <div className="font-display leading-none flex items-baseline justify-center" style={{ color: "var(--conifer)" }}>
        {show ? (
          <>
            <CountUp value={ratio} decimals={ratio >= 10 ? 0 : 1} className="text-6xl lg:text-7xl font-extralight" />
            <span className="text-2xl font-light ml-1">×</span>
          </>
        ) : (
          <span className="text-5xl font-extralight text-ink-faint">—</span>
        )}
      </div>

      <div className="micro mt-4 max-w-[8rem] leading-relaxed" style={{ color: show ? "var(--conifer)" : undefined }}>
        {caption}
      </div>
      {sub && <div className="text-[11px] text-ink-mute mt-2 metric-num">{sub}</div>}
    </div>
  );
}
