import { decodeTokPerSec, elapsedMs, ttftMs } from "../lib/raceMath";
import type { LaneState } from "../lib/useRace";
import type { LaneConfig } from "../config";
import { Meter } from "./primitives";

interface Props {
  lane: LaneConfig;
  state: LaneState;
  maxTokens: number;
  accent: string;
  leading: boolean;
  finishedFirst: boolean;
}

function Field({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div>
      <div className="micro">{label}</div>
      <div className="metric-num text-[15px] mt-1 text-ink" style={accent ? { color: accent } : undefined}>
        {value}
      </div>
    </div>
  );
}

export function RaceLane({ lane, state, maxTokens, accent, leading, finishedFirst }: Props) {
  const streaming = state.status === "streaming";
  const endAt = state.doneAt ?? (streaming ? performance.now() : state.startedAt);
  const elapsed = elapsedMs(state.startedAt, endAt);
  const ttft = ttftMs(state.startedAt, state.firstTokenAt);
  const tps = decodeTokPerSec(state.tokens, state.firstTokenAt, state.lastTokenAt);
  const progress = Math.min(100, (state.tokens / Math.max(maxTokens, 1)) * 100);

  const dotColor =
    state.status === "error" ? "var(--studio-fail)"
      : state.status === "idle" ? "var(--studio-ink-faint)"
        : accent;

  return (
    <div
      className="panel stripe relative flex flex-col min-h-0 overflow-hidden transition-colors duration-300"
      style={{
        borderInlineStartColor: leading ? accent : "var(--studio-rule)",
        background: leading ? "var(--studio-card-hover)" : "var(--studio-card)",
      }}
    >
      {/* Header */}
      <div className="flex items-start justify-between px-4 pt-3.5 pb-3 border-b hair">
        <div>
          <div className="flex items-center gap-2">
            <span
              className={`h-2 w-2 rounded-full ${streaming ? "animate-pulse-soft" : ""}`}
              style={{ background: dotColor }}
            />
            <span className="font-display text-xl font-medium tracking-tight text-ink">{lane.label}</span>
            {finishedFirst && (
              <span
                className="metric-num text-[9px] px-1.5 py-0.5 stripe"
                style={{
                  color: accent,
                  background: `color-mix(in oklch, ${accent} 14%, transparent)`,
                  borderInlineStartColor: accent,
                }}
              >
                FIRST
              </span>
            )}
          </div>
          <div className="text-[11px] text-ink-mute mt-1.5 metric-num">{lane.detail}</div>
        </div>
        <span className="micro mt-1 px-1.5 py-0.5 border hair text-ink-mute">B=1</span>
      </div>

      {/* Live metrics */}
      <div className="grid grid-cols-4 gap-2 px-4 py-3 border-b hair">
        <Field label="elapsed" value={`${(elapsed / 1000).toFixed(2)}s`} accent={accent} />
        <Field label="ttft" value={ttft == null ? "—" : `${ttft.toFixed(0)}ms`} />
        <Field label="tok/s" value={tps ? tps.toFixed(0) : "—"} />
        <Field label="tokens" value={String(state.tokens)} />
      </div>

      {/* Race progress */}
      <div className="px-4 py-3 border-b hair">
        <Meter value={progress} max={100} accent={accent} />
      </div>

      {/* Streamed output */}
      <div className="flex-1 min-h-0 overflow-y-auto px-4 py-3 text-[13px] leading-relaxed text-ink-soft">
        {state.error ? (
          <span style={{ color: "var(--studio-fail)" }}>error: {state.error}</span>
        ) : state.text ? (
          <span className="whitespace-pre-wrap">
            {state.text}
            {streaming && (
              <span className="inline-block w-[7px] h-4 -mb-0.5 ml-0.5 animate-pulse-soft" style={{ background: accent }} />
            )}
          </span>
        ) : (
          <span className="text-ink-faint italic">{streaming ? "prefilling…" : "idle"}</span>
        )}
      </div>
    </div>
  );
}
