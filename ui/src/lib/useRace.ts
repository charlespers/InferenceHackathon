import { useCallback, useRef, useState } from "react";
import { streamChat } from "./apiClient";
import type { Summary } from "../types";
import type { LaneConfig } from "../config";

export type LaneStatus = "idle" | "streaming" | "done" | "error";

export interface LaneState {
  text: string;
  status: LaneStatus;
  startedAt: number | null;
  firstTokenAt: number | null;
  lastTokenAt: number | null;
  doneAt: number | null;
  tokens: number;
  summary: Summary | null;
  error: string | null;
}

function blankLane(): LaneState {
  return {
    text: "", status: "idle", startedAt: null, firstTokenAt: null,
    lastTokenAt: null, doneAt: null, tokens: 0, summary: null, error: null,
  };
}

const now = () => performance.now();

/**
 * Drives two engines through the identical prompt at the same instant and records
 * wall-clock timestamps as each token arrives. The speedup the UI shows is therefore
 * measured here, not reported by the server.
 */
export function useRace(base: string, lanes: LaneConfig[]) {
  const lanesRef = useRef<Record<string, LaneState>>(
    Object.fromEntries(lanes.map((l) => [l.id, blankLane()])),
  );
  const [, force] = useState(0);
  const render = useCallback(() => force((n) => n + 1), []);
  const [running, setRunning] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const rafRef = useRef<number | null>(null);

  const tick = useCallback(() => {
    render();
    const live = Object.values(lanesRef.current).some((l) => l.status === "streaming");
    if (live) rafRef.current = requestAnimationFrame(tick);
    else { rafRef.current = null; setRunning(false); }
  }, [render]);

  const run = useCallback((prompt: string, maxTokens: number) => {
    if (!prompt.trim()) return;
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    const t0 = now();
    for (const l of lanes) {
      lanesRef.current[l.id] = { ...blankLane(), status: "streaming", startedAt: t0 };
    }
    render();
    setRunning(true);
    if (rafRef.current == null) rafRef.current = requestAnimationFrame(tick);

    for (const lane of lanes) {
      (async () => {
        const st = () => lanesRef.current[lane.id];
        try {
          const stream = streamChat(
            lane.base ?? base,
            {
              model: "qwen3-235b-a22b",
              temperature: 0.2,
              max_tokens: maxTokens,
              engine: lane.engine,
              messages: [{ role: "user", content: prompt }],
            },
            ctrl.signal,
          );
          for await (const chunk of stream) {
            const piece = chunk?.choices?.[0]?.delta?.content;
            if (piece) {
              const s = st();
              const t = now();
              s.text += piece;
              s.tokens += 1;
              if (s.firstTokenAt == null) s.firstTokenAt = t;
              s.lastTokenAt = t;
            }
            if (chunk?.x_summary) st().summary = chunk.x_summary;
          }
          const s = st();
          s.doneAt = now();
          s.status = "done";
        } catch (e: any) {
          const s = st();
          if (e?.name === "AbortError") { s.status = "idle"; return; }
          s.error = String(e?.message ?? e);
          s.doneAt = now();
          s.status = "error";
        }
      })();
    }
  }, [base, lanes, render, tick]);

  const cancel = useCallback(() => abortRef.current?.abort(), []);

  return { lanes: lanesRef.current, running, run, cancel };
}
