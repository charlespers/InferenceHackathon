import { useState } from "react";
import { useRace } from "../lib/useRace";
import { elapsedMs, speedup } from "../lib/raceMath";
import { LANES, DEFAULT_PROMPT } from "../config";
import { RaceLane } from "./RaceLane";
import { SpeedupBadge } from "./SpeedupBadge";
import { TestTimeCompute } from "./TestTimeCompute";

const ACCENT: Record<string, string> = { conifer: "var(--conifer)", vllm: "var(--baseline)" };

export function RaceView({ base }: { base: string }) {
  const race = useRace(base, LANES);
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [maxTokens, setMaxTokens] = useState(140);

  const conifer = race.lanes["conifer"];
  const vllm = race.lanes["vllm"];
  const bothDone = conifer.status === "done" && vllm.status === "done";

  const cTotal = elapsedMs(conifer.startedAt, conifer.doneAt);
  const vTotal = elapsedMs(vllm.startedAt, vllm.doneAt);
  const ratio = bothDone ? speedup(vTotal, cTotal) : 0;
  const budget = bothDone ? vTotal / 1000 : 0;

  const leaderId = bothDone
    ? (cTotal <= vTotal ? "conifer" : "vllm")
    : race.running
      ? (conifer.tokens === vllm.tokens ? null : conifer.tokens > vllm.tokens ? "conifer" : "vllm")
      : null;

  const firstDoneId =
    conifer.doneAt && (!vllm.doneAt || conifer.doneAt < vllm.doneAt) ? "conifer"
      : vllm.doneAt && (!conifer.doneAt || vllm.doneAt < conifer.doneAt) ? "vllm"
        : null;

  const submit = () => {
    if (race.running) { race.cancel(); return; }
    race.run(prompt, maxTokens);
  };

  return (
    <div className="flex flex-col gap-3 min-h-0 flex-1 animate-fade-up">
      {/* Prompt bar */}
      <div className="panel p-2 flex gap-2 items-stretch">
        <textarea
          className="flex-1 bg-paper border hair px-3 py-2 text-[13px] resize-none h-[52px] outline-none text-ink placeholder:text-ink-faint focus:border-conifer"
          placeholder="Prompt both engines…"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); submit(); }
          }}
        />
        <label className="flex flex-col justify-center px-3 text-[10px] text-ink-mute border hair micro">
          max tokens
          <input
            type="number" min={16} max={512} value={maxTokens}
            onChange={(e) => setMaxTokens(Math.max(16, +e.target.value || 16))}
            className="bg-transparent metric-num text-ink text-sm w-16 outline-none mt-1"
          />
        </label>
        <button
          onClick={submit}
          className="px-5 text-sm font-medium transition-colors stripe border hair"
          style={
            race.running
              ? { color: "var(--studio-fail)", background: "var(--studio-sunk)", borderInlineStartColor: "var(--studio-fail)" }
              : { color: "var(--studio-card)", background: "var(--conifer)", borderColor: "var(--conifer)", borderInlineStartColor: "var(--conifer-soft)" }
          }
        >
          {race.running ? "Stop" : "Run race"}
        </button>
      </div>

      {/* Lanes + speedup */}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_148px_1fr] gap-3 flex-1 min-h-0">
        <RaceLane
          lane={LANES[0]} state={conifer} maxTokens={maxTokens} accent={ACCENT.conifer}
          leading={leaderId === "conifer"} finishedFirst={firstDoneId === "conifer"}
        />
        <SpeedupBadge conifer={conifer} vllm={vllm} />
        <RaceLane
          lane={LANES[1]} state={vllm} maxTokens={maxTokens} accent={ACCENT.vllm}
          leading={leaderId === "vllm"} finishedFirst={firstDoneId === "vllm"}
        />
      </div>

      {/* Test-time compute */}
      <TestTimeCompute speedupRatio={ratio} budgetSeconds={budget} />
    </div>
  );
}
