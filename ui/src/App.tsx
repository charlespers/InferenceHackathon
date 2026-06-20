import { useState } from "react";
import { RaceView } from "./components/RaceView";
import { ConsoleView } from "./components/ConsoleView";
import { getDefaultBase } from "./config";

type View = "race" | "console";

function SegBtn({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`px-4 py-1.5 text-[12px] border-r last:border-r-0 hair transition-colors metric-num ${
        active ? "bg-sunk text-ink" : "text-ink-mute hover:text-ink"
      }`}
    >
      {children}
    </button>
  );
}

export default function App() {
  const [base, setBase] = useState(getDefaultBase());
  const [view, setView] = useState<View>("race");

  return (
    <div className="h-screen flex flex-col px-5 py-4 gap-3.5">
      <header className="flex items-center justify-between gap-4 flex-wrap">
        <div className="flex items-baseline gap-3">
          <div className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full" style={{ background: "var(--conifer)" }} />
            <h1 className="font-serif text-[28px] leading-none tracking-tight text-ink">Conifer</h1>
          </div>
          <span className="text-ink-mute text-[11px] hidden sm:inline metric-num uppercase tracking-wider">
            Inference, measured · Qwen3-235B-A22B · 8×H100 · B=1
          </span>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex border hair">
            <SegBtn active={view === "race"} onClick={() => setView("race")}>Race</SegBtn>
            <SegBtn active={view === "console"} onClick={() => setView("console")}>Console</SegBtn>
          </div>
          <input
            className="bg-paper border hair px-2.5 py-1.5 text-xs w-60 metric-num text-ink-soft outline-none focus:border-conifer"
            value={base}
            onChange={(e) => setBase(e.target.value)}
            aria-label="backend url"
          />
        </div>
      </header>

      {view === "race" ? <RaceView base={base} /> : <ConsoleView base={base} />}
    </div>
  );
}
