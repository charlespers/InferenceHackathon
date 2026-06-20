import { useCallback, useEffect, useState } from "react";
import { ChatPane } from "./components/ChatPane";
import { LatencyPanel } from "./components/LatencyPanel";
import { GpuExpertViz } from "./components/GpuExpertViz";
import { getTopology } from "./lib/apiClient";
import { getDefaultBase } from "./config";
import type { Telemetry, Summary, Topology } from "./types";

export default function App() {
  const [base, setBase] = useState(getDefaultBase());
  const [topology, setTopology] = useState<Topology | null>(null);
  const [dark, setDark] = useState(true);
  const [turn, setTurn] = useState<{ telemetry: Telemetry[]; summary: Summary | null }>({
    telemetry: [], summary: null,
  });
  const onTurn = useCallback((t: { telemetry: Telemetry[]; summary: Summary | null }) => setTurn(t), []);

  useEffect(() => {
    if (dark) {
      document.documentElement.classList.add("dark");
      document.documentElement.classList.remove("light");
    } else {
      document.documentElement.classList.remove("dark");
      document.documentElement.classList.add("light");
    }
  }, [dark]);

  useEffect(() => {
    let alive = true;
    const poll = () =>
      getTopology(base).then((t) => alive && setTopology(t)).catch(() => {});
    poll();
    const id = setInterval(poll, 2000);
    return () => { alive = false; clearInterval(id); };
  }, [base]);

  return (
    <div className="h-screen flex flex-col p-4 gap-3">
      <header className="flex items-center justify-between py-1">
          <div className="flex items-baseline gap-3">
            <h1 className="font-chat leading-none text-neutral-500 dark:text-neutral-400" style={{ fontSize: "20px" }}>
              typhoon
            </h1>
            <span className="text-neutral-600 dark:text-neutral-700 text-xs">·</span>
            <span className="text-xs text-neutral-500 dark:text-neutral-600">8×h100 · b=1</span>
          </div>
          <div className="flex items-center gap-3">
            <input
              className="bg-transparent border border-neutral-300 dark:border-neutral-800 rounded px-2 py-1 text-xs text-neutral-500 dark:text-neutral-500 w-56 focus:outline-none focus:border-neutral-400 dark:focus:border-neutral-600 transition-colors"
              value={base}
              onChange={(e) => setBase(e.target.value)}
              aria-label="backend url"
            />
            <button
              onClick={() => setDark(!dark)}
              className="text-xs text-neutral-500 dark:text-neutral-600 hover:text-neutral-300 dark:hover:text-neutral-400 transition-colors px-2 py-1 border border-neutral-300 dark:border-neutral-800 rounded"
            >
              {dark ? "light" : "dark"}
            </button>
          </div>
      </header>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-[1fr_400px] gap-3 min-h-0">
        <ChatPane base={base} onTurn={onTurn} />
        <div className="flex flex-col gap-3 min-h-0" data-testid="right-rail">
          <LatencyPanel telemetry={turn.telemetry} summary={turn.summary} />
          <GpuExpertViz telemetry={turn.telemetry} topology={topology} />
        </div>
      </div>
    </div>
  );
}
