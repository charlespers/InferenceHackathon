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
  const [turn, setTurn] = useState<{ telemetry: Telemetry[]; summary: Summary | null }>({
    telemetry: [], summary: null,
  });
  const onTurn = useCallback((t: { telemetry: Telemetry[]; summary: Summary | null }) => setTurn(t), []);

  useEffect(() => {
    let alive = true;
    getTopology(base).then((t) => alive && setTopology(t)).catch(() => alive && setTopology(null));
    return () => { alive = false; };
  }, [base]);

  return (
    <div className="h-screen flex flex-col p-4 gap-3">
      <header className="flex items-center justify-between">
        <h1 className="text-lg font-semibold tracking-tight">Inference Console
          <span className="text-neutral-500 text-sm font-normal"> · 8×H100 · B=1</span></h1>
        <input className="bg-neutral-900 border border-neutral-800 rounded px-2 py-1 text-xs w-72"
               value={base} onChange={(e) => setBase(e.target.value)} aria-label="backend url" />
      </header>
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-[1fr_420px] gap-3 min-h-0">
        <ChatPane base={base} onTurn={onTurn} />
        <div className="flex flex-col gap-3 min-h-0" data-testid="right-rail">
          <LatencyPanel telemetry={turn.telemetry} summary={turn.summary} />
          <GpuExpertViz telemetry={turn.telemetry} topology={topology} />
        </div>
      </div>
    </div>
  );
}
