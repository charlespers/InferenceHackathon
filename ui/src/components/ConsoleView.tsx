import { useCallback, useEffect, useState } from "react";
import { ChatPane } from "./ChatPane";
import { LatencyPanel } from "./LatencyPanel";
import { GpuExpertViz } from "./GpuExpertViz";
import { getTopology } from "../lib/apiClient";
import type { Telemetry, Summary, Topology } from "../types";

/** Single-stream instrument console: chat + live latency + GPU/expert routing. */
export function ConsoleView({ base }: { base: string }) {
  const [topology, setTopology] = useState<Topology | null>(null);
  const [turn, setTurn] = useState<{ telemetry: Telemetry[]; summary: Summary | null }>({
    telemetry: [], summary: null,
  });
  const onTurn = useCallback(
    (t: { telemetry: Telemetry[]; summary: Summary | null }) => setTurn(t),
    [],
  );

  useEffect(() => {
    let alive = true;
    getTopology(base).then((t) => alive && setTopology(t)).catch(() => alive && setTopology(null));
    return () => { alive = false; };
  }, [base]);

  return (
    <div className="flex-1 grid grid-cols-1 lg:grid-cols-[1fr_420px] gap-3 min-h-0 animate-fade-up">
      <ChatPane base={base} onTurn={onTurn} />
      <div className="flex flex-col gap-3 min-h-0" data-testid="right-rail">
        <LatencyPanel telemetry={turn.telemetry} summary={turn.summary} />
        <GpuExpertViz telemetry={turn.telemetry} topology={topology} />
      </div>
    </div>
  );
}
