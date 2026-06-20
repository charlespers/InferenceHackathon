import { useCallback, useRef, useState } from "react";
import { streamChat } from "./apiClient";
import type { ChatMessage, Telemetry, Summary } from "../types";

type Status = "idle" | "streaming" | "error";
interface SendOpts { model: string; temperature: number; max_tokens: number; }

export function useChatStream(base: string) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [telemetry, setTelemetry] = useState<Telemetry[]>([]);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const send = useCallback((text: string, opts: SendOpts) => {
    const history: ChatMessage[] = [...messages, { role: "user", content: text }];
    setMessages([...history, { role: "assistant", content: "" }]);
    setTelemetry([]);
    setSummary(null);
    setError(null);
    setStatus("streaming");
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    (async () => {
      try {
        for await (const chunk of streamChat(base, { ...opts, messages: history }, ctrl.signal)) {
          const piece = chunk?.choices?.[0]?.delta?.content;
          if (piece) {
            setMessages((prev) => {
              const next = prev.slice();
              const last = next[next.length - 1];
              next[next.length - 1] = { ...last, content: last.content + piece };
              return next;
            });
          }
          if (chunk?.x_telemetry) setTelemetry((p) => [...p, chunk.x_telemetry]);
          if (chunk?.x_summary) setSummary(chunk.x_summary);
        }
        setStatus("idle");
      } catch (e: any) {
        if (e?.name === "AbortError") { setStatus("idle"); return; }
        setError(String(e?.message ?? e));
        setStatus("error");
      }
    })();
  }, [base, messages]);

  const cancel = useCallback(() => abortRef.current?.abort(), []);

  return { messages, telemetry, summary, status, error, send, cancel };
}
