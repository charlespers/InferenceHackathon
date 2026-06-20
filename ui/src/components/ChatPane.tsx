import { useEffect, useState } from "react";
import Markdown from "react-markdown";
import { useChatStream } from "../lib/useChatStream";
import type { Telemetry, Summary } from "../types";

interface Props {
  base: string;
  onTurn: (t: { telemetry: Telemetry[]; summary: Summary | null }) => void;
}

export function ChatPane({ base, onTurn }: Props) {
  const chat = useChatStream(base);
  const [text, setText] = useState("");
  const [model, setModel] = useState("qwen3-235b-a22b");
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(256);

  useEffect(() => {
    onTurn({ telemetry: chat.telemetry, summary: chat.summary });
  }, [chat.telemetry, chat.summary, onTurn]);

  const streaming = chat.status === "streaming";
  const submit = () => {
    if (!text.trim() || streaming) return;
    chat.send(text.trim(), { model, temperature, max_tokens: maxTokens });
    setText("");
  };

  return (
    <div className="flex flex-col h-full panel overflow-hidden">
      <div className="flex gap-3 items-center px-3 py-2 border-b hair text-xs text-ink-mute">
        <input className="bg-paper border hair px-2 py-1 w-40 metric-num text-ink outline-none focus:border-conifer" value={model}
               onChange={(e) => setModel(e.target.value)} aria-label="model" />
        <label className="flex items-center gap-1.5 micro">temp
          <input type="number" step="0.1" min="0" max="2" value={temperature}
                 onChange={(e) => setTemperature(+e.target.value)}
                 className="bg-paper border hair px-2 py-1 w-16 metric-num text-ink outline-none" /></label>
        <label className="flex items-center gap-1.5 micro">max
          <input type="number" min="1" value={maxTokens}
                 onChange={(e) => setMaxTokens(+e.target.value)}
                 className="bg-paper border hair px-2 py-1 w-20 metric-num text-ink outline-none" /></label>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {chat.messages.map((m, i) => (
          <div key={i} className={m.role === "user" ? "text-ink" : "text-ink-soft"}>
            <div className="micro mb-1.5">{m.role}</div>
            <div className="text-[13px] leading-relaxed whitespace-pre-wrap">
              {m.role === "assistant" ? <Markdown>{m.content}</Markdown> : m.content}
            </div>
          </div>
        ))}
        {chat.error && <div className="text-sm" style={{ color: "var(--studio-fail)" }}>error: {chat.error}</div>}
      </div>

      <div className="border-t hair p-3 flex gap-2">
        <textarea
          className="flex-1 bg-paper border hair px-3 py-2 text-sm resize-none h-12 text-ink placeholder:text-ink-faint outline-none focus:border-conifer"
          placeholder="Message the model…" value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } }}
        />
        {streaming
          ? <button className="px-4 text-sm border hair stripe" style={{ color: "var(--studio-fail)", background: "var(--studio-sunk)", borderInlineStartColor: "var(--studio-fail)" }} onClick={chat.cancel}>Stop</button>
          : <button className="px-4 text-sm border" style={{ color: "var(--studio-card)", background: "var(--conifer)", borderColor: "var(--conifer)" }} onClick={submit}>Send</button>}
      </div>
    </div>
  );
}
