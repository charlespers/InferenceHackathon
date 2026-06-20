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
  const [model, setModel] = useState("qwen3-235b-fp8");
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
    <div className="flex flex-col h-full border border-neutral-800 rounded-lg overflow-hidden">
      <div className="flex gap-3 items-center px-3 py-2 border-b border-neutral-800 text-xs text-neutral-400">
        <input className="bg-neutral-900 rounded px-2 py-1 w-40" value={model}
               onChange={(e) => setModel(e.target.value)} aria-label="model" />
        <label className="flex items-center gap-1">temp
          <input type="number" step="0.1" min="0" max="2" value={temperature}
                 onChange={(e) => setTemperature(+e.target.value)}
                 className="bg-neutral-900 rounded px-2 py-1 w-16" /></label>
        <label className="flex items-center gap-1">max
          <input type="number" min="1" value={maxTokens}
                 onChange={(e) => setMaxTokens(+e.target.value)}
                 className="bg-neutral-900 rounded px-2 py-1 w-20" /></label>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {chat.messages.map((m, i) => (
          <div key={i} className={m.role === "user" ? "text-neutral-200" : "text-emerald-200"}>
            <div className="text-[10px] uppercase tracking-wide text-neutral-500 mb-1">{m.role}</div>
            <div className="prose prose-invert prose-sm max-w-none whitespace-pre-wrap">
              {m.role === "assistant" ? <Markdown>{m.content}</Markdown> : m.content}
            </div>
          </div>
        ))}
        {chat.error && <div className="text-red-400 text-sm">error: {chat.error}</div>}
      </div>

      <div className="border-t border-neutral-800 p-3 flex gap-2">
        <textarea
          className="flex-1 bg-neutral-900 rounded px-3 py-2 text-sm resize-none h-12"
          placeholder="Message the model…" value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } }}
        />
        {streaming
          ? <button className="px-4 rounded bg-red-600/80 text-sm" onClick={chat.cancel}>Stop</button>
          : <button className="px-4 rounded bg-emerald-600 text-sm" onClick={submit}>Send</button>}
      </div>
    </div>
  );
}
