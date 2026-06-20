import { useEffect, useRef, useState } from "react";
import Markdown from "react-markdown";
import { useChatStream } from "../lib/useChatStream";
import type { Telemetry, Summary } from "../types";

interface Props {
  base: string;
  onTurn: (t: { telemetry: Telemetry[]; summary: Summary | null }) => void;
}

function Logo({ className }: { className?: string }) {
  const cx = 80, cy = 80;
  const outerR = 64;
  const innerR = 30;
  const nArms = 6;
  const turns = 0.50;
  const steps = 80;


  const arms = Array.from({ length: nArms }, (_, arm) => {
    const baseAngle = (arm / nArms) * Math.PI * 2;
    return Array.from({ length: steps + 1 }, (__, i) => {
      const t = i / steps;
      const angle = baseAngle + t * turns * Math.PI * 2;
      const r = innerR + t * (outerR - innerR); // start at inner circle edge
      const x = (cx + r * Math.cos(angle)).toFixed(1);
      const y = (cy + r * Math.sin(angle)).toFixed(1);
      return i === 0 ? `M ${x},${y}` : `L ${x},${y}`;
    }).join(' ');
  });

  return (
    <svg viewBox="0 0 160 160" className={className}>
      <g className="text-neutral-400 dark:text-neutral-600" fill="none" stroke="currentColor">
        <circle cx={cx} cy={cy} r={outerR} strokeWidth="0.8" />
        <circle cx={cx} cy={cy} r={innerR} strokeWidth="0.7" />
        {arms.map((d, i) => <path key={i} d={d} strokeWidth="0.9" />)}
      </g>
    </svg>
  );
}

function InputBox({
  text, setText, onSubmit, onCancel, streaming,
  placeholder = "message the model…",
  className = "",
}: {
  text: string;
  setText: (v: string) => void;
  onSubmit: () => void;
  onCancel: () => void;
  streaming: boolean;
  placeholder?: string;
  className?: string;
}) {
  return (
    <div className={`flex gap-2 items-end ${className}`}>
      <textarea
        className="font-chat flex-1 bg-transparent text-sm resize-none focus:outline-none
                   text-neutral-900 dark:text-neutral-200
                   placeholder:text-neutral-400 dark:placeholder:text-neutral-600
                   min-h-[2rem] max-h-32"
        placeholder={placeholder}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); onSubmit(); } }}
        rows={1}
      />
      {streaming
        ? <button onClick={onCancel}
            className="text-[11px] px-3 py-1 rounded shrink-0
                       border border-red-300 dark:border-red-900
                       text-red-500 dark:text-red-600 transition-colors">stop</button>
        : <button onClick={onSubmit}
            className="text-[11px] px-3 py-1 rounded shrink-0
                       border border-black/20 dark:border-white/10
                       text-neutral-600 dark:text-neutral-500
                       hover:text-neutral-900 dark:hover:text-neutral-300 transition-colors">send</button>
      }
    </div>
  );
}

export function ChatPane({ base, onTurn }: Props) {
  const chat = useChatStream(base);
  const [text, setText] = useState("");
  const [model, setModel] = useState("qwen3-235b-fp8");
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(256);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    onTurn({ telemetry: chat.telemetry, summary: chat.summary });
  }, [chat.telemetry, chat.summary, onTurn]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chat.messages]);

  const streaming = chat.status === "streaming";
  const empty = chat.messages.length === 0;

  const submit = () => {
    if (!text.trim() || streaming) return;
    chat.send(text.trim(), { model, temperature, max_tokens: maxTokens });
    setText("");
  };

  return (
    <div className="flex flex-col h-full overflow-hidden rounded-xl
                    border border-black/10 dark:border-white/[0.05]
                    bg-white/30 dark:bg-white/[0.02]
                    backdrop-blur-sm">

      {/* toolbar — always visible */}
      <div className="flex gap-3 items-center px-4 py-2.5 shrink-0
                      border-b border-black/10 dark:border-white/[0.05]
                      text-[11px] text-neutral-500 dark:text-neutral-600">
        <input className="bg-transparent w-36 focus:outline-none
                          text-neutral-600 dark:text-neutral-500"
               value={model} onChange={(e) => setModel(e.target.value)}
               aria-label="model" />
        <span className="opacity-30">·</span>
        <label className="flex items-center gap-1.5">
          temp
          <input type="number" step="0.1" min="0" max="2" value={temperature}
                 onChange={(e) => setTemperature(+e.target.value)}
                 className="bg-transparent w-10 focus:outline-none
                            text-neutral-600 dark:text-neutral-500" />
        </label>
        <span className="opacity-30">·</span>
        <label className="flex items-center gap-1.5">
          max
          <input type="number" min="1" value={maxTokens}
                 onChange={(e) => setMaxTokens(+e.target.value)}
                 className="bg-transparent w-14 focus:outline-none
                            text-neutral-600 dark:text-neutral-500" />
        </label>
      </div>

      {empty ? (
        /* ── Empty state: logo + tagline + input all centered ── */
        <div className="flex-1 flex flex-col items-center justify-center gap-6 px-8 pb-8">
          <Logo className="w-44 h-44 opacity-60" />
          <p className="font-chat text-4xl text-neutral-500 dark:text-neutral-400 select-none" style={{ overflow: "visible" }}>
            {(() => {
              // 7 irregular polygonal shards per letter — jagged fracture lines across 3 rows.
              // At f≈0 they overlap perfectly (normal letter); at f→1 each piece drifts independently.
              const shards = [
                { clip: "polygon(0% 0%, 48% 0%, 42% 36%, 0% 32%)",                dx: -3, dy: -5, rot: -7 },
                { clip: "polygon(48% 0%, 100% 0%, 100% 34%, 42% 36%)",            dx:  2, dy: -6, rot:  5 },
                { clip: "polygon(0% 32%, 42% 36%, 37% 64%, 0% 60%)",              dx: -5, dy:  0, rot:  6 },
                { clip: "polygon(42% 36%, 72% 33%, 66% 64%, 37% 64%)",            dx:  0, dy: -4, rot: -5 },
                { clip: "polygon(72% 33%, 100% 34%, 100% 67%, 66% 64%)",          dx:  6, dy:  1, rot:  8 },
                { clip: "polygon(0% 60%, 37% 64%, 31% 100%, 0% 100%)",            dx: -4, dy:  5, rot: -6 },
                { clip: "polygon(37% 64%, 100% 67%, 100% 100%, 31% 100%)",        dx:  3, dy:  6, rot:  5 },
              ];
              return "Typhoon".split("").map((ch, i, arr) => {
                const t = i / (arr.length - 1);
                const f = Math.min(Math.pow(t, 2.6), 0.65); // cap so even last letter doesn't fully scatter
                return (
                  <span key={i} style={{
                    display: "inline-block",
                    position: "relative",
                    marginRight: `${4 + f * 8}px`,
                    opacity: Math.max(0.8, 1 - f * 0.15),
                  }}>
                    <span style={{ visibility: "hidden" }}>{ch}</span>
                    {shards.map((s, si) => (
                      <span key={si} style={{
                        position: "absolute",
                        inset: 0,
                        clipPath: s.clip,
                        transform: `translate(${s.dx * f * 2.4}px, ${s.dy * f * 2.1}px) rotate(${s.rot * f * 0.55}deg)`,
                        transformOrigin: "center center",
                        filter: undefined,
                      }}>{ch}</span>
                    ))}
                  </span>
                );
              });
            })()}
          </p>
          <div className="w-full max-w-lg rounded-xl px-4 py-3
                          border border-black/30 dark:border-white/[0.08]
                          bg-white/40 dark:bg-white/[0.04]">
            <InputBox text={text} setText={setText}
                      onSubmit={submit} onCancel={chat.cancel}
                      streaming={streaming} />
          </div>
        </div>
      ) : (
        /* ── Active: messages scrollable + input pinned at bottom ── */
        <>
          <div className="flex-1 overflow-y-auto flex flex-col min-h-0">
            <div className="px-6 py-5 space-y-6">
              {chat.messages.map((m, i) => (
                <div key={i} className="space-y-1.5">
                  <div className="text-[9px] uppercase tracking-widest
                                  text-neutral-400 dark:text-neutral-700">
                    {m.role}
                  </div>
                  <div className={`font-chat text-sm leading-relaxed ${
                    m.role === "user"
                      ? "text-neutral-700 dark:text-neutral-300"
                      : "text-neutral-900 dark:text-neutral-200"
                  }`}>
                    {m.role === "assistant"
                      ? <Markdown>{m.content}</Markdown>
                      : m.content}
                  </div>
                </div>
              ))}
              {chat.error && (
                <div className="font-chat text-red-600 dark:text-red-500 text-xs px-3 py-2 rounded
                                bg-red-50 dark:bg-red-950/30
                                border border-red-200 dark:border-red-900/40">
                  {chat.error}
                </div>
              )}
              <div ref={bottomRef} />
            </div>
          </div>

          <div className="px-4 py-3 shrink-0
                          border-t border-black/10 dark:border-white/[0.05]">
            <InputBox text={text} setText={setText}
                      onSubmit={submit} onCancel={chat.cancel}
                      streaming={streaming} />
          </div>
        </>
      )}
    </div>
  );
}
