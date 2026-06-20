import type { Topology, ChatMessage } from "../types";

export async function* parseSSE(stream: ReadableStream<Uint8Array>): AsyncGenerator<any> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 2);
      if (!frame.startsWith("data:")) continue;
      const payload = frame.slice(frame.indexOf(":") + 1).trim();
      if (payload === "[DONE]" || payload === "") continue;
      yield JSON.parse(payload);
    }
  }
}

export async function* streamChat(
  base: string,
  body: {
    model: string;
    messages: ChatMessage[];
    temperature: number;
    max_tokens: number;
    engine?: string;
  },
  signal: AbortSignal,
): AsyncGenerator<any> {
  const res = await fetch(`${base}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...body, stream: true }),
    signal,
  });
  if (!res.ok || !res.body) {
    const msg = await res.text().catch(() => res.statusText);
    throw new Error(`chat request failed (${res.status}): ${msg}`);
  }
  yield* parseSSE(res.body);
}

export async function getTopology(base: string): Promise<Topology> {
  const res = await fetch(`${base}/v1/topology`);
  if (!res.ok) throw new Error(`topology failed: ${res.status}`);
  return res.json();
}

export async function getModels(base: string): Promise<string[]> {
  const res = await fetch(`${base}/v1/models`);
  if (!res.ok) throw new Error(`models failed: ${res.status}`);
  const data = await res.json();
  return (data.data ?? []).map((m: any) => m.id);
}
