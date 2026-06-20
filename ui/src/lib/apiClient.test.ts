import { describe, it, expect } from "vitest";
import { parseSSE } from "./apiClient";

function streamFrom(chunks: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const c of chunks) controller.enqueue(enc.encode(c));
      controller.close();
    },
  });
}

describe("parseSSE", () => {
  it("parses framed events and skips DONE", async () => {
    const s = streamFrom([
      'data: {"choices":[{"delta":{"content":"a"}}]}\n\n',
      'data: {"x_summary":{"ttft_ms":1}}\n\n',
      "data: [DONE]\n\n",
    ]);
    const out: any[] = [];
    for await (const e of parseSSE(s)) out.push(e);
    expect(out).toHaveLength(2);
    expect(out[0].choices[0].delta.content).toBe("a");
    expect(out[1].x_summary.ttft_ms).toBe(1);
  });

  it("reassembles events split across chunk boundaries", async () => {
    const s = streamFrom(['data: {"choices":[{"de', 'lta":{"content":"hi"}}]}\n\n']);
    const out: any[] = [];
    for await (const e of parseSSE(s)) out.push(e);
    expect(out[0].choices[0].delta.content).toBe("hi");
  });
});
