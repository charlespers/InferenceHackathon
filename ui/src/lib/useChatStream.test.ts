import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";

const mockStreamChat = vi.fn();
vi.mock("./apiClient", () => ({ streamChat: (...a: any[]) => mockStreamChat(...a) }));

import { useChatStream } from "./useChatStream";

async function* fakeStream() {
  yield { choices: [{ delta: { content: "Hel" } }] };
  yield { choices: [{ delta: { content: "lo" } }], x_telemetry: { token_index: 0, t_ms: 8, experts: [{ layer: 0, expert_id: 1, gpu: 1 }], spec: { proposed: 4, accepted: 3 } } };
  yield { x_summary: { ttft_ms: 40, decode_tok_per_s: 125, prefill_tokens: 1, completion_tokens: 2, spec_accept_rate: 0.75 } };
}

describe("useChatStream", () => {
  beforeEach(() => mockStreamChat.mockReset());

  it("accumulates assistant content, telemetry, and summary", async () => {
    mockStreamChat.mockReturnValue(fakeStream());
    const { result } = renderHook(() => useChatStream("http://x"));
    act(() => result.current.send("hi", { model: "m", temperature: 0.7, max_tokens: 8 }));
    await waitFor(() => expect(result.current.status).toBe("idle"));
    const msgs = result.current.messages;
    expect(msgs[0]).toEqual({ role: "user", content: "hi" });
    expect(msgs[1]).toEqual({ role: "assistant", content: "Hello" });
    expect(result.current.telemetry).toHaveLength(1);
    expect(result.current.summary?.decode_tok_per_s).toBe(125);
  });

  it("sets error status on stream failure but keeps prior messages", async () => {
    mockStreamChat.mockImplementation(async function* () { throw new Error("boom"); });
    const { result } = renderHook(() => useChatStream("http://x"));
    act(() => result.current.send("hi", { model: "m", temperature: 0.7, max_tokens: 8 }));
    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(result.current.error).toContain("boom");
    expect(result.current.messages[0].content).toBe("hi");
  });
});
