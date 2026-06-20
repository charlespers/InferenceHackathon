import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

const send = vi.fn();
vi.mock("../lib/useChatStream", () => ({
  useChatStream: () => ({
    messages: [{ role: "user", content: "hi" }, { role: "assistant", content: "**yo**" }],
    telemetry: [], summary: null, status: "idle", error: null, send, cancel: vi.fn(),
  }),
}));

import { ChatPane } from "./ChatPane";

describe("ChatPane", () => {
  it("renders messages and sends on click", () => {
    render(<ChatPane base="http://x" onTurn={() => {}} />);
    expect(screen.getByText("hi")).toBeTruthy();
    expect(screen.getByText("yo")).toBeTruthy(); // markdown bold -> text node
    const input = screen.getByPlaceholderText(/message/i) as HTMLTextAreaElement;
    fireEvent.change(input, { target: { value: "hello there" } });
    fireEvent.click(screen.getByRole("button", { name: /send/i }));
    expect(send).toHaveBeenCalledWith("hello there", expect.objectContaining({ model: expect.any(String) }));
  });
});
