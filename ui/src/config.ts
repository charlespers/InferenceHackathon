export function getDefaultBase(): string {
  return (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://localhost:8000";
}

export interface LaneConfig {
  id: "conifer" | "vllm";
  /** Display name. */
  label: string;
  /** Backend `engine` profile to request (mock) — ignored by a real OpenAI server. */
  engine: string;
  /** Sub-label describing the serving setup. */
  detail: string;
  /** Per-lane base URL. Defaults to the shared base; set to race two real engines. */
  base?: string;
}

/**
 * The two contenders. Both hit B=1 (no batching). By default they share one backend
 * and differ only by `engine` profile; point `base` at distinct URLs to race a real
 * Conifer engine against a real vLLM server.
 */
export const LANES: LaneConfig[] = [
  {
    id: "conifer",
    label: "Conifer",
    engine: "conifer",
    detail: "8×H100 · expert-parallel MoE · speculative decode",
    base: import.meta.env.VITE_CONIFER_BASE as string | undefined,
  },
  {
    id: "vllm",
    label: "vLLM",
    engine: "vllm",
    detail: "8×H100 · OpenAI-compatible baseline · B=1",
    base: import.meta.env.VITE_VLLM_BASE as string | undefined,
  },
];

/** Reasoning-flavored default so the speed difference maps to task quality. */
export const DEFAULT_PROMPT =
  "A snail climbs a 12-meter well, going up 3 meters each day and slipping 2 meters each night. On which day does it reach the top? Think step by step, then give the answer.";

export interface TaskPreset {
  id: string;
  label: string;
  base: number;
  plateau: number;
  growth: number;
}

/** Illustrative single-pass accuracies + self-consistency ceilings per task family. */
export const TASK_PRESETS: TaskPreset[] = [
  { id: "gsm8k", label: "Grade-school math", base: 0.74, plateau: 0.94, growth: 0.55 },
  { id: "aime", label: "Competition math", base: 0.32, plateau: 0.66, growth: 0.32 },
  { id: "code", label: "Code (unit-tested)", base: 0.51, plateau: 0.82, growth: 0.45 },
];
