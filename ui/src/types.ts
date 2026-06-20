export interface Expert { layer: number; expert_id: number; gpu: number; }
export interface Telemetry {
  token_index: number;
  t_ms: number;
  experts: Expert[];
  spec: { proposed: number; accepted: number };
}
export interface Summary {
  ttft_ms: number;
  decode_tok_per_s: number;
  prefill_tokens: number;
  completion_tokens: number;
  spec_accept_rate: number;
}
export interface GpuInfo {
  id: number;
  name: string;
  mem_total_mb: number;
  mem_used_mb: number;
  utilization_pct: number;
  temp_c: number;
}
export interface Topology {
  gpus: GpuInfo[];
  num_layers: number;
  experts_per_layer: number;
  placement: Record<string, Record<string, number>>;
}
export interface ChatMessage { role: "user" | "assistant" | "system"; content: string; }
