export interface Expert { layer: number; expert_id: number; gpu: number; }
export interface Telemetry {
  token_index: number;
  t_ms: number;
  experts: Expert[];
  spec: { proposed: number; accepted: number };
}
export interface FloorBreakdown { weight: number; comms: number; kv: number; overhead: number; }
export interface Summary {
  ttft_ms: number;
  decode_tok_per_s: number;
  prefill_tokens: number;
  completion_tokens: number;
  spec_accept_rate: number;
  // Optimization telemetry (optional; server/optimization_telemetry.py). Absent on older backends.
  engine?: string;
  spec_enabled?: boolean;
  floor_breakdown_ms?: FloorBreakdown;   // weight/comms/kv/overhead ms per token (sums≈tpot)
  regime?: string;                       // "floor-bound" | "weight-bound"
  next_lever?: string;
  pct_of_ceiling?: number;               // decode_tok_per_s / ~2000 (fp8+spec ceiling)
  pct_of_roofline?: number;
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
