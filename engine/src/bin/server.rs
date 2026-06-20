/// Axum inference server — replaces server/main.py + server/backend.py.
///
/// Endpoints:
///   GET  /health
///   GET  /v1/models
///   GET  /v1/topology
///   POST /v1/chat/completions   (SSE proxy to vLLM, injects x_telemetry + x_summary)
///   GET  /api/tasks             (who is running what right now)
///
/// Task tracking: pass X-User header or "user" field in request body to identify
/// yourself. Visit /api/tasks in a browser to see live request status.
///
/// Run: cargo run --release --bin server
/// Env: VLLM_URL (default http://localhost:8001), PORT (default 8000)

use std::{
    collections::HashMap,
    convert::Infallible,
    sync::{Arc, Mutex},
    time::Instant,
};

use axum::{
    extract::State,
    http::HeaderMap,
    response::{
        sse::{Event, KeepAlive, Sse},
        Html, IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use futures_util::StreamExt;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::CorsLayer;

use engine::routing::sim::RoutingSimulator;

const VLLM_URL_DEFAULT: &str = "http://localhost:8001";
const PORT_DEFAULT: u16 = 8000;
const TASKS_PORT_DEFAULT: u16 = 9000;
const N_GPUS: usize = 8;
const N_LAYERS: usize = 94;
const N_EXPERTS: usize = 128;
const ROUTING_STATS_PATH: &str = "/alloc/data/routing_stats.json";
const PLACEMENT_PATH: &str = "/alloc/data/optimized_placement.json";
const ROUTING_SOCK: &str = "/tmp/vllm_routing.sock";

// ---------------------------------------------------------------------------
// Real routing reader — reads from Unix socket written by start_vllm.py.
// Buffers one full token's worth of layer records (94 entries).
// Falls back to None when socket isn't connected.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct RoutingReader {
    /// Completed token routing: outer = tokens, inner = per-layer expert lists.
    /// Shared ring buffer; Mutex<VecDeque> so proxy can pop tokens in order.
    buffer: Arc<Mutex<std::collections::VecDeque<Vec<Vec<u32>>>>>,
    connected: Arc<std::sync::atomic::AtomicBool>,
}

impl RoutingReader {
    fn new() -> Self {
        let reader = Self {
            buffer: Arc::new(Mutex::new(std::collections::VecDeque::new())),
            connected: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        };
        let r = reader.clone();
        tokio::spawn(async move { r.run().await });
        reader
    }

    async fn run(&self) {
        loop {
            match tokio::net::UnixStream::connect(ROUTING_SOCK).await {
                Ok(stream) => {
                    self.connected.store(true, std::sync::atomic::Ordering::Relaxed);
                    eprintln!("[routing_reader] connected to {ROUTING_SOCK}");
                    self.read_stream(stream).await;
                    self.connected.store(false, std::sync::atomic::Ordering::Relaxed);
                    eprintln!("[routing_reader] disconnected, retrying…");
                }
                Err(_) => {
                    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                }
            }
        }
    }

    async fn read_stream(&self, stream: tokio::net::UnixStream) {
        use tokio::io::{AsyncBufReadExt, BufReader};
        let reader = BufReader::new(stream);
        let mut lines = reader.lines();
        // Accumulate one token's 94 layers
        let mut token_layers: Vec<(usize, Vec<u32>)> = Vec::with_capacity(N_LAYERS);

        while let Ok(Some(line)) = lines.next_line().await {
            let Ok(record) = serde_json::from_str::<serde_json::Value>(&line) else { continue };
            let layer = record["layer"].as_u64().unwrap_or(0) as usize;
            let experts: Vec<u32> = record["experts"]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_u64().map(|e| e as u32)).collect())
                .unwrap_or_default();

            token_layers.push((layer, experts));

            // Once we have all N_LAYERS for this token, push to buffer
            if token_layers.len() >= N_LAYERS {
                let mut by_layer = vec![vec![]; N_LAYERS];
                for (l, e) in token_layers.drain(..) {
                    if l < N_LAYERS { by_layer[l] = e; }
                }
                self.buffer.lock().unwrap().push_back(by_layer);
                // Cap buffer at 64 tokens to avoid unbounded growth
                let mut buf = self.buffer.lock().unwrap();
                while buf.len() > 64 { buf.pop_front(); }
            }
        }
    }

    /// Pop the next token's routing data (94 layers × top-k experts).
    /// Returns None if the socket isn't connected or buffer is empty.
    fn pop_token(&self) -> Option<Vec<Vec<u32>>> {
        self.buffer.lock().unwrap().pop_front()
    }

    #[allow(dead_code)]
    fn is_connected(&self) -> bool {
        self.connected.load(std::sync::atomic::Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Task tracking
// ---------------------------------------------------------------------------

struct ActiveTask {
    user: String,
    prompt_preview: String,
    started_at: Instant,
    tokens: u32,
}

struct TaskTracker {
    active: HashMap<u64, ActiveTask>,
    next_id: u64,
    total_served: u64,
    server_started: Instant,
}

impl TaskTracker {
    fn new() -> Self {
        Self {
            active: HashMap::new(),
            next_id: 0,
            total_served: 0,
            server_started: Instant::now(),
        }
    }

    fn register(&mut self, user: String, prompt_preview: String) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.active.insert(id, ActiveTask { user, prompt_preview, started_at: Instant::now(), tokens: 0 });
        id
    }

    fn tick(&mut self, id: u64) {
        if let Some(t) = self.active.get_mut(&id) { t.tokens += 1; }
    }

    fn finish(&mut self, id: u64) {
        self.active.remove(&id);
        self.total_served += 1;
    }
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// GPU / vLLM live stats — polled every 2s so /api/tasks shows real GPU load
// regardless of whether requests come through this server or bypass it.
// ---------------------------------------------------------------------------

#[derive(Default, Clone)]
struct GpuProc {
    pid: u32,
    label: String,
    mem_mb: u64,
}

#[derive(Default, Clone)]
struct LiveStats {
    /// Per-GPU: (peak_util_pct, mem_used_mb, mem_total_mb).
    /// util is a rolling peak over ~6 s to avoid 0% snapshots between token bursts.
    gpus: Vec<(u64, u64, u64)>,
    /// Processes currently using any GPU, from nvidia-smi --query-compute-apps.
    gpu_procs: Vec<GpuProc>,
}

struct AppState {
    client: reqwest::Client,
    vllm_url: String,
    simulator: Mutex<RoutingSimulator>,
    /// Placement JSON cached for x_telemetry ({"placement": {"0": {"0": 2, ...}}})
    placement_json: Value,
    tasks: Mutex<TaskTracker>,
    /// Real routing data from vLLM hook (start_vllm.py). None when not connected.
    routing_reader: RoutingReader,
    /// Live GPU + vLLM metrics, refreshed every 2 s by a background task.
    live: Mutex<LiveStats>,
}

/// Parse a compute-apps name into something human-readable.
fn label_gpu_proc(name: &str) -> Option<String> {
    if name.contains("resource_tracker") { return None; }
    if name.contains("vllm") || name.contains("api_server") {
        return Some("vLLM API server".to_string());
    }
    if name.contains("spawn_main") {
        if let Some(tp) = name.split("fork_TP").nth(1).and_then(|s| s.split(')').next()) {
            return Some(format!("vLLM TP worker {tp}"));
        }
        return Some("vLLM worker".to_string());
    }
    if name.contains("python") { return Some("python".to_string()); }
    Some(name.split('/').last().unwrap_or(name).chars().take(40).collect())
}

/// Background task: refresh LiveStats every 2 s by reading directly from nvidia-smi.
/// No dependency on vLLM — shows whatever is running on the GPU.
async fn stats_poller(state: Arc<AppState>) {
    // Rolling peak utilization per GPU across 3 polls (~6 s) to avoid
    // catching the idle gaps between token bursts at B=1.
    let mut util_window: Vec<Vec<u64>> = Vec::new();

    loop {
        // Per-GPU: utilization + memory
        let smi_out = tokio::process::Command::new("nvidia-smi")
            .args(["--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"])
            .output().await;
        let current: Vec<(u64, u64, u64)> = match smi_out {
            Ok(o) if o.status.success() =>
                String::from_utf8_lossy(&o.stdout).lines()
                    .filter_map(|l| {
                        let p: Vec<&str> = l.split(',').map(str::trim).collect();
                        if p.len() < 3 { return None; }
                        Some((p[0].parse().unwrap_or(0), p[1].parse().unwrap_or(0), p[2].parse().unwrap_or(0)))
                    })
                    .collect(),
            _ => vec![],
        };

        util_window.push(current.iter().map(|(u, _, _)| *u).collect());
        if util_window.len() > 3 { util_window.remove(0); }

        let gpus: Vec<(u64, u64, u64)> = current.iter().enumerate().map(|(i, (_, used, total))| {
            let peak = util_window.iter().filter_map(|w| w.get(i)).copied().max().unwrap_or(0);
            (peak, *used, *total)
        }).collect();

        // Processes using GPU memory — ground truth for what's running, source of truth
        let apps_out = tokio::process::Command::new("nvidia-smi")
            .args(["--query-compute-apps=pid,used_memory,name", "--format=csv,noheader,nounits"])
            .output().await;
        let mut gpu_procs: Vec<GpuProc> = match apps_out {
            Ok(o) if o.status.success() => {
                let mut seen_pids = std::collections::HashSet::new();
                String::from_utf8_lossy(&o.stdout).lines()
                    .filter_map(|l| {
                        let p: Vec<&str> = l.splitn(3, ',').map(str::trim).collect();
                        if p.len() < 3 { return None; }
                        let pid: u32 = p[0].parse().ok()?;
                        if !seen_pids.insert(pid) { return None; }
                        let mem_mb: u64 = p[1].parse().unwrap_or(0);
                        let label = label_gpu_proc(p[2])?;
                        Some(GpuProc { pid, label, mem_mb })
                    })
                    .collect()
            }
            _ => vec![],
        };
        gpu_procs.sort_by(|a, b| b.mem_mb.cmp(&a.mem_mb));

        *state.live.lock().unwrap() = LiveStats { gpus, gpu_procs };

        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let vllm_url = std::env::var("VLLM_URL").unwrap_or_else(|_| VLLM_URL_DEFAULT.to_string());
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(PORT_DEFAULT);
    let tasks_port: u16 = std::env::var("TASKS_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(TASKS_PORT_DEFAULT);

    let simulator = RoutingSimulator::from_routing_stats(ROUTING_STATS_PATH)
        .unwrap_or_else(|e| {
            eprintln!("[server] routing stats unavailable ({e}), using uniform predictor");
            RoutingSimulator::default()
        });

    let placement_json: Value = std::fs::read_to_string(PLACEMENT_PATH)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| json!({"placement": {}}));

    let state = Arc::new(AppState {
        client: reqwest::Client::new(),
        vllm_url: vllm_url.clone(),
        simulator: Mutex::new(simulator),
        placement_json,
        tasks: Mutex::new(TaskTracker::new()),
        routing_reader: RoutingReader::new(),
        live: Mutex::new(LiveStats::default()),
    });

    // Background GPU + vLLM metrics poller
    tokio::spawn(stats_poller(Arc::clone(&state)));

    // Main proxy — port 8000
    let proxy_app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/topology", get(topology))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(CorsLayer::permissive())
        .with_state(Arc::clone(&state));

    // Monitor — port 9000 (only /api/tasks)
    let tasks_app = Router::new()
        .route("/api/tasks", get(tasks_handler))
        .layer(CorsLayer::permissive())
        .with_state(Arc::clone(&state));

    let proxy_addr = format!("0.0.0.0:{port}");
    let tasks_addr = format!("0.0.0.0:{tasks_port}");
    eprintln!("[server] proxy on {proxy_addr}  tasks on {tasks_addr}  vllm={vllm_url}");

    let proxy_listener = tokio::net::TcpListener::bind(&proxy_addr).await.unwrap();
    let tasks_listener = tokio::net::TcpListener::bind(&tasks_addr).await.unwrap();

    tokio::join!(
        axum::serve(proxy_listener, proxy_app),
        axum::serve(tasks_listener, tasks_app),
    ).0.unwrap();
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn models() -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{"id": "qwen3-235b-a22b", "object": "model"}]
    }))
}

async fn topology(State(state): State<Arc<AppState>>) -> Json<Value> {
    let gpus = query_gpus().await;
    let placement = state.placement_json["placement"].clone();
    let placement_source = if state.placement_json.get("stats").is_some() {
        "optimized"
    } else {
        "round-robin"
    };
    Json(json!({
        "gpus": gpus,
        "num_layers": N_LAYERS,
        "experts_per_layer": N_EXPERTS,
        "placement": placement,
        "placement_source": placement_source,
    }))
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(mut body): Json<Value>,
) -> Response {
    let wants_stream = body["stream"].as_bool().unwrap_or(false);

    // Identify the caller: X-User header, then "user" field in body, then "unknown"
    let user = headers
        .get("x-user")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .or_else(|| body["user"].as_str().map(str::to_string))
        .unwrap_or_else(|| "unknown".to_string());

    // Prompt preview from the last user message
    let prompt_preview = body["messages"]
        .as_array()
        .and_then(|msgs| msgs.iter().rev().find(|m| m["role"] == "user"))
        .and_then(|m| m["content"].as_str())
        .map(|s| {
            let trimmed = s.trim();
            if trimmed.len() > 80 { format!("{}…", &trimmed[..80]) } else { trimmed.to_string() }
        })
        .unwrap_or_default();

    let task_id = state.tasks.lock().unwrap().register(user, prompt_preview);

    // Always stream internally; non-streaming collects and wraps.
    body["stream"] = json!(true);
    // Ensure thinking mode disabled for Qwen3
    body["model"] = json!("/alloc/data/Qwen3-235B-A22B");
    {
        let ktw = body["chat_template_kwargs"].as_object_mut();
        if let Some(m) = ktw {
            m.entry("enable_thinking").or_insert(json!(false));
        } else {
            body["chat_template_kwargs"] = json!({"enable_thinking": false});
        }
    }

    let vllm_url = format!("{}/v1/chat/completions", state.vllm_url);
    // Channel carries raw JSON strings so we can read them back for non-streaming.
    let (tx, rx) = mpsc::channel::<String>(128);

    tokio::spawn(async move {
        proxy_and_inject(state, body, vllm_url, tx, task_id).await;
    });

    if wants_stream {
        // Convert String → Event for SSE
        let event_stream = ReceiverStream::new(rx)
            .map(|s| -> Result<Event, Infallible> { Ok(Event::default().data(s)) });
        Sse::new(event_stream).keep_alive(KeepAlive::default()).into_response()
    } else {
        // Collect all content for non-streaming response
        let mut rx = rx;
        let mut full_text = String::new();
        while let Some(data) = rx.recv().await {
            if let Ok(chunk) = serde_json::from_str::<Value>(&data) {
                if let Some(c) = chunk["choices"][0]["delta"]["content"].as_str() {
                    full_text.push_str(c);
                }
            }
        }
        Json(json!({
            "id": "chatcmpl-rust",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": full_text}, "finish_reason": "stop"}]
        })).into_response()
    }
}

// ---------------------------------------------------------------------------
// Core SSE proxy: forwards to vLLM and injects x_telemetry + x_summary
// ---------------------------------------------------------------------------

async fn proxy_and_inject(
    state: Arc<AppState>,
    body: Value,
    vllm_url: String,
    tx: mpsc::Sender<String>,
    task_id: u64,
) {
    let resp = match state.client.post(&vllm_url).json(&body).send().await {
        Ok(r) => r,
        Err(e) => {
            let _ = tx.send(
                json!({"error": format!("vLLM unreachable: {e}")}).to_string()
            ).await;
            return;
        }
    };

    let mut byte_stream = resp.bytes_stream();
    let mut buf = String::new();
    let t_start = Instant::now();
    let mut t_first: Option<Instant> = None;
    let mut token_count: u32 = 0;
    let mut hit_rate_sum = 0.0f32;
    let mut n_simulated = 0u32;

    'outer: while let Some(chunk_result) = byte_stream.next().await {
        let bytes = match chunk_result {
            Ok(b) => b,
            Err(_) => break,
        };
        buf.push_str(&String::from_utf8_lossy(&bytes));

        loop {
            let Some(frame_end) = buf.find("\n\n") else { break };
            let frame = buf[..frame_end].trim().to_string();
            buf = buf[frame_end + 2..].to_string();

            let data = match frame.strip_prefix("data:") {
                Some(d) => d.trim(),
                None => continue,
            };

            if data == "[DONE]" {
                break 'outer;
            }

            let mut chunk: Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let has_content = chunk["choices"][0]["delta"]["content"]
                .as_str()
                .map(|s| !s.is_empty())
                .unwrap_or(false);

            if has_content {
                let now = Instant::now();
                if t_first.is_none() { t_first = Some(now); }
                let t_ms = t_start.elapsed().as_secs_f64() * 1000.0;

                // Run predictor for this token.
                // If the vLLM routing hook (start_vllm.py) is connected, use
                // real per-layer expert selections; otherwise fall back to simulation.
                let hr = if let Some(real_routing) = state.routing_reader.pop_token() {
                    state.simulator.lock().unwrap().score_real_routing(&real_routing)
                } else {
                    state.simulator.lock().unwrap().simulate_token()
                };
                hit_rate_sum += hr;
                n_simulated += 1;

                chunk["x_telemetry"] = make_telemetry(
                    token_count, t_ms, &state.placement_json,
                );
                state.tasks.lock().unwrap().tick(task_id);
                token_count += 1;
            }

            if tx.send(chunk.to_string()).await.is_err() {
                return; // client disconnected
            }
        }
    }

    // Emit x_summary
    let elapsed = t_start.elapsed().as_secs_f64();
    let ttft_ms = t_first
        .map(|t| (t - t_start).as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let decode_elapsed = t_first
        .map(|t| elapsed - (t - t_start).as_secs_f64())
        .unwrap_or(elapsed);
    let decode_tps = if decode_elapsed > 1e-6 && token_count > 1 {
        (token_count - 1) as f64 / decode_elapsed
    } else {
        0.0
    };
    let predictor_hit_rate = if n_simulated > 0 {
        Some((hit_rate_sum / n_simulated as f32 * 1000.0).round() / 1000.0)
    } else {
        None
    };

    let summary = json!({
        "x_summary": {
            "ttft_ms": (ttft_ms * 10.0).round() / 10.0,
            "decode_tok_per_s": (decode_tps * 10.0).round() / 10.0,
            "completion_tokens": token_count,
            "spec_accept_rate": 0.0,
            "predictor_hit_rate": predictor_hit_rate,
        }
    });
    let _ = tx.send(summary.to_string()).await;
    state.tasks.lock().unwrap().finish(task_id);
}

// ---------------------------------------------------------------------------
// /api/tasks — live request monitor
// ---------------------------------------------------------------------------

async fn tasks_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    let tracker = state.tasks.lock().unwrap();
    let uptime = tracker.server_started.elapsed().as_secs();
    let live = state.live.lock().unwrap().clone();

    let wants_html = headers
        .get("accept")
        .and_then(|v| v.to_str().ok())
        .map(|v| v.contains("text/html"))
        .unwrap_or(false);

    let active: Vec<Value> = tracker.active.values().map(|t| {
        let elapsed = t.started_at.elapsed();
        json!({
            "user": t.user,
            "prompt": t.prompt_preview,
            "elapsed_s": elapsed.as_secs(),
            "tokens": t.tokens,
            "tok_per_s": if elapsed.as_secs_f64() > 0.5 {
                (t.tokens as f64 / elapsed.as_secs_f64() * 10.0).round() / 10.0
            } else { 0.0 },
        })
    }).collect();

    if wants_html {
        let gpu_rows = if live.gpus.is_empty() {
            "<tr><td colspan='3' style='color:#888'>nvidia-smi unavailable</td></tr>".to_string()
        } else {
            live.gpus.iter().enumerate().map(|(i, (util, used, total))| {
                let bar_width = (*util as usize).min(100);
                let bar_color = if *util > 80 { "#f0883e" } else if *util > 20 { "#3fb950" } else { "#444" };
                let bar = format!(
                    "<div style='background:#21262d;width:140px;display:inline-block;border-radius:3px;vertical-align:middle'>\
                     <div style='background:{bar_color};width:{bar_width}%;height:10px;border-radius:3px'></div></div>"
                );
                format!("<tr><td>GPU {i}</td><td>{bar} &nbsp;{util}%</td><td>{used} / {total} MB</td></tr>")
            }).collect::<Vec<_>>().join("\n")
        };

        let proc_rows = if live.gpu_procs.is_empty() {
            "<tr><td colspan='3' style='color:#888'>No GPU processes detected</td></tr>".to_string()
        } else {
            live.gpu_procs.iter().map(|p| format!(
                "<tr><td>{}</td><td style='color:#8b949e'>{}</td><td>{} MB</td></tr>",
                p.label, p.pid, p.mem_mb,
            )).collect::<Vec<_>>().join("\n")
        };

        let proxy_rows = if active.is_empty() {
            "<tr><td colspan='5' style='color:#888'>No active requests via port 8000</td></tr>".to_string()
        } else {
            active.iter().map(|r| format!(
                "<tr><td>{}</td><td>{}</td><td>{}s</td><td>{}</td><td>{} tok/s</td></tr>",
                r["user"].as_str().unwrap_or(""),
                r["prompt"].as_str().unwrap_or(""),
                r["elapsed_s"], r["tokens"], r["tok_per_s"],
            )).collect::<Vec<_>>().join("\n")
        };

        let html = format!(r#"<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="2">
<title>GPU Monitor</title>
<style>
  body {{ font-family: monospace; background: #0d1117; color: #e6edf3; padding: 24px; }}
  h2 {{ color: #58a6ff; margin-bottom: 4px; }}
  h3 {{ color: #58a6ff; margin: 20px 0 6px; font-size: 1em; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 8px; }}
  th {{ background: #161b22; color: #8b949e; padding: 6px 12px; text-align: left; font-weight: normal; }}
  td {{ padding: 6px 12px; border-bottom: 1px solid #21262d; }}
  .meta {{ color: #8b949e; margin-top: 16px; font-size: 0.85em; }}
</style>
</head><body>
<h2>GPU Monitor</h2>

<h3>Per-GPU Utilization (peak over 6 s)</h3>
<table>
  <tr><th>GPU</th><th>Compute</th><th>Memory</th></tr>
  {gpu_rows}
</table>

<h3>Processes on GPU</h3>
<table>
  <tr><th>Process</th><th>PID</th><th>GPU Mem</th></tr>
  {proc_rows}
</table>

<h3>Requests via proxy (port 8000)</h3>
<table>
  <tr><th>User</th><th>Prompt</th><th>Elapsed</th><th>Tokens</th><th>Speed</th></tr>
  {proxy_rows}
</table>

<p class="meta">Total served: {total} &nbsp;|&nbsp; Uptime: {uptime}s &nbsp;|&nbsp; Auto-refreshes every 2s</p>
</body></html>"#,
            gpu_rows = gpu_rows,
            proc_rows = proc_rows,
            proxy_rows = proxy_rows,
            total = tracker.total_served,
            uptime = uptime,
        );

        Html(html).into_response()
    } else {
        Json(json!({
            "gpus": live.gpus.iter().enumerate().map(|(i, (u, m, t))| json!({
                "id": i, "util_pct": u, "mem_used_mb": m, "mem_total_mb": t,
            })).collect::<Vec<_>>(),
            "gpu_procs": live.gpu_procs.iter().map(|p| json!({
                "pid": p.pid, "label": p.label, "mem_mb": p.mem_mb,
            })).collect::<Vec<_>>(),
            "proxy_active": active,
            "total_served": tracker.total_served,
            "uptime_s": uptime,
        })).into_response()
    }
}

// ---------------------------------------------------------------------------
// Telemetry helpers
// ---------------------------------------------------------------------------

fn make_telemetry(token_index: u32, t_ms: f64, placement_json: &Value) -> Value {
    let top_k = 8usize;
    let layer = token_index as usize % N_LAYERS;
    let layer_placement = &placement_json["placement"][layer.to_string()];

    let experts: Vec<Value> = (0..top_k)
        .map(|k| {
            let expert_id = (token_index as usize * 7 + k * 13) % N_EXPERTS;
            let gpu = layer_placement[expert_id.to_string()]
                .as_u64()
                .unwrap_or((expert_id % N_GPUS) as u64);
            json!({"layer": layer, "expert_id": expert_id, "gpu": gpu})
        })
        .collect();

    json!({
        "token_index": token_index,
        "t_ms": (t_ms * 100.0).round() / 100.0,
        "experts": experts,
        "spec": {"proposed": 8, "accepted": 0},
    })
}

async fn query_gpus() -> Vec<Value> {
    let output = tokio::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .await;

    match output {
        Ok(o) if o.status.success() => {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .filter_map(|line| {
                    let p: Vec<&str> = line.split(',').map(str::trim).collect();
                    if p.len() < 6 { return None; }
                    Some(json!({
                        "id":             p[0].parse::<u64>().unwrap_or(0),
                        "name":           p[1],
                        "mem_total_mb":   p[2].parse::<u64>().unwrap_or(0),
                        "mem_used_mb":    p[3].parse::<u64>().unwrap_or(0),
                        "utilization_pct":p[4].parse::<u64>().unwrap_or(0),
                        "temp_c":         p[5].parse::<u64>().unwrap_or(0),
                    }))
                })
                .collect()
        }
        _ => (0..N_GPUS as u64)
            .map(|i| json!({
                "id": i, "name": "H100 SXM5 80GB", "mem_total_mb": 81920,
                "mem_used_mb": 0, "utilization_pct": 0, "temp_c": 0,
            }))
            .collect(),
    }
}
