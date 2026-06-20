# Claude instructions for this repo

## Testing

Always read all tests across the codebase before making changes — including tests written by other team members. Run the full test suite (`cargo test --package engine`) and check for failures before and after any edit.

Test locations:
- `engine/src/routing/optimizer.rs` — placement optimizer tests
- `engine/src/routing/predictor.rs` — route predictor tests
- `engine/src/routing/scheduler.rs` — prefetch scheduler tests
- `engine/src/routing/stats.rs` — accuracy tracker tests
- `engine/src/spec/accept.rs` — speculative decoding acceptance tests
- `engine/src/spec/engine.rs` — spec engine integration tests
- `server/tests/` — FastAPI server tests

## Benchmarking

Whenever you run a benchmark, always display the results in full — never summarize or truncate the output. Show the complete table including per-request rows, then the summary stats (TTFT p50/p95/mean, decode tok/s p50/p95/mean, ms/tok, predictor hit%).

Run benchmarks with:
```bash
python3 tools/benchmark.py --base http://localhost:8000 --n 10 --tokens 100
```

Save results to a timestamped file so we can track progress:
```bash
python3 tools/benchmark.py --n 10 --tokens 100 --out /alloc/data/bench_$(date +%Y%m%d_%H%M%S).json
```

Tag benchmark requests so they appear in `/api/tasks`:
```bash
# Pass X-User header to identify who is running the benchmark
python3 tools/benchmark.py --n 10 --tokens 100  # benchmark.py should pass X-User automatically
```

After any optimization, run the benchmark before and after and show both results side by side so the team can see the delta.
