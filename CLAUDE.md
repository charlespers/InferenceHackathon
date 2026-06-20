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
