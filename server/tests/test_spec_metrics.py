from server.spec_metrics import parse_spec_counters, accept_stats, cumulative_accept

# Synthetic vLLM v1 /metrics scrapes (loose names; labels present to test the regex strips them).
BEFORE = """
# HELP vllm:spec_decode_num_accepted_tokens_total Accepted tokens.
vllm:spec_decode_num_accepted_tokens_total{model="q"} 100.0
vllm:spec_decode_num_draft_tokens_total{model="q"} 200.0
vllm:num_requests_running 1.0
"""
AFTER = """
vllm:spec_decode_num_accepted_tokens_total{model="q"} 130.0
vllm:spec_decode_num_draft_tokens_total{model="q"} 240.0
vllm:num_requests_running 1.0
"""


def test_parse_counters():
    c = parse_spec_counters(BEFORE)
    assert c["accepted"] == 100.0 and c["drafts"] == 200.0


def test_accept_stats_delta():
    # Δaccepted=30, Δdrafts=40 -> rate 0.75, tau 1.75
    st = accept_stats(BEFORE, AFTER)
    assert st["accept_rate"] == 0.75
    assert st["tau"] == 1.75
    assert st["accepted"] == 30.0 and st["drafts"] == 40.0


def test_no_drafts_window_is_safe():
    # spec off / no draft fired in the window -> neutral (rate 0, tau 1), no divide-by-zero
    st = accept_stats(BEFORE, BEFORE)
    assert st["accept_rate"] == 0.0 and st["tau"] == 1.0


def test_missing_counters_graceful():
    assert parse_spec_counters("") == {"accepted": 0.0, "drafts": 0.0}
    assert accept_stats("", "")["accept_rate"] == 0.0


def test_cumulative_accept():
    # after-only running average: 100/200 = 0.5, tau 1.5
    st = cumulative_accept(BEFORE)
    assert st["accept_rate"] == 0.5 and st["tau"] == 1.5
    assert cumulative_accept("")["accept_rate"] == 0.0   # graceful


def test_rate_clamped():
    # guard against counter weirdness producing >1
    weird_after = "vllm:spec_decode_num_accepted_tokens_total 1000.0\nvllm:spec_decode_num_draft_tokens_total 210.0\n"
    st = accept_stats(BEFORE, weird_after)
    assert st["accept_rate"] == 1.0
