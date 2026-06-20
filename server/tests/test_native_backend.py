from server.schemas import ChatRequest, ChatMessage
from server.topology import build_topology
from server import native_backend as nb


class _FakeStdin:
    def __init__(self):
        self.written = []

    def write(self, s):
        self.written.append(s)

    def flush(self):
        pass


class _FakeStdout:
    """Replays a fixed script of lines, like the --serve wire protocol would produce."""

    def __init__(self, lines):
        self._lines = iter(lines)

    def readline(self):
        return next(self._lines, "")


class _FakeProcess:
    def __init__(self, ready_lines, gen_lines):
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout(ready_lines + gen_lines)


def test_engine_waits_for_serve_ready(monkeypatch):
    fake = _FakeProcess(ready_lines=["rank 0: loading 94 real layers...\n", "SERVE_READY\n"], gen_lines=[])
    monkeypatch.setattr(nb.subprocess, "Popen", lambda *a, **k: fake)
    eng = nb.NativeEngineProcess()
    eng.start()
    assert eng._ready is True


def test_engine_raises_if_process_dies_before_ready(monkeypatch):
    fake = _FakeProcess(ready_lines=["rank 0: loading 94 real layers...\n"], gen_lines=[])  # no SERVE_READY, then EOF
    monkeypatch.setattr(nb.subprocess, "Popen", lambda *a, **k: fake)
    eng = nb.NativeEngineProcess()
    try:
        eng.start()
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "exited during startup" in str(e)


def test_engine_generate_streams_tokens_and_stops_at_done(monkeypatch):
    fake = _FakeProcess(ready_lines=["SERVE_READY\n"],
                        gen_lines=["TOK 11\n", "TOK 22\n", "TOK 33\n", "DONE\n"])
    monkeypatch.setattr(nb.subprocess, "Popen", lambda *a, **k: fake)
    eng = nb.NativeEngineProcess()
    eng.start()
    toks = list(eng.generate(prompt_ids=[1, 2, 3], max_new_tokens=3))
    assert toks == [11, 22, 33]
    assert fake.stdin.written == ["3 1 2 3\n"]


def test_engine_generate_raises_on_error_line(monkeypatch):
    fake = _FakeProcess(ready_lines=["SERVE_READY\n"], gen_lines=["ERROR request out of bounds\n"])
    monkeypatch.setattr(nb.subprocess, "Popen", lambda *a, **k: fake)
    eng = nb.NativeEngineProcess()
    eng.start()
    try:
        list(eng.generate(prompt_ids=[1], max_new_tokens=1))
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "request out of bounds" in str(e)


class _FakeTokenizer:
    def encode(self, text):
        return [ord(c) % 100 for c in text][:8]

    def decode(self, ids, skip_special_tokens=True):
        # Deterministic, monotonically-growing string so the delta-decoding logic in
        # NativeBackend.stream is exercised the same way a real BPE tokenizer would be.
        return "".join(chr(65 + (i % 26)) for i in ids)


def test_native_backend_stream_yields_deltas_and_summary(monkeypatch):
    fake_proc = _FakeProcess(ready_lines=["SERVE_READY\n"],
                             gen_lines=["TOK 0\n", "TOK 1\n", "DONE\n"])
    monkeypatch.setattr(nb.subprocess, "Popen", lambda *a, **k: fake_proc)
    monkeypatch.setattr(nb, "_engine", None)   # reset the module-level singleton between tests

    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _FakeTokenizer())

    backend = nb.NativeBackend()
    topo = build_topology(num_layers=4, experts_per_layer=16)
    req = ChatRequest(model="m", messages=[ChatMessage(role="user", content="hi")], max_tokens=2)
    out = list(backend.stream(req, topo))

    content_chunks = [c for c in out if "choices" in c]
    assert len(content_chunks) == 2
    full = "".join(c["choices"][0]["delta"]["content"] for c in content_chunks)
    assert full == backend.tokenizer.decode([0, 1])

    summary = out[-1]["x_summary"]
    assert summary["completion_tokens"] == 2
    assert summary["ttft_ms"] >= 0.0
