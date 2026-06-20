import sys, os, tempfile
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil.bench.cli import main, _unique_runid


def test_unique_runid_returns_free_id():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "n"))
        base = datetime.now().strftime("%Y%m%d-%H%M%S")
        open(os.path.join(d, "n", base + ".json"), "w").close()   # occupy this second
        rid = _unique_runid(d, "n")
        assert not os.path.exists(os.path.join(d, "n", rid + ".json"))  # always free


def test_cli_run_diagnose_export_flow():
    with tempfile.TemporaryDirectory() as d:
        main(["--results-dir", d, "run", "--name", "c", "--dtype", "1",
              "--decode", "16", "--repeats", "3"])
        main(["--results-dir", d, "diagnose", "--name", "c"])
        out = os.path.join(d, "o.md")
        main(["--results-dir", d, "export", "--name", "c", "--format", "md", "--out", out])
        assert os.path.exists(out)
        # a reproducibility manifest is written alongside the run
        runs = os.listdir(os.path.join(d, "c"))
        assert any(f.endswith(".manifest.json") for f in runs)


def test_cli_analytical_commands_do_not_raise():
    # plan / sweep / spec are analytical (no GPU, no stored run); must run clean
    main(["plan", "--dtype", "1", "--decode", "16"])
    main(["sweep", "--layout", "--dtype", "1"])
    main(["sweep", "--full", "--decode", "16"])
    main(["sweep", "--depths", "512,32768", "--dtype", "1"])
    main(["spec", "--alpha", "0.7", "--base-tok-s", "260"])


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
