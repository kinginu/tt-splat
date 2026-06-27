"""Make `import spike...` work whether run via pytest from the repo root or as a script."""
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def run_module(globs):
    """Run every test_* function in a module's globals; print PASS/FAIL; return exit code."""
    fns = [(k, v) for k, v in sorted(globs.items()) if k.startswith("test_") and callable(v)]
    fail = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS {name}")
        except Exception as e:  # noqa: BLE001
            # honor pytest.skip() in direct-run mode too
            if type(e).__name__ == "Skipped" or type(e).__module__.startswith("_pytest"):
                print(f"  SKIP {name}: {e}")
                continue
            fail += 1
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
    tag = "ALL PASS" if not fail else f"{fail} FAILED"
    print(f"  -> {tag}")
    return 1 if fail else 0
