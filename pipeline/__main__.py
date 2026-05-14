"""Dispatch for `python -m pipeline <subcommand> [args...]`.

Subcommands:
  run    Hydra-driven pipeline orchestrator (see pipeline/run.py).
  cli    Manifest CLI (init / status / list-pending / scan-videos / etc.).
  qc     Quality-control over inference outputs (Phase E).

Examples:
  python -m pipeline run +experiment=surgsam2_oob_whip
  python -m pipeline cli status
  python -m pipeline cli scan-videos /gpfs/.../frames_attempt2
"""
from __future__ import annotations

import sys


def _print_usage_and_exit(code: int = 1) -> None:
    print(__doc__)
    sys.exit(code)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        _print_usage_and_exit(0 if len(sys.argv) >= 2 else 1)

    cmd = sys.argv.pop(1)   # consume subcommand from argv

    if cmd == "run":
        from pipeline.run import main as run_main
        run_main()
    elif cmd == "cli":
        from pipeline.cli import main as cli_main
        sys.exit(cli_main())
    elif cmd == "qc":
        try:
            from pipeline.qc import main as qc_main
        except ImportError:
            print("pipeline.qc not implemented yet (Phase E).", file=sys.stderr)
            sys.exit(2)
        sys.exit(qc_main())
    else:
        print(f"unknown subcommand: {cmd}\n", file=sys.stderr)
        _print_usage_and_exit(1)


if __name__ == "__main__":
    main()
