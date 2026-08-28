"""Run one LEAF-EB LEAF-EB simulation in an isolated process."""

from __future__ import annotations

import argparse
import os
import sys
import traceback

os.environ["LEAF_WORKER"] = "1"

from src.core.simulation import run_external_worker_batch


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for one simulation worker."""

    parser = argparse.ArgumentParser(
        description="Run one isolated LEAF-EB LEAF-EB simulation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--simulation-id", type=int)
    parser.add_argument("--simulation-ids")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--monte-carlo", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Execute one worker and return a process exit code."""

    try:
        args = parse_arguments()
        if args.simulation_ids:
            simulation_ids = [
                int(value) for value in args.simulation_ids.split(",")
                if value.strip()]
        elif args.simulation_id is not None:
            simulation_ids = [args.simulation_id]
        else:
            raise ValueError(
                "One of --simulation-id or --simulation-ids is required.")
        run_external_worker_batch(
            args.config,
            simulation_ids,
            args.monte_carlo,
            args.run_id)
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
