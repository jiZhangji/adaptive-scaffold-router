#!/usr/bin/env python3
"""Wait for a continuous GPU-idle window, then run one command exactly once."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_gpu_csv(text: str) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise ValueError(f"unexpected nvidia-smi row: {line!r}")
        rows.append(
            {"index": int(parts[0]), "utilization": int(parts[1]), "memory_mib": int(parts[2])}
        )
    return rows


def query_gpus() -> list[dict[str, int]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_gpu_csv(result.stdout)


def selected_gpus(
    rows: list[dict[str, int]], indices: list[int]
) -> list[dict[str, int]]:
    by_index = {row["index"]: row for row in rows}
    missing = [index for index in indices if index not in by_index]
    if missing:
        raise ValueError(f"nvidia-smi did not return requested GPUs: {missing}")
    return [by_index[index] for index in indices]


def all_idle(
    rows: list[dict[str, int]], max_utilization: int, max_memory_mib: int
) -> bool:
    return all(
        row["utilization"] <= max_utilization and row["memory_mib"] <= max_memory_mib
        for row in rows
    )


def write_state(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def acquire_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"monitor lock already exists: {path}") from error
    os.write(descriptor, f"{os.getpid()}\n".encode())
    return descriptor


def run(args: argparse.Namespace) -> int:
    if not args.command:
        raise ValueError("a command is required after --")
    lock_descriptor = acquire_lock(args.lock_file)
    idle_since: float | None = None
    try:
        print(
            f"[{timestamp()}] monitoring GPUs {args.gpu_indices}; require "
            f"util<={args.max_utilization}%, memory<={args.max_memory_mib} MiB "
            f"for {args.idle_seconds} seconds",
            flush=True,
        )
        while True:
            try:
                rows = selected_gpus(query_gpus(), args.gpu_indices)
                idle = all_idle(rows, args.max_utilization, args.max_memory_mib)
            except Exception as error:  # keep monitoring across transient driver/SSH issues
                idle = False
                rows = []
                print(f"[{timestamp()}] GPU query failed: {error}", flush=True)
            now = time.monotonic()
            if idle:
                if idle_since is None:
                    idle_since = now
                idle_elapsed = now - idle_since
            else:
                idle_since = None
                idle_elapsed = 0.0
            payload = {
                "status": "waiting",
                "timestamp": timestamp(),
                "pid": os.getpid(),
                "gpu_indices": args.gpu_indices,
                "gpus": rows,
                "idle": idle,
                "idle_elapsed_seconds": round(idle_elapsed, 1),
                "idle_required_seconds": args.idle_seconds,
                "command": args.command,
            }
            write_state(args.state_file, payload)
            summary = " ".join(
                f"gpu{row['index']}:util={row['utilization']}%,mem={row['memory_mib']}MiB"
                for row in rows
            )
            print(
                f"[{payload['timestamp']}] {summary or 'no GPU data'}; "
                f"continuous_idle={idle_elapsed:.0f}/{args.idle_seconds}s",
                flush=True,
            )
            if idle_elapsed >= args.idle_seconds:
                break
            time.sleep(args.poll_seconds)

        write_state(
            args.state_file,
            {**payload, "status": "launching", "timestamp": timestamp()},
        )
        print(f"[{timestamp()}] idle window satisfied; launching: {args.command}", flush=True)
        completed = subprocess.run(args.command, cwd=args.workdir)
        write_state(
            args.state_file,
            {
                **payload,
                "status": "completed" if completed.returncode == 0 else "failed",
                "timestamp": timestamp(),
                "exit_code": completed.returncode,
            },
        )
        return completed.returncode
    finally:
        os.close(lock_descriptor)
        args.lock_file.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-indices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--idle-seconds", type=int, default=1800)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-utilization", type=int, default=10)
    parser.add_argument("--max-memory-mib", type=int, default=1024)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if args.idle_seconds < 0 or args.poll_seconds <= 0:
        parser.error("idle-seconds must be nonnegative and poll-seconds must be positive")
    return args


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except Exception as error:
        print(f"monitor failed: {error}", file=sys.stderr, flush=True)
        raise
