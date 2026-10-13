#!/usr/bin/env python3
"""Attach reference material by full question, refusing ambiguous source matches."""
import argparse
import json
from pathlib import Path

from rcst_source_join import attach_sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import pandas as pd

    roots = [json.loads(line) for line in args.roots.read_text(encoding="utf-8").splitlines() if line.strip()]
    output = attach_sources(roots, pd.read_parquet(args.source).to_dict(orient="records"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for row in output:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Joined {len(output)} roots; mathematical correctness still requires review.")


if __name__ == "__main__":
    main()
