"""Command-line interface: python main.py --help."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

from puzzle15 import DEFAULT_GOAL, DEFAULT_START, HEURISTICS, SearchResult
from puzzle15 import format_board, solve, validate_board


def parse_board(text: str) -> tuple[int, ...]:
    tokens = re.split(r"[\s,;]+", text.strip())
    try:
        return validate_board(tuple(int(token) for token in tokens))
    except ValueError as exc:
        raise ValueError(f"Invalid board: {exc}") from exc


def _json_board(value: object) -> tuple[int, ...]:
    if isinstance(value, list) and len(value) == 4 and all(
        isinstance(row, list) and len(row) == 4 for row in value
    ):
        value = [tile for row in value for tile in row]
    return validate_board(value)


def load_boards(path: Path) -> tuple[tuple[int, ...], tuple[int, ...]]:
    with path.open(encoding="utf-8-sig") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("The input JSON must be an object with start and goal boards.")
    return (
        _json_board(data.get("start", list(DEFAULT_START))),
        _json_board(data.get("goal", list(DEFAULT_GOAL))),
    )


def interactive_board(label: str, default: tuple[int, ...]) -> tuple[int, ...]:
    print(f"{label} default (0 = blank):\n{format_board(default)}")
    while True:
        text = input(f"{label}, 16 numbers or Enter to accept default: ").strip()
        if not text:
            return default
        try:
            return parse_board(text)
        except ValueError as exc:
            print(exc, file=sys.stderr)


def result_text(result: SearchResult, show_path: bool = True) -> str:
    lines = [
        "15-puzzle | A* search",
        f"Heuristic: {result.heuristic}",
        "Initial board:",
        format_board(result.start),
        "Goal board:",
        format_board(result.goal),
        f"Status: {result.status}",
    ]
    if result.status == "solved":
        lines.extend([
            f"Optimal moves: {result.cost}",
            "Moves (blank direction): " + (" ".join(result.moves) or "(none)"),
        ])
    elif result.status == "unsolvable":
        lines.append("The boards have different parity; no legal solution exists.")
    else:
        lines.append("Search stopped at its resource limit; no solution is claimed.")
    lines.extend([
        f"Initial h: {result.initial_h}",
        f"Expanded: {result.expanded}",
        f"Generated successors: {result.generated}",
        f"Discovered states: {result.discovered}",
        f"Reopened states: {result.reopened}",
        f"Peak heap entries: {result.frontier_peak}",
        f"Elapsed: {result.elapsed:.6f} s (including heuristic setup)",
    ])
    if show_path and result.status == "solved":
        lines.append("\nSolution path:")
        for index, board in enumerate(result.path):
            if index:
                old_blank = result.path[index - 1].index(0)
                tile = board[old_blank]
                label = f"Step {index}: blank {result.moves[index - 1]}, slide tile {tile}"
            else:
                label = "Step 0: initial board"
            lines.extend([label, format_board(board), ""])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find an optimal 15-puzzle solution with A*. Defaults match puzzle15.png."
    )
    parser.add_argument("--start", nargs="+", metavar="N", help="16 initial tiles; 0 = blank")
    parser.add_argument("--goal", nargs="+", metavar="N", help="16 goal tiles; 0 = blank")
    parser.add_argument("--input", type=Path, help="JSON file containing start and/or goal")
    parser.add_argument("--interactive", action="store_true", help="prompt for both boards")
    parser.add_argument("--heuristic", choices=HEURISTICS, default="walking-distance")
    parser.add_argument("--max-nodes", type=int, help="maximum expansion operations")
    parser.add_argument("--timeout", type=float, help="maximum elapsed seconds")
    parser.add_argument("--output", type=Path, help="save complete result as JSON")
    parser.add_argument("--text-output", type=Path, help="save complete path and statistics as text")
    parser.add_argument("--summary-only", action="store_true", help="omit boards along path from console")
    return parser


def _write_result(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def format_result_json(data: dict) -> str:
    """Keep each four-integer board row on one line in indented JSON."""
    text = json.dumps(data, indent=2)
    return re.sub(
        r"\[\n[ \t]+(\d+),\n[ \t]+(\d+),\n[ \t]+(\d+),\n[ \t]+(\d+)\n[ \t]+\]",
        r"[\1, \2, \3, \4]",
        text,
    ) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.interactive and (args.input is not None or args.start is not None or args.goal is not None):
            raise ValueError("Use --interactive on its own, without --input, --start or --goal.")
        start, goal = load_boards(args.input) if args.input is not None else (DEFAULT_START, DEFAULT_GOAL)
        if args.start is not None:
            start = parse_board(" ".join(args.start))
        if args.goal is not None:
            goal = parse_board(" ".join(args.goal))
        if args.interactive:
            start = interactive_board("Initial board", start)
            goal = interactive_board("Goal board", goal)
        # Detect conflicting output paths before creating any files.
        outputs = [path.resolve() for path in (args.output, args.text_output) if path is not None]
        if len(set(outputs)) != len(outputs):
            raise ValueError("JSON and text output paths must be different.")
        if args.input is not None and args.input.resolve() in outputs:
            raise ValueError("An output path must not overwrite the input JSON.")
        result = solve(start, goal, heuristic=args.heuristic, max_nodes=args.max_nodes, timeout=args.timeout)
        print(result_text(result, show_path=not args.summary_only))
        if args.output is not None:
            _write_result(args.output, format_result_json(result.to_dict()))
            print(f"JSON saved: {args.output}")
        if args.text_output is not None:
            _write_result(args.text_output, result_text(result) + "\n")
            print(f"Text saved: {args.text_output}")
        return 0 if result.status == "solved" else 1
    except (ValueError, OSError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except (KeyboardInterrupt, EOFError):
        print("Search/input interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
