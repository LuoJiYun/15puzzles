"""Reproducible report measurements; never changes the saved default solution."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify_path(result):
    if result.status != "solved":
        assert result.cost is None and not result.path and not result.moves
        return {"solved_path_checked": False, "empty_failure_path": True}
    assert result.path[0] == result.start and result.path[-1] == result.goal
    assert len(result.moves) == result.cost == len(result.path) - 1
    offsets = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
    for before, direction, after in zip(result.path, result.moves, result.path[1:]):
        blank = before.index(0)
        row, column = divmod(blank, 4)
        dr, dc = offsets[direction]
        next_row, next_column = row + dr, column + dc
        assert 0 <= next_row < 4 and 0 <= next_column < 4
        target = 4 * next_row + next_column
        expected = list(before)
        expected[blank], expected[target] = expected[target], expected[blank]
        assert tuple(expected) == after
    return {"solved_path_checked": True, "transitions_checked": result.cost}


def child(payload):
    from puzzle15 import solve

    result = solve(**payload)
    print(json.dumps({"result": result.to_dict(), "verification": verify_path(result)}))


def run_isolated(payload):
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child", json.dumps(payload)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
    )
    output = json.loads(completed.stdout)
    output["child_process_wall_seconds"] = time.perf_counter() - started
    return output


def flatten(rows):
    return [value for row in rows for value in row]


def move(board, direction):
    delta = {"U": -4, "D": 4, "L": -1, "R": 1}[direction]
    board = list(board)
    blank = board.index(0)
    target = blank + delta
    board[blank], board[target] = board[target], board[blank]
    return board


def main():
    from puzzle15 import DEFAULT_GOAL, DEFAULT_START

    saved_path = ROOT / "results" / "default_solution.json"
    saved_bytes = saved_path.read_bytes()
    saved = json.loads(saved_bytes)
    saved_verification = verify_path(
        SimpleNamespace(
            status=saved["status"],
            cost=saved["cost"],
            start=tuple(flatten(saved["start"])),
            goal=tuple(flatten(saved["goal"])),
            path=[tuple(flatten(board)) for board in saved["path"]],
            moves=saved["moves"],
        )
    )
    heuristics = ["manhattan", "linear-conflict", "walking-distance"]
    randomizer = random.Random(20261018)
    raw = []
    order = []
    for round_number in range(1, 6):
        round_order = heuristics.copy()
        randomizer.shuffle(round_order)
        for name in round_order:
            order.append(name)
            item = run_isolated({"heuristic": name})
            result = item.pop("result")
            assert result["status"] == "solved" and result["cost"] == 41
            raw.append(
                {
                    "round": round_number,
                    "execution_index": len(raw) + 1,
                    "heuristic": name,
                    "status": result["status"],
                    "cost": result["cost"],
                    "statistics": result["statistics"],
                    "moves": result["moves"],
                    **item,
                }
            )
    summaries = {}
    for name in heuristics:
        records = [record for record in raw if record["heuristic"] == name]
        elapsed = [record["statistics"]["elapsed_seconds"] for record in records]
        fixed_metrics = {
            key: records[0]["statistics"][key]
            for key in records[0]["statistics"]
            if key != "elapsed_seconds"
        }
        assert all(
            all(record["statistics"][key] == value for key, value in fixed_metrics.items())
            for record in records
        )
        summaries[name] = {
            "runs": len(records),
            "status": "solved",
            "cost": 41,
            "fixed_statistics": fixed_metrics,
            "elapsed_seconds_raw": elapsed,
            "elapsed_seconds_median": statistics.median(elapsed),
            "elapsed_seconds_min": min(elapsed),
            "elapsed_seconds_max": max(elapsed),
        }
    canonical = list(DEFAULT_GOAL)
    custom_goal = [6, 2, 11, 15, 9, 0, 3, 8, 7, 1, 4, 12, 5, 10, 14, 13]
    custom_start = move(move(custom_goal, "U"), "L")
    unsolvable = canonical.copy()
    unsolvable[0], unsolvable[1] = unsolvable[1], unsolvable[0]
    cases = [
        ("initial_equals_goal", {"start": canonical, "goal": canonical, "max_nodes": 0, "timeout": 0}),
        ("one_move", {"start": move(canonical, "U"), "goal": canonical}),
        ("unsolvable_swap_1_2", {"start": unsolvable, "goal": canonical}),
        ("arbitrary_interior_blank_goal", {"start": custom_start, "goal": custom_goal}),
        ("node_limit_0", {"max_nodes": 0}),
        ("node_limit_1", {"max_nodes": 1}),
        ("timeout_0", {"timeout": 0}),
    ]
    problem2 = json.loads((ROOT / "examples" / "problem2.json").read_text(encoding="utf-8"))
    problem2_payload = {
        "start": flatten(problem2["start"]),
        "goal": flatten(problem2["goal"]),
        "timeout": 3,
        "max_nodes": 100000,
    }
    cases.append(("examples_problem2", problem2_payload))
    case_results = [
        {"case": name, "input": payload, **run_isolated(payload)} for name, payload in cases
    ]
    started_tests = time.perf_counter()
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )
    tests_wall = time.perf_counter() - started_tests
    assert tests.returncode == 0, tests.stdout + tests.stderr
    spec = importlib.util.spec_from_file_location("report_test_solver", ROOT / "tests" / "test_solver.py")
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
    bfs_evidence = []
    for name, goal in (("canonical", test_module.CANONICAL_GOAL), ("custom", test_module.CUSTOM_GOAL)):
        distances = test_module.exact_distances(goal, 8)
        bfs_evidence.append(
            {
                "goal_name": name,
                "goal": goal,
                "maximum_depth": 8,
                "unique_states": len(distances),
                "states_by_depth": dict(sorted(Counter(distances.values()).items())),
                "heuristics_checked_by_test": heuristics,
            }
        )
    md, wd = summaries["manhattan"], summaries["walking-distance"]
    output = {
        "schema_version": 1,
        "experiment_local_datetime": datetime.now().astimezone().isoformat(),
        "environment": {
            "python_version": sys.version,
            "python_implementation": platform.python_implementation(),
            "executable": sys.executable,
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "processor_identifier_environment": os.environ.get("PROCESSOR_IDENTIFIER"),
            "logical_cpu_count": os.cpu_count(),
        },
        "protocol": {
            "runs_per_heuristic": 5,
            "fresh_child_process_each_run": True,
            "walking_distance_cache_cold_at_each_solve": True,
            "execution_mode": "sequential isolated child processes",
            "order_random_seed": 20261018,
            "execution_order": order,
            "timing": "SearchResult.elapsed includes validation, heuristic setup, search and reconstruction; excludes interpreter startup and imports. child_process_wall_seconds includes process overhead.",
            "default_start": DEFAULT_START,
            "default_goal": DEFAULT_GOAL,
            "production_source_sha256": {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in ("puzzle15.py", "heuristics.py", "main.py")
            },
        },
        "default_benchmark_raw": raw,
        "default_benchmark_summary": summaries,
        "manhattan_to_walking_distance": {
            "expansion_ratio": md["fixed_statistics"]["expanded"] / wd["fixed_statistics"]["expanded"],
            "expansion_reduction_fraction": 1 - wd["fixed_statistics"]["expanded"] / md["fixed_statistics"]["expanded"],
            "median_time_ratio": md["elapsed_seconds_median"] / wd["elapsed_seconds_median"],
        },
        "saved_default_artifact": {
            "path": "results/default_solution.json",
            "sha256": hashlib.sha256(saved_bytes).hexdigest(),
            "status": saved["status"],
            "cost": saved["cost"],
            "heuristic": saved["heuristic"],
            "statistics": saved["statistics"],
            "moves": saved["moves"],
            "selected_path_boards": {str(step): saved["path"][step] for step in (0, 10, 20, 30, 41)},
            "verification": saved_verification,
            "untouched_after_experiments": saved_bytes == saved_path.read_bytes(),
        },
        "edge_case_runs": case_results,
        "unittest": {
            "command": "python -m unittest discover -s tests -v",
            "returncode": tests.returncode,
            "process_wall_seconds": tests_wall,
            "stdout": tests.stdout,
            "stderr": tests.stderr,
        },
        "bfs_admissibility_coverage_from_test_helpers": bfs_evidence,
    }
    destination = ROOT / "results" / "report_experiments.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summaries, "ratios": output["manhattan_to_walking_distance"], "bfs": bfs_evidence, "unittest": tests.stderr, "cases": [{"case": case["case"], "status": case["result"]["status"], "cost": case["result"]["cost"], "statistics": case["result"]["statistics"]} for case in case_results], "environment": output["environment"], "saved_untouched": output["saved_default_artifact"]["untouched_after_experiments"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(json.loads(sys.argv[2]))
    else:
        main()
