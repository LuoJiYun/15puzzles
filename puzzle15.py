"""Optimal A* search for the 4 x 4 sliding puzzle, using only Python's stdlib."""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from math import isfinite
from time import perf_counter
from typing import Iterable

from heuristics import make_packed_heuristic


Board = tuple[int, ...]
DEFAULT_START: Board = (11, 9, 4, 15, 1, 3, 0, 12, 7, 5, 8, 6, 13, 2, 10, 14)
DEFAULT_GOAL: Board = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0)
HEURISTICS = ("manhattan", "linear-conflict", "walking-distance")


def validate_board(board: Iterable[int]) -> Board:
    """Require each integer 0..15 exactly once; 0 represents the blank."""
    try:
        values = tuple(board)
    except TypeError as exc:
        raise ValueError("A board must contain 16 integers, including 0 for the blank.") from exc
    if len(values) != 16 or any(type(value) is not int for value in values):
        raise ValueError("A board must contain exactly 16 integers.")
    if set(values) != set(range(16)):
        raise ValueError("A board must contain each number from 0 to 15 exactly once.")
    return values


def _parity(board: Board) -> int:
    inversions = sum(
        left > right
        for index, left in enumerate(board)
        if left != 0
        for right in board[index + 1 :]
        if right != 0
    )
    blank_row_from_bottom = 4 - board.index(0) // 4
    return (inversions + blank_row_from_bottom) % 2


def is_solvable(start: Iterable[int], goal: Iterable[int] = DEFAULT_GOAL) -> bool:
    """Two valid boards are connected iff their even-width parity agrees."""
    return _parity(validate_board(start)) == _parity(validate_board(goal))


def pack_board(board: Board) -> int:
    return sum(tile << (4 * index) for index, tile in enumerate(board))


def unpack_board(state: int) -> Board:
    return tuple((state >> (4 * index)) & 15 for index in range(16))


def format_board(board: Iterable[int]) -> str:
    values = validate_board(board)
    return "\n".join(
        " ".join(f"{tile:2d}" if tile else " ." for tile in values[row : row + 4])
        for row in range(0, 16, 4)
    )


def _legal_moves(blank: int) -> tuple[tuple[str, int], ...]:
    row, column = divmod(blank, 4)
    options = []
    if row > 0:
        options.append(("U", blank - 4))
    if row < 3:
        options.append(("D", blank + 4))
    if column > 0:
        options.append(("L", blank - 1))
    if column < 3:
        options.append(("R", blank + 1))
    return tuple(options)


NEIGHBORS = tuple(_legal_moves(blank) for blank in range(16))


@dataclass
class SearchResult:
    status: str
    start: Board
    goal: Board
    heuristic: str
    path: list[Board] = field(default_factory=list)
    moves: list[str] = field(default_factory=list)
    expanded: int = 0
    generated: int = 0
    discovered: int = 0
    reopened: int = 0
    frontier_peak: int = 0
    initial_h: int | None = None
    elapsed: float = 0.0

    @property
    def cost(self) -> int | None:
        return len(self.moves) if self.status == "solved" else None

    def to_dict(self) -> dict:
        def rows(board: Board) -> list[list[int]]:
            return [list(board[index : index + 4]) for index in range(0, 16, 4)]

        return {
            "algorithm": "A*",
            "status": self.status,
            "heuristic": self.heuristic,
            "start": rows(self.start),
            "goal": rows(self.goal),
            "cost": self.cost,
            "move_convention": "U/D/L/R indicate the blank moving up/down/left/right",
            "moves": self.moves,
            "path": [rows(board) for board in self.path],
            "statistics": {
                "initial_h": self.initial_h,
                "expanded": self.expanded,
                "generated": self.generated,
                "discovered": self.discovered,
                "reopened": self.reopened,
                "frontier_peak": self.frontier_peak,
                "elapsed_seconds": self.elapsed,
            },
        }


def solve(
    start: Iterable[int] = DEFAULT_START,
    goal: Iterable[int] = DEFAULT_GOAL,
    *,
    heuristic: str = "walking-distance",
    max_nodes: int | None = None,
    timeout: float | None = None,
) -> SearchResult:
    """Find a shortest path with f = g + h; goal success occurs only on heap pop.

    max_nodes limits expansion operations, including re-expansions. timeout is
    a wall-clock limit in seconds, including heuristic setup. Neither is enabled
    by default. A failed/limited search returns no solution path or cost.
    """
    started = perf_counter()
    start_board, goal_board = validate_board(start), validate_board(goal)
    if heuristic not in HEURISTICS:
        raise ValueError(f"Unknown heuristic: {heuristic!r}. Choose from {HEURISTICS}.")
    if max_nodes is not None and (type(max_nodes) is not int or max_nodes < 0):
        raise ValueError("max_nodes must be a nonnegative integer or None.")
    if timeout is not None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ValueError("timeout must be a finite nonnegative number or None.")
        if not isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be a finite nonnegative number or None.")

    result = SearchResult("unsolvable", start_board, goal_board, heuristic)

    def finish(status: str) -> SearchResult:
        result.status = status
        result.elapsed = perf_counter() - started
        return result

    if _parity(start_board) != _parity(goal_board):
        return finish("unsolvable")
    if start_board == goal_board:
        result.path = [start_board]
        result.discovered = result.frontier_peak = 1
        result.initial_h = 0
        return finish("solved")

    start_state, goal_state = pack_board(start_board), pack_board(goal_board)
    evaluate = make_packed_heuristic(goal_board, heuristic)
    result.initial_h = evaluate(start_state)
    # Negating g prefers deeper nodes when f ties, without changing optimality.
    frontier = [(result.initial_h, 0, start_state, start_board.index(0))]
    best_g = {start_state: 0}
    parents: dict[int, tuple[int, str]] = {}
    closed: set[int] = set()
    result.discovered = result.frontier_peak = 1

    while frontier:
        if timeout is not None and perf_counter() - started >= timeout:
            return finish("timeout")
        _, negative_g, state, blank = heappop(frontier)
        g = -negative_g
        # Older heap entries survive decreases of g and must be ignored.
        if best_g[state] != g:
            continue
        if state == goal_state:
            reverse_path = [goal_state]
            reverse_moves = []
            while state != start_state:
                state, move = parents[state]
                reverse_path.append(state)
                reverse_moves.append(move)
            result.path = [unpack_board(item) for item in reversed(reverse_path)]
            result.moves = list(reversed(reverse_moves))
            return finish("solved")
        if max_nodes is not None and result.expanded >= max_nodes:
            return finish("node_limit")
        if state in closed:
            result.reopened += 1
        closed.add(state)
        result.expanded += 1

        for move, target in NEIGHBORS[blank]:
            result.generated += 1
            tile = (state >> (4 * target)) & 15
            successor = state ^ (tile << (4 * target)) ^ (tile << (4 * blank))
            next_g = g + 1
            previous_g = best_g.get(successor)
            if previous_g is not None and next_g >= previous_g:
                continue
            if previous_g is None:
                result.discovered += 1
            # Improved closed states are eligible for expansion again. This
            # preserves optimality even for an admissible inconsistent h.
            best_g[successor] = next_g
            parents[successor] = (state, move)
            heappush(frontier, (next_g + evaluate(successor), -next_g, successor, target))
        result.frontier_peak = max(result.frontier_peak, len(frontier))

    return finish("unsolvable")
