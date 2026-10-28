"""Admissible heuristics for the 4-by-4 sliding puzzle.

Walking distance counts vertical and horizontal moves in separate abstractions.
Linear conflict uses the minimum tiles that must leave each line (length minus
longest increasing subsequence), so overlapping conflicts are not double-counted.
Packed boards store position ``i`` in bits ``4*i`` through ``4*i+3``.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import deque
from functools import lru_cache
from typing import Callable


Board = tuple[int, ...]
Heuristic = Callable[[Board], int]
PackedHeuristic = Callable[[int], int]
HEURISTIC_NAMES = ("manhattan", "linear-conflict", "walking-distance")
_BLANK_SHIFT = 48


def _goal_coordinates(goal: Board) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if len(goal) != 16 or set(goal) != set(range(16)):
        raise ValueError("The goal must contain every integer from 0 through 15 once.")
    positions = [0] * 16
    for position, tile in enumerate(goal):
        positions[tile] = position
    return (
        tuple(position // 4 for position in positions),
        tuple(position % 4 for position in positions),
    )


def _distance_table(
    goal_rows: tuple[int, ...], goal_columns: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(
            abs(position // 4 - goal_rows[tile])
            + abs(position % 4 - goal_columns[tile])
            if tile else 0
            for tile in range(16)
        )
        for position in range(16)
    )


def _minimum_removals(sequence: list[int]) -> int:
    tails: list[int] = []
    for value in sequence:
        position = bisect_left(tails, value)
        if position == len(tails):
            tails.append(value)
        else:
            tails[position] = value
    return len(sequence) - len(tails)


def _linear_conflict(
    board: Board,
    goal_rows: tuple[int, ...],
    goal_columns: tuple[int, ...],
    distances: tuple[tuple[int, ...], ...],
) -> int:
    result = sum(distances[position][tile] for position, tile in enumerate(board))
    for line in range(4):
        row = [
            goal_columns[tile]
            for tile in board[4 * line : 4 * line + 4]
            if tile and goal_rows[tile] == line
        ]
        column = [
            goal_rows[board[position]]
            for position in range(line, 16, 4)
            if board[position] and goal_columns[board[position]] == line
        ]
        result += 2 * (_minimum_removals(row) + _minimum_removals(column))
    return result


@lru_cache(maxsize=4)
def _walking_distance_table(goal_blank_line: int) -> dict[int, int]:
    """BFS over tile counts by current line and required goal line.

    An abstract move moves one tile count into the blank's line from an adjacent
    line. Counts need three bits each; the blank line occupies the next two bits.
    Each of the four possible goal blank lines yields 24,964 abstract states.
    """
    root = goal_blank_line << _BLANK_SHIFT
    for line in range(4):
        root += (3 if line == goal_blank_line else 4) << (3 * (line * 4 + line))
    table = {root: 0}
    queue = deque([root])
    while queue:
        current = queue.popleft()
        blank_line = current >> _BLANK_SHIFT
        following_distance = table[current] + 1
        for source_line in (blank_line - 1, blank_line + 1):
            if not 0 <= source_line < 4:
                continue
            for goal_line in range(4):
                source_shift = 3 * (source_line * 4 + goal_line)
                if not (current >> source_shift) & 7:
                    continue
                following = (
                    current
                    - (1 << source_shift)
                    + (1 << (3 * (blank_line * 4 + goal_line)))
                    + ((source_line - blank_line) << _BLANK_SHIFT)
                )
                if following not in table:
                    table[following] = following_distance
                    queue.append(following)
    return table


def _walking_distance_components(
    goal_rows: tuple[int, ...], goal_columns: tuple[int, ...]
) -> tuple[
    dict[int, int], dict[int, int],
    tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...],
]:
    row_table = _walking_distance_table(goal_rows[0])
    column_table = _walking_distance_table(goal_columns[0])
    row_contributions = tuple(
        tuple(
            1 << (3 * ((position // 4) * 4 + goal_rows[tile]))
            if tile else (position // 4) << _BLANK_SHIFT
            for tile in range(16)
        )
        for position in range(16)
    )
    column_contributions = tuple(
        tuple(
            1 << (3 * ((position % 4) * 4 + goal_columns[tile]))
            if tile else (position % 4) << _BLANK_SHIFT
            for tile in range(16)
        )
        for position in range(16)
    )
    return row_table, column_table, row_contributions, column_contributions


def make_heuristic(goal: Board, name: str = "walking-distance") -> Heuristic:
    """Return a heuristic for valid tuple boards and an arbitrary valid goal."""
    goal_rows, goal_columns = _goal_coordinates(goal)
    if name == "walking-distance":
        row_table, column_table, row_parts, column_parts = _walking_distance_components(
            goal_rows, goal_columns
        )

        def walking_distance(board: Board) -> int:
            rows = sum(row_parts[position][tile] for position, tile in enumerate(board))
            columns = sum(
                column_parts[position][tile] for position, tile in enumerate(board)
            )
            return row_table[rows] + column_table[columns]

        return walking_distance
    distances = _distance_table(goal_rows, goal_columns)
    if name == "manhattan":
        return lambda board: sum(
            distances[position][tile] for position, tile in enumerate(board)
        )
    if name == "linear-conflict":
        return lambda board: _linear_conflict(board, goal_rows, goal_columns, distances)
    raise ValueError(f"Unknown heuristic {name!r}; choose one of {HEURISTIC_NAMES}.")


def make_packed_heuristic(
    goal: Board, name: str = "walking-distance"
) -> PackedHeuristic:
    """Return the same heuristic for boards packed into a 64-bit integer."""
    goal_rows, goal_columns = _goal_coordinates(goal)
    if name == "walking-distance":
        row_table, column_table, row_parts, column_parts = _walking_distance_components(
            goal_rows, goal_columns
        )

        def walking_distance(board: int) -> int:
            rows = columns = 0
            for position in range(16):
                tile = board & 15
                rows += row_parts[position][tile]
                columns += column_parts[position][tile]
                board >>= 4
            return row_table[rows] + column_table[columns]

        return walking_distance
    distances = _distance_table(goal_rows, goal_columns)
    if name == "manhattan":

        def manhattan(board: int) -> int:
            result = 0
            for position in range(16):
                result += distances[position][board & 15]
                board >>= 4
            return result

        return manhattan
    if name == "linear-conflict":
        return lambda board: _linear_conflict(
            tuple((board >> (4 * position)) & 15 for position in range(16)),
            goal_rows, goal_columns, distances,
        )
    raise ValueError(f"Unknown heuristic {name!r}; choose one of {HEURISTIC_NAMES}.")
