"""Fast marching travel-time search on a cost map, and route tracing down its gradient.

Same scheme as MARE's ``fast_marching.py`` (from the DyMu planner of J. R. Sanchez Ibanez and G. Paz Delgado): a front
spreads from the goal solving the Eikonal equation |grad T| = cost on the 4-connected lattice with the first-order
upwind update; impassable cells are never entered (T stays infinite there). This implementation keeps the front in a
binary heap with lazy deletion instead of a sorted list.

Route tracing: small steps down the bilinearly interpolated unit gradient of T from the start; when the descent stalls
or would cross an unreached cell, a cell-by-cell steepest-descent walk over the 8 neighbours (no corner cutting
through impassable cells) is used instead, which always reaches the goal from a reached cell.

Cells are (row, column); T is in cost x cells (multiply by the spacing for cost x metres).
"""
from __future__ import annotations

import heapq
import math

import numpy as np

FOUR_NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT_NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
TRACE_STEP_CELLS = 0.4
TRACE_STALL_STEPS = 40
TRACE_GOAL_CELLS = 1.0


def eikonal_update(horizontal_minimum: float, vertical_minimum: float, cost: float) -> float:
    """T of a cell from its smallest horizontal and vertical neighbour T and its own cost (first-order upwind)."""
    if math.isinf(horizontal_minimum):
        return vertical_minimum + cost
    if math.isinf(vertical_minimum):
        return horizontal_minimum + cost
    difference = horizontal_minimum - vertical_minimum
    if abs(difference) >= cost:
        return min(horizontal_minimum, vertical_minimum) + cost
    return 0.5 * (horizontal_minimum + vertical_minimum + math.sqrt(2.0 * cost * cost - difference * difference))


def travel_time(cost: np.ndarray, impassable: np.ndarray, goal: tuple[int, int],
                start: tuple[int, int] | None = None) -> np.ndarray:
    """T map (float64, same shape as ``cost``): least travel cost from each cell to ``goal``; inf where unreachable.

    The search stops once ``start`` is accepted (cells farther than the start keep partial or infinite values)."""
    rows, columns = cost.shape
    if impassable[goal]:
        raise ValueError(f"goal cell {goal} is impassable")
    times = np.full(cost.shape, np.inf)
    accepted = np.asarray(impassable, dtype=bool).copy()
    cost_values = np.asarray(cost, dtype=np.float64)
    times[goal] = 0.0
    heap = [(0.0, goal[0], goal[1])]
    while heap:
        time_value, row, column = heapq.heappop(heap)
        if accepted[row, column] or time_value > times[row, column]:
            continue                                   # a stale heap entry (the cell was improved or accepted)
        accepted[row, column] = True
        if start is not None and (row, column) == tuple(start):
            break
        for row_step, column_step in FOUR_NEIGHBOURS:
            neighbour_row, neighbour_column = row + row_step, column + column_step
            if not (0 <= neighbour_row < rows and 0 <= neighbour_column < columns):
                continue
            if accepted[neighbour_row, neighbour_column]:
                continue
            horizontal = min(times[neighbour_row, neighbour_column - 1] if neighbour_column > 0 else np.inf,
                             times[neighbour_row, neighbour_column + 1] if neighbour_column < columns - 1 else np.inf)
            vertical = min(times[neighbour_row - 1, neighbour_column] if neighbour_row > 0 else np.inf,
                           times[neighbour_row + 1, neighbour_column] if neighbour_row < rows - 1 else np.inf)
            candidate = eikonal_update(horizontal, vertical, cost_values[neighbour_row, neighbour_column])
            if candidate < times[neighbour_row, neighbour_column]:
                times[neighbour_row, neighbour_column] = candidate
                heapq.heappush(heap, (candidate, neighbour_row, neighbour_column))
    return times


def unit_gradient_field(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(gradient_row, gradient_column) unit vectors of T per cell; one-sided next to infinite cells, 0 where undefined."""
    padded = np.pad(times, 1, constant_values=np.inf)
    centre = padded[1:-1, 1:-1]
    gradients = []
    for minus, plus in ((padded[:-2, 1:-1], padded[2:, 1:-1]), (padded[1:-1, :-2], padded[1:-1, 2:])):
        with np.errstate(invalid="ignore"):
            central = (plus - minus) / 2.0
            forward = plus - centre
            backward = centre - minus
        finite_minus, finite_plus = np.isfinite(minus), np.isfinite(plus)
        gradient = np.where(finite_minus & finite_plus, central,
                            np.where(finite_plus, forward, np.where(finite_minus, backward, 0.0)))
        gradients.append(np.where(np.isfinite(centre) & np.isfinite(gradient), gradient, 0.0))
    magnitude = np.hypot(gradients[0], gradients[1])
    safe = np.where(magnitude > 0, magnitude, 1.0)
    return np.where(magnitude > 0, gradients[0] / safe, 0.0), np.where(magnitude > 0, gradients[1] / safe, 0.0)


def _bilinear(field: np.ndarray, row: float, column: float) -> float:
    """Bilinear value of a per-cell field at a fractional (row, column) inside the lattice."""
    rows, columns = field.shape
    row = min(max(row, 0.0), rows - 1.0)
    column = min(max(column, 0.0), columns - 1.0)
    row_low, column_low = min(int(row), rows - 2) if rows > 1 else 0, min(int(column), columns - 2) if columns > 1 else 0
    row_fraction, column_fraction = row - row_low, column - column_low
    top = field[row_low, column_low] * (1 - column_fraction) + field[row_low, min(column_low + 1, columns - 1)] * column_fraction
    bottom = field[min(row_low + 1, rows - 1), column_low] * (1 - column_fraction) + \
        field[min(row_low + 1, rows - 1), min(column_low + 1, columns - 1)] * column_fraction
    return float(top * (1 - row_fraction) + bottom * row_fraction)


def trace_cells(times: np.ndarray, impassable: np.ndarray, start: tuple[int, int],
                goal: tuple[int, int]) -> np.ndarray | None:
    """Route (N, 2 fractional cells) by steepest descent of T over 8 neighbours; None if the start was not reached."""
    rows, columns = times.shape
    row, column = start
    if not np.isfinite(times[row, column]):
        return None
    path = [(row, column)]
    for _ in range(rows * columns):
        if (row, column) == tuple(goal):
            return np.asarray(path, dtype=np.float64)
        best, best_time = None, times[row, column]
        for row_step, column_step in EIGHT_NEIGHBOURS:
            neighbour_row, neighbour_column = row + row_step, column + column_step
            if not (0 <= neighbour_row < rows and 0 <= neighbour_column < columns):
                continue
            if row_step and column_step and (impassable[row + row_step, column] or impassable[row, column + column_step]):
                continue
            if times[neighbour_row, neighbour_column] < best_time:
                best, best_time = (neighbour_row, neighbour_column), times[neighbour_row, neighbour_column]
        if best is None:
            return None
        row, column = best
        path.append(best)
    return None


def trace_route(times: np.ndarray, impassable: np.ndarray, start: tuple[int, int],
                goal: tuple[int, int]) -> np.ndarray | None:
    """Route (N, 2 fractional cells, start first, goal last) down the gradient of T; None when unreachable."""
    if not np.isfinite(times[start]):
        return None
    gradient_row, gradient_column = unit_gradient_field(times)
    rows, columns = times.shape
    position = np.array(start, dtype=np.float64)
    path = [position.copy()]
    best_time, since_best = np.inf, 0
    maximum_steps = int(4 * (rows + columns) / TRACE_STEP_CELLS) + 100
    for _ in range(maximum_steps):
        cell = (int(round(position[0])), int(round(position[1])))
        if not (0 <= cell[0] < rows and 0 <= cell[1] < columns) or not np.isfinite(times[cell]):
            return trace_cells(times, impassable, start, goal)
        if times[cell] < best_time:
            best_time, since_best = times[cell], 0
        else:
            since_best += 1
            if since_best > TRACE_STALL_STEPS:
                return trace_cells(times, impassable, start, goal)
        step = np.array([_bilinear(gradient_row, *position), _bilinear(gradient_column, *position)])
        norm = math.hypot(*step)
        if not norm > 0:
            return trace_cells(times, impassable, start, goal)
        position = position - TRACE_STEP_CELLS * step / norm
        path.append(position.copy())
        if math.hypot(position[0] - goal[0], position[1] - goal[1]) < TRACE_GOAL_CELLS:
            path.append(np.array(goal, dtype=np.float64))
            route = np.asarray(path)
            cells = np.rint(route).astype(np.int64)
            if np.any(impassable[cells[:, 0], cells[:, 1]][1:]):
                return trace_cells(times, impassable, start, goal)
            return route
    return trace_cells(times, impassable, start, goal)
