"""
maze_generator.py  —  Procedural maze generation using recursive backtracker.

Completely standalone — no Isaac Lab or torch dependency.
Can be tested independently:
    python maze_generator.py          # prints an ASCII maze to terminal

The maze is represented as a 2-D grid of cells.  Each cell knows which of
its four walls are still standing.  The generator carves passages by removing
walls between adjacent cells.

Outputs used by iris_maze_env.py:
    - get_wall_segments()  →  list of (x, y, width, depth) wall rectangles
                               in world-frame metres, ready to spawn as USD prims
    - get_spawn_corners()  →  3 world-frame (x, y) positions near 3 maze corners
                               guaranteed to be inside a corridor
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Cell and wall definitions
# ---------------------------------------------------------------------------

NORTH = 0
SOUTH = 1
EAST  = 2
WEST  = 3

OPPOSITE = {NORTH: SOUTH, SOUTH: NORTH, EAST: WEST, WEST: EAST}

# Direction → (row_delta, col_delta)
DELTA = {
    NORTH: (-1,  0),
    SOUTH: ( 1,  0),
    EAST:  ( 0,  1),
    WEST:  ( 0, -1),
}


@dataclass
class Cell:
    walls: List[bool] = field(default_factory=lambda: [True, True, True, True])
    # walls[NORTH], walls[SOUTH], walls[EAST], walls[WEST]
    visited: bool = False


# ---------------------------------------------------------------------------
# Maze generator
# ---------------------------------------------------------------------------

class MazeGenerator:
    """
    Generates a perfect maze (exactly one path between any two cells) using
    the recursive backtracker / depth-first search algorithm.

    Parameters
    ----------
    rows, cols      : number of maze cells (not pixels, not metres)
    cell_size       : width/height of each cell in metres
    wall_thickness  : thickness of each wall in metres
    seed            : random seed — same seed → same maze layout every time
                      None → random each call (used during training)

    World-frame origin is the centre of the maze.
    """

    def __init__(
        self,
        rows: int       = 10,
        cols: int       = 10,
        cell_size: float       = 1.0,
        wall_thickness: float  = 0.2,
        wall_height: float     = 2.0,
        seed: int | None       = None,
    ):
        self.rows           = rows
        self.cols           = cols
        self.cell_size      = cell_size
        self.wall_thickness = wall_thickness
        self.wall_height    = wall_height
        self.rng            = random.Random(seed)

        # Allocate grid
        self._grid: List[List[Cell]] = [
            [Cell() for _ in range(cols)] for _ in range(rows)
        ]
        self._carve_passages(0, 0)

    # ------------------------------------------------------------------
    # Maze generation
    # ------------------------------------------------------------------

    def _carve_passages(self, row: int, col: int):
        """Recursive DFS — removes walls to create corridors."""
        self._grid[row][col].visited = True

        directions = [NORTH, SOUTH, EAST, WEST]
        self.rng.shuffle(directions)

        for d in directions:
            dr, dc = DELTA[d]
            nr, nc = row + dr, col + dc

            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if not self._grid[nr][nc].visited:
                    # Remove wall between current cell and neighbour
                    self._grid[row][col].walls[d] = False
                    self._grid[nr][nc].walls[OPPOSITE[d]] = False
                    self._carve_passages(nr, nc)

    # ------------------------------------------------------------------
    # Wall segment extraction
    # ------------------------------------------------------------------

    def get_wall_segments(self) -> List[Tuple[float, float, float, float]]:
        """
        Returns a list of wall rectangles in world-frame metres.
        Each entry: (centre_x, centre_y, size_x, size_y)
        The caller adds wall_height separately when spawning USD prims.

        Strategy: iterate every cell, emit its SOUTH and EAST walls if they
        are still standing.  Also emit the full perimeter.  This avoids
        double-counting shared walls between adjacent cells.

        World frame: origin at maze centre, +X = East, +Y = North.
        """
        cs = self.cell_size
        wt = self.wall_thickness
        # Total maze extent
        total_x = self.cols * cs
        total_y = self.rows * cs
        # Origin offset: top-left corner of the grid in world coords
        ox = -total_x / 2.0
        oy = -total_y / 2.0

        walls: List[Tuple[float, float, float, float]] = []

        def add(cx, cy, sx, sy):
            walls.append((cx, cy, sx, sy))

        # --- Perimeter walls ---
        # South perimeter (bottom edge, y = oy)
        add(ox + total_x / 2, oy,             total_x + wt, wt)
        # North perimeter (top edge,    y = oy + total_y)
        add(ox + total_x / 2, oy + total_y,   total_x + wt, wt)
        # West perimeter  (left edge,   x = ox)
        add(ox,               oy + total_y/2, wt, total_y + wt)
        # East perimeter  (right edge,  x = ox + total_x)
        add(ox + total_x,     oy + total_y/2, wt, total_y + wt)

        # --- Interior walls ---
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self._grid[r][c]

                # Cell bottom-left corner in world coords
                # Note: row 0 is at the top (north), so we flip y
                cell_x = ox + c * cs
                cell_y = oy + (self.rows - 1 - r) * cs

                # SOUTH wall of this cell = bottom horizontal segment
                if cell.walls[SOUTH]:
                    wx = cell_x + cs / 2.0
                    wy = cell_y
                    add(wx, wy, cs + wt, wt)

                # EAST wall of this cell = right vertical segment
                if cell.walls[EAST]:
                    wx = cell_x + cs
                    wy = cell_y + cs / 2.0
                    add(wx, wy, wt, cs + wt)

        return walls

    # ------------------------------------------------------------------
    # Spawn positions — 3 corners
    # ------------------------------------------------------------------

    def get_spawn_positions(self) -> List[Tuple[float, float]]:
        """
        Returns 3 world-frame (x, y) spawn positions, one near each of
        three different corners of the maze.  Each position is placed at
        the centre of the corner cell, guaranteed to be inside a corridor.

        Corners used: bottom-left, bottom-right, top-left.
        The top-right corner is reserved as a natural "goal" reference.
        """
        cs   = self.cell_size
        ox   = -(self.cols * cs) / 2.0
        oy   = -(self.rows * cs) / 2.0
        half = cs / 2.0

        # (row, col) of three corners
        corners = [
            (self.rows - 1, 0),              # bottom-left
            (self.rows - 1, self.cols - 1),  # bottom-right
            (0, 0),                           # top-left
        ]

        positions = []
        for r, c in corners:
            # World centre of this cell
            world_x = ox + c * cs + half
            world_y = oy + (self.rows - 1 - r) * cs + half
            positions.append((world_x, world_y))

        return positions

    # ------------------------------------------------------------------
    # ASCII debug print
    # ------------------------------------------------------------------

    def print_ascii(self):
        """Prints the maze to stdout — useful for quick sanity checks."""
        cs = self.cell_size

        # Top border
        print("+" + ("---+" * self.cols))

        for r in range(self.rows):
            # Cell interiors and east walls
            row_str = "|"
            for c in range(self.cols):
                row_str += "   "
                row_str += "|" if self._grid[r][c].walls[EAST] else " "
            print(row_str)

            # South walls
            south_str = "+"
            for c in range(self.cols):
                south_str += "---+" if self._grid[r][c].walls[SOUTH] else "   +"
            print(south_str)

    def get_maze_bounds(self) -> Tuple[float, float]:
        """Returns (total_width, total_height) of the maze in metres."""
        return self.cols * self.cell_size, self.rows * self.cell_size


# ---------------------------------------------------------------------------
# Quick test — run this file directly to see a generated maze
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42

    print(f"\nMaze (seed={seed}, 10x10 cells, 1.0 m cell size):\n")
    gen = MazeGenerator(
        rows=10, cols=10,
        cell_size=1.0, wall_thickness=0.2, wall_height=2.0,
        seed=seed,
    )
    gen.print_ascii()

    walls = gen.get_wall_segments()
    spawns = gen.get_spawn_positions()

    print(f"\nTotal wall segments: {len(walls)}")
    print(f"Spawn positions:     {spawns}")
    print(f"Maze bounds:         {gen.get_maze_bounds()} m")