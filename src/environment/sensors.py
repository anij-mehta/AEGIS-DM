import math

class SensoryLayer:
    def __init__(self, grid_size):
        self.grid_size = grid_size

    def get_line_of_sight(self, start_pos, end_pos, blocked_cells):
        """Bresenham's Line Algorithm for grid-based LoS[cite: 61]."""
        x0, y0 = start_pos
        x1, y1 = end_pos
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            if (x0, y0) == (x1, y1): return True
            if (x0, y0) in blocked_cells: return False
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy