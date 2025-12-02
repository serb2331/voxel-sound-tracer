from math import sqrt
import numpy as np

# -------------------------
# Scene / material setup
# -------------------------

# -------------------------
# Room creation (Cartesian coordinates)
# -------------------------

def make_room(w=40, h=30, wall_value=1):
    """
    Create a rectangular room with walls and an inner obstacle.
    Uses Cartesian coordinates: grid[y, x] where y=0 is BOTTOM, x=0 is LEFT.
    
    Args:
        w: width in voxels
        h: height in voxels
        wall_value: value to use for walls
    
    Returns:
        2D numpy array of shape (h, w)
    """
    grid = np.zeros((h, w), dtype=np.int8)
    
    # Bottom wall (y=0)
    grid[0, :] = wall_value
    # Top wall (y=h-1)
    grid[-1, :] = wall_value
    # Left wall (x=0)
    grid[:, 0] = wall_value
    # Right wall (x=w-1)
    grid[:, -1] = wall_value
    
    # Add an inner obstacle (Cartesian: y from bottom)
    # This places obstacle at voxels y=12-15 (from bottom), x=18-21
    grid[12:16, 18:22] = wall_value
    
    return grid


def distance(a, b):
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)