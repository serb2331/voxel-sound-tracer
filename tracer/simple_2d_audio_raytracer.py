import numpy as np
from constants import DEFAULT_ABSORPTION, VOXEL_SIZE
from scene.scene import distance


# -------------------------
# Utility functions
# -------------------------

def world_to_voxel(pos):
    """Convert world position (x, y) to voxel coordinates (vx, vy).
    World: x right, y up from bottom-left.
    Grid: grid[vy, vx], vy=0 at bottom.
    """
    return int(np.floor(pos[0] / VOXEL_SIZE)), int(np.floor(pos[1] / VOXEL_SIZE))


def trace_ray_2d(grid, start_position, direction, max_distance=100.0, max_bounces=5, absorption=DEFAULT_ABSORPTION):
    """
    Correct 2D DDA voxel traversal with reflection.
    Returns list of hit events (including early reflections).
    """
    events = []
    ray_origin = start_position
    pos = np.array(ray_origin, dtype=np.float64)
    dir_vec = np.array(direction, dtype=np.float64)
    dir_vec = dir_vec / np.linalg.norm(dir_vec + 1e-30)  # normalize safely

    h, w = grid.shape  # h = num rows (y), w = num cols (x)

    remaining_energy = 1.0
    previous_traveled = 0.0
    ray_traveled = 0.0

    for bounce in range(max_bounces + 1):
        # Current voxel
        vx, vy = world_to_voxel(pos)

        # If starting outside or exactly on boundary in a bad way, reject early
        if not (0 <= vx < w and 0 <= vy < h):
            break
        if grid[vy, vx] != 0:  # started inside wall?
            break

        # Determine stepping direction
        step_x = 1 if dir_vec[0] > 0 else -1
        step_y = 1 if dir_vec[1] > 0 else -1

        # Distance to next voxel boundary in each direction
        if abs(dir_vec[0]) < 1e-12:
            tMaxX = np.inf
            tDeltaX = np.inf
        else:
            next_boundary_x = (vx + (1 if step_x > 0 else 0)) * VOXEL_SIZE
            tMaxX = (next_boundary_x - pos[0]) / dir_vec[0]
            tDeltaX = VOXEL_SIZE / abs(dir_vec[0])

        if abs(dir_vec[1]) < 1e-12:
            tMaxY = np.inf
            tDeltaY = np.inf
        else:
            next_boundary_y = (vy + (1 if step_y > 0 else 0)) * VOXEL_SIZE
            tMaxY = (next_boundary_y - pos[1]) / dir_vec[1]
            tDeltaY = VOXEL_SIZE / abs(dir_vec[1])

        hit_something = False
        
        # DDA traversal
        while (previous_traveled + ray_traveled) < max_distance:
            vx, vy = world_to_voxel(pos)  # recompute in case of floating point drift

            # Check if we left the grid
            if not (0 <= vx < w and 0 <= vy < h):
                break

            # Advance to next voxel boundary
            if tMaxX < tMaxY:
                dt = tMaxX
                tMaxX += tDeltaX
                vx += step_x
                hit_normal = np.array([-step_x, 0.0])  # left/right wall
                hit_face = 'vertical'
            else:
                dt = tMaxY
                tMaxY += tDeltaY
                vy += step_y
                hit_normal = np.array([0.0, -step_y])  # top/bottom wall
                hit_face = 'horizontal'

            # Move position (stop just before crossing into new voxel)
            pos = np.array(ray_origin, dtype=np.float64) + dir_vec * (dt + 1e-8)  # tiny epsilon to enter new voxel
            ray_traveled = distance(ray_origin, pos)

            # Now in new voxel (vx, vy)
            if not (0 <= vx < w and 0 <= vy < h):
                break

            # Did we hit a wall?
            if grid[vy, vx] != 0:
                # Back pos up to exact hit point (on boundary)
                pos -= dir_vec * 1e-7  # pull back slightly from wall

                events.append({
                    'pos': pos.copy(),
                    'distance_traveled': ray_traveled,
                    'hit_voxel': (vx, vy),
                    'hit_normal': hit_normal,
                    'hit_face': hit_face,
                    'remaining_energy': remaining_energy,
                    'bounce': bounce
                })

                remaining_energy *= (1.0 - absorption)
                if remaining_energy < 1e-4:
                    return events

                # === SPECULAR REFLECTION ===
                
                dir_vec = dir_vec - 2.0 * np.dot(dir_vec, hit_normal) * hit_normal
                dir_vec /= np.linalg.norm(dir_vec) + 1e-30

                # Update the origin of the specular ray
                ray_origin = pos
                
                # Update the previous travelled amount
                previous_traveled += ray_traveled
                
                # Nudge forward into free space to avoid self-intersection
                # pos += dir_vec * 1e-6

                hit_something = True
                break  # go to next bounce

        if not hit_something:
            break  # escaped without hitting anything this bounce

    return events
