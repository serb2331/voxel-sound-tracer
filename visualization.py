
# -------------------------
# Visualization
# -------------------------
from matplotlib import pyplot as plt

from constants import VOXEL_SIZE


def visualize_room_with_rays(grid, listener, sources=None, ray_events=None,
                             cmap='gray_r', figsize=(10, 10)):
    """
    Visualize the room, listener, sources, and ray paths in Cartesian coordinates.
    
    Args:
        grid: 2D voxel grid (Cartesian: bottom-left origin)
        listener: (x, y) listener position in world coordinates
        sources: list of (x, y) source positions, or single (x, y) tuple
        ray_events: dict mapping angles to event lists from trace_ray_2d
        cmap: colormap for grid visualization
        figsize: figure size tuple
    """
    plt.figure(figsize=figsize)
    
    # Grid dimensions in world coordinates
    grid_height = grid.shape[0]  # rows
    grid_width = grid.shape[1]   # columns
    world_width = grid_width * VOXEL_SIZE
    world_height = grid_height * VOXEL_SIZE
    
    # Display grid with correct orientation
    # extent = [left, right, bottom, top] in data coordinates
    plt.imshow(grid, cmap=cmap, interpolation='nearest',
               extent=[0, world_width, 0, world_height],
               origin='lower')  # origin='lower' ensures (0,0) is at bottom-left
    
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Room + Specular Ray Tracing (Cartesian Coordinates)")
    plt.grid(True, alpha=0.3)
    
    lx, ly = listener
    
    # Plot listener
    plt.scatter(lx, ly, c='red', marker='x', s=150, linewidths=3, 
                label='Listener', zorder=5)
    
    # Plot sources
    if sources is not None:
        # Handle both single source and list of sources
        if isinstance(sources, (list, tuple)) and len(sources) == 2 and isinstance(sources[0], (int, float)):
            # Single source as (x, y)
            sources = [sources]
        
        for idx, (sx, sy) in enumerate(sources):
            plt.scatter(sx, sy, c='blue', marker='o', s=100, 
                       label=f'Source {idx+1}' if len(sources) > 1 else 'Source',
                       zorder=5)
    
    # Plot ray paths with reflections
    if ray_events:
        num_rays = len(ray_events)
        # Sample a subset of rays for visualization if too many
        max_rays_to_plot = 100
        
        if num_rays > max_rays_to_plot:
            # Sample rays uniformly
            angles_to_plot = list(ray_events.keys())[::num_rays//max_rays_to_plot]
        else:
            angles_to_plot = list(ray_events.keys())
        
        for angle in angles_to_plot:
            events = ray_events[angle]
            if not events:
                continue
            
            # Start path at listener origin
            pts_x = [lx]
            pts_y = [ly]
            
            # Append all reflection points in order
            for ev in events:
                ex, ey = ev['pos']
                pts_x.append(ex)
                pts_y.append(ey)
            
            # Draw the full reflective polyline
            plt.plot(pts_x, pts_y, ':', linewidth=0.5, alpha=0.6)
            
            # Mark each reflection point
            for i, ev in enumerate(events):
                ex, ey = ev['pos']
                plt.scatter(ex, ey, c='orange', s=10, alpha=0.7, zorder=3)
    
    plt.legend(loc='upper right')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
