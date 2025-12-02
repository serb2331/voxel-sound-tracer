from math import cos, pi, sin
import sys
import numpy as np

from renderer.audio_render import import_and_convolve_file
from scene.scene import make_room
from sir.sir import analyze_sir, compute_sir, print_sir_analysis
from tracer.simple_2d_audio_raytracer import trace_ray_2d, world_to_voxel
from visualization import visualize_room_with_rays



# -------------------------
# Test/Debug Functions
# -------------------------
def debug_ray_trace():
    """Debug function to test ray tracing with Cartesian coordinates."""
    grid = make_room()
    listener_pos = np.array([5.0, 8.0])
    source_pos = np.array([16.0, 10.0])
    
    print(f"Grid shape: {grid.shape} (height={grid.shape[0]}, width={grid.shape[1]})")
    print(f"Grid is in Cartesian: (0,0) = bottom-left, y-axis points UP")
    print(f"Listener at world: {listener_pos}, voxel: {world_to_voxel(listener_pos)}")
    print(f"Source at world: {source_pos}, voxel: {world_to_voxel(source_pos)}")
    print()
    
    # Test rays in cardinal directions
    test_angles = [0.0, pi/2, pi, 3*pi/2]
    angle_names = ["East (0°, +X)", "North (90°, +Y)", "West (180°, -X)", "South (270°, -Y)"]
    
    for angle, name in zip(test_angles, angle_names):
        dir_vec = np.array([cos(angle), sin(angle)])
        events = trace_ray_2d(grid, listener_pos, dir_vec, max_bounces=2)
        print(f"{name}:")
        print(f"  Direction: ({dir_vec[0]:.3f}, {dir_vec[1]:.3f})")
        if events:
            for i, ev in enumerate(events):
                print(f"  Bounce {i}: pos=({ev['pos'][0]:.2f}, {ev['pos'][1]:.2f}), "
                      f"voxel={ev['hit_voxel']}, dist={ev['distance_traveled']:.2f}m")
        else:
            print(f"  No wall hits (ray escaped grid)")
        print()


if __name__ == "__main__":


    # -------------------------------------------------------
    # Command-line arguments
    # Usage: python script.py input.wav output.wav
    # -------------------------------------------------------
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_sound_file> <output_sound_file>")
        sys.exit(1)

    # use "./sound_files/input_files/voice.wav" and "./sound_files/output_files/voice.wav"
    input_sound_file_path = sys.argv[1]
    output_sound_file_path = sys.argv[2]
    
    # -------------------------
    # Example run
    # -------------------------

    # listener and source positions in world coords (meters)
    # choose positions inside the grid (x,y) where x right, y down
    listener_pos = np.array([10.0, 15.0])   # meters
    source_pos   = np.array([30.0, 15.0]) # meters
    grid = make_room(40, 30)
    
    sample_rate = 8000

    sir, ray_details = compute_sir(grid, source_pos, listener_pos, n_rays=15000, max_time=1.0, sample_rate=sample_rate, max_bounces=5)

    # visualize_room_with_rays(grid, listener=listener_pos, sources=[source_pos], ray_events=ray_details)
    
    analyze_sir(sir, sample_rate=sample_rate)
    
    print_sir_analysis(sir, sample_rate=sample_rate)
    
    # Import and apply SIR to wav file sample
    import_and_convolve_file(input_sound_file_path, output_sound_file_path, sir=sir)
