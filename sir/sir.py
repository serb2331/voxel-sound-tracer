from math import cos, pi, sin
import numpy as np

from constants import C, CAPTURE_RADIUS
from tracer.simple_2d_audio_raytracer import distance, trace_ray_2d

# -------------------------
# Impulse response accumulation
# -------------------------
def compute_sir(grid, source, listener, n_rays=4096, max_time=2.0, sample_rate=16000, max_bounces=3):
    """
    Monte Carlo ray tracing from LISTENER to detect SOURCE.
    Checks if ray segments (paths between consecutive bounces) pass near the source.
    
    Args:
        grid: 2D voxel grid in Cartesian coordinates
        source: (x, y) position of sound source in world coordinates
        listener: (x, y) position of listener in world coordinates
        n_rays: number of rays to trace
        max_time: maximum time to simulate (seconds)
        sample_rate: samples per second for output
        max_bounces: maximum number of reflections per ray
    
    Returns:
        sir: numpy array of impulse response samples
        ray_details: dictionary of ray angles and their events
    """
    sir_len = int(max_time * sample_rate)
    sir = np.zeros(sir_len, dtype=np.float32)
    
    origin = np.array(listener, dtype=float)
    destination = np.array(source, dtype=float)
    
    ray_details = {}
    
    # Emit rays uniformly over 2*pi
    angles = np.linspace(0, 2*pi, n_rays, endpoint=False)
    
    # Capture radius for source detection
    capture_radius = CAPTURE_RADIUS  # meters
    
    for a in angles:
        dir_vec = np.array([cos(a), sin(a)])
        events = trace_ray_2d(grid, origin, dir_vec, max_distance=100.0, max_bounces=max_bounces)
        ray_details[a] = events
        
        # Build list of path segments to check
        # Segment 0: listener -> first bounce (or listener -> infinity if no bounces)
        # Segment i: bounce[i-1] -> bounce[i]
        
        path_points = [origin]  # Start with listener position
        
        # Add all bounce positions
        for ev in events:
            path_points.append(ev['pos'])
        
        # Now check each segment between consecutive points
        found_source = False
        
        for i in range(len(path_points)):
            # Define segment from point i to point i+1
            segment_start = path_points[i]
            
            if i < len(path_points) - 1:
                # Segment between two known points (listener->bounce or bounce->bounce)
                segment_end = path_points[i + 1]
                segment_vec = segment_end - segment_start
                segment_length = np.linalg.norm(segment_vec)
                
                if segment_length < 1e-9:
                    continue
                
                segment_dir = segment_vec / segment_length
                
                # Check if source is along this segment
                to_source = destination - segment_start
                projection = np.dot(to_source, segment_dir)
                
                # Source must project onto the segment (not beyond it)
                if 0 <= projection <= segment_length:
                    # Calculate perpendicular distance from segment to source
                    closest_point = segment_start + segment_dir * projection
                    perpendicular_dist = distance(closest_point, destination)
                    
                    if perpendicular_dist <= capture_radius:
                        # Ray passes near source!
                        # Calculate total path length and energy
                        if i == 0:
                            # Direct path segment (no prior bounces)
                            total_path_length = projection + perpendicular_dist
                            energy = 1.0
                        else:
                            # Path includes bounces
                            total_path_length = events[i-1]['distance_traveled'] + projection + perpendicular_dist
                            energy = events[i-1]['remaining_energy']
                        
                        # Calculate time of arrival
                        t = total_path_length / C
                        idx = int(round(t * sample_rate))
                        
                        if 0 <= idx < sir_len:
                            # Amplitude: energy / distance (inverse distance law for 2D)
                            # amp = energy / max(0.1, total_path_length)
                            # sir[idx] += amp
                            
                            # Amplitude calculation for 2D ray tracing:
                            # - Energy accounts for absorption from bounces
                            # - Geometric spreading in 2D: sqrt(1/r) is more accurate than 1/r
                            # - We'll normalize by solid angle later, not per-ray
                            geometric_factor = 1.0 / np.sqrt(max(1.0, total_path_length))
                            amp = energy * geometric_factor
                            sir[idx] += amp
                        
                        found_source = True
                        break  # Only count first detection ZZper ray
            
            else:
                # Last segment: extends from last bounce to infinity
                # Check if source is in front of this segment
                if i > 0:
                    # Get direction after last bounce (from last event's reflection)
                    # We need to reconstruct the direction, but we don't have it stored
                    # For now, skip the infinite segment check
                    # (In practice, most detections happen on finite segments)
                    pass
        
        # Alternative: If no detection found, check direct line from listener to source
        # only if there are no bounces (empty events)
        if not found_source and len(events) == 0:
            # Direct path with no obstructions
            direct_vec = destination - origin
            direct_dist = np.linalg.norm(direct_vec)
            
            if direct_dist < 1e-9:
                continue
            
            direct_dir = direct_vec / direct_dist
            
            # Check if ray direction is close to direct direction
            angle_diff = np.arccos(np.clip(np.dot(dir_vec, direct_dir), -1.0, 1.0))
            
            # If ray is within capture angle (rough approximation)
            capture_angle = np.arctan(capture_radius / direct_dist)
            
            if angle_diff <= capture_angle:
                t = direct_dist / C
                idx = int(round(t * sample_rate))
                
                if 0 <= idx < sir_len:
                    # amp = 1.0 / max(0.1, direct_dist)
                    
                    geometric_factor = 1.0 / np.sqrt(max(1.0, direct_dist))
                    amp = geometric_factor
                    sir[idx] += amp
    
    # Normalize by number of rays (Monte Carlo)
    # sir /= float(n_rays)
    
    # Normalize by solid angle coverage per ray (2π / n_rays)
    # This accounts for the discrete sampling of directions
    solid_angle_per_ray = (2.0 * np.pi) / n_rays
    sir *= solid_angle_per_ray
    
    # Scale to reasonable amplitude range (empirical adjustment)
    # The peak should be around 0.1-1.0 for good convolution results
    if np.max(sir) > 0:
        sir *= (0.5 / np.max(sir))  # Normalize peak to 0.5
    
    return sir, ray_details


def analyze_sir(sir, sample_rate):
    """
    Analyze the spatial impulse response to diagnose issues.
    
    Args:
        sir: Spatial impulse response
        sample_rate: Sample rate used
    
    Returns:
        Dictionary with analysis metrics
    """
    non_zero = np.nonzero(sir)[0]
    
    if len(non_zero) == 0:
        return {
            'num_arrivals': 0,
            'peak_amplitude': 0.0,
            'first_arrival_time': None,
            'last_arrival_time': None,
            'energy': 0.0,
            'sparsity': 0.0
        }
    
    first_idx = non_zero[0]
    last_idx = non_zero[-1]
    
    return {
        'num_arrivals': len(non_zero),
        'peak_amplitude': np.max(np.abs(sir)),
        'mean_amplitude': np.mean(np.abs(sir[non_zero])),
        'first_arrival_time': first_idx / sample_rate,
        'last_arrival_time': last_idx / sample_rate,
        'energy': np.sum(sir ** 2),
        'sparsity': len(non_zero) / len(sir),
        'first_10_arrivals': [(i / sample_rate, sir[i]) for i in non_zero[:10]]
    }
    
    
def print_sir_analysis(sir, sample_rate):
    """Print human-readable analysis of the SIR."""
    analysis = analyze_sir(sir, sample_rate)
    
    print("=" * 60)
    print("SPATIAL IMPULSE RESPONSE ANALYSIS")
    print("=" * 60)
    print(f"Total arrivals:      {analysis['num_arrivals']}")
    print(f"Sparsity:            {analysis['sparsity']*100:.2f}%")
    print(f"Peak amplitude:      {analysis['peak_amplitude']:.6f}")
    print(f"Mean amplitude:      {analysis['mean_amplitude']:.6f}")
    print(f"Total energy:        {analysis['energy']:.6f}")
    print(f"First arrival:       {analysis['first_arrival_time']*1000:.2f} ms")
    print(f"Last arrival:        {analysis['last_arrival_time']*1000:.2f} ms")
    print(f"Response duration:   {(analysis['last_arrival_time']-analysis['first_arrival_time'])*1000:.2f} ms")
    print("\nFirst 10 arrivals:")
    for t, amp in analysis['first_10_arrivals']:
        print(f"  {t*1000:7.2f} ms: {amp:.6f}")
    print("=" * 60)
    