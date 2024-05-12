import gpxpy
import matplotlib.pyplot as plt
import numpy as np
def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

def analyze_and_visualize_gpx(file_path, split_distance=1000):
    with open(file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)

    distances = []
    elevations = []
    times = []

    for track in gpx.tracks:
        for segment in track.segments:
            current_split_distance = 0
            current_split_elevation_gain = 0
            current_split_start = segment.points[0]

            for point_index in range(1, len(segment.points)):
                previous_point = segment.points[point_index - 1]
                current_point = segment.points[point_index]
                distance = previous_point.distance_3d(current_point)
                current_split_distance += distance

                if current_point.elevation and previous_point.elevation:
                    elevation_change = current_point.elevation - previous_point.elevation
                    if elevation_change > 0:
                        current_split_elevation_gain += elevation_change

                if current_split_distance >= split_distance:
                    distances.append(current_split_distance)
                    elevations.append(current_split_elevation_gain)
                    if current_point.time and current_split_start.time:
                        duration = (current_point.time - current_split_start.time).total_seconds()
                        times.append(duration)

                    # Reset for next split
                    current_split_distance = 0
                    current_split_elevation_gain = 0
                    current_split_start = current_point

            # Add the last split if it didn't reach the full distance
            if current_split_distance > 0:
                distances.append(current_split_distance)
                elevations.append(current_split_elevation_gain)
                if current_split_start.time and segment.points[-1].time:
                    duration = (segment.points[-1].time - current_split_start.time).total_seconds()
                    times.append(duration)

    # Output split stats
    for i, distance in enumerate(distances):
        print(f'Split {i+1}: Distance = {distance / 1000:.2f} km, Elevation Gain = {elevations[i]:.2f} m')
        if times:
            print(f'         Time = {format_time(times[i])}')

def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
def analyze_run(file_path):
    with open(file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)

    total_stats = {
        'distance': 0,
        'max_elevation': 0,
        'min_elevation': float('inf'),
        'elevation_gain': 0,
        'duration': 0,
        'start_date': None
    }

    for track in gpx.tracks:
        for segment in track.segments:
            if not total_stats['start_date'] and segment.points and segment.points[0].time:
                total_stats['start_date'] = segment.points[0].time.strftime('%Y-%m-%d')

            segment_distance = segment.length_3d()
            total_stats['distance'] += segment_distance

            segment_elevations = [point.elevation for point in segment.points if point.elevation is not None]
            if segment_elevations:
                max_elevation = max(segment_elevations)
                min_elevation = min(segment_elevations)
                elevation_gain = 0
                for i in range(1, len(segment_elevations)):
                    elevation_change = segment_elevations[i] - segment_elevations[i - 1]
                    if elevation_change > 0:
                        elevation_gain += elevation_change

                total_stats['max_elevation'] = max(total_stats['max_elevation'], max_elevation)
                total_stats['min_elevation'] = min(total_stats['min_elevation'], min_elevation)
                total_stats['elevation_gain'] += elevation_gain

            segment_times = [point.time for point in segment.points if point.time is not None]
            if segment_times:
                total_stats['duration'] += (segment_times[-1] - segment_times[0]).total_seconds()

    return total_stats

def calculate_speeds(segment):
    speeds = []
    for i in range(1, len(segment.points)):
        point1 = segment.points[i - 1]
        point2 = segment.points[i]
        distance = point1.distance_3d(point2)  # distance in meters
        time_diff = (point2.time - point1.time).total_seconds() / 3600  # time in hours
        if time_diff > 0:
            speed = distance / (time_diff * 1000)  # speed in km/h
            speeds.append(speed)
    return speeds

def detect_intervals(speeds, threshold=10):
    changes = []
    for i in range(1, len(speeds)):
        if abs(speeds[i] - speeds[i - 1]) > threshold:  # Threshold of 10 km/h change
            changes.append(i)
    return changes

def define_splits(segment, changes):
    splits = []
    last_index = 0
    for change_index in changes:
        splits.append((last_index, change_index))
        last_index = change_index
    splits.append((last_index, len(segment.points) - 1))  # Add the final split
    return splits
def analyze_gpx_for_intervals(file_path, speed_change_threshold=10):
    with open(file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)

    interval_data = []
    for track in gpx.tracks:
        for segment in track.segments:
            speeds = calculate_speeds(segment)
            changes = detect_intervals(speeds, speed_change_threshold)
            splits = define_splits(segment, changes)

            # Collect data for each split
            for start, end in splits:
                # Calculate distance and time for each split
                split_distance = sum(segment.points[j].distance_3d(segment.points[j + 1]) for j in range(start, end))
                start_time = segment.points[start].time
                end_time = segment.points[end].time
                duration = (end_time - start_time).total_seconds()

                interval_data.append({
                    'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'distance': split_distance,
                    'duration': duration,
                    'type': 'Warm-up' if start == 0 else ('Cool-down' if end == len(segment.points) - 1 else 'Interval')
                })

    return interval_data

def analyze_splits(file_path, split_distance=1000):
    with open(file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)

    split_stats = {
        'distances': [],
        'elevations': [],
        'times': [],
        'timestamps': []
    }

    for track in gpx.tracks:
        for segment in track.segments:
            current_split_distance = 0
            current_split_elevation_gain = 0
            current_split_start = segment.points[0]

            for point_index in range(1, len(segment.points)):
                previous_point = segment.points[point_index - 1]
                current_point = segment.points[point_index]
                distance = previous_point.distance_3d(current_point)
                current_split_distance += distance

                if current_point.elevation and previous_point.elevation:
                    elevation_change = current_point.elevation - previous_point.elevation
                    if elevation_change > 0:
                        current_split_elevation_gain += elevation_change

                if current_split_distance >= split_distance:
                    split_stats['distances'].append(current_split_distance)
                    split_stats['elevations'].append(current_split_elevation_gain)
                    if current_point.time and current_split_start.time:
                        duration = (current_point.time - current_split_start.time).total_seconds()
                        split_stats['times'].append(duration)
                        split_stats['timestamps'].append(current_split_start.time.strftime('%Y-%m-%d %H:%M:%S'))

                    # Reset for next split
                    current_split_distance = 0
                    current_split_elevation_gain = 0
                    current_split_start = current_point

            # Handling the last split if it exists
            if current_split_distance > 0:
                split_stats['distances'].append(current_split_distance)
                split_stats['elevations'].append(current_split_elevation_gain)
                if current_split_start.time and segment.points[-1].time:
                    duration = (segment.points[-1].time - current_split_start.time).total_seconds()
                    split_stats['times'].append(duration)
                    split_stats['timestamps'].append(current_split_start.time.strftime('%Y-%m-%d %H:%M:%S'))
    print(f"Total Splits: {len(split_stats['distances'])}")
    print(f"Total Distance: {sum(split_stats['distances']) / 1000:.2f} km")
    print(f"Total Elevation Gain: {sum(split_stats['elevations']):.2f} m")
    if split_stats['times']:
        print(f"Total Duration: {format_time(sum(split_stats['times']))}")
    return split_stats


def compare_and_visualize_runs(file_path1, file_path2):
    stats1 = analyze_run(file_path1)
    stats2 = analyze_run(file_path2)

    print("Run Comparison:")
    print(f"Date: Run1 = {stats1['start_date']}, Run2 = {stats2['start_date']}")
    print(f"Total Distance: Run1 = {stats1['distance'] / 1000:.2f} km, Run2 = {stats2['distance'] / 1000:.2f} km")
    print(f"Max Elevation: Run1 = {stats1['max_elevation']} m, Run2 = {stats2['max_elevation']} m")
    print(f"Min Elevation: Run1 = {stats1['min_elevation']} m, Run2 = {stats2['min_elevation']} m")
    print(f"Elevation Gain: Run1 = {stats1['elevation_gain']} m, Run2 = {stats2['elevation_gain']} m")
    print(f"Total Duration: Run1 = {format_time(stats1['duration'])}, Run2 = {format_time(stats2['duration'])}")

def plot_splits_comparison(run1_stats, run2_stats):
    # Ensure equal length of data
    min_length = min(len(run1_stats['times']), len(run2_stats['times']))
    indexes = np.arange(min_length) + 1  # Split numbers

    # Convert times from seconds to minutes
    times1 = [t / 60 for t in run1_stats['times'][:min_length]]
    times2 = [t / 60 for t in run2_stats['times'][:min_length]]

    # Combine all times for both runs to establish the ticks
    all_times = sorted(set(times1 + times2))

    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.35  # the width of the bars

    # Plotting data
    rects1 = ax.bar(indexes - width/2, times1, width, label='Run 1')
    rects2 = ax.bar(indexes + width/2, times2, width, label='Run 2')

    ax.set_xlabel('Splits')
    ax.set_ylabel('Time per Split (minutes and seconds)')
    ax.set_title('Comparison of Split Times for Two Runs')
    ax.set_xticks(indexes)
    ax.set_xticklabels([f'Split {i}' for i in indexes])

    # Function to format minutes into "Xm Ys"
    def format_ticks(t):
        minutes = int(t)
        seconds = int((t - minutes) * 60)
        return f"{minutes}m {seconds:02d}s"

    # Apply the unique times as y-ticks and format them
    ax.set_yticks(all_times)
    ax.set_yticklabels([format_ticks(t) for t in all_times])

    # Enhance grid visibility for better reading
    ax.grid(which='both', color='gray', linestyle=':', linewidth=0.5)

    ax.legend()
    plt.tight_layout()
    plt.show()

    
    
data_file1 = 'data/gpx/2024-04-25-full-run.gpx'
data_file2 = 'data/gpx/route_2024-05-09_4.47pm.gpx'
interval_analysis = analyze_gpx_for_intervals(data_file1, speed_change_threshold=8)
for interval in interval_analysis:
    print(interval)
# print("Analysis for Run 1:")
# compare_and_visualize_runs(data_file1, data_file2)
# run1_stats=analyze_splits(data_file1, split_distance=1000)
# print("\nAnalysis for Run 2:")
# run2_stats=analyze_splits(data_file2, split_distance=1000)
# plot_splits_comparison(run1_stats, run2_stats)