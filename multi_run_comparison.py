import os
import gpxpy
import matplotlib.pyplot as plt
import numpy as np

running_speed_threshold = 1.5  # m/s

def list_gpx_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.gpx')]

def extract_data_from_gpx(file_path, split_distance=1000):
    with open(file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)

    splits_data = []
    start_time = None

    for track in gpx.tracks:
        for segment in track.segments:
            if segment.points:
                if not start_time:
                    start_time = segment.points[0].time.strftime('%Y-%m-%d %H:%M')
            current_split_distance = 0
            current_split_time = 0
            current_split_start = segment.points[0]

            for point in segment.points[1:]:
                distance = current_split_start.distance_3d(point)
                current_split_distance += distance

                if point.time and current_split_start.time:
                    time_difference = (point.time - current_split_start.time).total_seconds()
                    current_split_time += time_difference

                if current_split_distance >= split_distance:
                    splits_data.append(current_split_time)  # Capture split time
                    current_split_distance = 0  # Resetting split distance
                    current_split_time = 0  # Resetting split time
                    current_split_start = point  # Starting new split

            # Adding remaining distance and time not reaching the next split
            if current_split_distance > 0:
                splits_data.append(current_split_time)  # Capture remaining split time

    return start_time, splits_data

def format_time(seconds):
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{int(minutes)}m {int(seconds)}s"

def aggregate_and_plot(directory):
    files = list_gpx_files(directory)
    session_labels = []
    all_split_times = []

    for file in files:
        if os.path.basename(file) in files_to_ignore:
            continue
        session_label, split_times = extract_data_from_gpx(file)
        # Flatten list to handle multiple split times
        all_split_times.extend(split_times)
        session_labels.extend([session_label] * len(split_times))

    sorted_data = sorted(zip(session_labels, all_split_times), key=lambda x: x[0])
    session_labels, all_split_times = zip(*sorted_data)
    # Plotting
    plt.figure(figsize=(14, 6))
    plt.bar(session_labels, all_split_times, color='blue')
    plt.title('Time per Split per Session')
    plt.xlabel('Session')
    plt.ylabel('Split Time')
    plt.xticks(rotation=45, ha="right")
    plt.gca().set_yticklabels([format_time(t) for t in plt.gca().get_yticks()])  # Applying formatting

    plt.tight_layout()
    plt.show()

# Usage
files_to_ignore = ['route_2024-01-23_4.24pm.gpx', 'route_2024-01-23_4.24pm.gpx']
directory_path = 'data/gpx'
aggregate_and_plot(directory_path)
