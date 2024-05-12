import gpxpy
import gpxpy.gpx
import matplotlib.pyplot as plt


# Function to parse GPX file and extract information
def analyze_gpx(file_path):
    with open(file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)

    # Extracting track information
    for track in gpx.tracks:
        for segment in track.segments:
            print(f'Track: {track.name}')
            segment_distance = segment.length_3d()  # Distance in meters
            print(f'Total Distance: {segment_distance / 1000:.2f} km')
            
            # Elevation data
            elevations = [point.elevation for point in segment.points if point.elevation is not None]
            if elevations:
                max_elevation = max(elevations)
                min_elevation = min(elevations)
                elevation_gain = sum(max(0, b - a) for a, b in zip(elevations, elevations[1:]))
                print(f'Max Elevation: {max_elevation} m')
                print(f'Min Elevation: {min_elevation} m')
                print(f'Total Elevation Gain: {elevation_gain} m')

            # Time analysis (if timestamps are available)
            times = [point.time for point in segment.points if point.time is not None]
            if times:
                total_duration = (times[-1] - times[0]).total_seconds() / 3600  # Duration in hours
                print(f'Total Duration: {total_duration:.2f} hours')

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

                    current_split_distance = 0
                    current_split_elevation_gain = 0
                    current_split_start = current_point
                    
            for i, split in enumerate(split_distances):
                print(f'Split {i+1}: Distance = {split / 1000:.6f} km, Elevation Gain = {split_elevations[i]:.6f} m')
                if split_times:
                    print(f'         Time = {split_times[i]/3600:.6f} hours')
    # Visualizations
    fig, axs = plt.subplots(2, 1, figsize=(10, 15))
    

    # Plot Elevation Gain
    axs[0].plot(range(1, len(elevations)+1), elevations, marker='o', color='green')
    axs[0].set_title('Elevation Gain per Split (m)')
    axs[0].set_xlabel('Split')
    axs[0].set_ylabel('Elevation (m)')
    # Plot Time in Minutes and Seconds
    axs[1].plot(range(1, len(times)+1), times, marker='o', color='red')
    axs[1].set_title('Time per Split (minutes and seconds)')
    axs[1].set_xlabel('Split')
    axs[1].set_ylabel('Time (s)')

    # Convert seconds to MM:SS format
    axs[1].set_yticks(range(int(min(times)), int(max(times)) + 1, 10))  # Setting ticks every minute
    axs[1].set_yticklabels([f'{int(t//60)}:{int(t%60):02d}' for t in axs[1].get_yticks()])  # Formatting labels as MM:SS


    plt.tight_layout()
    plt.show()

data_file1 = 'data/gpx/route_2024-04-25_3.41pm.gpx'
# Example usage
analyze_gpx(data_file1)
analyze_and_visualize_gpx(data_file1, split_distance=1000)
