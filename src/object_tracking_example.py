"""
Object Detection and Tracking Example using Ultralytics YOLO

This script demonstrates object detection and tracking capabilities using Ultralytics YOLO.
It processes a video file and tracks objects across frames, displaying trajectories.

Based on: https://docs.ultralytics.com/es/modes/track/
"""

from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


def track_objects_with_trajectories(
    video_path: str,
    model_name: str = "yolo11n.pt",
    tracker_config: str = "botsort.yaml",
    show_video: bool = True,
    save_output: bool = True,
    max_trajectory_length: int = 30
):
    """
    Track objects in a video and visualize their trajectories.
    
    Args:
        video_path: Path to the input video file
        model_name: YOLO model to use (e.g., 'yolo11n.pt', 'yolo11n-seg.pt', 'yolo11n-pose.pt')
        tracker_config: Tracker configuration file (e.g., 'botsort.yaml', 'bytetrack.yaml')
        show_video: Whether to display the video in a window
        save_output: Whether to save the output video
        max_trajectory_length: Maximum number of points to keep in trajectory history
    """
    # Load the YOLO model
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)
    
    # Open video file
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    # Setup video writer if saving output
    out = None
    if save_output:
        output_path = video_path.parent / f"{video_path.stem}_tracked{video_path.suffix}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"Saving output to: {output_path}")
    
    # Track history for each object ID
    track_history = defaultdict(lambda: [])
    
    frame_count = 0
    
    print("Starting object tracking...")
    print("Press 'q' to quit, 'p' to pause/resume")
    
    paused = False
    
    while cap.isOpened():
        if not paused:
            success, frame = cap.read()
            if not success:
                break
            
            frame_count += 1
            
            # Perform tracking with persistence
            results = model.track(frame, persist=True, tracker=tracker_config)
            
            # Get tracking results
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                
                # Draw annotated frame
                annotated_frame = results[0].plot()
                
                # Draw trajectories
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    center_x = float(x)
                    center_y = float(y)
                    
                    # Add point to track history
                    track = track_history[track_id]
                    track.append((center_x, center_y))
                    
                    # Limit trajectory length
                    if len(track) > max_trajectory_length:
                        track.pop(0)
                    
                    # Draw trajectory line
                    if len(track) > 1:
                        points = np.array(track, dtype=np.int32)
                        cv2.polylines(
                            annotated_frame,
                            [points],
                            isClosed=False,
                            color=(230, 230, 230),
                            thickness=2
                        )
                    
                    # Draw current position circle
                    cv2.circle(
                        annotated_frame,
                        (int(center_x), int(center_y)),
                        radius=5,
                        color=(0, 255, 0),
                        thickness=-1
                    )
            else:
                # No detections, just use the frame
                annotated_frame = frame
            
            # Add frame counter
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_count}/{total_frames} | Objects: {len(track_history)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Save frame if needed
            if save_output and out is not None:
                out.write(annotated_frame)
            
            # Display frame
            if show_video:
                cv2.imshow("YOLO Object Tracking", annotated_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('p'):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
        else:
            # When paused, still check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('p'):
                paused = not paused
                print("Resumed")
    
    # Cleanup
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"\nTracking complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Unique objects tracked: {len(track_history)}")
    if save_output:
        print(f"Output saved to: {output_path}")


def simple_tracking_example(video_path: str, model_name: str = "yolo11n.pt"):
    """
    Simple tracking example without trajectory visualization.
    
    Args:
        video_path: Path to the input video file
        model_name: YOLO model to use
    """
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)
    
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    print(f"Processing video: {video_path}")
    print("Press 'q' to quit")
    
    # Perform tracking - this will display the video automatically
    results = model.track(
        source=str(video_path),
        save=True,
        show=True,
        tracker="botsort.yaml"  # or "bytetrack.yaml"
    )
    
    print("Tracking complete!")


if __name__ == "__main__":
    # Get the path to exampleVideo1.mp4 relative to this script
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    video_path = project_root / "exampleVideo1.mp4"
    
    print("=" * 60)
    print("Ultralytics YOLO Object Detection and Tracking")
    print("=" * 60)
    print()
    
    # Example 1: Tracking with trajectory visualization
    print("Running tracking with trajectory visualization...")
    try:
        track_objects_with_trajectories(
            video_path=str(video_path),
            model_name="yolo11n.pt",  # You can use 'yolo11n-seg.pt' or 'yolo11n-pose.pt'
            tracker_config="botsort.yaml",  # or "bytetrack.yaml"
            show_video=True,
            save_output=True,
            max_trajectory_length=30
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Looking for video at: {video_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    # Uncomment to run the simple tracking example instead:
    # simple_tracking_example(str(video_path), model_name="yolo11n.pt")

