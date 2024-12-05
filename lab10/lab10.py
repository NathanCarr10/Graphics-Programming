import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Open the input video file
video_path = "istockphoto-1423119278-640_adpp_is.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter to save the output video
output_path = "output_tracked_video.mp4"  # Output video file path
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 files
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Dictionary to track the movement of objects
object_positions = {}  # {track_id: previous_x_center}

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Run YOLO tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Iterate through detections
        for detection in results[0].boxes:
            track_id = int(detection.id)  # Unique Track ID
            class_id = int(detection.cls)  # Class ID
            bbox = detection.xyxy[0].tolist()  # Bounding box [x_min, y_min, x_max, y_max]
            x_min, y_min, x_max, y_max = bbox

            # Calculate the center of the bounding box
            x_center = (x_min + x_max) / 2

            # Determine the movement direction
            direction = None
            if track_id in object_positions:
                previous_x_center = object_positions[track_id]
                if x_center > previous_x_center:
                    direction = "Left to Right"
                elif x_center < previous_x_center:
                    direction = "Right to Left"

            # Update the tracked position of the object
            object_positions[track_id] = x_center

            # Annotate the frame with movement direction
            label = f"ID: {track_id}, Dir: {direction or 'Stationary'}"
            cv2.putText(frame, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Save the annotated frame to the output video
        out.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("YOLO Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Tracking complete. Output video saved as {output_path}")