import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Open the video file
video_path = "istockphoto-1423119278-640_adpp_is.mp4"
cap = cv2.VideoCapture(video_path)

# Dictionary to track the movement of objects
object_positions = {}  # {track_id: previous_x_center}

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
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
            y_center = (y_min + y_max) / 2

            # Determine the movement direction (if applicable)
            direction = None
            if track_id in object_positions:
                previous_x_center = object_positions[track_id]
                if x_center > previous_x_center:
                    direction = "Left to Right"
                elif x_center < previous_x_center:
                    direction = "Right to Left"

            # Update the tracked position of the object
            object_positions[track_id] = x_center

            # Print information about the object
            print(f"Track ID: {track_id}, Class: {class_id}, BBox: {bbox}, Direction: {direction}")

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()