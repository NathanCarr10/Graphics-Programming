import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# Open the video file
video_path = "istockphoto-1423119278-640_adpp_is.mp4"
cap = cv2.VideoCapture(video_path)

# Dictionary to track unique IDs for each class
tracked_objects = {}

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Iterate through detections and track unique IDs per class
        for detection in results[0].boxes:
            class_id = int(detection.cls)  # Class ID
            track_id = int(detection.id)  # Unique Track ID

            if class_id not in tracked_objects:
                tracked_objects[class_id] = set()
            
            # Add the track ID to the set for the corresponding class
            tracked_objects[class_id].add(track_id)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # Print the number of unique tracked objects for each class
        print("Tracked Objects:")
        for class_id, unique_ids in tracked_objects.items():
            print(f"Class {class_id}: {len(unique_ids)} unique detections")

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()