import os
import cv2
import face_recognition
import numpy as np


# Function to load images from the subfolders
def load_face_images_from_folders(folder_path):
    known_face_encodings = []
    known_face_names = []

    for user_folder in os.listdir(folder_path):
        user_folder_path = os.path.join(folder_path, user_folder)
        if os.path.isdir(user_folder_path):
            for filename in os.listdir(user_folder_path):
                file_path = os.path.join(user_folder_path, filename)
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        # Load the image file and get face encodings
                        image = face_recognition.load_image_file(file_path)
                        encodings = face_recognition.face_encodings(image)
                        for encoding in encodings:
                            known_face_encodings.append(encoding)
                            known_face_names.append(user_folder)
                    except Exception as e:
                        print(f"Error processing file {filename} in {user_folder}: {e}")
    return known_face_encodings, known_face_names


# Function to capture frames from the video feed and detect faces
def detect_faces_from_video(video_capture):
    ret, frame = video_capture.read()
    if not ret:
        return None, None, None

    # Convert the frame from BGR to RGB
    rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])

    # Find face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    return frame, face_locations, face_encodings


# Function to compare detected faces with known faces and label them
def label_detected_faces(known_face_encodings, known_face_names, face_encodings, face_locations, frame):
    face_names = []
    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        # Compare the detected face with known face encodings
        matches = face_recognition.compare_faces(known_face_encodings, encoding)
        name = "Unknown"

        # Check if any known face encodings match
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Label the face with the name
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        face_names.append(name)
    return frame, face_names


# Main function to orchestrate everything
def main():
    folder_path = "FacesTing"  # Path to your folder containing subfolders of images

    # Load known face encodings from folders
    known_face_encodings, known_face_names = load_face_images_from_folders(folder_path)

    # Start the video capture
    video_capture = cv2.VideoCapture(0)

    while True:
        # Detect faces from the video
        frame, face_locations, face_encodings = detect_faces_from_video(video_capture)
        if frame is None:
            break

        # Label detected faces and draw rectangles
        frame, face_names = label_detected_faces(known_face_encodings, known_face_names, face_encodings, face_locations,
                                                 frame)

        # Display the resulting frame
        cv2.imshow('Camera Feed', frame)

        # Press 'q' to exit the camera feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()


# Run the main function
if __name__ == "__main__":
    main()

# face detection using face_cascade
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the frame
    #faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    #if len(faces) > 0:
        # Draw rectangles around the detected faces
        #for (x, y, width, height) in faces:
            # Draw a rectangle around each detected face
            #cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)  # Blue rectangle with 2 px thickness


#while True:
    # Capture frame-by-frame
 #   ret, frame = video_capture.read()

    # Convert the frame from BGR to RGB
  #  rgb_frame = frame[:, :, ::-1]

    # Find all the face locations in the current frame
   # face_locations = face_recognition.face_locations(rgb_frame)

    # Draw rectangles around each detected face
    #for (top, right, bottom, left) in face_locations:
        # Draw a green rectangle around the face
     #   cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)


    # Display the resulting frame
   # cv2.imshow('Camera Feed', frame)
    # Press 'q' to exit the camera feed
    #if cv2.waitKey(1) & 0xFF == ord('q'):
     #   break
# Release the camera and close all OpenCV windows
#video_capture.release()
#cv2.destroyAllWindows()