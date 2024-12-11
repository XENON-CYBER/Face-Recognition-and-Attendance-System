import cv2
import os
import time

# Ask the user for their name
user_name = input("Enter your name: ")

# Set the directory where you want to save the images
save_dir = user_name
os.makedirs(save_dir, exist_ok=True)

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

# Number of images to capture
num_images = 25
count = 0

print("Capturing images...")

while count < num_images:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Check if the frame was captured correctly
    if not ret:
        print("Error: Failed to capture an image.")
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If a face is detected, adjust the window size to the face
    if len(faces) > 0:  # <- This line's indentation is now corrected
        # Get the first detected face (x, y, w, h)
        (x, y, w, h) = faces[0]

        # Crop the frame to the face region
        face_frame = frame[y:y+h, x:x+w]

        # Display the face region
        cv2.imshow('Capture Face', face_frame)

        # Save the image of the face
        image_path = os.path.join(save_dir, f'{user_name}_{count + 1}.jpg')
        cv2.imwrite(image_path, face_frame)
        print(f"Image saved: {image_path}")
        count += 1

        # Wait for 5 seconds before capturing the next image
        time.sleep(5.5)

        # Exit the loop if 'q' is pressed during capturing
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting.")
            break
    else:
        # Display the full frame if no face is detected
        cv2.imshow('Capture Face', frame)

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

print(f"Captured {count} images.")
