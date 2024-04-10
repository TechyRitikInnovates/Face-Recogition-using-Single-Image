import face_recognition
import cv2
import os
import numpy as np

def load_known_faces(known_faces_folder):
    known_face_encodings = []
    known_face_names = []

    # Loop through each file in the folder
    for file_name in os.listdir(known_faces_folder):
        # Load the image
        image_path = os.path.join(known_faces_folder, file_name)
        image = face_recognition.load_image_file(image_path)

        # Encode the face
        face_encoding = face_recognition.face_encodings(image)
        if len(face_encoding) > 0:
            known_face_encodings.append(face_encoding[0])
            known_face_names.append(os.path.splitext(file_name)[0])  # Extracting the name without extension

    return known_face_encodings, known_face_names

def compare_faces(known_face_encodings, known_face_names, frame):
    # Convert the frame from BGR to RGB
    rgb_frame = frame[:, :, ::-1]

    # Find faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    if face_locations:
        # Encode the face found in the frame
        face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]

        # Compare with known faces
        distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        # Find the closest match
        min_distance_index = np.argmin(distances)
        min_distance = distances[min_distance_index]
        min_distance_name = known_face_names[min_distance_index]

        # Adjust this threshold as needed
        if min_distance < 0.6:
            print(f"Match found: {min_distance_name} (Distance: {min_distance})")
        else:
            print("No matching face detected.")

def main():
    # Folder containing known faces
    known_faces_folder = "./Known_Faces"
    known_face_encodings, known_face_names = load_known_faces(known_faces_folder)

    # Start capturing video from webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # Compare faces with known faces
        compare_faces(known_face_encodings, known_face_names, frame)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
