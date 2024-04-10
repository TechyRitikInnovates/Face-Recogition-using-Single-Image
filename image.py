import face_recognition
import numpy as np

def compare_faces(image_path1, image_path2):
    # Load the images
    image1 = face_recognition.load_image_file(image_path1)
    image2 = face_recognition.load_image_file(image_path2)

    # Encode faces
    face_encoding1 = face_recognition.face_encodings(image1)
    face_encoding2 = face_recognition.face_encodings(image2)

    if len(face_encoding1) == 0:
        print("No face detected in the first image.")
        return
    if len(face_encoding2) == 0:
        print("No face detected in the second image.")
        return

    # Take the first face encoding from each list
    face_encoding1 = face_encoding1[0]
    face_encoding2 = face_encoding2[0]

    # Compare faces
    distance = face_recognition.face_distance([face_encoding1], face_encoding2)

    # Adjust this threshold as needed
    if distance < 0.6:
        print("Same face detected!")
    else:
        print("No matching face detected.")

# Paths to the images
image_path1 = "./faces/1.webp"
image_path2 = "./faces/2.jpeg"

# Compare faces in both images
compare_faces(image_path1, image_path2)