import os
import cv2
import face_recognition
import shutil
import numpy as np
from sklearn.cluster import DBSCAN
import streamlit as st

# Paths
KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
OUTPUT_DIR = "organized_faces"

# Parameters
TOLERANCE = 0.6
IMAGE_RESIZE_WIDTH = 800

# Ensure folders exist
for folder in [KNOWN_FACES_DIR, UNKNOWN_FACES_DIR, OUTPUT_DIR]:
    os.makedirs(folder, exist_ok=True)

# Load known faces
def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_path = os.path.join(KNOWN_FACES_DIR, person_name)
        if os.path.isdir(person_path):
            for filename in os.listdir(person_path):
                file_path = os.path.join(person_path, filename)
                try:
                    image = face_recognition.load_image_file(file_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        known_face_encodings.append(encodings[0])
                        known_face_names.append(person_name)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    return known_face_encodings, known_face_names

# Process and segregate unknown faces
def segregate_faces(known_face_encodings, known_face_names):
    face_encodings_list = []
    file_paths_list = []

    for filename in os.listdir(UNKNOWN_FACES_DIR):
        file_path = os.path.join(UNKNOWN_FACES_DIR, filename)
        if not filename.lower().endswith(("jpg", "jpeg", "png")):
            continue

        try:
            image = face_recognition.load_image_file(file_path)
            if image.shape[1] > IMAGE_RESIZE_WIDTH:
                image = cv2.resize(image, (IMAGE_RESIZE_WIDTH, int(image.shape[0] * (IMAGE_RESIZE_WIDTH / image.shape[1]))))

            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            for face_encoding in face_encodings:
                distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(distances) if len(distances) > 0 else None

                if best_match_index is not None and distances[best_match_index] <= TOLERANCE:
                    name = known_face_names[best_match_index]
                else:
                    name = None

                if name:
                    person_folder = os.path.join(OUTPUT_DIR, name)
                    os.makedirs(person_folder, exist_ok=True)
                    shutil.copy(file_path, os.path.join(person_folder, filename))
                else:
                    face_encodings_list.append(face_encoding)
                    file_paths_list.append(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if face_encodings_list:
        clustering = DBSCAN(metric="euclidean", eps=0.6, min_samples=1).fit(face_encodings_list)
        cluster_labels = clustering.labels_
        unrecognized_dir = os.path.join(OUTPUT_DIR, "Unrecognized")
        for label, file_path in zip(cluster_labels, file_paths_list):
            cluster_folder = os.path.join(unrecognized_dir, f"Cluster_{label}")
            os.makedirs(cluster_folder, exist_ok=True)
            shutil.copy(file_path, os.path.join(cluster_folder, os.path.basename(file_path)))

# Streamlit UI
st.title("Face Segregation Tool")

# Tab 1: Add Known Faces
st.header("Step 1: Add Known Faces")
with st.expander("Upload a photo"):
    name = st.text_input("Enter your name")
    uploaded_files = st.file_uploader("Upload your photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if st.button("Save to Known Faces"):
        if name and uploaded_files:
            person_folder = os.path.join(KNOWN_FACES_DIR, name)
            os.makedirs(person_folder, exist_ok=True)
            for file in uploaded_files:
                file_path = os.path.join(person_folder, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.read())
            st.success(f"Saved {len(uploaded_files)} photos for {name}.")

with st.expander("Capture photo from camera"):
    name_camera = st.text_input("Enter your name for camera capture")
    if st.button("Capture"):
        cap = cv2.VideoCapture(0)
        st.text("Press 's' to save or 'q' to quit without saving.")
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image.")
                break
            cv2.imshow("Capture Photo", frame)
            key = cv2.waitKey(1)
            if key == ord("s"):  # Save photo
                person_folder = os.path.join(KNOWN_FACES_DIR, name_camera)
                os.makedirs(person_folder, exist_ok=True)
                file_path = os.path.join(person_folder, f"{name_camera}.jpg")
                cv2.imwrite(file_path, frame)
                st.success(f"Photo saved for {name_camera}.")
                break
            elif key == ord("q"):  # Quit without saving
                break
        cap.release()
        cv2.destroyAllWindows()

# Tab 2: Upload Unknown Faces
st.header("Step 2: Upload Unknown Faces")
unknown_files = st.file_uploader("Upload unknown photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
if st.button("Save Unknown Faces"):
    for file in unknown_files:
        file_path = os.path.join(UNKNOWN_FACES_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
    st.success(f"Saved {len(unknown_files)} unknown photos.")

# Tab 3: Run Segregation
st.header("Step 3: Run Face Segregation")
if st.button("Start Segregation"):
    known_face_encodings, known_face_names = load_known_faces()
    if not known_face_encodings:
        st.error("No known faces available. Add known faces first.")
    else:
        segregate_faces(known_face_encodings, known_face_names)
        st.success("Face segregation complete. Check the organized_faces folder.")
