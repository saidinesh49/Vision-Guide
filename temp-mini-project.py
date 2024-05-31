import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import pyttsx3
import sqlite3
import os #Working Original ( User options + detection features ) V2

# Initialize YOLO for object detection
model = YOLO("yolov8l.pt")

# Initialize face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Face-Recognition/recognizer/trainingdata.yml")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('Face-Recognition/haarcascade_frontalface_default.xml')

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to get profile from database
def get_profile(id):
    conn = sqlite3.connect("Face-Recognition/sqlite.db")
    cursor = conn.execute("SELECT * FROM STUDENTS WHERE id=?", (id,))
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile

def draw_button(frame, text, position, size=(200, 50)):
    (x, y) = position
    (w, h) = size
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), -1)
    cv2.putText(frame, text, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def check_button_click(position, size, mouse_pos):
    (x, y) = position
    (w, h) = size
    (mx, my) = mouse_pos
    return x <= mx <= x + w and y <= my <= y + h

def environment_read():
    cam = cv2.VideoCapture(0)
    global button_clicked
    button_clicked = False
    
    def on_mouse(event, x, y, flags, param):
        global button_clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            if check_button_click((10, 10), (100, 40), (x, y)):
                button_clicked = True

    cv2.namedWindow("Environment Read")
    cv2.setMouseCallback("Environment Read", on_mouse)

    while True:
        ret, frame = cam.read()
        for result in model.track(source=frame, stream=True):
            frame = result.orig_img
            detections = sv.Detections.from_yolov8(result)

            for _, confidence, class_id, _ in detections:
                class_name = model.model.names[class_id]

                if class_name == 'person':
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                        profile = get_profile(id)

                        if profile is not None:
                            speech = f"Hey {profile[1]}"
                            engine.say(speech)
                            engine.runAndWait()
                            cv2.putText(frame, "Name:" + str(profile[1]), (x, y + h + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
                        else:
                            print("Detected object:", class_name, "with confidence:", confidence)
                            speech_text = f"Detected object: {class_name}"
                            engine.say(speech_text)
                            engine.runAndWait()

                else:
                    print("Detected object:", class_name, "with confidence:", confidence)
                    speech_text = f"Detected object: {class_name}"
                    engine.say(speech_text)
                    engine.runAndWait()

            draw_button(frame, "Back", (10, 10), (100, 40))
            cv2.imshow("Environment Read", frame)

            if cv2.waitKey(1) & 0xFF == 27 or button_clicked:  # Press 'ESC' to quit or 'Back' button clicked
                break

    cam.release()
    cv2.destroyAllWindows()

def add_person():
    face_detect = cv2.CascadeClassifier('Face-Recognition/haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0)

    def insert_or_update(id, name, age):
        conn = sqlite3.connect("Face-Recognition/sqlite.db")
        cmd = "SELECT * FROM STUDENTS WHERE ID=" + str(id)
        cursor = conn.execute(cmd)
        is_record_exist = 0
        for row in cursor:
            is_record_exist = 1
        if is_record_exist == 1:
            conn.execute("UPDATE STUDENTS SET NAME=?, AGE=? WHERE ID=?", (name, age, id))
        else:
            conn.execute("INSERT INTO STUDENTS (ID, NAME, AGE) values (?, ?, ?)", (id, name, age))
            speech=f"{name} was added successfully"
            engine.say(speech)
            engine.runAndWait()
        conn.commit()
        conn.close()

    id = input("Enter User Id: ")
    name = input("Enter User Name: ")
    age = input("Enter User Age: ")

    insert_or_update(id, name, age)

    sample_num = 0
    global button_clicked
    button_clicked = False
    
    def on_mouse(event, x, y, flags, param):
        global button_clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            if check_button_click((10, 10), (100, 40), (x, y)):
                button_clicked = True

    cv2.namedWindow("Add Person")
    cv2.setMouseCallback("Add Person", on_mouse)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            sample_num += 1
            cv2.imwrite(f"Face-Recognition/dataset/user.{id}.{sample_num}.jpg", gray[y:y + h, x:x + w])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.waitKey(100)
        draw_button(img, "Back", (10, 10), (100, 40))
        cv2.imshow("Add Person", img)
        if cv2.waitKey(1) & 0xFF == 27 or button_clicked:  # Press 'ESC' or 'Back' button to quit
            break
        if sample_num > 20:
            break

    cam.release()
    cv2.destroyAllWindows()

    # Train the model with new data
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    path = "Face-Recognition/dataset"

    def get_images_with_id(path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        faces = []
        ids = []
        for image_path in image_paths:
            face_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            id = int(os.path.split(image_path)[-1].split(".")[1])
            faces.append(face_img)
            ids.append(id)
            cv2.imshow("Training", face_img)
            cv2.waitKey(10)
        return np.array(ids), faces

    ids, faces = get_images_with_id(path)
    recognizer.train(faces, ids)
    recognizer.save("Face-Recognition/recognizer/trainingdata.yml")
    cv2.destroyAllWindows()

def remove_face():
    conn = sqlite3.connect("Face-Recognition/sqlite.db")
    cursor = conn.cursor()
    name = input("Enter the name of the person to remove: ")
    cursor.execute("SELECT ID FROM STUDENTS WHERE NAME=?", (name,))
    rows = cursor.fetchall()
    if rows:
        for row in rows:
            id = row[0]
            cursor.execute("DELETE FROM STUDENTS WHERE ID=?", (id,))
            # Remove images from dataset
            dataset_path = "Face-Recognition/dataset"
            images = [f for f in os.listdir(dataset_path) if f.startswith(f"user.{id}.")]
            for image in images:
                os.remove(os.path.join(dataset_path, image))
        conn.commit()
        speech=f"{name} removed successfully"
        engine.say(speech)
        engine.runAndWait()
        print(f"Removed all data for {name}.")
    else:
        print(f"No record found for {name}.")
    conn.close()

def main():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Options")

    def on_mouse(event, x, y, flags, param):
        global button_clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            if check_button_click((50, 50), (200, 50), (x, y)):
                button_clicked = "environment_read"
            elif check_button_click((50, 150), (200, 50), (x, y)):
                button_clicked = "add_person"
            elif check_button_click((50, 250), (200, 50), (x, y)):
                button_clicked = "navigate_me"
            elif check_button_click((50, 350), (200, 50), (x, y)):
                button_clicked = "remove_face"

    cv2.setMouseCallback("Options", on_mouse)
    global button_clicked
    button_clicked = None

    while True:
        ret, frame = cam.read()
        # Remove the black background, use the camera feed as the background
        # frame[:] = (0, 0, 0)
        draw_button(frame, "Environment Read", (50, 50))
        draw_button(frame, "Add Person", (50, 150))
        draw_button(frame, "Navigate Me", (50, 250))
        draw_button(frame, "Remove Face", (50, 350))

        if button_clicked == "environment_read":
            environment_read()
            button_clicked = None
        elif button_clicked == "add_person":
            add_person()
            button_clicked = None
        elif button_clicked == "navigate_me":
            print("Navigate Me functionality not yet implemented.")
            button_clicked = None
        elif button_clicked == "remove_face":
            remove_face()
            button_clicked = None

        cv2.imshow("Options", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to quit
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()