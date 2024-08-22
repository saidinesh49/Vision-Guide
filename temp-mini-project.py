import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import sqlite3
import os

model = YOLO("yolov8l.pt")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Face-Recognition/recognizer/trainingdata.yml")

face_cascade = cv2.CascadeClassifier('Face-Recognition/haarcascade_frontalface_default.xml')

engine = pyttsx3.init()

def get_profile(id):
    conn = sqlite3.connect("Face-Recognition/sqlite.db")
    cursor = conn.execute("SELECT * FROM STUDENTS WHERE id=?", (id,))
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile

def person_exists(name):
    conn = sqlite3.connect("Face-Recognition/sqlite.db")
    cursor = conn.execute("SELECT * FROM STUDENTS WHERE NAME=?", (name.upper(),))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

def draw_button(frame, text, position, size=(200, 50), padding=10, color="#C509EB"):
    (x, y) = position
    (w, h) = size
    
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    
    button_width = max(text_size[0] + 2 * padding, w)
    button_height = max(text_size[1] + 2 * padding, h)
    
    text_x = x + (button_width - text_size[0]) // 2
    text_y = y + (button_height + text_size[1]) // 2
    
    cv2.rectangle(frame, (x, y), (x + button_width, y + button_height), (175, 9, 235), -1)
    cv2.rectangle(frame, (x, y), (x + button_width, y + button_height), (0, 0, 0), 2)  # Black border
    
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def check_button_click(position, size, mouse_pos):
    (x, y) = position
    (w, h) = size
    (mx, my) = mouse_pos
    return x <= mx <= x + w and y <= my <= y + h

def estimate_distance(box, real_height=1.7, focal_length=700):
    x1, y1, x2, y2 = box
    pixel_height = y2 - y1
    distance = (real_height * focal_length) / pixel_height
    return distance

def environment_read():
    cam = cv2.VideoCapture(0)
    button_clicked = False
    
    def on_mouse(event, x, y, flags, param):
        nonlocal button_clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            if check_button_click((10, 10), (200, 50), (x, y)):
                button_clicked = True
    
    cv2.namedWindow("Environment Read", cv2.WINDOW_NORMAL) 
    cv2.setMouseCallback("Environment Read", on_mouse)
    
    while True:
        ret, frame = cam.read()
        if not ret:
            continue
        
        results = model(frame)
        detections = results[0].boxes

        for box in detections:
            conf = box.conf[0]
            cls = int(box.cls[0])
            xyxy = box.xyxy[0]

            if conf > 0.5:
                label = f'{model.names[cls]} {conf:.2f}'
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if model.names[cls] == 'person':
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    pro,face,ex,ey,ew,eh=False,[],0,0,0,0
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                        profile = get_profile(id)
                        if profile is not None:
                            pro,face=True,profile
                            ex,ey,ew,eh=x,y,w,h
                            break

                    x,y,w,h=ex,ey,ew,eh
                    if pro:
                        profile=face
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.putText(frame, str(profile[1]), (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        engine.say(str(profile[1]))
                        engine.runAndWait()
                    else:
                        cv2.putText(frame, "Unknown", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        engine.say(f"Person")
                        engine.runAndWait()
                else:
                    engine.say(model.names[cls])
    
        engine.runAndWait()
        draw_button(frame, "Back", (10, 10), (200, 50))
        cv2.imshow("Environment Read", frame)
    
        if cv2.waitKey(1) & 0xFF == 27 or button_clicked:  
            revokeMainfun()
            break


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
            conn.execute("UPDATE STUDENTS SET NAME=?, AGE=? WHERE ID=?", (name.upper(), age, id))
        else:
            conn.execute("INSERT INTO STUDENTS (ID, NAME, AGE) values (?, ?, ?)", (id, name.upper(), age))
            speech = f"{name} was added successfully"
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
            if check_button_click((10, 10), (200, 50), (x, y)):
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
        draw_button(img, "Back", (10, 10), (200, 50))
        cv2.imshow("Add Person", img)
        if cv2.waitKey(1) & 0xFF == 27 or button_clicked: 
            break
        if sample_num > 20:
            break

    cam.release()
    cv2.destroyAllWindows()

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
    main()

def remove_face():
    conn = sqlite3.connect("Face-Recognition/sqlite.db")
    cursor = conn.cursor()
    name = input("Enter the name of the person to remove: ")
    cursor.execute("SELECT ID FROM STUDENTS WHERE NAME=?", (name.upper(),))
    rows = cursor.fetchall()
    if rows:
        for row in rows:
            id = row[0]
            cursor.execute("DELETE FROM STUDENTS WHERE ID=?", (id,))
            dataset_path = "Face-Recognition/dataset"
            for file in os.listdir(dataset_path):
                if file.startswith(f"user.{id}."):
                    os.remove(os.path.join(dataset_path, file))
            conn.commit()
            print(f"Removed {name} from the database.")
            speech = f"{name} was removed successfully"
            engine.say(speech)
            engine.runAndWait()
    else:
        print(f"No person named {name} found in the database.")
        speech = f"No person named {name} found in the database"
        engine.say(speech)
        engine.runAndWait()
    conn.close()
    cv2.destroyAllWindows()
    main()

def revokeMainfun():
    cv2.VideoCapture(0).release()
    cv2.destroyAllWindows()
    main()
    return 

def navigate_to_person(name):
    if not person_exists(name):
        print(f"No person found with the name {name}.")
        engine.say(f"No person found with the name {name}.")
        engine.runAndWait()
        return

    cam = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cam.read()
        if not ret:
            continue
        
        results = model(frame)
        detections = results[0].boxes

        target_coordinates = None
        for box in detections:
            conf = box.conf[0]
            cls = int(box.cls[0])
            xyxy = box.xyxy[0]

            if conf > 0.5 and model.names[cls] == 'person':
                x1, y1, x2, y2 = map(int, xyxy)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                    profile = get_profile(id)
                    if profile is not None and profile[1] == name:
                        target_coordinates = (x, y, w, h)
                        distance = estimate_distance((x, y, x + w, y + h))
                        cv2.putText(frame, f"Distance: {distance:.2f}m", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        break

        if target_coordinates:
            x, y, w, h = target_coordinates
            frame_center_x = frame.shape[1] // 2
            object_center_x = x + w // 2
            direction = ""

            if distance <= 11:  
                direction = f"You have succesfully reached {name}."
                engine.say(direction)
                engine.runAndWait()
                # cam.release()
                # cv2.destroyAllWindows()
                revokeMainfun()
                break
            else:
                if object_center_x < frame_center_x - 100:
                    direction = "left"
                elif object_center_x > frame_center_x + 100:
                    direction = "right"
                else:
                    direction = "forward"

                direction = f"Move {direction}. Distance to {name} is {distance:.2f} meters."

            engine.say(direction)
            engine.runAndWait()
        else:
            engine.say(f"Searching for {name}.")
            engine.runAndWait()

        cv2.imshow("Navigate to Person", frame)

        if cv2.waitKey(1) & 0xFF == 27:  
            break

    cam.release()
    cv2.destroyAllWindows()

def main():
    global button_clicked
    button_clicked = False
    
    def on_mouse(event, x, y, flags, param):
        global button_clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            if check_button_click((10, 10), (200, 50), (x, y)):
                button_clicked = "environment_read"
            elif check_button_click((10, 70), (200, 50), (x, y)):
                button_clicked = "add_person"
            elif check_button_click((10, 130), (200, 50), (x, y)):
                button_clicked = "remove_face"
            elif check_button_click((10, 190), (200, 50), (x, y)):
                button_clicked = "navigate_to_person"

    cv2.namedWindow("Main Menu")
    cv2.setMouseCallback("Main Menu", on_mouse)

    while True:
        frame = np.zeros((600, 800, 3), np.uint8)
        draw_button(frame, "Read Environment", (10, 10), (200, 50))
        draw_button(frame, "Add Person", (10, 70), (200, 50))
        draw_button(frame, "Remove Person", (10, 130), (200, 50))
        draw_button(frame, "Navigate to Person", (10, 190), (200, 50))
        cv2.imshow("Main Menu", frame)

        if cv2.waitKey(1) & 0xFF == 27 or button_clicked:
            break

    if button_clicked:
        cv2.destroyWindow("Main Menu")
    if button_clicked == "environment_read":
        environment_read()
    elif button_clicked == "add_person":
        add_person()
    elif button_clicked == "remove_face":
        remove_face()
    elif button_clicked == "navigate_to_person":
        target_name = input("Enter the name of the person to navigate to: ")
        target_name=target_name.upper()
        if person_exists(target_name):
            navigate_to_person(target_name)
        else:
            print(f"No person named {target_name} found in the database.")
            speech = f"No person named {target_name} found in the database"
            engine.say(speech)
            engine.runAndWait()
            main()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



# Version (V - 2.1)