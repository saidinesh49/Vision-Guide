import cv2
import numpy as np
import pyttsx3
from ultralytics import YOLO
from heapq import heappop, heappush
import math

# Initialize YOLO for object detection
model = YOLO("yolov8l.pt")

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Pathfinding with A* algorithm
def heuristic(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def a_star(array, start, goal):
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []

    heappush(oheap, (fscore[start], start))

    while oheap:
        current = heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data[::-1]

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    continue
            else:
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heappush(oheap, (fscore[neighbor], neighbor))

    return False

def mock_depth_data(shape):
    return np.random.randint(1, 255, size=shape).astype(np.uint8)

def navigate_to_target(target_name):
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Unable to access the camera.")
        return

    height, width = 480, 640  # Standard resolution for the camera
    depth_image = mock_depth_data((height, width))

    while True:
        ret, color_image = cam.read()
        if not ret:
            continue

        results = model(color_image)
        target_found = False
        target_center = None

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                if class_name == 'person':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    if conf > 0.5:
                        target_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        distance = depth_image[target_center[1], target_center[0]]
                        if distance > 0:
                            print(f"Found {target_name} at distance: {distance} (mock units)")
                            target_found = True
                            break

        if target_found and target_center:
            start = (height // 2, width // 2)
            goal = target_center
            path = a_star(depth_image, start, goal)

            if path:
                for point in path:
                    cv2.circle(color_image, (point[1], point[0]), 1, (0, 0, 255), -1)

                engine.say("Target found. Navigating...")
                engine.runAndWait()
            else:
                engine.say("Path not found. Please try again.")
                engine.runAndWait()
        else:
            engine.say("Target not found. Please try again.")
            engine.runAndWait()

        cv2.imshow('Navigation', color_image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
