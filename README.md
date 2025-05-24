# Vision Guide - Navigation Assistance for the Visually Impaired

## About  
A deep learning model that helps visually impaired individuals navigate their surroundings effortlessly.

## Features  
- **Real-time object detection** – Uses trained AI models to provide real-time feedback.
- **Face Recognition** – Detects and identifies faces for enhanced interaction.
- **Navigation Assistance** – Helps users safely move through environments.

## 🛠️ Setup  
### Clone the Repository  
```bash
git clone https://github.com/saidinesh49/Vision-Guide.git
cd Vision-Guide
```
```python
python temp-mini-project.py
```
## User-Guide
- **Adding a New Face to the Model** – After running the script, a popup window will appear where you can add a face for recognition. Use a webcam or other camera utilities to capture your face. Before doing so, you must provide details so the model can associate your face with a name—this data is stored in a SQLite server for future recognition.
- **Environment Detection** – Recognizes surrounding objects while also running facial recognition mode. If trained faces exist in the model, it will detect and identify them.
- **Navigation Assistance** – Utilizes object detection to maintain a clear pathway and uses facial recognition to enhance guidance.
