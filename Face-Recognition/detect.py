import cv2
import numpy as np
import os
import sqlite3


import pyttsx3


facedetect=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam=cv2.VideoCapture(0)

engine=pyttsx3.init()

recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer/trainingdata.yml")

def getprofile(id):
    conn=sqlite3.connect("sqlite.db")
    cursor=conn.execute("SELECT * FROM STUDENTS WHERE id=?",(id,))
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile

while(True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        id,conf=recognizer.predict(gray[y:y+h,x:x+w])
        profile=getprofile(id)
        print(profile)
        if(profile!=None):
            speech=f"{profile[1]}"
            engine.say(speech)
            engine.runAndWait()
            cv2.putText(img,"Name:"+str(profile[1]),(x,y+h+20),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,127),2)

    cv2.imshow("FACE",img)
    if(cv2.waitKey(1)==ord('q')):
        break

cam.release()
engine.stop()
cv2.destroyAllWindows()

