import matplotlib.pyplot as plt
import ultralytics
import cv2
from ultralytics import YOLO
import os
import numpy as np

model_path= 'Notbook/best.pt'
model = YOLO(model_path)
# results=model.predict(source=0)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    
    ret, frame = cap.read()

    results = model.predict(frame,conf=0.7)
    
    annotated_frame = results[0].plot()
    cv2.imshow('YOLO', annotated_frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()   
cv2.destroyAllWindows()


