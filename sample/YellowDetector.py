import cv2
import numpy as np
from m_rcnn import *

url = 'rtsp://192.168.254.106:554/user=admin_password=Vokk1212*_channel=0_stream=0.sdp?real_stream'

model = "C:\\Users\\jemar\\Downloads\\EnokiSet\\enoki_mask_rcnn_object_0005.h5"
config = CustomConfig()

net = cv2.dnn.readNet(model, config)

cap = cv2.VideoCapture(url)
width, height = 1280, 720


while True:

    ret, frame = cap.read()
    
    frame = cv2.resize(frame, (width, height))
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])


    mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        color = (0, 255, 0) 
        label = "Eraser"
        cv2.putText(frame, label, (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


    cv2.imshow('Yellow Image with Bounding Box', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
