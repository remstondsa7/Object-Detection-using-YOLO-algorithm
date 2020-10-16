import cv2
import argparse
import numpy as np
import time


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


    
cap= cv2.VideoCapture("video.mp4")
font = cv2.FONT_HERSHEY_PLAIN
starting_time=time.time()
frame_id=0
scale = 0.00392
classes = None

with open("yolov3.txt" , 'r') as f:
    classes = [line.strip() for line in f.readlines()]
while True:
    _,frame=cap.read()
    Height,Width,channels=frame.shape
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    blob = cv2.dnn.blobFromImage(frame, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in range(len(boxes)):
        if i in indices:
            x,y,w,h=boxes[i]
            label = str(classes[class_ids[i]])
            color = COLORS[class_ids[i]]
            confidence=confidences[i]
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, label+" "+str(round(confidence,2)),(x,y+30),font, 1,(255,255,255), 2)
    elapsed_time=time.time()-starting_time
    fps=frame_id/elapsed_time
    cv2.putText(frame,"FPS:"+str(round(fps,2)),(10,50),font,2,(0,0,0),1)

    cv2.imshow("VIDEO",frame)
    key=cv2.waitKey(1)
    if key==27:
        break;
    
cap.release()
cv2.destroyAllWindows()
