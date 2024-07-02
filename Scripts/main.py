import cv2
from ultralytics import YOLO
import cvzone
import math
from sort import *
import numpy as np


cap = cv2.VideoCapture("../Videos/waveclip_street_footage.mp4")

model = YOLO("../Model-Weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("../Videos/mask_resized2.png")

tracker = Sort(max_age=20, min_hits=3, iou_threshold=.3)

limits = [19, 280, 567, 277]
total_count = []

while True:
    success, img = cap.read()

    imgRegion = cv2.bitwise_and(img, mask)

    results = model(imgRegion, stream=True, device='cuda')

    detections = np.empty((0, 5))

    cv2.imshow("Original Video", img)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x, y, w, h = int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1)

            conf = math.ceil(box.conf[0] * 100)/100
            current_class = classNames[int(box.cls[0])]

            if current_class in ["car", "motorbike", "bus", "truck"] and conf > .3:
                cvzone.cornerRect(img, bbox=(x, y, w, h), l=7, rt=5)
                cvzone.putTextRect(img=img,
                                   text=f"{current_class} {conf}",
                                   pos=(max(0, x), max(35, y)),
                                   scale=0.5,
                                   thickness=1,
                                   offset=5)
                current_array = np.array([
                    int(x1), int(y1),
                    int(x2), int(y2)
                    , conf])

                detections = np.vstack((detections, current_array))

    resultsTracker = tracker.update(detections)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, object_id = result
        print(result)
        x, y, w, h = int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1)
        # cvzone.cornerRect(img, bbox=(x, y, w, h), l=7, rt=3, colorR=(255, 0, 0))
        # cvzone.putTextRect(img=img,
        #                    text=f"{int(object_id)}",
        #                    pos=(max(0, x), max(35, y)),
        #                    scale=0.5,
        #                    thickness=1,
        #                    offset=5)
        cx, cy = int(x1) + w//2, int(y1) + h//2
        # cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 50 < cy < limits[3] + 50:
            if total_count.count(object_id) == 0:
                total_count.append(object_id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    cvzone.putTextRect(img, f"Count: {len(total_count)}", (60, 60))

    cv2.imshow("Video Region", imgRegion)
    cv2.imshow("Running Yolo", img)
    cv2.waitKey(1)
