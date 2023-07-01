import numpy as np
import datetime
import cv2
import math
from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.tools import generate_detections as gdet
from typing import Any

from helper import create_video_writer, cornerRect

class VehicleCounter:
    def __init__(self, video_path: str, res_path, yolo_weights_path: str,
                model_path: str, classes_path: str):
        
        self.video_path = video_path
        self.res_path = res_path
        self.yolo_weights_path = yolo_weights_path
        self.model_path = model_path
        self.classes_path = classes_path
        
        self.conf_threshold = 0.5
        self.max_cosine_distance = 0.4
        self.nn_budget = None
        self.tracker = None
        
        self.class_names = []
        self.desired_classes = ["car", "truck", "bus", "motorbike"]
        
        self.limits = [241, 639, 912, 641]
        self.limits2 = [1056, 601, 1675, 589]
        
        self.entering_list = []
        self.leaving_list = []
        self.entering_cnt = [0, 0, 0, 0]
        self.leaving_cnt = [0, 0, 0, 0]
        
        self.cap = None
    
    def initialize(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.writer = create_video_writer(self.cap, self.res_path)
        self.encoder = gdet.create_box_encoder(self.model_path, batch_size=1)
        self.model_path = YOLO(self.yolo_weights_path)
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(metric, max_age=60, n_init=4)
        
        with open(self.classes_path, "r") as f:
            self.class_names = f.read().strip().split("\n")
        
    def process_video(self):
        while True:
            start = datetime.datetime.now()
            ret, frame = self.cap.read()
            frame = cv2.resize(frame, (1920, 1080))
            if not ret:
                print("End of the video file...")
                break
            
            results = self.model_path(frame)
            for result in results:
                bboxes = []
                confidences = []
                class_ids = []
                # loop over the detections
                for data in result.boxes.data.tolist():
                    x1, y1, x2, y2, confidence, class_id = data
                    x = int(x1)
                    y = int(y1)
                    w = int(x2) - int(x1)
                    h = int(y2) - int(y1)
                    class_id = int(class_id)
                    confidence = math.ceil((confidence * 100)) / 100
                    
                    if confidence > self.conf_threshold:
                        bboxes.append([x, y, w, h])
                        confidences.append(confidence)
                        class_ids.append(class_id)
            #prepare to update             
            names = [self.class_names[class_id] for class_id in class_ids]           
            features = self.encoder(frame, bboxes)
            detections = []
            
            for bbox, conf, class_name, feature in zip(bboxes, confidences, names, features):
                detections.append(Detection(bbox, conf, class_name, feature))
            
            self.tracker.predict()
            tracks = self.tracker.update(detections)
            cv2.line(frame, (self.limits[0], self.limits[1]), (self.limits[2], self.limits[3]), (0, 0, 255), 5)
            cv2.line(frame, (self.limits2[0], self.limits2[1]), (self.limits2[2], self.limits2[3]), (0, 0, 255), 5)

            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                track_id = track.track_id
                class_name = track.get_class()
                if class_name in self.desired_classes:
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    w, h = x2 - x1, y2 - y1
                    cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(0, 128, 255), colorC=(0, 0, 255))
                    class_id = self.class_names.index(class_name)

                    #create rectangle around the object based on bbox just updated
                    pos = ((x1, y1), (x2, y2))

                    cv2.rectangle(frame, pos[0], pos[1], (0, 128, 255), 1)
                    text = f'{class_name} {conf}'
                    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
                    pt1 = (pos[0][0], pos[0][1] - text_size[1] - 10) 
                    pt2 = (pos[0][0] + text_size[0] + 10, pos[0][1]) 
                    cv2.rectangle(frame, pt1, pt2, (0, 128, 255), -1)
                    cv2.putText(frame, text, (pos[0][0] + 5, pos[0][1] - 5),
                                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
                        
                    #create the center of the rectangle surrounding the object
                    cx, cy = x1 + w // 2, y1 + h // 2
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
                        
                    if self.limits[0] < cx < self.limits[2] and self.limits[1] - 5 < cy < self.limits[1] + 5:
                        if self.entering_list.count(track_id) == 0:
                            if class_name == "car":
                                self.entering_cnt[0] += 1
                                self.entering_list.append(track_id)
                            elif class_name == "truck":
                                self.entering_cnt[1] += 1
                                self.entering_list.append(track_id)
                            elif class_name == "bus":
                                self.entering_cnt[2] += 1
                                self.entering_list.append(track_id)
                            elif class_name == "motorbike":
                                self.entering_cnt[3] += 1
                                self.entering_list.append(track_id)
                            cv2.line(frame, (self.limits[0], self.limits[1]), (self.limits[2], self.limits[3]), (0, 255, 0), 5)

                    elif self.limits2[0] < cx < self.limits2[2] and self.limits2[1] - 5 < cy < self.limits2[1] + 5:
                        if self.leaving_list.count(track_id) == 0:
                            if class_name == "car":
                                self.leaving_cnt[0] += 1
                                self.leaving_list.append(track_id)
                            if class_name == "truck":
                                self.leaving_cnt[1] += 1
                                self.leaving_list.append(track_id)
                            if class_name == "bus":
                                self.leaving_cnt[2] += 1
                                self.leaving_list.append(track_id)
                            if class_name == "motorbike":
                                self.leaving_cnt[3] += 1
                                self.leaving_list.append(track_id)
                            cv2.line(frame, (self.limits2[0], self.limits2[1]), (self.limits2[2], self.limits2[3]), (0, 255, 0), 5)

            cv2.putText(frame, f'Entering: {len(self.entering_list)}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f'Leaving: {len(self.leaving_list)}', (1700, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            
            cv2.putText(frame, f'Car: {self.entering_cnt[0]}', (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f'Truck: {self.entering_cnt[1]}', (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f'Bus: {self.entering_cnt[2]}', (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f'Motorbike: {self.entering_cnt[3]}', (50, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            
            cv2.putText(frame, f'Car: {self.leaving_cnt[0]}', (1700, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f'Truck: {self.leaving_cnt[1]}', (1700, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f'Bus: {self.leaving_cnt[2]}', (1700, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f'Motorbike: {self.leaving_cnt[3]}', (1700, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)    
            

            end = datetime.datetime.now()
            fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
            cv2.putText(frame, fps, (800, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
            cv2.imshow("Vehicles Counting", frame)
            self.writer.write(frame)
            if cv2.waitKey(1) == ord("q"):
                break

        self.cap.release()
        self.writer.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    
    VIDEO_PATH = "vehicle-counting.mp4"
    RES_PATH = "output.mp4"
    YOLO_WEIGHT = "yolov8l.pt"
    MODEL_PATH = "config/mars-small128.pb"
    CLASSES_PATH = "config/coco.names"
    car_counter = VehicleCounter(VIDEO_PATH, RES_PATH, YOLO_WEIGHT, MODEL_PATH, CLASSES_PATH)
    car_counter.initialize() 
    car_counter.process_video()
