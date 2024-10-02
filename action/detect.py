import cv2
import os
import sys
from ultralytics import YOLO
import moviepy.editor as moviepy
from predict import *
from drawBox import *

def non_max_suppression(boxes, threshold):
    order = boxes.copy()
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        for j in order:
            # Calculate the IoU between the two boxes
            box_i = i.copy()
            box_j = j.copy()
            intersection = max(0, min(box_i[2], box_j[2]) - max(box_i[0], box_j[0])) * \
                           max(0, min(box_i[3], box_j[3]) - max(box_i[1], box_j[1]))
            union = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1]) + \
                    (box_j[2] - box_j[0]) * (box_j[3] - box_j[1]) - intersection
            iou = intersection / union

            # Remove boxes with IoU greater than the threshold
            if iou > threshold:
                order.remove(j)
    return keep

def detect(pathVideo):

    #path_save=os.path.normpath(os.path.join(current_path, '..','static', 'save'))
    relative_path = os.path.normpath(os.path.join(current_path, '..','static', 'save'))

    os.makedirs(os.path.join(path_save, 'video'), exist_ok=True)

    current_path=os.path.dirname(os.path.abspath(__file__))
    #link model traffic
    model_traffic_name='yolov8l.pt'     
    ################model_traffic_name='model_traffic.pt'  
    model_traffic_path=os.path.normpath(os.path.join(current_path, '..','model', model_traffic_name))
    # link model plate
    model_plate_name='yolov8l.pt'    
    ##############model_plate_name='model_plate.pt'   
    model_plate_path=os.path.normpath(os.path.join(current_path, '..','model', model_plate_name))

    # get model
    model = YOLO(model_traffic_path)
    model1 = YOLO(model_plate_path)
   

    name_video = pathVideo.split('/')[-1]
    

    cap = cv2.VideoCapture(pathVideo)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video = cv2.VideoWriter(os.path.join(path_save, 'video', name_video.split('.')[0]+'.avi'), 
                        cv2.VideoWriter_fourcc(*'XVID'), 10, (width, height))
    
    plate_content_dict = {}
    list_images = []

    count = 0
    t = 0
    while cap.isOpened():
        count += 1
        if count%15 != 0:
            continue

        ret, frame = cap.read()
        if not ret: 
            break

        results = predict_traffic(model, frame)
        results1 = predict_plate(model1, frame)

        if results is not None and results1 is not None :
            results = non_max_suppression(results, 0.45)

            if t == 0:
                lenTraffic = (results[0][3] - results[0][1])
                if lenTraffic <= height*0.15:
                    t = results[0][3] + lenTraffic*2
                else:
                    t = results[0][3] + lenTraffic*1.25
            
            # cv2.line(frame, (0,int(t)), (10000, int(t)), (255, 0, 0), thickness=1, lineType=8, shift=0)

            frame, labelTraffic = draw_box_traffic(frame, results)
            frame, plate_content_dict, list_images = draw_box_plate(frame, results1, labelTraffic, t, plate_content_dict,\
                                                        path_save, list_images, relative_path)

        video.write(frame)
        # cv2.imshow('video', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
        
    cap.release()
    video.release()

    os.remove(pathVideo)
    # print('Done, save video: {}'.format(os.path.join(path_save, source, name_video.split('.')[0]+'.avi')))
    path_video = '{}'.format(os.path.join(path_save, 'video', name_video.split('.')[0]+'.avi'))
    clip = moviepy.VideoFileClip(path_video)
    clip.write_videofile(pathVideo)
    os.remove(path_video)
    relative_path += "/video/" + name_video
    return list_images, relative_path

if __name__ == '__main__' :
    current_path=os.path.dirname(os.path.abspath(__file__))
    name_video='video1.mp4'     
    pathVideo=os.path.normpath(os.path.join(current_path, '..','video_source', name_video))
    #pathVideo = "D:/TraficRedLight/ClientServer/static/videos/abcd.mp4"
    #pathVideo=os.path.normpath(os.path.join(current_path, '..','static', 'videos'))
    path_save=os.path.normpath(os.path.join(current_path, '..','static', 'save'))
    detect(pathVideo, path_save)

