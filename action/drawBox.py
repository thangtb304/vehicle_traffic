import cv2
import random
import os
import sys
from deep_sort_realtime.deepsort_tracker import DeepSort
from predict import *
from ultralytics import YOLO

tracker = DeepSort()

current_path=os.path.dirname(os.path.abspath(__file__))
model_char_name='model_char.pt'     
model_char_path=os.path.normpath(os.path.join(current_path, '..','model', model_char_name))
model_char = YOLO(model_char_path)

def get_label_traffic(label):
    if str(label) == '0':
        return 'green'
    if str(label) == '1' or str(label) == '2':
        return 'red'
    
def draw_box_traffic(image, boxs):
    reLabel = 1
    for i, (left, top, right, bottom, score, label) in enumerate(boxs):
        label = 1 if label == 2 else label
        reLabel = label
        c1, c2 = (left, top), (right, bottom)
        
        color = (0, 0, 255) if label in [1, 2] else (0, 255, 0)  # Red for label 1 or 2, green otherwise
        cv2.rectangle(image, c1, c2, color=color, thickness=2)
        cv2.putText(image, "{} {:.2f}".format(get_label_traffic(label), score), 
                    (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
    return image, reLabel


def draw_box_plate(image, boxs, labelTraffic, positionLine, plate_content_dict, path_save, list_images, relative_path):
    os.makedirs(os.path.join(path_save, 'images'), exist_ok=True)

    tracks = tracker.update_tracks(boxs, frame=image)
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        bbox = track.to_ltrb()

        if track_id not in plate_content_dict:
            image_plate = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            plate_content_dict[track_id] = predict_character(model_char, image_plate)

        if int(bbox[1]) <= positionLine:
            color = (0, 0, 255) if labelTraffic == 1 else (0, 255, 0)
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.putText(image, "#" + str(plate_content_dict[track_id]), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
         
            if labelTraffic == 1:
                image_filename = os.path.join(path_save, 'images', f"{plate_content_dict[track_id]}.jpg")
                if not os.path.isfile(image_filename) and plate_content_dict[track_id]:
                    captured_image = image[:int(positionLine) + 20, :]
                    cv2.imwrite(image_filename, captured_image)
                    list_images.append([os.path.join(relative_path, 'images', f"{plate_content_dict[track_id]}.jpg"), plate_content_dict[track_id]])
    return image, plate_content_dict, list_images

