import cv2
from ultralytics import YOLO
import os

current_path=os.path.dirname(os.path.abspath(__file__))
video_name='video2.mp4'
#video_path=os.path.join(current_path,'video',video_name)
video_path=os.path.normpath(os.path.join(current_path, '..','video', video_name))
# Tải mô hình YOLOv8l đã huấn luyện
model_name='traffic-lights.pt'    #'yolov8l.pt' 
model_path=os.path.normpath(os.path.join(current_path, '..','model', model_name))
model = YOLO(model_path)

print(model_path)
##Đọc video
cap = cv2.VideoCapture(video_path)

# Giả sử trạng thái đèn đỏ và vị trí vạch dừng
red_light_active = True  # Đèn đỏ đang hoạt động
stop_line_y = 400  # Vị trí y của vạch dừng

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Nhận diện đối tượng trong frame với YOLOv8
    results = model(frame)

    # Lặp qua các đối tượng được nhận diện
    for result in results:
        boxes = result.boxes  # bounding boxes
        for box in boxes:
            # Lấy tọa độ bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            label = result.names[box.cls[0].item()]  # Lấy nhãn đối tượng
            confidence = box.conf[0].item()  # Độ tin cậy của dự đoán
            
            # Nếu phương tiện vi phạm (vượt qua vạch dừng khi đèn đỏ)
            if red_light_active and y2 > stop_line_y and label == "car":
                # Vẽ bounding box màu đỏ cho phương tiện vi phạm
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Màu đỏ
                cv2.putText(frame, f'{label} {confidence:.2f} VI PHẠM', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                # Vẽ bounding box màu xanh cho phương tiện không vi phạm
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Màu xanh
                cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Hiển thị frame đã xử lý
    cv2.imshow('Red Light Violation Detection', frame)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
