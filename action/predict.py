char_label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',\
          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',\
              'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def predict_traffic(model_traffic, image):
    res = []
    results = model_traffic(image)
    for box in results.xyxy[0]:
        left, top, right, bottom = map(int, box[:4])  # Chuyển đổi tọa độ về int
        score = float(box[4])  # Chuyển score sang float
        label = box[5]
        
        if score >= 0.3:
            label = 1 if label == 2 else label  # Thay đổi label nếu cần
            res.append([left, top, right, bottom, score, label])
    return res


def predict_plate(model_plate, image):
    res = []
    results = model_plate(image)
    
    if len(results.xyxy[0]) > 0:
        for box in results.xyxy[0]:
            left, top, right, bottom = map(int, box[:4])  # Chuyển tất cả giá trị về int trong một dòng
            score = float(box[4])
            label=box[5]
            if score >= 0.45:
                width = right - left
                height = bottom - top
                image_plate = image[top:bottom, left:right]  # Cắt ảnh biển số
                res.append(([left, top, width, height], score, 'plate', image_plate))  # Thêm ảnh biển số vào kết quả
    return res


def predict_character(model_char, image_plate):    
    res = []
    res_up=[]
    res_low=[]
    height, width = image_plate.shape[:2]
    
    def extract_results(results):
        return [
            (*map(int, box[:4]), float(box[4]), box[5])
            for box in results.xyxy[0]
        ] if len(results.xyxy[0]) > 0 else []

    if width > 2 * height:
        results = model_char(image_plate)
        res = extract_results(results)
    else:
        new_height = height // 2
        upper_half = image_plate[:new_height, :]
        lower_half = image_plate[new_height:, :]

        results_up = model_char(upper_half)
        results_low = model_char(lower_half)

        res_up += extract_results(results_up) 
        res_up.sort(key=lambda x: x[0])

        res_low += extract_results(results_low)
        res_low.sort(key=lambda x: x[0])

        res =res_up + res_low
    plate = "".join(char_label[i[5]] for i in res)
    return plate



