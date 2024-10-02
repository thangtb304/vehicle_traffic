import os
current_path=os.path.dirname(os.path.abspath(__file__))
model_traffic_name='model_traffic.pt'     
model_traffic_path=os.path.normpath(os.path.join(current_path, '..','model', model_traffic_name))
# link model plate
model_plate_name='model_plate.pt'     
model_plate_path=os.path.normpath(os.path.join(current_path, '..','model', model_plate_name))



print(model_plate_path)