from utils.bbox import draw_boxes,BoundBox
from keras.models import model_from_json
import cv2
import numpy as np

box = [BoundBox(582, 274, 700, 321,None,[.7])]

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
# np.load = lambda *a,**k: np.load(*a, allow_pickle=True, **k)
label_map = np.load('label_map.npy',allow_pickle=True).item()

im = cv2.imread("input_2.jpeg")
cv2.imshow("input",im)
labels=["number_plate"]
draw_boxes(im, box, loaded_model, label_map, labels, 0.5)

cv2.imshow("See here",im)
cv2.waitKey()