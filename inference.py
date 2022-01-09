import sys
import numpy as np
import cv2
import glob

list_argv = sys.argv[1:]

if len(list_argv) == 4:
  dict_argv = {list_argv[0]: list_argv[1], list_argv[2]: list_argv[3]}
  thresh = 0.8
else:
  dict_argv = {list_argv[0]: list_argv[1], list_argv[2]: list_argv[3], list_argv[4]: list_argv[5]}
  thresh = float(dict_argv['-thresh'])

labels_path = 'obj.names'
config_path = 'yolo-obj.cfg'
weights_path = 'yolo-obj_best.weights'

font_scale = 1
thickness = 1

labels = open(labels_path).read().strip().split('\n')

np.random.seed(4)
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

path_in = dict_argv['-images_dir']
path_out = dict_argv['-output_dir']

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

for file in glob.glob(path_in + '*.jpg'):
  output_file = path_out + file.split('/')[-1]
  
  image = cv2.imread(file)
  h, w = image.shape[:2]
  
  blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
  net.setInput(blob)
  layer_outputs = net.forward(ln)

  boxes, confidences, class_ids = [], [], []

  for output in layer_outputs:
    for detection in output:
      scores = detection[5:]
      class_id = np.argmax(scores)
      confidence = scores[class_id]

      if confidence > thresh:
        box = detection[:4] * np.array([w, h, w, h])
        (centerX, centerY, width, height) = box.astype('int')
        x = int(centerX - (width / 2))
        y = int(centerY - (height / 2))
        boxes.append([x, y, int(width), int(height)])
        confidences.append(float(confidence))
        class_ids.append(class_id)
        
  idxs = cv2.dnn.NMSBoxes(boxes, confidences, thresh, thresh)

  if len(idxs) > 0:
    for i in idxs.flatten():
      (x, y) = (boxes[i][0], boxes[i][1])
      (w, h) = (boxes[i][2], boxes[i][3])

      color = [int(c) for c in colors[class_ids[i]]]

      cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
      text = f'{labels[class_ids[i]]}: {confidences[i]:.2f}'
      cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

  cv2.imwrite(output_file, image)