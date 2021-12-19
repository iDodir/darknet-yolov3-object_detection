# darknet-yolov3-object_detection

class_id = 0, name = protective vest, ap = 96.59%   	 (TP = 1663, FP = 201)
class_id = 1, name = hard hat, ap = 92.85%   	 (TP = 1326, FP = 183)
class_id = 2, name = person, ap = 97.75%   	 (TP = 1706, FP = 150)

for conf_thresh = 0.25, precision = 0.90, recall = 0.95, F1-score = 0.92
for conf_thresh = 0.25, TP = 4695, FP = 534, FN = 242, average IoU = 71.73 %

IoU threshold = 50 %, used Area-Under-Curve for each unique Recall
mean average precision (mAP@0.50) = 0.957311, or 95.73 %
Total Detection Time: 48 Seconds

Set -points flag:
 `-points 101` for MS COCO
 `-points 11` for PascalVOC 2007 (uncomment `difficult` in voc.data)
 `-points 0` (AUC) for ImageNet, PascalVOC 2010-2012, your custom dataset

 mean_average_precision (mAP@0.50) = 0.957311
