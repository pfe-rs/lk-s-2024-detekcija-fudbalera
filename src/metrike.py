import torch
import torchvision.transforms as T
import torchvision
#funkc za čitanje anotacija
# funkcija za računanje IoU (Intersection over Union)
# u sustini gleda preklapanje predvidjenog i pravog BB

def calculate_iou(box1, box2):
    # box format: [xmin, ymin, xmax, ymax]
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # presek
    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    #unija
    box1_area = (x2 - x1+1) * (y2 - y1+1)
    box2_area = (x4 - x3+1) * (y4 - y3+1)
    union_area = box1_area + box2_area - inter_area


    iou = inter_area / union_area

    return iou


# poređenje prediction sa ground truth
def compare_predictions_with_annotations(predictions, annotations, CLASSES, threshold=0.5):
    tp = {}
    fp = {}
    fn = {}

    for cls_name in CLASSES:
        tp[cls_name] = 0
        fp[cls_name] = 0
        fn[cls_name] = 0

    for image_id, pred in predictions.items():
        if image_id in annotations:
            pred_boxes = pred['boxes']
            pred_labels = pred['labels']

            target_boxes = annotations[image_id]['boxes']
            target_labels = annotations[image_id]['labels']

            for cls_idx, cls_name in enumerate(CLASSES):
                pred_mask = pred_labels == cls_idx
                target_mask = target_labels == cls_idx

                pred_boxes_cls = pred_boxes[pred_mask]
                target_boxes_cls = target_boxes[target_mask]

                pred_boxes_cls = pred_boxes_cls.cpu().numpy() if torch.cuda.is_available() else pred_boxes_cls.numpy()
                target_boxes_cls = target_boxes_cls.cpu().numpy() if torch.cuda.is_available() else target_boxes_cls.numpy()

                for pred_box in pred_boxes_cls:
                    found_match = False
                    for target_box in target_boxes_cls:
                        iou = calculate_iou(pred_box, target_box)
                        if iou > threshold:
                            tp[cls_name] += 1
                            found_match = True
                            break
                    if not found_match:
                        fp[cls_name] += 1

                for target_box in target_boxes_cls:
                    found_match = False
                    for pred_box in pred_boxes_cls:
                        iou = calculate_iou(pred_box, target_box)
                        if iou > threshold:
                            found_match = True
                            break
                    if not found_match:
                        fn[cls_name] += 1

    return tp, fp, fn
