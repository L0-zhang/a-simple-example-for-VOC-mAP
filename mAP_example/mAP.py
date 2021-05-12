# -*- coding: utf-8 -*-
"""
Created on Wed May 12 19:37:59 2021
"""

import os
from tqdm import tqdm
import numpy as np
import torch

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

class_names = ['hat', 'person']


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    plt.scatter(mrec, mpre)
    plt.plot(mrec, mpre, color='green')
    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    plt.xticks(ticks=np.arange(0, 1.125, step=0.125),)
    plt.xlabel("recall")
    plt.ylabel("precision")
    for j in range(len(mrec)-1, 0, -1):
        if mpre[j] != mpre[j-1]:
            plt.plot([mrec[j-1], mrec[j]], [mpre[j], mpre[j]], color='orange')
            plt.plot([mrec[j-1], mrec[j-1]],
                     [mpre[j], mpre[j-1]], color='orange')
        else:
            plt.plot([mrec[j-1], mrec[j]], [mpre[j], mpre[j]], color='orange')
    plt.show()


    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,
                                          0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,
                                          0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def add_box(boxes, color='blue'):
    if boxes is not None:
        # Rescale boxes to original image
        # output = rescale_boxes(output, cfg.img_size, img.shape[:2])
        for cls_box, x1, y1, x2, y2 in boxes:

            box_w = x2 - x1
            box_h = y2 - y1

            # Create a Rectangle patch
            bbox = patches.Rectangle(
                (x1, y1), box_w, box_h, linewidth=1, edgecolor=color, facecolor="none")
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            if color == 'blue':
                plt.text(
                    x1,
                    y1,
                    s='p' if class_names[int(cls_box)] == 'person' else 'h',
                    color="white",
                    # verticalalignment="baseline"if class_list[int(cls_box)]=='person' else 'bottom',
                    # horizontalalignment='left',
                    bbox={"color": color, "pad": 0},
                )
            else:
                plt.text(
                    x2,
                    y2,
                    s='p' if class_names[int(cls_box)] == 'person' else 'h',
                    color="white",
                    # verticalalignment="baseline"if class_list[int(cls_box)]=='person' else 'bottom',
                    # horizontalalignment='left',
                    bbox={"color": color, "pad": 1},
                )


def letterbox(img, new_shape=(412, 412), color=(114, 114, 114)):
    shape = img.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img


if __name__ == "__main__":

    iou_thres = 0.5

    print("Compute mAP...")

    sample_metrics = []  # List of tuples (TP, confs, pred)
    path = os.getcwd()

    # read gt labels
    targets = torch.tensor(np.loadtxt(
        os.path.join(path, "gt_boxes\\target.txt")))
    # Extract labels
    labels = targets[:, 1].tolist()

    # read pred result
    pred_result = []
    for i in range(3):
        pred_result.append(np.loadtxt(
            os.path.join(path, f"pred_boxes\\{i+1}.txt")))
    outputs = [torch.tensor(output) for output in pred_result]

    # # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    for img_i, output in enumerate(outputs):

        # Create plot
        img_path = os.path.join(path, f"image\\{img_i+1}.jpg")

        # img=Image.open(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = letterbox(img)

        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of gt
        gt_Imgi = targets[targets[:, 0] == img_i]
        if gt_Imgi is not None:
            add_box(gt_Imgi[:, 1:], color='red')

        # Draw bounding boxes and labels of detections
        pred_boxes = torch.cat((output[:, -1:], output[:, :4]), dim=1)
        if pred_boxes is not None:
            add_box(pred_boxes, color='blue')

        # # Save generated image with detections
        ax.axis("off")
        ax.autoscale_view()
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        #
        # filename = os.path.basename(path).split(".")[0]
        output_path = os.path.join(path, f"{img_i+1}.jpg")
        plt.savefig(output_path, bbox_inches="tight", dpi=200, pad_inches=0.0)

        batch_metrics = []
        for sample_i in range(len(outputs)):  # sample_i：batch_id

            if outputs[sample_i] is None:
                continue

            output = outputs[sample_i]
            pred_boxes = output[:, :4]
            pred_scores = output[:, 4]
            pred_labels = output[:, -1]

            true_positives = np.zeros(pred_boxes.shape[0])

            annotations = targets[targets[:, 0] == sample_i][:, 1:]
            target_labels = annotations[:, 0] if len(annotations) else []
            if len(annotations):
                detected_boxes = []
                target_boxes = annotations[:, 1:]

                for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                    # If targets are found break
                    if len(detected_boxes) == len(annotations):
                        break

                    # Ignore if label is not one of the target labels
                    if pred_label not in target_labels:
                        continue

                    iou, box_index = bbox_iou(
                        pred_box.unsqueeze(0), target_boxes).max(0)
                    if iou >= iou_thres and box_index not in detected_boxes:
                        true_positives[pred_i] = 1
                        detected_boxes += [box_index]
            batch_metrics.append([true_positives, pred_scores, pred_labels])
        sample_metrics += batch_metrics
        plt.show()

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*batch_metrics))]

    print('tp:', true_positives)
    print('pred_scores:', pred_scores)
    print('pred_labels:', pred_labels.astype(int))
    print('labels:', list(map(int, labels)))

    # precision, recall, AP, f1, ap_class = ap_per_class(
    #     true_positives, pred_scores, pred_labels, labels)

    tp, conf, pred_cls, target_cls = true_positives, pred_scores, pred_labels, labels

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects

        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs

            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            # compute AP and plot_PRcurve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    ap_class = unique_classes.astype("int32")

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {ap[i]}")

    print(f"mAP: {ap.mean()}")

    plt.show()
