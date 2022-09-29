from PIL import Image, ImageDraw
import torch
import numpy as np


def intersection_over_union(boxes_preds, boxes_labels):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """
    box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
    box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    union = box1_area + box2_area - intersection

    return intersection / (union + 1e-6)

def non_max_suppression(bboxes, iou_threshold, conf_threshold):
    """
    Does Non Max Suppression given bboxes
    :param bboxes: list of lists containing all bboxes with each bboxes specified as [class_pred, prob_score, x1, y1, x2, y2]
    :param iou_threshold: threshold where predicted bboxes is correct
    :param threshold: threshold to remove predicted bboxes (independent of IoU)
    :return: bboxes after performing NMS given a specific IoU threshold
    """
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[0] > conf_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[0], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [box for box in bboxes if intersection_over_union(torch.tensor(chosen_box[1:]),torch.tensor(box[1:])) < iou_threshold]
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def convert2imagesize(predictions):
    """
    Convert coordinates in prediction that are relative to the cell to the one that is relative to the image
    :param predictions: Tensor output of the network
    :return: convert x and y to the the position relative to the image
    """
    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    bboxes = predictions[..., 1:5]
    scores = predictions[..., 0:1]
    x_cells = predictions.shape[2]
    y_cells = predictions.shape[1]
    x_cell_indices = torch.arange(x_cells).repeat(batch_size, y_cells , 1).unsqueeze(-1).type(torch.float32)
    y_cell_indices = torch.arange(y_cells).repeat(batch_size, x_cells, 1).unsqueeze(-1).permute(0, 2, 1, 3).type(torch.float32)

    x = (bboxes[...,0:1]+x_cell_indices)/x_cells
    y = (bboxes[...,1:2]+y_cell_indices)/y_cells
    w = bboxes[..., 2:3]
    h = bboxes[..., 3:4]
    converted = torch.cat((predictions[...,0:1],x, y, w, h), dim=-1)
    return converted


def plot_bbox(image, bboxes, color):
    """
    plot rectangle inside the Image
    :param image: numpy image in format HxWxC
    :param bboxes: list of bounding boxes in format x y w h
    :param color: string with color
    :return: numpy image in same format as the input image with bboxes drawn
    """
    image1 = Image.fromarray(image)
    imgdraw = ImageDraw.Draw(image1)
    for bbox in bboxes:
        l, r, t, b = xywh2lrtb(bbox[1], bbox[2], bbox[3], bbox[4])
        l, r, t, b = position2pixels(image1.size[0], image1.size[1], l, r, t, b)
        imgdraw.rectangle([(l, t), (r, b)], outline=color)
    return np.asarray(image1)

def xywh2lrtb(x, y, w, h):
    """
    Convert x,y,w,h to left right top bottom
    :param x: x position of the center
    :param y: y position of the center
    :param w: width of the bounding box
    :param h: height of the bounding box
    :return: left right top and bottom coordinates
    """
    l = x - w / 2
    r = x + w / 2
    t = y - h / 2
    b = y + h / 2
    return l, r, t, b

def position2pixels(size_x,size_y,x0,x1,y0,y1):
    """
    Convert coordinates from percentage to pixels
    :param size_x: width of the image in pixels
    :param size_y: height of the image in pixels
    :param x0: x coordinate to convert
    :param x1: x coordinate to convert
    :param y0: y coordinate to convert
    :param y1: y coordinate to convert
    :return: converted coordinates to pixels
    """
    pix_x0 = int(x0*size_x)
    pix_x1 = int(x1*size_x)
    pix_y0 = int(y0*size_y)
    pix_y1 = int(y1*size_y)
    return pix_x0, pix_x1, pix_y0, pix_y1
