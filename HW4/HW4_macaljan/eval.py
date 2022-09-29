import model, dataset
import torch
import os
import utils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def threshold_detections(list_of_boxes, th):
    """
    Count number of detections with the confidence bigger and lower than threshold
    :param list_of_boxes: list of bounding boxes
    :param th: threshold to divide the list of boxes
    :return: sum of boxes that has bigger confidence then th, sum of boxes that has lower confidence then th
    """
    bigger_then = np.asarray(list_of_boxes)[:, 0] > th
    return sum(bigger_then), sum(~bigger_then)


def create_potential_detections(pred_boxes_orig, lab_boxes_orig, iou_th=0.6):
    """
    :param pred_boxes_orig: list of predicted boxes
    :param lab_boxes_orig:  list of label boxes
    :param iou_th: intersection over union threshold to decide if the prediction is accurate enough to be considered as detected
    :return: correct: list of all label box with assigned prediction (if there is)
             wrong: list of all predictions that does not corresponds to any ground truth bounding box
    """
    pred_boxes = pred_boxes_orig.copy()
    lab_boxes = lab_boxes_orig.copy()
    correct = []
    wrong = []
    for l_box in lab_boxes:
        p_i = 0
        predicted = 0
        for p_box in pred_boxes:
            iou = utils.intersection_over_union(torch.tensor(l_box[1:]), torch.tensor(p_box[1:]))
            if iou > iou_th:
                correct.append(p_box)
                pred_boxes.pop(p_i)
                predicted = 1
            p_i += 1
        if predicted == 0:
            correct.append([0, 0, 0, 0, 0])
    for p in pred_boxes:
        wrong.append(p)
    return correct, wrong


def eval_weights(weights_path, dataset_path, export_graph_path, export_data_path, conf_threshold_toplot):
    '''
    :param weights_path:
    :param dataset_path:
    :param export_path:
    :param conf_threshold_toplot:
    :return:
    # DO NOT CHANGE ANYTHING INSIDE THIS FUNCTION
    '''
    # create directories if not exist
    if not os.path.exists(export_data_path):
        os.makedirs(export_data_path)
    if not os.path.exists(export_graph_path):
        os.makedirs(export_graph_path)

    # load dataset
    testingDataset = dataset.myDataset(dataset_path)
    testingloader = torch.utils.data.DataLoader(testingDataset, batch_size=1, shuffle=False, num_workers=3)

    # load model and weights
    net = model.YoloTiny()
    net.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    all_iou_correct = {}
    all_iou_wrong = {}
    for i, data in enumerate(testingloader):
        input = data['image'].to(device)
        label = data['label'].to(device)
        # forward pass
        with torch.no_grad():
            output = net(input)

        # output to the list of boxes
        conv_output = utils.convert2imagesize(output[0:1, ...])
        output_list = conv_output.flatten(end_dim=-2).tolist()
        pred_boxes = utils.non_max_suppression(output_list, 0.01, 0.01)
        pred_boxes_toplot = utils.non_max_suppression(output_list, iou_threshold=0.2,
                                                      conf_threshold=conf_threshold_toplot)
        # label to the list of boxes
        label_output = utils.convert2imagesize(label[0:1, ...])
        label_list = label_output.flatten(end_dim=-2).tolist()
        label_boxes = utils.non_max_suppression(label_list, 0.2, 0.5)

        # plot and save image with gt and predicted bbox
        numpy_im = (np.asarray(input.detach().cpu().permute((0, 2, 3, 1))).astype(np.uint8)[0])
        numpy_im = utils.plot_bbox(numpy_im, pred_boxes_toplot, 'green')
        numpy_im = utils.plot_bbox(numpy_im, label_boxes, 'red')
        pil_im = Image.fromarray(numpy_im)
        pil_im.save(export_data_path + '/' + data['im_path'][0].split('/')[-1])

        # prepare predictions and labels
        for iou in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            correct, wrong = create_potential_detections(pred_boxes, label_boxes, iou_th=iou)
            for c in correct:
                try:
                    all_iou_correct[iou].append(c)
                except:
                    all_iou_correct[iou] = []
                    all_iou_correct[iou].append(c)

            # print('wrong%f:' % iou, wrong)
            for w in wrong:
                try:
                    all_iou_wrong[iou].append(w)
                except:
                    all_iou_wrong[iou] = []
                    all_iou_wrong[iou].append(w)

    # compute precision and recall
    all_AP = []
    for iou in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        p = []
        r = []
        for th in np.arange(0.00, 1.001, 0.001):
            FP, TN = threshold_detections(all_iou_wrong[iou], th)
            TP, FN = threshold_detections(all_iou_correct[iou], th)
            precision = TP / (TP + FP + 1e-6)
            recall = TP / (TP + FN)
            p.append(precision)
            r.append(recall)
        p.insert(0, 0)
        r.insert(0, max(r))
        p.insert(0, 0)
        r.insert(0, 1)
        rec = np.asarray(r)
        p_int = []
        p_max = 0
        # smooth the pr curve
        for p_i in p:
            if p_i > p_max:
                p_int.append(p_i)
                p_max = p_i
            else:
                p_int.append(p_max)
        prec = np.asarray(p_int)
        # compute average precision
        AP = np.sum((rec[:-1] - rec[1:]) * prec[:-1])
        plt.plot(r, p_int, label='IoU: ' + str(iou) + ' AP:  %.3f' % AP)
        all_AP.append(AP)
    # compute mean average precision
    all_AP = np.asarray(all_AP).mean()
    plt.title('mAP@0.2:0.1:0.7 = %.4f' % all_AP)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.savefig(export_graph_path + '/pr_curve.pdf')

    # plt.savefig(export_graph_path + f'/pr_curve{num}.pdf')
    # print("Saved: " + export_graph_path + f'/pr_curve{num}.pdf')


if __name__ == '__main__':
    # This is the only part you can change

    # The images with the prediction will be saved in this path
    export_data_path = "prediction"
    # In this path will be saved pr_curve
    export_graph_path = "graph"

    # Path where the weights are saved
    # epoch = 30
    # weights_path = f"net-2021-11-30_01:03:07/net_epoch_{epoch}"
    # weights_path = "net-2021-11-30_16:36:02/net_epoch_0.pth"
    weights_path = "weights.pth"

    # Path where the validation dataset is saved
    dataset_path = "survivor_dataset/validation/"
    # set threshold to plot predictions
    conf_threshold_toplot = 1

    eval_weights(weights_path, dataset_path, export_graph_path, export_data_path, conf_threshold_toplot)

    '''
        max_iter = 24
        for i in range(1, max_iter):
            try:
                # Path where the weights are saved
                weights_path = f"net-2021-11-30_11:12:15/net_epoch_{i * 10}"

                eval_weights(weights_path, dataset_path, export_graph_path, export_data_path, conf_threshold_toplot, i * 10)
            except:
                print("Exception:", i * 10)
        '''






