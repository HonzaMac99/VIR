import model, dataset
import torch
import os
import utils
from loss import YoloLoss
import numpy as np
import random
from tensorboardX import SummaryWriter
from time import gmtime, strftime

if __name__ == '__main__':
    training_data_path = "survivor_dataset/training/"
    validation_data_path = "survivor_dataset/validation/"
    batch_size = 16
    epochs = 30

    learning_rate = 0.001
    weight_decay = 1e-6

    # chose the gpu by "cuda:<GPU_NUMBER>"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainingDataset = dataset.myDataset(training_data_path, augment=False)
    validationDataset = dataset.myDataset(validation_data_path, augment=False)

    directory = os.path.abspath(os.path.dirname(__file__))
    model = model.YoloTiny()
    model.load_state_dict(torch.load(directory + '/weights.pth', map_location='cpu'))

    model.to(device)
    loss_fc = YoloLoss()

    trainloader = torch.utils.data.DataLoader(trainingDataset, batch_size=batch_size, shuffle=True, num_workers=6)
    valloader = torch.utils.data.DataLoader(validationDataset, batch_size=1, shuffle=False, num_workers=6)

    # Set a optimizer
    # Adam, SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.01, last_epoch=-1)

    runtime = strftime("%Y-%m-%d_%H:%M:%S", gmtime())

    if not os.path.exists('net-' + runtime):
        os.makedirs('net-' + runtime)

    writer = SummaryWriter('net-' + runtime + '/')

    # initialize how many samples has been seen
    trn_seen = 0
    val_seen = 0

    # enumerate trough epochs
    for epoch in range(epochs):
        total_loss_val = 0
        total_loss_train = 0

        # enumerate training dataset
        for i, data in enumerate(trainloader):
            image = data['image'].to(device)
            label = data['label'].to(device)

            trn_seen += 1

            # Do a forward pass
            # print(f"maximal value of pixel in batch: {image.max()}")
            predictions = model(image)

            # Compute loss
            loss, box_loss, obj_loss, noobj_loss = loss_fc(predictions, label)
            total_loss_train += loss

            # Do the backward pass and update the gradients
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            # If you will use tensorboard uncomment this:
            writer.add_scalar('data/loss', loss / batch_size, trn_seen)
            writer.add_scalar('data/loss_box', box_loss / batch_size, trn_seen)
            writer.add_scalar('data/loss_obj', obj_loss / batch_size, trn_seen)
            writer.add_scalar('data/loss_noobj', noobj_loss / batch_size, trn_seen)

            print('e' + str(epoch) + ' - ' + str(int(i * batch_size)) + '/' + str(int(trainingDataset.size)) + ': %5.3f' % loss.item())
        scheduler.step()


        # If you will use tensorboard uncomment this:
        writer.add_scalar('data/epoch_train_loss', total_loss_train / (batch_size*trainingDataset.size), epoch)

        converted_output = utils.convert2imagesize(predictions[0:1, ...])
        output_list = converted_output.flatten(end_dim=-2).tolist()
        pred_boxes = utils.non_max_suppression(output_list, iou_threshold=0.9, conf_threshold=0.3)

        converted_label = utils.convert2imagesize(label[0:1, ...])
        label_list = converted_label.flatten(end_dim=-2).tolist()
        label_boxes = utils.non_max_suppression(label_list, iou_threshold=0.9, conf_threshold=0.5)

        numpy_im = (np.asarray(image.detach().cpu().permute((0, 2, 3, 1))).astype(np.uint8)[0])
        numpy_im = utils.plot_bbox(numpy_im, label_boxes, 'red')
        numpy_im = utils.plot_bbox(numpy_im, pred_boxes, 'green')
        im = np.moveaxis(np.asarray(numpy_im), -1, 0)  # HxWxC -> CxHxW
        writer.add_image('data/images', im / 255, trn_seen)


        # enumerate validation dataset
        to_show = random.randint(0, validationDataset.size)
        for i, data in enumerate(valloader):
            input_img = data['image'].to(device)
            label = data['label'].to(device)
            val_seen += 1

            # Do a forward pass (use "with torch.no_grad():")
            with torch.no_grad():
                predictions = model(input_img)

            # Compute loss
            loss, box_loss, obj_loss, noobj_loss = loss_fc(predictions, label)
            total_loss_val += loss

            print('e' + str(epoch) + ' - ' + str(i) + '/' + str(int(validationDataset.size)) + ': %5.3f' % loss.item())


            # If you will use tensorboard uncomment this:
            if i == to_show:
                converted_output = utils.convert2imagesize(predictions[0:1, ...])
                output_list = converted_output.flatten(end_dim=-2).tolist()
                pred_boxes = utils.non_max_suppression(output_list, iou_threshold=0.9, conf_threshold=0.3)

                converted_label = utils.convert2imagesize(label[0:1, ...])
                label_list = converted_label.flatten(end_dim=-2).tolist()
                label_boxes = utils.non_max_suppression(output_list, iou_threshold=0.9, conf_threshold=0.5)

                numpy_im = (np.asarray(image.detach().cpu().permute((0, 2, 3, 1))).astype(np.uint8)[0])
                numpy_im = utils.plot_bbox(numpy_im, label_boxes, 'red')
                numpy_im = utils.plot_bbox(numpy_im, pred_boxes, 'green')
                im = np.moveaxis(np.asarray(numpy_im), -1, 0)  # HxWxC -> CxHxW
                writer.add_image('data/val_images', im / 255, val_seen)

            writer.add_scalar('data/val_loss', loss / batch_size, val_seen)
            writer.add_scalar('data/val_loss_box', box_loss / batch_size, val_seen)
            writer.add_scalar('data/val_loss_obj', obj_loss / batch_size, val_seen)
            writer.add_scalar('data/val_loss_noobj', noobj_loss / batch_size, val_seen)

        writer.add_scalar('data/epoch_val_loss', total_loss_val/validationDataset.size , epoch) # save validation epoch loss


        # save every 10. epoch

        if epoch % 10 == 0 or epoch == 1:
            torch.save(model.state_dict(), 'net-' + runtime + '/net_epoch_%d.pth' % epoch)

        model.save_model()
