from torch.utils.data import dataloader
from torchmetrics import metric
from tqdm import tqdm
import wandb
import pyjson5
import numpy as np

import torch

from initializers import (
    init_metrics,
    init_loss,
    init_optimizer,
    init_lr_scheduler,
)


CLASS_LABELS = {0: 'Unburnt', 1: 'Burnt', 2: 'Other events'}


def train1epoch(model, train_dataloader, loss_function, optimizer, learning_rate):
    model.train()
    total_batches = len(train_dataloader)

    for (pre_images, post_images, labels) in tqdm(train_dataloader):
        optimizer.zero_grad()
        predictions = model(pre_images, post_images)
        batch_loss = loss_function(predictions, labels)
        batch_loss.backward()
        optimizer.step()

    learning_rate.step()
   
    return model


def eval1epoch(model, dataloader, loss_function):
    model.eval()
    total_batches = len(dataloader)
    running_loss = 0.0
    metrics = init_metrics()
    with torch.no_grad():
        for (pre_images, post_images, labels) in tqdm(dataloader):
            predictions = model(pre_images, post_images)
            batch_loss = loss_function(predictions, labels).item()
            running_loss += batch_loss
            metrics.update(predictions, labels)


    total_loss = running_loss / total_batches
    metrics.compute()

    return total_loss, metrics


def wandb_log_metrics(loss, metrics, learning_rate, epoch, rep_i, learning_stage):
    accuracies = metrics['accuracy']
    f1_scores = metrics['f1']
    precisions = metrics['precision']
    recalls = metrics['recall']
    ious = metrics['iou']

    mean_iou = ious.mean()

    log_dict = {
        f'({rep_i}) Epoch': epoch,
        f'({rep_i}) {learning_stage} Loss': loss,
        f'({rep_i}) {learning_stage} Accuracy ({CLASS_LABELS[0]})': 100 * accuracies[0],
        f'({rep_i}) {learning_stage} Accuracy ({CLASS_LABELS[1]})': 100 * accuracies[1],
        f'({rep_i}) {learning_stage} F-Score ({CLASS_LABELS[0]})': 100 * f1_scores[0],
        f'({rep_i}) {learning_stage} F-Score ({CLASS_LABELS[1]})': 100 * f1_scores[1],
        f'({rep_i}) {learning_stage} Precision ({CLASS_LABELS[0]})': 100 * precisions[0],
        f'({rep_i}) {learning_stage} Precision ({CLASS_LABELS[1]})': 100 * precisions[1],
        f'({rep_i}) {learning_stage} Recall ({CLASS_LABELS[0]})': 100 * recalls[0],
        f'({rep_i}) {learning_stage} Recall ({CLASS_LABELS[1]})': 100 * recalls[1],
        f'({rep_i}) {learning_stage} IoU ({CLASS_LABELS[0]})': 100 * ious[0],
        f'({rep_i}) {learning_stage} IoU ({CLASS_LABELS[1]})': 100 * ious[1],
        f'({rep_i}) {learning_stage} MeanIoU': mean_iou * 100,
        f'({rep_i}) lr': learning_rate
            }

    wandb.log(log_dict)
    return


def train_change_detection(model, device, class_weights, run_path, init_epoch, train_loader, val_loader, validation_id,
                           gsd, checkpoint, configs, model_configs, rep_i, wandb=None):
    '''
    Train a model for Change Detection using a single satellite source.
    '''
    print(f'\n===== REP {rep_i} =====\n')

    save_every = configs['save_every_n_epochs']
    save_last_epoch = configs['save_last_epoch']

    criterion = init_loss(configs, 'train', device, class_weights, model_configs=model_configs)
    optimizer = init_optimizer(model, checkpoint, configs, model_configs)
    lr_scheduler = init_lr_scheduler(optimizer, checkpoint, configs, model_configs)

    if configs['load_state_path'] is None:
        best_val = 0.0
    else:
        best_val = load_checkpoint(configs['load_state_path'])['burnt_value']

    if configs['mixed_precision?']:
        scaler = torch.cuda.amp.GradScaler()

    last_epoch = init_epoch + configs['#epochs'] + 1

    print_frequency = configs['print_frequency']

    #begin training
    for epoch in range(init_epoch, last_epoch):
        print(f'=== Epoch: {epoch} ===')
        print('---BackPropagation---')
        model = train1epoch(model, train_loader, criterion, optimizer, lr_scheduler, print_frequency) #update the weights
        learning_rate = (lr_scheduler.get_last_lr())[0]

        print('---Validating for Underfitting---')
        train_loss, train_metrics = eval1epoch(model. train_loader, criterion, device, print_frequency)  #metrics for underfitting checks.
        print(f'Mean Train Loss: {train_loss:.6f}')
        wandb_log_metrics(train_loss, train_metrics, learning_rate, epoch, rep_i, 'train')

        print('---Validating for Overfitting---')
        val_loss, val_metrics = eval1epoch(model, val_loader, criterion, device, print_frequency)  #metrics for overfitting checks.
        print(f'Mean Validation Loss: {val_loss:.6f}')
        wandb_log_metrics(val_loss, val_metrics, learning_rate, epoch, rep_i, 'validation')


        if (save_every != -1) and ((epoch >= save_last_epoch) or (epoch % save_every == 0)):
            save_checkpoint() #in case I want multiple checkpoints during a repetition


    save_checkpoint() # Save at least once at the end of this repetition

    return





def get_sample_index_in_batch(batch_size, idx):
    '''
    Takes as input the index of an individual sample (as it is mapped by the Dataset object) and
    calculates the batch index it is contained into, as well as its index inside the batch.
    '''
    batch_idx = (idx // batch_size)
    idx_in_batch = (idx % batch_size)

    return batch_idx, idx_in_batch





           
"""
def eval_change_detection(model, device, class_weights, init_epoch, loader, validation_id, gsd,
                          mode, configs, model_configs, rep_i, wandb=None, run_path=None):
    cm, iou = initialize_metrics(configs, device)


    criterion = init_loss(configs, 'val', device, class_weights, model_configs=model_configs)

    total_loss = 0.0
    total_iters = 0

    bands_idx = list(configs['datasets']['selected_bands'][configs['datasets']['data_source']].values())

    batch_idx, idx_in_batch = get_sample_index_in_batch(configs['datasets']['batch_size'], validation_id)


    model.eval()

    with tqdm(initial=0, total=len(loader)) as pbar:
        for index, batch in enumerate(loader):
            with torch.cuda.amp.autocast(enabled=configs['train']['mixed_precision']):
                with torch.no_grad():
                    if configs['datasets']['data_source'] == 'mod':
                        before_img = batch['MOD_before_image'][:, bands_idx, :, :]
                        after_img = batch['MOD_after_image'][:, bands_idx, :, :]
                    else:
                        before_img = batch['S2_before_image'][:, bands_idx, :, :]
                        after_img = batch['S2_after_image'][:, bands_idx, :, :]
                    label = batch['label']

                    before_img = before_img.to(device)
                    after_img = after_img.to(device)
                    label = label.to(device).long()

                    output = model(before_img, after_img)

                    if configs['method'] == 'changeformer':
                        output = output[-1]

                    predictions = output.argmax(1).to(dtype=torch.int8)

                    loss = criterion(output, label)

                    # Note: loss.item() is averaged across all training examples of the current batch
                    # so we multiply by the batch size to obtain the unaveraged current loss
                    total_loss += (loss.item() * before_img.size(0))

                    cm.compute(predictions, label)
                    iou.update(predictions, label)

                    if configs['train']['log_landcover_metrics']:
                        clc = batch['clc_mask'].to(device)
                        lc_logger.compute(predictions, label, clc)

                    if index % configs['train']['print_freq'] == 0:
                        pbar.set_description(f'{mode} Loss: {total_loss:.4f}')

                    if configs['wandb']['activate'] and (index == batch_idx):
                        # Note: permute() is used because wandb Image requires channel-last format
                        before_img_wand = before_img[idx_in_batch].permute(1, 2, 0).detach().cpu()
                        after_img_wand = after_img[idx_in_batch].permute(1, 2, 0).detach().cpu()

                        label_wand = label[idx_in_batch].detach().cpu()
                        prediction_wand = predictions[idx_in_batch].detach().cpu()

            pbar.update(1)

    acc = cm.accuracy()
    score = cm.f1_score()
    prec = cm.precision()
    rec = cm.recall()
    ious = iou.compute()
    mean_iou = ious[:2].mean()

    if configs['train']['log_landcover_metrics']:
        lc_stats = lc_logger.get_metrics()

    print(f'VAL F1-score: {score[1].item()}')

    selected_bands_idx = {band: order_id for order_id, (band, _) in enumerate(configs['datasets']['selected_bands'][configs['datasets']['data_source']].items())}

    if configs['datasets']['data_source'] == 'sen2':
        if gsd['sen2'] == '10':
            if set(['B08', 'B04', 'B03']) <= set(configs['datasets']['selected_bands']['sen2'].keys()):
                # NIR, Red, Green
                bands_to_plot = [selected_bands_idx[band] for band in ['B08', 'B04', 'B03']]
            else:
                # Plot the first band
                bands_to_plot = list(selected_bands_idx.values())[0]
        else:
            if set(['B8A', 'B04', 'B03']) <= set(configs['datasets']['selected_bands']['sen2'].keys()):
                # NIR, Red, Green
                bands_to_plot = [selected_bands_idx[band] for band in ['B8A', 'B04', 'B03']]
            else:
                # Plot the first band
                bands_to_plot = list(selected_bands_idx.values())[0]
    elif configs['datasets']['data_source'] == 'mod':
        if set(['B02', 'B01', 'B04']) <= configs['datasets']['selected_bands']['mod'].keys():
            # NIR, Red, Green
            bands_to_plot = [selected_bands_idx[band] for band in ['B02', 'B01', 'B04']]
        else:
            # Plot the first band
            bands_to_plot = list(selected_bands_idx.values())[0]

    if configs['wandb']['activate']:
        if len(bands_to_plot) == 3:
            before_img_log = wandb.Image(
                (before_img_wand[:, :, bands_to_plot] * 255).int().numpy(),
                caption='Before',
                masks={
                    "predictions": {
                        "mask_data": prediction_wand.float().numpy(),
                        "class_labels": CLASS_LABELS
                    },
                    "ground_truth": {
                        "mask_data": label_wand.float().numpy(),
                        "class_labels": CLASS_LABELS
                    },
                }
            )
            after_img_log = wandb.Image(
                (after_img_wand[:, :, bands_to_plot] * 255).int().numpy(),
                caption='After',
                masks={
                    "predictions": {
                        "mask_data": prediction_wand.float().numpy(),
                        "class_labels": CLASS_LABELS
                    },
                    "ground_truth": {
                        "mask_data": label_wand.float().numpy(),
                        "class_labels": CLASS_LABELS
                    },
                }
            )
            wandb.log({f'({rep_i}) {mode} Before image': before_img_log})
            wandb.log({f'({rep_i}) {mode} After image': after_img_log})
        else:
            before_img_red_log = wandb.Image(
                (before_img_wand[:, :, bands_to_plot[1]] * 255).int().numpy(),
                caption='Before (Red)',
                masks={
                    "predictions": {
                        "mask_data": prediction_wand.float().numpy(),
                        "class_labels": CLASS_LABELS
                    },
                    "ground_truth": {
                        "mask_data": label_wand.float().numpy(),
                        "class_labels": CLASS_LABELS
                    },
                }
            )
            before_img_nir_log = wandb.Image(
                (before_img_wand[:, :, bands_to_plot[0]] * 255).int().numpy(),
                caption='Before (NIR)',
                masks={
                    "predictions": {
                        "mask_data": prediction_wand.float().numpy(),
                        "class_labels": CLASS_LABELS
                    },
                    "ground_truth": {
                        "mask_data": label_wand.float().numpy(),
                        "class_labels": CLASS_LABELS
                    },
                }
            )
            after_img_red_log = wandb.Image(
                (after_img_wand[:, :, bands_to_plot[1]] * 255).int().numpy(),
                caption='After (Red)',
                masks={
                    "predictions": {
                        "mask_data": prediction_wand.float().numpy(),
                        "class_labels": CLASS_LABELS
                    },
                    "ground_truth": {
                        "mask_data": label_wand.float().numpy(),
                        "class_labels": CLASS_LABELS
                    },
                }
            )
            after_img_nir_log = wandb.Image(
                (after_img_wand[:, :, bands_to_plot[0]] * 255).int().numpy(),
                caption='After (NIR)',
                masks={
                    "predictions": {
                        "mask_data": prediction_wand.float().numpy(),
                        "class_labels": CLASS_LABELS
                    },
                    "ground_truth": {
                        "mask_data": label_wand.float().numpy(),
                        "class_labels": CLASS_LABELS
                    },
                }
            )
            wandb.log({f'({rep_i}) {mode} Before image (Red)': before_img_red_log})
            wandb.log({f'({rep_i}) {mode} Before image (NIR)': before_img_nir_log})
            wandb.log({f'({rep_i}) {mode} After image (Red)': after_img_red_log})
            wandb.log({f'({rep_i}) {mode} After image (NIR)': after_img_nir_log})

    if configs['wandb']['activate']:
        wandb.log({
            f'({rep_i}) {mode} F-Score ({CLASS_LABELS[0]})': 100 * score[0].item(),
            f'({rep_i}) {mode} F-Score ({CLASS_LABELS[1]})': 100 * score[1].item(),
            f'({rep_i}) {mode} IoU ({CLASS_LABELS[0]})': 100 * ious[0],
            f'({rep_i}) {mode} IoU ({CLASS_LABELS[1]})': 100 * ious[1],
            f'({rep_i}) {mode} Precision ({CLASS_LABELS[0]})': 100 * prec[0].item(),
            f'({rep_i}) {mode} Precision ({CLASS_LABELS[1]})': 100 * prec[1].item(),
            f'({rep_i}) {mode} Recall ({CLASS_LABELS[0]})': 100 * rec[0].item(),
            f'({rep_i}) {mode} Recall ({CLASS_LABELS[1]})': 100 * rec[1].item(),
            f'({rep_i}) {mode} Accuracy ({CLASS_LABELS[0]})': 100 * acc[0].item(),
            f'({rep_i}) {mode} Accuracy ({CLASS_LABELS[1]})': 100 * acc[1].item(),
            f'({rep_i}) {mode} MeanIoU': 100 * mean_iou.item(),
            f'({rep_i}) {mode} Loss': total_loss / len(loader)
        })
    elif mode == 'test':
        print(f'({rep_i}) {mode} F-Score ({CLASS_LABELS[0]}): {100 * score[0].item()}')
        print(f'({rep_i}) {mode} F-Score ({CLASS_LABELS[1]}): {100 * score[1].item()}')
        print(f'({rep_i}) {mode} IoU ({CLASS_LABELS[0]}): {100 * ious[0]}')
        print(f'({rep_i}) {mode} IoU ({CLASS_LABELS[1]}): {100 * ious[1]}')
        print(f'({rep_i}) {mode} Precision ({CLASS_LABELS[0]}): {100 * prec[0].item()}')
        print(f'({rep_i}) {mode} Precision ({CLASS_LABELS[1]}): {100 * prec[1].item()}')
        print(f'({rep_i}) {mode} Recall ({CLASS_LABELS[0]}): {100 * rec[0].item()}')
        print(f'({rep_i}) {mode} Recall ({CLASS_LABELS[1]}): {100 * rec[1].item()}')
        print(f'({rep_i}) {mode} Accuracy ({CLASS_LABELS[0]}): {100 * acc[0].item()}')
        print(f'({rep_i}) {mode} Accuracy ({CLASS_LABELS[1]}): {100 * acc[1].item()}')
        print(f'({rep_i}) {mode} MeanIoU {100 * mean_iou.item()}')

        if configs['train']['log_landcover_metrics']:
            print('')
            for lc_id, lc_info in lc_stats.items():
                print(f'{lc_id}: {lc_info}')

    if mode == 'test':
        res = {
            'precision': (100 * prec[0].item(), 100 * prec[1].item()),
            'recall': (100 * rec[0].item(), 100 * rec[1].item()),
            'accuracy': (100 * acc[0].item(), 100 * acc[1].item()),
            'f1': (100 * score[0].item(), 100 * score[1].item()),
            'iou': (100 * ious[0].item(), 100 * ious[1].item())
        }

        if configs['train']['log_landcover_metrics']:
            res['lc_stats'] = lc_stats

        return res
    else:
        return 100 * acc.nanmean(), 100 * score.nanmean(), 100 * mean_iou, score[1]
"""
