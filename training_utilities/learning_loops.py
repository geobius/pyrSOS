from torch.utils.data import dataloader
from torchmetrics import metric
from tqdm import tqdm
import wandb
import pyjson5
import numpy as np

import torch
import pdb
from .initializers import (
    init_metrics,
    init_loss,
    init_optimizer,
    init_lr_scheduler,
)


CLASS_LABELS = {0: 'Unburnt', 1: 'Burnt', 2: 'Other events'}


def train1epoch(model, train_dataloader, loss_function, optimizer, scheduler, device):
    model.train()

    for (pre_images, post_images, labels, _) in tqdm(train_dataloader):
        pre_images = pre_images.to(device=device)
        post_images = post_images.to(device=device)
        labels = labels.to(device=device)

        optimizer.zero_grad()
        predictions = model(pre_images, post_images)
        batch_loss = loss_function(predictions, labels)
        batch_loss.backward()
        optimizer.step()

    scheduler.step()

    return model


def eval1epoch(model, dataloader, loss_function, device):
    model.eval()

    total_batches = len(dataloader)
    running_loss = 0.0
    metrics = init_metrics().to(device=device)
    with torch.no_grad():
        for (pre_images, post_images, labels, _) in tqdm(dataloader):
            pre_images = pre_images.to(device=device)
            post_images = post_images.to(device=device)
            labels = labels.to(device=device)

            logit_outputs = model(pre_images, post_images)
            probabilities = torch.softmax(logit_outputs, dim=1)
            masks = torch.argmax(probabilities, dim=1)

            batch_loss = loss_function(logit_outputs, labels)
            running_loss += batch_loss.item()
            metrics.update(masks, labels)

    total_loss = running_loss / total_batches
    final_metrics = metrics.compute()

    return total_loss, final_metrics


def wandb_log_metrics(loss, metrics, learning_rate, epoch, rep_i, learning_stage, should_log):
    Dice_scores = metrics['Dice']
    iou_scores = metrics['iou']

    mean_iou = iou_scores.mean()

    log_dict = {
        f'({rep_i}) Epoch': epoch,
        f'({rep_i}) {learning_stage} Loss': loss,
        f'({rep_i}) {learning_stage} Dice ({CLASS_LABELS[0]})': 100 * Dice_scores[0].item(),
        f'({rep_i}) {learning_stage} Dice ({CLASS_LABELS[1]})': 100 * Dice_scores[1].item(),
        f'({rep_i}) {learning_stage} IoU ({CLASS_LABELS[0]})': 100 * iou_scores[0].item(),
        f'({rep_i}) {learning_stage} IoU ({CLASS_LABELS[1]})': 100 * iou_scores[1].item(),
        f'({rep_i}) {learning_stage} MeanIoU': 100 * mean_iou.item(),
        f'({rep_i}) lr': learning_rate
            }
    if should_log:
        wandb.log(log_dict)

    return
