from torch.utils.data import dataloader
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryJaccardIndex)
from tqdm import tqdm
import wandb
import pyjson5
import numpy as np

import torch
import pdb
from .initializers import (
    init_loss,
    init_optimizer,
    init_lr_scheduler,
)


CLASS_LABELS = {0: 'Unburnt', 1: 'Burnt'}


def make_metrics_table():

    pyrsos_metrics = MetricCollection({
        "iou": BinaryJaccardIndex(),
        "precision": BinaryPrecision(),
        "recall": BinaryRecall(),
        "f1": BinaryF1Score()
    })

    return pyrsos_metrics


def train1epoch(model, train_dataloader, loss_function, optimizer, scheduler, device):
    model.train()

    total_batches = len(train_dataloader)
    running_loss = 0.0
    metrics = make_metrics_table().to(device=device)

    for (pre_images, post_images, labels) in tqdm(train_dataloader):
        pre_images = pre_images.to(device=device)
        post_images = post_images.to(device=device)
        labels = labels.to(device=device)

        optimizer.zero_grad()
        logit_outputs = model(pre_images, post_images)
        batch_loss = loss_function(logit_outputs, labels)
        running_loss += batch_loss.item()
        batch_loss.backward()
        optimizer.step()

        probabilities = torch.softmax(logit_outputs, dim=1)
        masks = torch.argmax(probabilities, dim=1)
        metrics.update(masks, labels)

    total_loss = running_loss / total_batches
    final_metrics = metrics.compute()

    scheduler.step()

    return total_loss, final_metrics


def eval1epoch(model, dataloader, loss_function, device):
    model.eval()

    total_batches = len(dataloader)
    running_loss = 0.0
    metrics = make_metrics_table().to(device=device)
    with torch.no_grad():
        for (pre_images, post_images, labels) in tqdm(dataloader):
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


def wandb_log_metrics(loss, metrics, learning_rate, epoch, learning_stage, should_log):
    iou_score = metrics['iou'].item()
    precision_score = metrics['precision'].item()
    recall_score = metrics['recall'].item()
    f1_score = metrics['f1'].item()

    log_dict = {
        'Epoch': epoch,
        f'{learning_stage} Loss': loss,
        f'{learning_stage} IoU ({CLASS_LABELS[1]})': 100 * iou_score,
        f'{learning_stage} Precision ({CLASS_LABELS[1]})': 100 * precision_score,
        f'{learning_stage} Recall ({CLASS_LABELS[1]})': 100 * recall_score,
        f'{learning_stage} F1 ({CLASS_LABELS[1]})': 100 * f1_score,
        'lr': learning_rate
            }
    if should_log:
        wandb.log(log_dict)

    return
