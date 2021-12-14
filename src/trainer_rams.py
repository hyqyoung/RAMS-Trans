import os
import sys
import numpy as np
import time
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
#import torch.utils
# from data_loader import get_cub_train_set, get_cub_test_set,\
# get_inat_train_set,get_inat_val_set,get_Air_train_set,get_Air_test_set
from checkpoint import load_checkpoint
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
from config import AugmentConfig
from torch.autograd import Variable
from linformer import Linformer
from model_rams import VisionTransformer, CONFIGS
# from model_baseline import VisionTransformer, CONFIGS
from torch.optim.lr_scheduler import StepLR
import json
# from sklearn.metrics import roc_auc_score
import math
import torch.nn.functional as F
from dataset import get_loader
import ml_collections

config = AugmentConfig()
# get_data = utils.get_ImageNetdata()
device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)


def main():
  logger.info("Logger is set - training start")

  # set default gpu device id
  torch.cuda.set_device(config.gpus[0])
  np.random.seed(config.seed)
  cudnn.benchmark = True
  os.environ['PYTHONHASHSEED'] = str(config.seed)
  torch.manual_seed(config.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed_all(config.seed)
  torch.cuda.manual_seed(config.seed)
  torch.backends.cudnn.deterministic = True

  logger.info("create model")
  args = CONFIGS['ViT-B_16']
  model = VisionTransformer(args, 448, zero_head=True, num_classes=200, smoothing_value=0.0)
  # model = VisionTransformer(args, 304, zero_head=True, num_classes=5089, smoothing_value=0.0)
  # load checkpoint
  model.load_from(np.load(your_checkpoint), logger)
  # send model to device
  model = model.to(device)
  if len(config.gpus) > 1:
      model = torch.nn.DataParallel(model, device_ids=config.gpus)
  # model size
  mb_params = utils.count_parameters_in_MB(model)
  logger.info("Model size = {:.3f} MB".format(mb_params))

  def get_args():
    config = ml_collections.ConfigDict()
    config.local_rank = -1
    config.train_batch_size = 16
    config.eval_batch_size = 16
    config.data_root = data_root
    config.dataset = "CUB_200_2011"
    return config

  args = get_args()

  train_loader, valid_loader = get_loader(args)

  # training criterion
  logger.info("create criterion and optimizer")
  criterion = nn.CrossEntropyLoss()

  optimizer3 = torch.optim.SGD(model.parameters(),
            lr=config.lr3,
            weight_decay=config.weight_decay,
            momentum=config.momentum)

  lr_scheduler3 = utils.WarmupCosineSchedule(optimizer3, 
                                    warmup_steps=config.warmup_steps, 
                                    t_total=config.train_steps3)

  best_top1 = 0.
  result_train = {}
  result_valid = {}
  epochs3 = config.train_steps3//len(train_loader)
  for epoch in range(1, epochs3+1):
    current_lr = optimizer3.param_groups[0]['lr']
  
    logging.info('Epoch: %d lr %e', epoch, current_lr)
    train_accTop1_class1, train_accTop5_class1, train_accTop1_class2, train_accTop5_class2,train_accTop1_class3, train_accTop5_class3, train_accTop1_class4, train_accTop5_class4, train_loss = \
    train(train_loader, model, optimizer3, lr_scheduler3, criterion, epoch, epochs3)#, alpha_optimizer, lr_scheduler_alpha)
    result_train['epoch{}'.format(epoch)] = {
                                        "train_accTop1_class":train_accTop1_class1, 
                                        "train_accTop5_class":train_accTop5_class1, 
                                        "train_loss":train_loss}

    valid_accTop1_class1, valid_accTop5_class1, valid_accTop1_class2, valid_accTop5_class2, valid_accTop1_class3, valid_accTop5_class3, valid_accTop1_class4, valid_accTop5_class4, valid_loss = \
    validate(valid_loader, model, criterion, epoch, epochs3)
    result_train['epoch{}'.format(epoch)] = {
                                        "valid_accTop1_class":valid_accTop1_class1, 
                                        "valid_accTop5_class":valid_accTop5_class1, 
                                        "valid_loss":valid_loss}

      # save
    if best_top1 < valid_accTop1_class1:
        best_top1 = valid_accTop1_class1
        is_best = True
    else:
        is_best = False
    utils.save_checkpoint(model, config.path, is_best)
    # lr_scheduler.step()

    print("")
  logger.info("Phase3 Over!")

  with open(os.path.join(config.path, 'result_train.json'), 'w') as f:
      json.dump(result_train, f)
  with open(os.path.join(config.path, 'result_valid.json'), 'w') as f:
      json.dump(result_valid, f)
  logger.info("Final best Acc_Top1@1 = {:.4%}".format(best_top1))


def train(train_loader, model, optimizer, lr_scheduler, criterion, epoch, epochs):#, alpha_optimizer, lr_scheduler_alpha):
  top1_class1 = utils.AverageMeter()
  top5_class1 = utils.AverageMeter()
  top1_class2 = utils.AverageMeter()
  top5_class2 = utils.AverageMeter()
  top1_class3 = utils.AverageMeter()
  top5_class3 = utils.AverageMeter()
  top1_class4 = utils.AverageMeter()
  top5_class4 = utils.AverageMeter()
  losses = utils.AverageMeter()

  # cur_step = epoch*len(train_loader)
  cur_lr = optimizer.param_groups[0]['lr']
  logger.info("Epoch {} LR {}".format(epoch, cur_lr))
  writer.add_scalar('train/lr', cur_lr, epoch)

  model.train()

  # for step, (data, gt_order, gt_family, gt_genus, gt_class) in enumerate(train_loader):
  for step, (X,y) in enumerate(train_loader):
      X, y_class = X.to(device), y.to(device)
      N = X.size(0)

      optimizer.zero_grad()
      logits1, logits2, logits3, logits4 = model(X, device)#, y_class)
      # logits1 = model(X, device)
      loss1 = criterion(logits1, y_class)
      # loss2 = criterion(logits2, y_class)
      # loss3 = criterion(logits3, y_class)
      # loss3 = criterion(logits3, y_class.unsqueeze(1).repeat(1, 7).view(-1))
      # loss4 = criterion(logits4, y_class)
      loss = loss1
      loss.backward()
      # gradient clipping
      nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      lr_scheduler.step()
      prec1_class1, prec5_class1 = utils.accuracy(logits1, y_class, topk=(1, 5))
      prec1_class2, prec5_class2 = utils.accuracy(logits2, y_class, topk=(1, 5))
      prec1_class3, prec5_class3 = utils.accuracy(logits2, y_class, topk=(1, 5))
      prec1_class4, prec5_class4 = utils.accuracy(logits2, y_class, topk=(1, 5))

      losses.update(loss.item(), N)
      top1_class1.update(prec1_class1.item(), N)
      top5_class1.update(prec5_class1.item(), N)
      top1_class2.update(prec1_class2.item(), N)
      top5_class2.update(prec5_class2.item(), N)
      top1_class3.update(prec1_class3.item(), N)
      top5_class3.update(prec5_class3.item(), N)
      top1_class4.update(prec1_class4.item(), N)
      top5_class4.update(prec5_class4.item(), N)

      if step % config.print_freq == 0 or step == len(train_loader)-1:
          logger.info(
              "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f}"
              "Prec@(cla)({top1_class1.avg:.3%}, {top5_class1.avg:.3%})"
              "Prec@(cla)({top1_class2.avg:.3%}, {top5_class2.avg:.3%})"
              "Prec@(cla)({top1_class3.avg:.3%}, {top5_class3.avg:.3%})"
              "Prec@(cla)({top1_class4.avg:.3%}, {top5_class4.avg:.3%})".format(
                  epoch, epochs, step, len(train_loader)-1, losses=losses,
                  top1_class1=top1_class1, top5_class1=top5_class1,
                  top1_class2=top1_class2, top5_class2=top5_class2,
                  top1_class3=top1_class3, top5_class3=top5_class3,
                  top1_class4=top1_class4, top5_class4=top5_class4))

  logger.info("Train: [{:3d}/{}] Final Prec/cla1@1 {:.4%},{:.4%} \
    Final Prec/cla2@1 {:.4%},{:.4%} Final Prec/cla3@1 {:.4%},{:.4%} Final Prec/cla4@1 {:.4%},{:.4%}" \
    .format(epoch, epochs, top1_class1.avg, top5_class1.avg, top1_class2.avg, top5_class2.avg,
      top1_class3.avg, top5_class3.avg, top1_class4.avg, top5_class4.avg))
  return top1_class1.avg, top5_class1.avg, top1_class2.avg, top5_class2.avg, top1_class3.avg, top5_class3.avg, top1_class4.avg, top5_class4.avg, losses.avg




def validate(valid_loader, model, criterion, epoch, epochs):#, cur_step):
  top1_class1 = utils.AverageMeter()
  top5_class1 = utils.AverageMeter()
  top1_class2 = utils.AverageMeter()
  top5_class2 = utils.AverageMeter()
  top1_class3 = utils.AverageMeter()
  top5_class3 = utils.AverageMeter()
  top1_class4 = utils.AverageMeter()
  top5_class4 = utils.AverageMeter()
  losses = utils.AverageMeter()

  model.eval()

  with torch.no_grad():
      # for step, (data, gt_order, gt_family, gt_genus, gt_class) in enumerate(valid_loader):
      for step, (X, y) in enumerate(valid_loader):
          X, y_class = X.to(device), y.to(device)
          N = X.size(0)
          logits1, logits2, logits3, logits4 = model(X, device)#, y_class)
          # logits1 = model(X, device)
          loss1 = criterion(logits1, y_class)
          # loss2 = criterion(logits2, y_class)
          # loss3 = criterion(logits3, y_class)
          # loss3 = criterion(logits3, y_class.unsqueeze(1).repeat(1, 7).view(-1))
          # loss4 = criterion(logits4, y_class)
          loss = loss1
          prec1_class1, prec5_class1 = utils.accuracy(logits1, y_class, topk=(1, 5))
          prec1_class2, prec5_class2 = utils.accuracy(logits2, y_class, topk=(1, 5))
          prec1_class3, prec5_class3 = utils.accuracy(logits2, y_class, topk=(1, 5))
          prec1_class4, prec5_class4 = utils.accuracy(logits2, y_class, topk=(1, 5))

          losses.update(loss.item(), N)
          top1_class1.update(prec1_class1.item(), N)
          top5_class1.update(prec5_class1.item(), N)
          top1_class2.update(prec1_class2.item(), N)
          top5_class2.update(prec5_class2.item(), N)
          top1_class3.update(prec1_class3.item(), N)
          top5_class3.update(prec5_class3.item(), N)
          top1_class4.update(prec1_class4.item(), N)
          top5_class4.update(prec5_class4.item(), N)

          if step % config.print_freq == 0 or step == len(valid_loader)-1:
            logger.info(
              "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f}"
              "Prec@(cla)({top1_class1.avg:.3%}, {top5_class1.avg:.3%})"
              "Prec@(cla)({top1_class2.avg:.3%}, {top5_class2.avg:.3%})"
              "Prec@(cla)({top1_class3.avg:.3%}, {top5_class3.avg:.3%})"
              "Prec@(cla)({top1_class4.avg:.3%}, {top5_class4.avg:.3%})".format(
                  epoch, epochs, step, len(valid_loader)-1, losses=losses,
                  top1_class1=top1_class1, top5_class1=top5_class1,
                  top1_class2=top1_class2, top5_class2=top5_class2,
                  top1_class3=top1_class3, top5_class3=top5_class3,
                  top1_class4=top1_class4, top5_class4=top5_class4))

  logger.info("Valid: [{:3d}/{}] Final Prec/cla1@1 {:.4%},{:.4%} \
    Final Prec/cla2@1 {:.4%},{:.4%} Final Prec/cla3@1 {:.4%},{:.4%} Final Prec/cla4@1 {:.4%},{:.4%}" \
    .format(epoch, epochs, top1_class1.avg, top5_class1.avg, top1_class2.avg, top5_class2.avg,
      top1_class3.avg, top5_class3.avg, top1_class4.avg, top5_class4.avg))
  return top1_class1.avg, top5_class1.avg, top1_class2.avg, top5_class2.avg, top1_class3.avg, top5_class3.avg, top1_class4.avg, top5_class4.avg, losses.avg


if __name__ == '__main__':
  main()(torch)