# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-8-9
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import sys
import time
import random
import logging
import traceback

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.util import *
from lib.metrics import *
from lib.dataset import MLDataset
from models.factory import create_model


class Trainer(object):
    def __init__(self, cfg, world_size, rank):
        super(Trainer, self).__init__()
        self.distributed = world_size > 1
        batch_size = cfg.batch_size // world_size if self.distributed else cfg.batch_size

        train_dataset = MLDataset(cfg.train_path, cfg, training=True)
        val_dataset = MLDataset(cfg.test_path, cfg, training=False)
        if self.distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        else:
            self.train_sampler = val_sampler = None
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(self.train_sampler is None),
                                       num_workers=4, sampler=self.train_sampler)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=val_sampler)

        torch.cuda.set_device(rank)
        self.model = create_model(cfg.model, cfg=cfg)
        self.model.cuda(rank)
        if cfg.pretrained_model is not None:
            pretrained_state_dict = torch.load(cfg.pretrained_model)
            pretrained_state_dict = {k[7:]:v for k, v in pretrained_state_dict.items() if not k.startswith('module.fc')}
            self.model.load_state_dict(pretrained_state_dict, strict=False)
            print('loaded pretrained model from {}!'.format(cfg.pretrained_model))
        self.ema_model = ModelEma(self.model, decay=cfg.ema_decay)
        if self.distributed:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[rank], find_unused_parameters=True)

        if cfg.true_weight_decay:
            parameters = add_weight_decay(self.model, cfg.weight_decay)
            cfg.weight_decay = 0.0
        else:
            parameters = self.model.parameters()
        self.optimizer = get_optimizer(parameters, cfg)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, cfg, steps_per_epoch=len(self.train_loader))
        self.criterion = get_loss_fn(cfg)
        self.voc_mAP = VOCmAP(cfg.num_classes, year='2012', ignore_path=cfg.ignore_path)
        self.voc_ema_mAP = VOCmAP(cfg.num_classes, year='2012', ignore_path=cfg.ignore_path)

        self.cfg = cfg
        self.best_mAP = 0
        self.global_step = 0
        self.notdist_or_rank0 = (not self.distributed) or (self.distributed and rank == 0)
        if self.notdist_or_rank0:
            self.logger = get_logger(cfg.log_path, __name__)
            self.logger.info(train_dataset.transform)
            self.logger.info(val_dataset.transform)
            self.writer = SummaryWriter(log_dir=cfg.exp_dir)

    def run(self):
        patience = 0
        for epoch in range(self.cfg.max_epochs):
            if self.distributed:
                self.train_sampler.set_epoch(epoch)
            self.train(epoch)
            mAP = self.validation(epoch)
            if self.best_mAP < mAP and self.notdist_or_rank0:
                torch.save(self.ema_model.state_dict(), self.cfg.ckpt_ema_best_path)
                self.best_mAP = mAP
                patience = 0
            else:
                patience += 1
            if self.cfg.estop and patience > 2:
                break

        if self.notdist_or_rank0:
            self.logger.info('\ntraining over, best validation score: {} mAP'.format(self.best_mAP))

    def train(self, epoch):
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp)
        self.model.train()
        for _, batch in enumerate(self.train_loader):
            batch_begin = time.time()
            imgs = batch['img'].cuda()
            targets = batch['target'].cuda()
            with torch.cuda.amp.autocast(enabled=self.cfg.amp):
                ret = self.model(imgs, y=targets)
            
            loss = ret['ce_loss']
            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            dur = time.time() - batch_begin
            self.lr_scheduler.step()
            self.ema_model.update(self.model)

            if self.global_step % (len(self.train_loader) // 6) == 0 and self.notdist_or_rank0:
                lr = get_lr(self.optimizer)
                self.writer.add_scalar('Loss/train', loss, self.global_step)
                self.writer.add_scalar('lr', lr, self.global_step)
                self.logger.info('TRAIN [epoch {}] loss: {:4f} lr:{:.6f} time:{:.4f}'.format(epoch, loss, lr, dur))

            self.global_step += 1

    @torch.no_grad()
    def validation(self, epoch):
        self.model.eval()
        self.ema_model.eval()
        self.voc_mAP.reset()
        self.voc_ema_mAP.reset()
        for batch in self.val_loader:
            imgs = batch['img'].cuda()
            targets = batch['target'].cuda()
            
            logits = self.model(imgs)['logits']
            scores = torch.sigmoid(logits)
            logits = self.ema_model(imgs)['logits']
            ema_scores = torch.sigmoid(logits)
            if self.distributed:
                scores = concat_all_gather(scores)
                ema_scores = concat_all_gather(ema_scores)
                targets = concat_all_gather(targets)

            targets = targets.cpu().numpy()
            scores = scores.detach().cpu().numpy()
            self.voc_mAP.update(scores, targets)
            ema_scores = ema_scores.detach().cpu().numpy()
            self.voc_ema_mAP.update(ema_scores, targets)
        
        if self.distributed:
            dist.barrier()
        _, mAP = self.voc_mAP.compute()
        _, ema_mAP = self.voc_ema_mAP.compute()
        if self.notdist_or_rank0:
            self.writer.add_scalar('mAP/val', mAP, self.global_step)
            self.writer.add_scalar('ema_mAP/val', ema_mAP, self.global_step)
            self.logger.info("VALID [epoch {}] mAP: {:.4f} ema_mAP: {:.4f} best mAP: {:.4f}"
                             .format(epoch, mAP, ema_mAP, self.best_mAP))

        return ema_mAP


def main_worker(local_rank, ngpus_per_node, cfg, port=None):
    world_size = ngpus_per_node  # only single node is enough.
    if ngpus_per_node > 1:
        init_method = 'tcp://127.0.0.1:{}'.format(port)
        dist.init_process_group(backend='nccl', init_method=init_method, world_size=world_size, rank=local_rank)
    trainer = Trainer(cfg, world_size, local_rank)
    trainer.run()


if __name__ == "__main__":
    args = get_args()
    cfg = prepare_env(args, sys.argv)

    try:
        ngpus_per_node = torch.cuda.device_count()
        if ngpus_per_node > 1:
            port = 12345 + random.randint(0, 1000)
            setup_seed(cfg.seed)
            mp.spawn(main_worker, args=(ngpus_per_node, cfg, port,), nprocs=ngpus_per_node)
        else:
            setup_seed(cfg.seed)
            main_worker(0, ngpus_per_node, cfg)
    except (Exception, KeyboardInterrupt):
        print(traceback.format_exc())
        if not os.path.exists(cfg.ckpt_ema_best_path):
            clear_exp(cfg.exp_dir)
