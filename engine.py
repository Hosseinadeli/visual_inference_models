# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import copy

import torch
from tqdm import tqdm


import util.misc as utils

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
                       
from models.matcher import build_matcher

import numpy as np
from scipy.stats import pearsonr as corr

import wandb


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    print_freq = 100
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=print_freq, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=print_freq))  #, fmt='{value:.2f}'
    header = 'Epoch: [{}]'.format(epoch)
    

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        #samples = torch.cat((samples,samples,samples), dim=1)
        
        samples = tuple(samples.cuda())
#         print('samples')
#         print(len(samples))
        samples = nested_tensor_from_tensor_list(samples)
    
        '''           
        targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:

         "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                   objects in the target) containing the class labels

        '''
        # TODO there may be a better way to do this 
        if type(targets) is dict:
            targets = [dict(zip(targets,t)) for t in zip(*targets.values())]
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

        else:
            target = {}
            target["labels"] = targets.cuda()
            targets = [target]


        outputs = model(samples)
        loss_dict = criterion(outputs, samples, targets)
        weight_dict = criterion.weight_dict
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
        

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value)  #, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        metric_logger.update(class_error=loss_value) #loss_dict_reduced['loss_recon']
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate_reconstruction(model, criterion, data_loader, args, lh_challenge_rois=None, rh_challenge_rois=None, return_all=False):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('recon_error', utils.SmoothedValue(window_size=100)) #, fmt='{value:.2f}'
    header = 'Test:'

    corr_all = 0
    total_samples = 0
    # test_matcher = build_matcher(args)
    all_resps = []
    device = args.device 
    
    batch_corr = []
    lh_f_pred_val = []
    rh_f_pred_val = []
    
    lh_fmri_val = []
    rh_fmri_val = []
    
    for samples, targets in metric_logger.log_every(data_loader, 100, header):

        samples = tuple(samples.cuda())
        samples = nested_tensor_from_tensor_list(samples)
        '''           
        targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:

         "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                   objects in the target) containing the class labels

        '''

        if type(targets) is dict:
            targets = [dict(zip(targets,t)) for t in zip(*targets.values())]
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

        else:
            target = {}
            target["labels"] = targets.cuda()
            targets = [target]
        
        outputs = model(samples)
    
        loss_dict = criterion(outputs, samples, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values())) #,
                             # **loss_dict_reduced_scaled,
                             # **loss_dict_reduced_unscaled)
                
        metric_logger.update(recon_error=loss_dict_reduced['loss_recon'])
        
        recon_loss = loss_dict_reduced['loss_recon']
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    print(f'loss: {recon_loss}')
    
    return recon_loss


@torch.no_grad()
def evaluate_nsd(model, criterion, data_loader, args, lh_challenge_rois=None, rh_challenge_rois=None, return_all=False):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=100)) #, fmt='{value:.2f}'
    header = 'Test:'

    corr_all = 0
    total_samples = 0
    # test_matcher = build_matcher(args)
    all_resps = []
    device = args.device 
    
    batch_corr = []
    lh_f_pred_val = []
    rh_f_pred_val = []
    
    lh_fmri_val = []
    rh_fmri_val = []

    for samples, targets in metric_logger.log_every(data_loader, 100, header):
        #samples = torch.cat((samples,samples,samples), dim=1)
        
        samples = tuple(samples.cuda())
        samples = nested_tensor_from_tensor_list(samples)

        if type(targets) is dict:
            targets = [dict(zip(targets,t)) for t in zip(*targets.values())]
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]


        outputs = model(samples)
    
        lh_f = targets[0]['lh_f']
        rh_f = targets[0]['rh_f']

        lh_fmri_val.append(lh_f.cpu().numpy())
        rh_fmri_val.append(rh_f.cpu().numpy())

        lh_f_pred = outputs['lh_f_pred']
        rh_f_pred = outputs['rh_f_pred']
        
        if args.readout_res != 'hemis':
            lh_f_pred = outputs['lh_f_pred'][:,:,:args.roi_nums]
            rh_f_pred = outputs['rh_f_pred'][:,:,:args.roi_nums]
        
            lh_challenge_rois_b = torch.tile(lh_challenge_rois[:,:,None], (1,1,lh_f_pred.shape[0])).permute(2,1,0)
            rh_challenge_rois_b = torch.tile(rh_challenge_rois[:,:,None], (1,1,rh_f_pred.shape[0])).permute(2,1,0)

            lh_f_pred = torch.sum(torch.mul(lh_challenge_rois_b, lh_f_pred), dim=2)
            rh_f_pred = torch.sum(torch.mul(rh_challenge_rois_b, rh_f_pred), dim=2)
        
        lh_f_pred_val.append(lh_f_pred.cpu().numpy())
        rh_f_pred_val.append(rh_f_pred.cpu().numpy())
    
        loss_dict = criterion(outputs, samples, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values())) #,
                             # **loss_dict_reduced_scaled,
                             # **loss_dict_reduced_unscaled)
                
        metric_logger.update(class_error=loss_dict_reduced['loss_mse_fmri'])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    return np.concatenate(lh_f_pred_val, axis=0), np.concatenate(rh_f_pred_val, axis=0), np.concatenate(lh_fmri_val, axis=0), np.concatenate(rh_fmri_val, axis=0), loss_dict_reduced['loss_mse_fmri']



@torch.no_grad()
def evaluate_classification(model, criterion, data_loader, args, lh_challenge_rois=None, rh_challenge_rois=None, return_all=False):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=100)) #, fmt='{value:.2f}'
    header = 'Test:'

    corr_all = 0
    total_samples = 0
    # test_matcher = build_matcher(args)
    all_resps = []
    device = args.device 
    
    batch_corr = []
    lh_f_pred_val = []
    rh_f_pred_val = []
    
    lh_fmri_val = []
    rh_fmri_val = []
    
    for samples, targets in metric_logger.log_every(data_loader, 100, header):

        samples = tuple(samples.cuda())

        samples = nested_tensor_from_tensor_list(samples)
    
        '''           
        targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:

         "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                   objects in the target) containing the class labels

        '''

        if type(targets) is dict:
            targets = [dict(zip(targets,t)) for t in zip(*targets.values())]
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
            labels = targets[0]['labels'].cuda()

        else:
            target = {}
            labels = targets.cuda()
            target["labels"] = targets.cuda()
            targets = [target]
        
        
        # targets = [dict(zip(targets,t)) for t in zip(*targets.values())]
        # targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

 #         targets = [{'labels':torch.tensor([y_t])} for y_t in tuple(targets)]  # y_t.topk(2)[1]
#         targets = [{k: v.cuda() for k, v in t.items()} for t in targets]       
        
#         targets = [dict(zip(targets,t)) for t in zip(*targets.values())]
#         targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

#         lh_fmri_val.append(lh_f.numpy())
#         rh_fmri_val.append(rh_f.numpy())
        
#         targets =  {'lh_f':lh_f.cuda(), 'rh_f':rh_f.cuda()}
        
#         for t in range(len(targets)):
#             boxes = targets[t]['boxes']
#             targets[t]['boxes']= boxes[boxes.sum(dim=1) != 0]

        outputs = model(samples)
    
    
#         lh_f = targets['lh_f']
#         rh_f = targets['rh_f']
        
                
#         lh_f_pred = outputs['lh_f_pred'][:,:,:args.roi_nums]
#         rh_f_pred = outputs['rh_f_pred'][:,:,:args.roi_nums]
        
#         if args.readout_res != 'hemis':
        
#             lh_challenge_rois_b = torch.tile(lh_challenge_rois[:,:,None], (1,1,lh_f_pred.shape[0])).permute(2,1,0)
#             rh_challenge_rois_b = torch.tile(rh_challenge_rois[:,:,None], (1,1,rh_f_pred.shape[0])).permute(2,1,0)

#             lh_f_pred = torch.sum(torch.mul(lh_challenge_rois_b, lh_f_pred), dim=2)
#             rh_f_pred = torch.sum(torch.mul(rh_challenge_rois_b, rh_f_pred), dim=2)
        
#         lh_f_pred_val.append(lh_f_pred.cpu().numpy())
#         rh_f_pred_val.append(rh_f_pred.cpu().numpy())
        
        
        
        
#         s_lh_corr = np.zeros(lh_f_pred.shape[0])
#         s_rh_corr = np.zeros(lh_f_pred.shape[0])
        
#         for s in range(lh_f_pred.shape[0]):
#             s_lh_corr = corr(lh_f[s].cpu().numpy(), lh_f_pred[s].cpu().numpy())[0]
#             s_rh_corr = corr(rh_f[s].cpu().numpy(), rh_f_pred[s].cpu().numpy())[0]
            
# #             print(s_lh_corr)
# #             print(s_rh_corr)
            
#             batch_corr.append(s_lh_corr)
#             batch_corr.append(s_rh_corr)

#         batch_lh_corr = np.mean(s_lh_corr) 
#         batch_rh_corr = np.mean(s_rh_corr) 
    
        loss_dict = criterion(outputs, samples, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values())) #,
                             # **loss_dict_reduced_scaled,
                             # **loss_dict_reduced_unscaled)
                
                
        metric_logger.update(class_error=loss_dict_reduced['loss_ce'])
        
#         print(y.shape)
        
#         yy = y.topk(2, sorted=False)
#         print(yy[1])
        
#         indices = test_matcher(outputs, targets)
#         m = [indices[i][1] for i in range(len(indices))]
#         m = torch.stack(m)

        pred_logits = outputs['pred_logits']

        P = pred_logits.topk(1,dim=1)[1].flatten(0)
        
#         print(f'P.shape : {P.shape}')
#         print(P)
#         print(labels)
        
        corr = 1* (labels == P)
        
#         all_resps.append(corr)
#         corr = torch.sum(torch.prod(corr,1))
        
        corr_all = corr_all + torch.sum(corr)
        total_samples = total_samples + len(labels)
        
#         P = torch.gather(P, 1, m.to(args.device))

#         corr = 1* (y.topk(2, sorted=False)[1].to(args.device) == P)
        
        #indices = test_matcher(samples, outputs, targets)
        # indices = [(torch.as_tensor([0], dtype=torch.int64), torch.as_tensor([0], dtype=torch.int64)) for i in range(len(targets))]
#         indices = [(torch.as_tensor([0], dtype=torch.int64), torch.as_tensor([0], dtype=torch.int64)) for i in range(len(targets))]
#         m_p = [indices[i][0] for i in range(len(indices))]
#         m_y = [indices[i][1] for i in range(len(indices))]
#         m_p = torch.stack(m_p)
#         m_y = torch.stack(m_y)

#         pred_logits = outputs['pred_logits']
#         P = pred_logits.topk(1,dim=1)[1].flatten(1)

#         P = torch.gather(P, 1, m_p.to(args.device))

# #         y_b = y.topk(2, sorted=False)[1].to(args.device)
# #         y_b = torch.gather(y_b, 1, m_y.to(args.device))
        
#         y_b = y #[:,None]
        
# #         print(P.shape)
# #         print(y_b.shape)
# #         print(P)
# #         print(y_b)
        

#         corr = 1* (y_b == P)
        
#         all_resps.append(corr)
#         corr = torch.sum(torch.prod(corr,1))
        
        
#         corr_all = corr_all + corr
#         total_samples = total_samples + len(y)
        

#         orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
#         results = postprocessors['bbox'](outputs, orig_target_sizes)
#         if 'segm' in postprocessors.keys():
#             target_sizes = torch.stack([t["size"] for t in targets], dim=0)
#             results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
#         res = {target['image_id'].item(): output for target, output in zip(targets, results)}
#         if coco_evaluator is not None:
#             coco_evaluator.update(res)

#         if panoptic_evaluator is not None:
#             res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
#             for i, target in enumerate(targets):
#                 image_id = target["image_id"].item()
#                 file_name = f"{image_id:012d}.png"
#                 res_pano[i]["image_id"] = image_id
#                 res_pano[i]["file_name"] = file_name

#             panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    
    acc = corr_all / total_samples
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    print(f'acc: {acc}')
    
    return acc
    
#     batch_corr = np.array(batch_corr)
#     #batch_corr = batch_corr[~np.isnan(batch_corr)]
#     batch_corr = np.nan_to_num(batch_corr, copy=True, nan=0.0, posinf=None, neginf=None)
    
    #return np.concatenate(lh_f_pred_val, axis=0), np.concatenate(rh_f_pred_val, axis=0), np.concatenate(lh_fmri_val, axis=0), np.concatenate(rh_fmri_val, axis=0), loss_dict_reduced['loss_mse_fmri']


@torch.no_grad()
def evaluate_class(model, criterion, data_loader, args=None, lh_challenge_rois=None, rh_challenge_rois=None, return_all=False):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=100)) #, fmt='{value:.2f}'
    header = 'Test:'

    corr_all = 0
    total_samples = 0

    record = {'top1': 0, 'top5': 0, 'loss': 0}
    
    for samples, targets in metric_logger.log_every(data_loader, 100, header):
        #samples = torch.cat((samples,samples,samples), dim=1)
        samples = tuple(samples.cuda())
        samples = nested_tensor_from_tensor_list(samples)
    
        '''           
        targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:

         "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                   objects in the target) containing the class labels

        '''

        # TODO fix this or make it an option for just custom loaders 
        if type(targets) is dict:
            targets = [dict(zip(targets,t)) for t in zip(*targets.values())]
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
            labels = targets[0]['labels'].cuda()

        else:
            target = {}
            labels = targets.cuda()
            target["labels"] = targets.cuda()
            targets = [target]

        outputs = model(samples)
    
    
        loss_dict = criterion(outputs, samples, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values())) #,
                             # **loss_dict_reduced_scaled,
                             # **loss_dict_reduced_unscaled)
                
                
        metric_logger.update(class_error=loss_dict_reduced['loss_ce'])

        record['loss'] += loss_dict_reduced['loss_ce']

        pred_logits = outputs['pred_logits']

        p1, p5 = accuracy(pred_logits, labels, topk=(1, 5))
        record['top1'] += p1
        record['top5'] += p5

        P = pred_logits.topk(1,dim=1)[1].flatten(0)
        corr = 1* (labels == P)
        corr_all = corr_all + torch.sum(corr)
        total_samples = total_samples + len(labels)

    acc = corr_all / total_samples
        
    for key in record:
        record[key] /= total_samples
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    print(f'acc: {acc}')
    print('top 1 and top 5 accuracies', record)
    
    return record


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = [correct[:k].sum().item() for k in topk]
        return res


@torch.no_grad()
def test_vggface2(model, criterion, data_loader):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=100)) #, fmt='{value:.2f}'
    header = 'Test:'

    corr_all = 0
    total_samples = 0

    record = {'top1': 0, 'top5': 0, 'loss': 0}
    
    for samples, targets in metric_logger.log_every(data_loader, 100, header):
        #samples = torch.cat((samples,samples,samples), dim=1)
        samples = tuple(samples.cuda())
        samples = nested_tensor_from_tensor_list(samples)
    
        '''           
        targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:

         "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                   objects in the target) containing the class labels

        '''

        # TODO fix this or make it an option for just custom loaders 
        if type(targets) is dict:
            targets = [dict(zip(targets,t)) for t in zip(*targets.values())]
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
            labels = targets[0]['labels'].cuda()

        else:
            target = {}
            labels = targets.cuda()
            target["labels"] = targets.cuda()
            targets = [target]

        outputs = model(samples)
    
    
        loss_dict = criterion(outputs, samples, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values())) #,
                             # **loss_dict_reduced_scaled,
                             # **loss_dict_reduced_unscaled)
                
                
        metric_logger.update(class_error=loss_dict_reduced['loss_ce'])

        record['loss'] += loss_dict_reduced['loss_ce']

        pred_logits = outputs['pred_logits']

        p1, p5 = accuracy(pred_logits, labels, topk=(1, 5))
        record['top1'] += p1
        record['top5'] += p5

        P = pred_logits.topk(1,dim=1)[1].flatten(0)
        corr = 1* (labels == P)
        corr_all = corr_all + torch.sum(corr)
        total_samples = total_samples + len(labels)

    acc = corr_all / total_samples
        
    for key in record:
        record[key] /= total_samples
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    print(f'acc: {acc}')
    print('top 1 and top 5 accuracies', record)
    
    return record



@torch.no_grad()
def test_nsd(model, criterion, data_loader, args, lh_challenge_rois, rh_challenge_rois, return_all=False):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=100)) #, fmt='{value:.2f}'
    header = 'Test:'

    corr_all = 0
    total_samples = 0
    all_resps = []
    device = args.device 
    
    lh_f_pred_all = []
    rh_f_pred_all = []
    
    for i,samples in tqdm(enumerate(data_loader), total=len(data_loader)):

        samples = tuple(samples.cuda())
        samples = nested_tensor_from_tensor_list(samples)

        outputs = model(samples)
        
        lh_f_pred = outputs['lh_f_pred']
        rh_f_pred = outputs['rh_f_pred']
        
        if args.readout_res != 'hemis':
            lh_f_pred = outputs['lh_f_pred'][:,:,:args.roi_nums]
            rh_f_pred = outputs['rh_f_pred'][:,:,:args.roi_nums]
        
            lh_challenge_rois_b = torch.tile(lh_challenge_rois[:,:,None], (1,1,lh_f_pred.shape[0])).permute(2,1,0)
            rh_challenge_rois_b = torch.tile(rh_challenge_rois[:,:,None], (1,1,rh_f_pred.shape[0])).permute(2,1,0)

            lh_f_pred = torch.sum(torch.mul(lh_challenge_rois_b, lh_f_pred), dim=2)
            rh_f_pred = torch.sum(torch.mul(rh_challenge_rois_b, rh_f_pred), dim=2)
        
        lh_f_pred_all.append(lh_f_pred.cpu().numpy())
        rh_f_pred_all.append(rh_f_pred.cpu().numpy())
        
    return np.vstack(lh_f_pred_all), np.vstack(rh_f_pred_all)

