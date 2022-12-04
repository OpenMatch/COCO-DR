from collections import defaultdict
import torch
import torch.nn as nn
import numpy as np 
import math
import logging 
import torch.distributed as dist
import random 
import copy 

class DROGreedyLoss(nn.Module):
    def __init__(self, args, n_groups, alpha, eps, ema = 0.1, weight_ema=False, weight_cutoff=True, fraction=None):
        super(DROGreedyLoss, self).__init__()
        self.args = args
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.alpha = alpha
        self.ema = ema
        self.eps = eps
        self.weight_cutoff = weight_cutoff
        self.weight_ema = weight_ema
        self.n_groups = n_groups
       
        self.id2group = {str(x): f"group{x}" for x in range(n_groups)}
        self.n_splits = n_groups
    

        self.register_buffer('h_fun', torch.ones(self.n_splits))
        self.register_buffer('sum_losses', torch.zeros(self.n_splits))  # historical loss sum over category
        if fraction is not None:
            self.register_buffer('fraction', torch.from_numpy(fraction).float())
            self.register_buffer('count_cat', None)
        else:
            self.register_buffer('count_cat', torch.ones(self.n_splits))

        self.idx_dict = defaultdict(lambda: len(self.idx_dict))  # autoincrementing index.
        for i in range(self.n_groups):
            _ = self.idx_dict['[' + str(i) + ']']

    def reset(self):
        self.h_fun.fill_(1.)
        self.sum_losses.fill_(0.)
        if self.count_cat is not None:
            self.count_cat.fill_(1.)

    def reset_loss(self):
        self.h_fun.fill_(1.)
        self.sum_losses.fill_(0.)

    def forward(self, losses, g, w = None):
        if w is not None:
            losses = losses * w
        batch_size = losses.size(0)
        one_vec = losses.new_ones(batch_size)
        zero_vec = losses.new_zeros(self.n_groups)
        s = g

        # n_groups
        gdro_losses = zero_vec.scatter_add(0, s, losses)
        robust_loss = (gdro_losses * self.h_fun).sum() / batch_size

        with torch.no_grad():
            if self.training:
                # aggregate different s and losses
                s_agg = self.gather_tensors(s.squeeze().contiguous())[0]         
                losses_agg = self.gather_tensors(losses.squeeze().contiguous())[0]    
                one_vec = s_agg.new_ones(s_agg.size(0))            
                gdro_counts_agg = zero_vec.scatter_add(0, s_agg, one_vec.float()).float() # every group's sample number in this batch
                gdro_losses_agg = zero_vec.scatter_add(0, s_agg, losses_agg).detach()
                ######## Combine results with different pids #########
                gdro_losses = gdro_losses_agg.div(gdro_counts_agg + (gdro_counts_agg == 0).float()) # group's loss
                
                ######################################################
                valid_idx = gdro_counts_agg.gt(0) # group exists in the current batch
                self.sum_losses[valid_idx] = self.sum_losses[valid_idx].mul(1 - self.ema).add(gdro_losses[valid_idx], alpha=self.ema)
                # sum_losses: total loss for all groups
                # count_cat: total number of samples for all groups
                if self.count_cat is not None:
                    self.count_cat.mul_(1 - self.ema).add_(gdro_counts_agg, alpha=self.ema)
                self.update_mw(weight_ema = self.weight_ema)
                
            zero_vec = losses.new_zeros(self.n_groups)
            one_vec = losses.new_ones(batch_size)
            group_counts = zero_vec.scatter_add(0, g, one_vec).float()
            group_losses = zero_vec.scatter_add(0, g, losses)
            group_losses.div_(group_counts + (group_counts == 0).float())
     
        return robust_loss, group_losses, group_counts


    def update_mw(self, weight_ema = False):
        # version that uses EMA. (sum_losses is EMA running loss, count_cat is EMA running sum)
        past_losses = self.sum_losses
        if self.count_cat is not None:
            past_frac = self.count_cat / self.count_cat.sum()
        else:
            past_frac = self.fraction
        sorted_losses, sort_id = torch.sort(past_losses, descending=True)
        sorted_frac = past_frac[sort_id]
        cutoff_count = torch.sum(torch.cumsum(sorted_frac, 0) < self.alpha)
        if cutoff_count == len(sorted_frac):
            cutoff_count = len(sorted_frac) - 1
        if weight_ema:
            h_fun_tmp = torch.ones(self.h_fun.size()).to(self.h_fun.device) * self.eps 
            h_fun_tmp[sort_id[:cutoff_count]] = 1.0 / self.alpha 
            leftover_mass = 1.0 - sorted_frac[:cutoff_count].sum().div(self.alpha)
            tiebreak_fraction = leftover_mass / sorted_frac[cutoff_count]  # check!
            h_fun_tmp[sort_id[cutoff_count]] = max(tiebreak_fraction, self.eps)
            if self.weight_cutoff:
                # h_fun_tmp.mul_(self.alpha)
                h_fun_tmp.clamp_(min = self.eps)
            if self.h_fun is not None:
                self.h_fun = self.h_fun * (1 - self.ema) + h_fun_tmp * self.ema
            else:
                self.h_fun = h_fun_tmp 
        else:
            self.h_fun = self.h_fun.new_full(self.h_fun.size(), self.eps) # minumum weight
            self.h_fun[sort_id[:cutoff_count]] = 1.0 / self.alpha 
            leftover_mass = 1.0 - sorted_frac[:cutoff_count].sum().div(self.alpha)
            tiebreak_fraction = leftover_mass / sorted_frac[cutoff_count]  # check!
            self.h_fun[sort_id[cutoff_count]] = max(tiebreak_fraction, self.eps)


    def output_state(self):
        h_fun = self.h_fun.detach().cpu().numpy()
        sum_loss = self.sum_losses.detach().cpu().numpy()
    
    def _gather_tensor(self, t: torch.Tensor):
        all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(all_tensors, t)
        all_tensors[self.args.local_rank] = t
        return all_tensors

    def gather_tensors(self, *tt: torch.Tensor):
        tt = [torch.cat(self._gather_tensor(t)) for t in tt]
        return tt     


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.

class iDROLoss(DROGreedyLoss):
    def __init__(self, args, n_groups, alpha, eps, ema, rho, reg = 0, weight_ema=False, weight_cutoff=True, fraction=None):
        super(iDROLoss, self).__init__(args, n_groups, alpha, eps, ema, weight_ema, weight_cutoff, fraction)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.alpha = alpha
        self.ema = ema
        self.eps = eps
        self.rho = rho
        self.reg = reg
        self.tol = 1e-5
        self.max_iter = 1200
        self.weight_cutoff = weight_cutoff
        self.para_name = {}

    def _params(self, model):
        params = []
        if self.args.model_size == 'large':
            select = ['layer.23', 'layer.22'] # (only use the last 2 layers mainly due to efficiency)
        else:
            select = ['layer.10', 'layer.11', 'layer.9'] # only use the last 3 layers 

        if len(self.para_name) == 0:
            for name, param in model.named_parameters():
                for s in select:
                    if (name.find(s) >= 0):
                        params.append(param)
                        self.para_name[name] = 1
                        break
        else:
            params = [param for (name, param) in model.named_parameters() if name in self.para_name]
        return params

    def _get_grad(self, params, gdro_losses_agg, gdro_counts_agg):
        all_grads = [None] * self.n_groups
        for li in range(self.n_groups):
            if gdro_counts_agg[li] > 0:
                grad = list(torch.autograd.grad(gdro_losses_agg[li], params, retain_graph=True))
                all_grads[li] = torch.cat([g.view(-1) for g in grad])
                assert all_grads[li] is not None   
                device = gdro_losses_agg.device 
                embedding_dim = all_grads[li].shape[0]
        for i in range(self.n_groups):
            if all_grads[i] is None:
                all_grads[i] = torch.zeros(embedding_dim).to(device)
        return all_grads
    
    def _gather_tensor(self, t):
        all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(all_tensors, t)
        all_tensors[self.args.local_rank] = t
        return all_tensors

    def gather_tensors(self, tt):
        tt = [torch.cat(self._gather_tensor(t)) for t in tt]
        return tt
   
    def forward(self, model, losses, g):
        batch_size = losses.size(0)
        one_vec = losses.new_ones(batch_size)
        zero_vec = losses.new_zeros(self.n_groups)
        s = g
        gdro_sum_losses = zero_vec.scatter_add(0, s, losses)
        if self.training:
            gdro_counts_agg = zero_vec.scatter_add(0, s, one_vec).float() # every group's sample number in this batch
            gdro_losses_agg = gdro_sum_losses.div(gdro_counts_agg + (gdro_counts_agg == 0).float()) # group's loss

        robust_loss = (gdro_losses_agg * self.h_fun).sum()

        gdro_loss_mask = (gdro_counts_agg > 0).detach().float()
        params = self._params(model)
        all_grads = self._get_grad(params, gdro_losses_agg, gdro_counts_agg)
        all_grads = torch.cat(all_grads).reshape(self.n_groups, -1) # (n_class * embedding_dim)
        dist.all_reduce(all_grads)
        all_grads = all_grads.detach() 
        
        grad_norm = torch.linalg.norm(all_grads, dim = -1, keepdim = True)
        all_grads = all_grads/(1e-12 + grad_norm)
        RTG = torch.matmul(all_grads, all_grads.T)
       
        _gl = torch.pow(gdro_losses_agg.detach().unsqueeze(-1), self.alpha)
        RTG = torch.mm(_gl, _gl.t()) * RTG
        _exp = self.rho * (torch.mean(RTG, dim = 0)) 
        
        _exp = _exp * gdro_loss_mask
        # to avoid overflow
        _exp -= _exp.max()

        weight = torch.exp(_exp)
        
        self.h_fun = torch.pow(self.h_fun, self.ema) * weight * ((gdro_counts_agg != 0).float())
        self.h_fun = self.h_fun / self.h_fun.sum()
        self.h_fun = torch.clamp(self.h_fun, min=self.eps)
        ####### end of the algo #########
        
        return robust_loss, gdro_losses_agg.detach(), gdro_counts_agg.detach()
