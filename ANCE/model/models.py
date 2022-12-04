import sys
sys.path += ['../']
import torch
from torch import nn
from transformers import (
    RobertaConfig,
    RobertaModel,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertModel,
    BertTokenizer,
    BertConfig,
    BertForSequenceClassification
)
import torch.distributed as dist
import numpy as np
import torch.nn.functional as F
from data.process_fn import triple_process_fn, triple2dual_process_fn
from .dro_loss import DROGreedyLoss, iDROLoss, AverageMeter, iDROLoss
import logging 
logger = logging.getLogger(__name__)

def mt_update(t_params, s_params, average="exponential", alpha=0.995, step=None):
    for (t_name, t_param), (s_name, s_param) in zip(t_params, s_params):
        if t_name != s_name:
            print("t_name != s_name: {} {}".format(t_name, s_name))
            raise ValueError
        param_new = s_param.data.to(t_param.device)
        if average == "exponential":
            t_param.data.add_( (1-alpha)*(param_new-t_param.data) )
        elif average == "simple":
            virtual_decay = 1 / float(step)
            diff = (param_new - t_param.data) * virtual_decay
            t_param.data.add_(diff)
            

class EmbeddingMixin:
    """
    Mixin for common functions in most embedding models. Each model should define its own bert-like backbone and forward.
    We inherit from RobertaModel to use from_pretrained 
    """
    def __init__(self, model_argobj):
        if model_argobj is None:
            self.use_mean = False
        else:
            self.use_mean = model_argobj.use_mean
        print("Using mean:", self.use_mean)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d

    def masked_mean_or_first(self, emb_all, mask):
        # emb_all is a tuple from bert - sequence output, pooler
        assert isinstance(emb_all, tuple)
        if self.use_mean:
            return self.masked_mean(emb_all[0], mask)
        else:
            return emb_all[0][:, 0]

    def query_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")

    def body_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")


class NLL(EmbeddingMixin):
    def forward_model(
                self,
                query_ids,
                attention_mask_q,
                input_ids_a=None,
                attention_mask_a=None,
                input_ids_b=None,
                attention_mask_b=None,
                is_query=True,
                group_ids = None,
            ):
        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)

        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)
               
        logit_matrix = torch.cat([(q_embs * a_embs).sum(-1).unsqueeze(1),
                                (q_embs * b_embs).sum(-1).unsqueeze(1)], dim=1)  # [B, 2]
           
        lsm = F.log_softmax(logit_matrix, dim=1)
        
        loss = -1.0 * lsm[:, 0]
        return_loss = loss
        accs = torch.argmax(logit_matrix, dim = 1)
        # loss, acc, group_losses, group_accs, group_counts = model(x, y, g, d, w)
        # print(loss.mean(), vat_loss)
        
        if group_ids is None:
            return return_loss, accs, logit_matrix
        else:
            return loss, accs, logit_matrix

class RobertaDot_NLL_LN(NLL, RobertaForSequenceClassification):
    """None
    Compress embedding to 200d, then computes NLL loss.
    """

    def __init__(self, config, model_argobj=None):
        NLL.__init__(self, model_argobj)
        RobertaForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768)
        self.norm = nn.LayerNorm(768)
        self.use_moco = False
        self.apply(self._init_weights)
        self.total = 0
        self.correct = 0

    def add_group_loss(self, n_groups, n_domains, alpha, eps, ema = 0.1, weight_ema = True):
        self.loss = DROGreedyLoss(n_groups, n_domains, alpha, eps, ema, weight_ema)
        self.n_groups = n_groups
        self.accum_loss = AverageMeter()
        self.accum_group_loss = [AverageMeter() for _ in range(n_groups)]

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)
    
    def forward(self,
            query_ids,
            attention_mask_q,
            input_ids_a=None,
            attention_mask_a=None,
            input_ids_b=None,
            attention_mask_b=None,
            is_query=True,
            noise_level = 0.0,
            group_ids = None,
            qids = None,
            weights = None):
        
        loss, train_acc, logits = self.forward_model(
            query_ids,
            attention_mask_q,
            input_ids_a,
            attention_mask_a,
            input_ids_b,
            attention_mask_b,
            is_query,
            noise_level,
            group_ids,)
        
        train_size = torch.LongTensor([train_acc.shape[0]]).to(train_acc.device)
        # print(train_size)
        torch.distributed.all_reduce(train_size, op=torch.distributed.reduce_op.SUM)
        acc = torch.sum(1-train_acc)
        torch.distributed.all_reduce(acc, op=torch.distributed.reduce_op.SUM)

        if group_ids is None:
            return loss, train_acc, logits
        else:
            robust_loss, group_losses, group_counts = self.loss(loss, group_ids, weights)
            self.accum_loss.update(robust_loss.item(), loss.size(0))
            for i in range(self.n_groups):
                self.accum_group_loss[i].update(group_losses[i].item(), group_counts[i].item())
            return robust_loss, train_acc, group_losses, group_counts

    def output_state(self):
        h_fun = {self.loss.id2group[str(i)]:  self.loss.h_fun.detach().cpu().numpy()[i] for i in range(self.n_groups)}
       
        sum_loss = {self.loss.id2group[str(i)]: self.accum_group_loss[i].avg for i in range(self.n_groups)}
        return h_fun, sum_loss        

##############################################################################
class BertDot_NLL_LN(NLL, BertForSequenceClassification):
    """None
    Compress embedding to 200d, then computes NLL loss.
    """

    def __init__(self, config, model_argobj=None):
        NLL.__init__(self, model_argobj)
        BertForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768)
        self.norm = nn.LayerNorm(768)
        self.use_moco = False
        self.apply(self._init_weights)
        self.total = 0
        self.correct = 0
        self.prob_diff = []
        self.dro_type = 'erm'

    def add_group_loss(self, args, n_groups, dro_type, alpha, eps, ema = 0.1, rho = 0.1, weight_ema = True):
        if dro_type == 'dro-greedy':
            self.dro_type = dro_type
            self.loss = DROGreedyLoss(args, n_groups, alpha, eps, ema, weight_ema)
        elif dro_type == 'idro':
            self.dro_type = dro_type
            self.loss = iDROLoss(args, n_groups, alpha, eps, ema, rho)
        else:
            logger.info("Warning! No training strategy selected")

        self.n_groups = n_groups
        self.accum_loss = AverageMeter()
        self.accum_group_loss = [AverageMeter() for _ in range(n_groups)]

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask)
        query1 = outputs1[0][:, 0]
        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)

    def forward(self,
            query_ids,
            attention_mask_q,
            input_ids_a=None,
            attention_mask_a=None,
            input_ids_b=None,
            attention_mask_b=None,
            is_query=True,
            group_ids = None,
            weights = None):
        
        loss, train_acc, logits = self.forward_model(
                query_ids,
                attention_mask_q,
                input_ids_a,
                attention_mask_a,
                input_ids_b,
                attention_mask_b,
                is_query,
                group_ids
            )
        
        train_size = torch.LongTensor([train_acc.shape[0]]).to(train_acc.device)
        torch.distributed.all_reduce(train_size, op=torch.distributed.reduce_op.SUM)
        self.total += train_size.item()
        if group_ids is None:
            loss = loss * weights
            loss = loss.mean()
            return loss, train_acc, logits
        else:
            if self.dro_type == 'idro':
                robust_loss, group_losses, group_counts = self.loss(self.bert, loss, group_ids)
            else:
                robust_loss, group_losses, group_counts = self.loss(loss, group_ids, weights)

            self.accum_loss.update(robust_loss.item(), loss.size(0))
            for i in range(self.n_groups):
                self.accum_group_loss[i].update(group_losses[i].item(), group_counts[i].item())
            # print({'loss': loss, 'h_fun': self.loss.h_fun, 'sum_loss':self.loss.sum_losses, 'count_cat' :self.loss.count_cat, 'robust_loss': robust_loss, 'gLOss':self.accum_group_loss, 'group_counts':group_counts})
            return robust_loss, train_acc, group_losses, group_counts

    def output_state(self):
        h_fun = {self.loss.id2group[str(i)]:  self.loss.h_fun.detach().cpu().numpy()[i] for i in range(self.n_groups)}
       
        sum_loss = {self.loss.id2group[str(i)]: self.accum_group_loss[i].avg for i in range(self.n_groups)}
        # sum_loss = self.accum_loss.detach().cpu().numpy()
        return h_fun, sum_loss   

    def _gather_tensor(self, t: torch.Tensor):
        all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(all_tensors, t)
        all_tensors[self.args.local_rank] = t
        return all_tensors

    def gather_tensors(self, *tt: torch.Tensor):
        tt = [torch.cat(self._gather_tensor(t)) for t in tt]
        return tt     

##############################################################################
# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


class NLL_MultiChunk(EmbeddingMixin):
    def forward(
            self,
            query_ids,
            attention_mask_q,
            input_ids_a=None,
            attention_mask_a=None,
            input_ids_b=None,
            attention_mask_b=None,
            is_query=True):
        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)

        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)

        [batchS, full_length] = input_ids_a.size()
        chunk_factor = full_length // self.base_len

        # special handle of attention mask -----
        attention_mask_body = attention_mask_a.reshape(
            batchS, chunk_factor, -1)[:, :, 0]  # [batchS, chunk_factor]
        inverted_bias = ((1 - attention_mask_body) * (-9999)).float()

        a12 = torch.matmul(
            q_embs.unsqueeze(1), a_embs.transpose(
                1, 2))  # [batch, 1, chunk_factor]
        logits_a = (a12[:, 0, :] + inverted_bias).max(dim=-
                                                      1, keepdim=False).values  # [batch]
        # -------------------------------------

        # special handle of attention mask -----
        attention_mask_body = attention_mask_b.reshape(
            batchS, chunk_factor, -1)[:, :, 0]  # [batchS, chunk_factor]
        inverted_bias = ((1 - attention_mask_body) * (-9999)).float()

        a12 = torch.matmul(
            q_embs.unsqueeze(1), b_embs.transpose(
                1, 2))  # [batch, 1, chunk_factor]
        logits_b = (a12[:, 0, :] + inverted_bias).max(dim=-
                                                      1, keepdim=False).values  # [batch]
        # -------------------------------------

        logit_matrix = torch.cat(
            [logits_a.unsqueeze(1), logits_b.unsqueeze(1)], dim=1)  # [B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0 * lsm[:, 0]
        return (loss.mean(),)


class RobertaDot_CLF_ANN_NLL_MultiChunk(NLL_MultiChunk, RobertaDot_NLL_LN):
    def __init__(self, config):
        RobertaDot_NLL_LN.__init__(self, config)
        self.base_len = 512

    def body_emb(self, input_ids, attention_mask):
        [batchS, full_length] = input_ids.size()
        chunk_factor = full_length // self.base_len

        input_seq = input_ids.reshape(
            batchS,
            chunk_factor,
            full_length //
            chunk_factor).reshape(
            batchS *
            chunk_factor,
            full_length //
            chunk_factor)
        attention_mask_seq = attention_mask.reshape(
            batchS,
            chunk_factor,
            full_length //
            chunk_factor).reshape(
            batchS *
            chunk_factor,
            full_length //
            chunk_factor)

        outputs_k = self.roberta(input_ids=input_seq,
                                 attention_mask=attention_mask_seq)

        compressed_output_k = self.embeddingHead(
            outputs_k[0])  # [batch, len, dim]
        compressed_output_k = self.norm(compressed_output_k[:, 0, :])

        [batch_expand, embeddingS] = compressed_output_k.size()
        complex_emb_k = compressed_output_k.reshape(
            batchS, chunk_factor, embeddingS)

        return complex_emb_k  # size [batchS, chunk_factor, embeddingS]


# --------------------------------------------------
ALL_MODELS = ()
# sum(
#     (
#         tuple(conf.pretrained_config_archive_map.keys())
#         for conf in (
#             RobertaConfig,
#         )
#     ),
#     (),
# )


default_process_fn = triple_process_fn


class MSMarcoConfig:
    def __init__(self, name, model, process_fn=default_process_fn, use_mean=True, tokenizer_class=RobertaTokenizer, config_class=RobertaConfig):
        self.name = name
        self.process_fn = process_fn
        self.model_class = model
        self.use_mean = use_mean
        self.tokenizer_class = tokenizer_class
        self.config_class = config_class


configs = [
    MSMarcoConfig(name="rdot_nll",
                model=RobertaDot_NLL_LN,
                use_mean=False,
                ),
    MSMarcoConfig(name="rdot_nll_multi_chunk",
                model=RobertaDot_CLF_ANN_NLL_MultiChunk,
                use_mean=False,
                ),
    MSMarcoConfig(name="rdot_nll_condenser",
                model=BertDot_NLL_LN,
                tokenizer_class=BertTokenizer,
                config_class=BertConfig,
                use_mean=False,
                ),
]

MSMarcoConfigDict = {cfg.name: cfg for cfg in configs}
