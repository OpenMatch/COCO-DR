import sys
sys.path += ['../']
import torch
from torch import nn
from transformers import (
    RobertaConfig,
    RobertaModel,
    RobertaForSequenceClassification,
    BertForSequenceClassification,
    RobertaTokenizer,
    BertModel,
    BertTokenizer,
    BertConfig
)
import torch.nn.functional as F
from data.process_fn import triple_process_fn, triple2dual_process_fn
# from .dro_greedy_loss import DROGreedyLoss, AverageMeter
# from .loss import SymKlCriterion
import numpy as np 

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
            noise_level = 0.0,
            group_ids = None
            ):
        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)

        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)
        if noise_level > 0:
            # print("NOISE", noise_level)
            noise_q = torch.randn(q_embs.shape).to(q_embs.device) / torch.sqrt(torch.sum(q_embs**2, -1).unsqueeze(1))
            noise_a = torch.randn(a_embs.shape).to(a_embs.device) / torch.sqrt(torch.sum(a_embs**2, -1).unsqueeze(1))
            noise_b = torch.randn(b_embs.shape).to(b_embs.device) / torch.sqrt(torch.sum(b_embs**2, -1).unsqueeze(1))
            q_embs = q_embs + 26.8 * noise_level * noise_q
            a_embs = a_embs + 26.8 * noise_level * noise_a
            b_embs = b_embs + 26.8 * noise_level * noise_b

        logit_matrix = torch.cat([(q_embs * a_embs).sum(-1).unsqueeze(1),
                                  (q_embs * b_embs).sum(-1).unsqueeze(1)], dim=1)  # [B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0 * lsm[:, 0]
        # loss, acc, group_losses, group_accs, group_counts = model(x, y, g, d, w)
        if group_ids is None:
            return (loss.mean(),)
        else:
            return loss

class RobertaDot_NLL_LN(NLL, RobertaForSequenceClassification):
    """None
    Compress embedding to 200d, then computes NLL loss.
    """

    def __init__(self, config, model_argobj=None):
        NLL.__init__(self, model_argobj)
        RobertaForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768)
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)
    
    # def add_group_loss(self, n_groups, n_domains, alpha, eps, ema = 0.1, weight_ema = True):
    #     self.loss = DROGreedyLoss(n_groups, n_domains, alpha, eps, ema, weight_ema)
    #     self.n_groups = n_groups
    #     self.accum_loss = AverageMeter()
    #     self.accum_group_loss = [AverageMeter() for _ in range(n_groups)]

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
        
        loss = self.forward_model(
            query_ids,
            attention_mask_q,
            input_ids_a,
            attention_mask_a,
            input_ids_b,
            attention_mask_b,
            is_query,
            noise_level,
            group_ids)
        # print(loss, weights)
        # exit(0)
        if group_ids is None:
            return loss
        else:
            robust_loss, group_losses, group_counts = self.loss(loss, group_ids, weights)
            self.accum_loss.update(robust_loss.item(), loss.size(0))
            for i in range(self.n_groups):
                self.accum_group_loss[i].update(group_losses[i].item(), group_counts[i].item())
            # print({'loss': loss, 'h_fun': self.loss.h_fun, 'sum_loss':self.loss.sum_losses, 'count_cat' :self.loss.count_cat, 'robust_loss': robust_loss, 'gLOss':self.accum_group_loss, 'group_counts':group_counts})
            return robust_loss, group_losses, group_counts

    def output_state(self):
        h_fun = {self.loss.id2group[str(i)]:  self.loss.h_fun.detach().cpu().numpy()[i] for i in range(self.n_groups)}
       
        sum_loss = {self.loss.id2group[str(i)]: self.accum_group_loss[i].avg for i in range(self.n_groups)}
        # sum_loss = self.accum_loss.detach().cpu().numpy()
        return h_fun, sum_loss        


##############################################################################

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


class HFBertEncoder(BertModel):
    def __init__(self, config):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'
        self.init_weights()
    @classmethod
    def init_encoder(cls, args, dropout: float = 0.1):
        cfg = BertConfig.from_pretrained("bert-base-uncased")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        return cls.from_pretrained("bert-base-uncased", config=cfg)
    def forward(self, input_ids, attention_mask):
        hidden_states = None
        sequence_output, pooled_output = super().forward(input_ids=input_ids,
                                                         attention_mask=attention_mask)
        pooled_output = sequence_output[:, 0, :]
        return sequence_output, pooled_output, hidden_states
    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size


class BiEncoder(nn.Module):
    """ Bi-Encoder model component. Encapsulates query/question and context/passage encoders.
    """
    def __init__(self, args):
        super(BiEncoder, self).__init__()
        self.question_model = HFBertEncoder.init_encoder(args)
        self.ctx_model = HFBertEncoder.init_encoder(args)
    def query_emb(self, input_ids, attention_mask):
        sequence_output, pooled_output, hidden_states = self.question_model(input_ids, attention_mask)
        return pooled_output
    def body_emb(self, input_ids, attention_mask):
        sequence_output, pooled_output, hidden_states = self.ctx_model(input_ids, attention_mask)
        return pooled_output
    def forward(self, query_ids, attention_mask_q, input_ids_a = None, attention_mask_a = None, input_ids_b = None, attention_mask_b = None):
        if input_ids_b is None:
            q_embs = self.query_emb(query_ids, attention_mask_q)
            a_embs = self.body_emb(input_ids_a, attention_mask_a)
            return (q_embs, a_embs)
        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)
        logit_matrix = torch.cat([(q_embs*a_embs).sum(-1).unsqueeze(1), (q_embs*b_embs).sum(-1).unsqueeze(1)], dim=1) #[B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0*lsm[:,0]
        return (loss.mean(),)
        

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
        # self.vat_criterion = SymKlCriterion()
        self.total = 0
        self.correct = 0
        self.prob_diff = []
    
    def set_zero(self):
        self.total = 0
        self.correct = 0
    
    def set_diff_zero(self):
        self.prob_diff = []

    # def add_group_loss(self, n_groups, n_domains, alpha, eps, ema = 0.1, weight_ema = True):
    #     self.loss = DROGreedyLoss(n_groups, n_domains, alpha, eps, ema, weight_ema)
    #     self.n_groups = n_groups
    #     self.accum_loss = AverageMeter()
    #     self.accum_group_loss = [AverageMeter() for _ in range(n_groups)]

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask)
        query1 = outputs1[0][:, 0]
        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)
    
    def init_queue(self, dim = 768, K = 65536, momentum_update = False):
        print(f"Init Moco Queue with dim{dim}, k{K}")
        self.use_moco = True
        self.K = K
        self.register_buffer("queue", torch.randn(dim, self.K)) # using random 65536 embeddings?
        self.queue = nn.functional.normalize(self.queue, dim=0) * 26.8
        # print(self.queue)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _dequeue_and_enqueue(self, keys):
        # print("DeQueue and Enqueue Part!")
        keys = concat_all_gather(keys)
        # print("keys", keys.shape)
        batch_size = keys.shape[0]   # (B * 768)
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        # print(f"Rank {dist.get_rank()}, Updating {ptr} - {ptr + batch_size} embedding!")
        # print(keys.T)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

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
            weights = None,
            use_vat_q = False,
            use_vat_d = False,
            vat_eps = 1e-3,
            vat_gamma = 1,
            vat_lambda = 1):
        
        loss, train_acc, logits = self.forward_model(
            query_ids,
            attention_mask_q,
            input_ids_a,
            attention_mask_a,
            input_ids_b,
            attention_mask_b,
            is_query,
            noise_level,
            group_ids,
            use_vat_q = use_vat_q,
            use_vat_d = use_vat_d,
            vat_eps = vat_eps,
            vat_gamma = vat_gamma,
            vat_lambda = vat_lambda)
        
        train_size = torch.LongTensor([train_acc.shape[0]]).to(train_acc.device)
        # print(train_size)
        torch.distributed.all_reduce(train_size, op=torch.distributed.reduce_op.SUM)
        acc = torch.sum(1-train_acc)
        # print('before', acc)
        torch.distributed.all_reduce(acc, op=torch.distributed.reduce_op.SUM)
        # print('after', acc)
        if use_vat_q or use_vat_d:
            prob_diff = loss[-1]
            # torch.distributed.all_reduce(prob_diff, op=torch.distributed.reduce_op.SUM)
            # print(prob_diff, prob_diff.item(), )
            self.prob_diff += prob_diff #.append(prob_diff.item())
            pass
        self.total += train_size.item()
        self.correct += acc.item() 

        if group_ids is None:
            return loss, train_acc, logits
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


# --------------------------------------------------
ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            RobertaConfig,
        )
    ),
    (),
)


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
    MSMarcoConfig(name="dpr",
                model=BiEncoder,
                tokenizer_class=BertTokenizer,
                config_class=BertConfig,
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
