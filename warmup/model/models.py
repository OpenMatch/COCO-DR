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
import logging 
logger = logging.getLogger(__name__)


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
        self.prob_diff = []
    
    def set_zero(self):
        self.total = 0
        self.correct = 0
    
    def set_diff_zero(self):
        self.prob_diff = []

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
            group_ids = None):
        
        loss, train_acc, logits = self.forward_model(
            query_ids,
            attention_mask_q,
            input_ids_a,
            attention_mask_a,
            input_ids_b,
            attention_mask_b,
            is_query,
            group_ids)
        
        train_size = torch.LongTensor([train_acc.shape[0]]).to(train_acc.device)
        # print(train_size)
        torch.distributed.all_reduce(train_size, op=torch.distributed.reduce_op.SUM)
        acc = torch.sum(1-train_acc)
        torch.distributed.all_reduce(acc, op=torch.distributed.reduce_op.SUM)
        
        self.total += train_size.item()
        self.correct += acc.item() 

        return loss, train_acc, logits
       

    def output_state(self):
        h_fun = {self.loss.id2group[str(i)]:  self.loss.h_fun.detach().cpu().numpy()[i] for i in range(self.n_groups)}
       
        sum_loss = {self.loss.id2group[str(i)]: self.accum_group_loss[i].avg for i in range(self.n_groups)}
        # sum_loss = self.accum_loss.detach().cpu().numpy()
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
        
    def set_zero(self):
        self.total = 0
        self.correct = 0
    
    def set_diff_zero(self):
        self.prob_diff = []


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
            group_ids = None):
        
        loss, train_acc, logits = self.forward_model(
            query_ids,
            attention_mask_q,
            input_ids_a,
            attention_mask_a,
            input_ids_b,
            attention_mask_b,
            is_query,
            group_ids)
        
        train_size = torch.LongTensor([train_acc.shape[0]]).to(train_acc.device)
        torch.distributed.all_reduce(train_size, op=torch.distributed.reduce_op.SUM)
        acc = torch.sum(1-train_acc)
        # print('before', acc)
        torch.distributed.all_reduce(acc, op=torch.distributed.reduce_op.SUM)
        # print('after', acc)
        self.total += train_size.item()
        self.correct += acc.item() 
        loss = loss 
        loss = loss.mean()
        return loss, train_acc, logits
 

    def output_state(self):
        h_fun = {self.loss.id2group[str(i)]:  self.loss.h_fun.detach().cpu().numpy()[i] for i in range(self.n_groups)}
       
        sum_loss = {self.loss.id2group[str(i)]: self.accum_group_loss[i].avg for i in range(self.n_groups)}
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
