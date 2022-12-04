import sys
sys.path += ['../']
import pandas as pd
from sklearn.metrics import roc_curve, auc
import gzip
import copy
import torch
from torch import nn
import torch.distributed as dist
from tqdm import tqdm, trange
import os
from os import listdir
from os.path import isfile, join
import json
import logging
import random
import pytrec_eval
import pickle
import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from multiprocessing import Process
from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset
import re
from model.models import MSMarcoConfigDict, ALL_MODELS
from typing import List, Set, Dict, Tuple, Callable, Iterable, Any


logger = logging.getLogger(__name__)


class InputFeaturesPair(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(
            self,
            input_ids_a,
            attention_mask_a=None,
            token_type_ids_a=None,
            input_ids_b=None,
            attention_mask_b=None,
            token_type_ids_b=None,
            label=None):

        self.input_ids_a = input_ids_a
        self.attention_mask_a = attention_mask_a
        self.token_type_ids_a = token_type_ids_a

        self.input_ids_b = input_ids_b
        self.attention_mask_b = attention_mask_b
        self.token_type_ids_b = token_type_ids_b

        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def getattr_recursive(obj, name):
    for layer in name.split("."):
        if hasattr(obj, layer):
            obj = getattr(obj, layer)
        else:
            return None
    return obj


def barrier_array_merge(
        args,
        data_array,
        merge_axis=0,
        prefix="",
        load_cache=False,
        only_load_in_master=False,
        load_from_public=False):
    # data array: [B, any dimension]
    # merge alone one axis

    if args.local_rank == -1:
        return data_array

    if load_from_public:
        write_dir = args.public_ann_data_dir            
    else:
        write_dir = args.output_dir

    if not load_cache:
        rank = args.rank
        # if args.local_rank not in [-1, 0]:
        #     dist.barrier()
        # if is_first_worker():
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)
        # if args.local_rank == 0:
            # dist.barrier()  # directory created
        print(rank, os.path.exists(write_dir), os.listdir(write_dir)[0])
        assert os.path.exists(write_dir), "The Folder has not been created"
        pickle_path = os.path.join(
            write_dir,
            "{1}_data_obj_{0}.pb".format(
                str(rank),
                prefix))
        with open(pickle_path, 'wb') as handle:
            pickle.dump(data_array, handle, protocol=4)

        # make sure all processes wrote their data before first process
        # collects it
        dist.barrier()

    data_array = None

    data_list = []

    # return empty data
    if only_load_in_master:
        if not is_first_worker():
            dist.barrier()
            return None

    for i in range(
            args.world_size):  # TODO: dynamically find the max instead of HardCode
        pickle_path = os.path.join(
            write_dir, #args.output_dir,
            "{1}_data_obj_{0}.pb".format(
                str(i),
                prefix))
        try:
            with open(pickle_path, 'rb') as handle:
                b = pickle.load(handle)
                data_list.append(b)
        except BaseException:
            continue

    data_array_agg = np.concatenate(data_list, axis=merge_axis)
    dist.barrier()
    return data_array_agg


def pad_input_ids(input_ids, max_length,
                  pad_on_left=False,
                  pad_token=0):
    padding_length = max_length - len(input_ids)
    padding_id = [pad_token] * padding_length

    if padding_length <= 0:
        input_ids = input_ids[:max_length]
    else:
        if pad_on_left:
            input_ids = padding_id + input_ids
        else:
            input_ids = input_ids + padding_id

    return input_ids


def pad_ids(input_ids, attention_mask, token_type_ids, max_length,
            pad_on_left=False,
            pad_token=0,
            pad_token_segment_id=0,
            mask_padding_with_zero=True):
    padding_length = max_length - len(input_ids)
    padding_id = [pad_token] * padding_length
    padding_type = [pad_token_segment_id] * padding_length
    padding_attention = [0 if mask_padding_with_zero else 1] * padding_length

    if padding_length <= 0:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        token_type_ids = token_type_ids[:max_length]
    else:
        if pad_on_left:
            input_ids = padding_id + input_ids
            attention_mask = padding_attention + attention_mask
            token_type_ids = padding_type + token_type_ids
        else:
            input_ids = input_ids + padding_id
            attention_mask = attention_mask + padding_attention
            token_type_ids = token_type_ids + padding_type

    return input_ids, attention_mask, token_type_ids


# to reuse pytrec_eval, id must be string
def convert_to_string_id(result_dict):
    string_id_dict = {}

    # format [string, dict[string, val]]
    for k, v in result_dict.items():
        _temp_v = {}
        for inner_k, inner_v in v.items():
            _temp_v[str(inner_k)] = inner_v

        string_id_dict[str(k)] = _temp_v

    return string_id_dict


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def is_first_worker():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def concat_key(all_list, key, axis=0):
    return np.concatenate([ele[key] for ele in all_list], axis=axis)


def get_checkpoint_no(checkpoint_path):
    nums = re.findall(r'\d+', checkpoint_path)
    return int(nums[-1]) if len(nums) > 0 else 0

def get_latest_group_result(result_dir, n_groups = 9):
    if n_groups == 9:
        id2group = {"0": "scidocs", "1": "fiqa", "2": "scifact", "3": "nfcorpus", "4": "trec-covid", \
                "5": "climate-fever", "6": "nq", "7": "hotpotqa", "8": "bioasq"}
    else:
        id2group = {str(x): f"group{x}" for x in range(n_groups)}
    ANN_PREFIX = "ann_group_ndcg_"
    result_data_path_marco = os.path.join(result_dir, 'msmarco')
    if not os.path.exists(result_data_path_marco):
        return {}
    files = list(next(os.walk(result_data_path_marco))[2])
    num_start_pos = len(ANN_PREFIX)
    data_no_list = [int(s[num_start_pos:])
                    for s in files if s[:num_start_pos] == ANN_PREFIX]
    result = {}
    if len(data_no_list) > 0:
        data_no = max(data_no_list)
        with open(os.path.join(result_data_path_marco, ANN_PREFIX + str(data_no)), 'r') as f:
            ndcg_group_json = json.load(f)
        for x in ndcg_group_json:
            result[id2group[x]] = ndcg_group_json[x]["final_ndcg"]
        return result
    else:
        return {}


def get_latest_ann_data(ann_data_path, result_dir = None):
    if not result_dir:
        result_dir = ann_data_path # previous settings
    ANN_PREFIX = "ann_ndcg_"
    result_data_path_marco = os.path.join(result_dir, 'msmarco')
    # print(result_data_path_marco, os.path.exists(result_data_path_marco))
    if not os.path.exists(result_data_path_marco):
        return -1, None, None, None
    files = list(next(os.walk(result_data_path_marco))[2])
    num_start_pos = len(ANN_PREFIX)
    data_no_list = [int(s[num_start_pos:])
                    for s in files if s[:num_start_pos] == ANN_PREFIX]
    # print(data_no_list)
    # exit()
    if len(data_no_list) > 0:
        data_no = max(data_no_list)
        with open(os.path.join(result_data_path_marco, ANN_PREFIX + str(data_no)), 'r') as f:
            ndcg_json = json.load(f)
        if result_dir == ann_data_path:
            return data_no, os.path.join(
                ann_data_path, "ann_training_data_" + str(data_no)), ndcg_json
        else:
            return data_no, os.path.join(
                ann_data_path, "msmarco/ann_training_data_" + str(data_no)), ndcg_json
    return -1, None, None, None


def numbered_byte_file_generator(base_path, file_no, record_size):
    for i in range(file_no):
        with open('{}_split{}'.format(base_path, i), 'rb') as f:
            while True:
                b = f.read(record_size)
                if not b:
                    # eof
                    break
                yield b

def chartoid_file_generator(base_path, file_no):
    for i in range(file_no):
        with open('{}charid_split{}'.format(base_path, i), 'r') as f:
            while True:
                b = str(f.readline()).strip('\n')                
                if not b or b in [' ', '', '\n']:
                    # eof
                    break
                # print(b.split(":"), ":".join(b.split(":")[1:]))
                try:
                    idx, char = b.split(":")
                    yield int(idx), char
                except:
                    b_split = b.split(":")
                    yield int(b_split[0]),  ":".join(b.split(":")[1:])

class EmbeddingCache:
    def __init__(self, base_path, group= False, seed=-1):
        self.base_path = base_path
        self.group = group
        with open(base_path + '_meta', 'r') as f:
            meta = json.load(f)
            self.dtype = np.dtype(meta['type'])
            self.total_number = meta['total_number']
            ######################################################
            self.record_size = int(
                meta['embedding_size']) * self.dtype.itemsize + 4
            ###################################################### # CHange when adding group
        if seed >= 0:
            self.ix_array = np.random.RandomState(
                seed).permutation(self.total_number)
        else:
            self.ix_array = np.arange(self.total_number)
        self.f = None

    def open(self):
        self.f = open(self.base_path, 'rb')

    def close(self):
        self.f.close()

    def read_single_record(self):
        record_bytes = self.f.read(self.record_size)
        passage_len = int.from_bytes(record_bytes[:4], 'big')
        passage = np.frombuffer(record_bytes[4:], dtype=self.dtype)
        return passage_len, passage
    

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __getitem__(self, key):
        if key < 0 or key > self.total_number:
            raise IndexError(
                "Index {} is out of bound for cached embeddings of size {}".format(
                    key, self.total_number))
        self.f.seek(key * self.record_size)
        return self.read_single_record()

    def __iter__(self):
        self.f.seek(0)
        for i in range(self.total_number):
            new_ix = self.ix_array[i]
            yield self.__getitem__(new_ix)

    def __len__(self):
        return self.total_number

class StreamingDataset(IterableDataset):
    def __init__(self, elements, fn, distributed=True, size = -1):
        super().__init__()
        self.elements = elements
        self.fn = fn
        self.num_replicas=-1 
        self.distributed = distributed
        self.size = size
        if self.size > 0:
            self.weight = np.ones(size)
            self.accum_losses = None

    def __iter__(self):
        if dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            print("Not running in distributed mode")
        for i, element in enumerate(self.elements):
            if self.distributed and self.num_replicas != -1 and i % self.num_replicas != self.rank:
                continue
            records = self.fn(element, i)
            for rec in records:
                if self.size <= 0:
                    yield rec
                else:
                    rec = rec #+ (torch.tensor(self.weight[rec[-1]], dtype = torch.float), ) # (update the weight)
                    yield rec


def tokenize_to_file(args, i, num_process, in_path, out_path, line_fn):

    configObj = MSMarcoConfigDict[args.model_type]
    
    tokenizer = configObj.tokenizer_class.from_pretrained(
        args.model_name_or_path,
        do_lower_case=True,
        cache_dir=None,
    )

    with open(in_path, 'r', encoding='utf-8') if in_path[-2:] != "gz" else gzip.open(in_path, 'rt', encoding='utf8') as in_f,\
            open('{}_split{}'.format(out_path, i), 'wb') as out_f:
        for idx, line in enumerate(in_f):
            if idx % num_process != i:
                continue
            out_f.write(line_fn(args, line, tokenizer))


def multi_file_process(args, num_process, in_path, out_path, line_fn):
    processes = []
    for i in range(num_process):
        p = Process(
            target=tokenize_to_file,
            args=(
                args,
                i,
                num_process,
                in_path,
                out_path,
                line_fn,
            ))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return [data]

    world_size = dist.get_world_size()
    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

def tokenize_to_file_bm25(args, i, num_process, in_path, out_path, line_fn):
    configObj = MSMarcoConfigDict[args.model_type]
    tokenizer = configObj.tokenizer_class.from_pretrained(
        args.model_name_or_path,
        do_lower_case=True,
        cache_dir=None,
    )

    with open(in_path, 'r', encoding='utf-8') if in_path[-2:] != "gz" else gzip.open(in_path, 'rt', encoding='utf8') as in_f,\
            open('{}_{}.jsonl'.format(out_path, i), 'a') as out_f:
        for idx, line in enumerate(in_f):
            if idx % num_process != i:
                continue
            p_id, p_text = line_fn(args, line, tokenizer, idx)
            json_file = {"id": p_id, "contents": p_text}
            out_f.write(json.dumps(json_file) + '\n')


def multi_file_process_bm25(args, num_process, in_path, out_path, line_fn):
    processes = []
    for i in range(num_process):
        p = Process(
            target=tokenize_to_file_bm25,
            args=(
                args,
                i,
                num_process,
                in_path,
                out_path,
                line_fn,
            ))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


def multi_file_process_json(args, num_process, in_path, out_path, line_fn, batch = None):
    processes = []
    for i in range(num_process):
        p = Process(
            target=tokenize_to_file_for_json,
            args=(
                args,
                i,
                num_process,
                in_path,
                out_path,
                line_fn,
                batch
            ))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


def tokenize_to_file_for_json(args, i, num_process, in_path, out_path, line_fn, batch=None):
    if args.model_name == 'fbv': # fbv or ance
        print("loading fbv, from %s"%(args.model_name_or_path))
        # configObj = L1ConfigDict[args.model_type]
        # tokenizer = configObj.tokenizer_class.from_pretrained(
        #     # 'bert-base-multilingual-uncased'
        #     args.model_name_or_path,
        #     do_lower_case=False,
        # #cache_dir=args.cache_dir if args.cache_dir else None,
        # )
        # # print(len(tokenizer.get_vocab()))
        assert 0
    else:
        configObj = MSMarcoConfigDict[args.model_type]
        tokenizer = configObj.tokenizer_class.from_pretrained(
            args.model_name_or_path,
            do_lower_case=True,
            cache_dir=None,
        )

    with open(in_path, 'r', encoding='utf-8') if in_path[-2:] != "gz" else gzip.open(in_path, 'rt', encoding='utf8') as in_f,\
            open('{}_split{}'.format(out_path, i), 'ab') as out_f, open('{}charid_split{}'.format(out_path, i), 'a') as out_f2:
        for idx, line in enumerate(in_f):
            if idx % num_process != i:
                continue
            if batch is not None:
                write_idx = batch * 1000 + idx
            else:
                write_idx = idx
            buffer, p_id = line_fn(args, line, tokenizer, write_idx)
            # print(buffer,  int.from_bytes(buffer[:8], 'big'),int.from_bytes(buffer[8:12], 'big'))
            out_f.write(buffer)
            out_f2.write("%d:%s\n"%(write_idx, p_id))
