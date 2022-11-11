# Copyright 2021 Condenser Author All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
from transformers import AutoTokenizer
from multiprocessing import Pool
from tqdm import tqdm
import nltk
import random
nltk.download('punkt')

parser = argparse.ArgumentParser()
parser.add_argument('--file', required=True)
parser.add_argument('--save_to', required=True)
parser.add_argument('--tokenizer', required=True)

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)


def encode_one(line, maxlen = 30):
    item = json.loads(line)
    group_id = item["group"]
    spans = item['spans']
    spans = nltk.sent_tokenize(spans.strip())
    if len(spans) <= 1: 
        return None
    
    tokenized = [
        tokenizer(
            s,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"] for s in spans
    ]

    tokenized_spans = []
    tokenized_span = []
    for span in tokenized:
        if len(span) > 0:
            if len(span) + len(tokenized_span) > maxlen or random.random() <= 0.1:
                if len(tokenized_span) > 0:
                    tokenized_spans.append(tokenized_span)
                tokenized_span = []
            tokenized_span.extend(span)
    
    if len(tokenized_span) > 0:
        tokenized_spans.append(tokenized_span)
    
    if len(tokenized_spans) < 2: 
        return None

    tokenized = [span for span in tokenized_spans if len(span) > 0]

    if len(tokenized) <= 1 or len(tokenized[0]) <= 2:
        return None
    return json.dumps({'spans': tokenized, "group": group_id})


with open(args.save_to, 'w') as f:
    with Pool() as p:
        all_tokenized = p.imap_unordered(
            encode_one,
            tqdm(open(args.file)),
            chunksize=500,
        )
        for x in all_tokenized:
            if x is None:
                continue
            f.write(x + '\n')
