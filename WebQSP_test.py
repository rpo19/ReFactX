# jq '.Questions[].Parses[].Answers[].AnswerType' WebQSP.test.json | sort | uniq -c
#   16805 "Entity"
#      53 "Value"

# '.Questions[].RawQuestion'

# multiple answers for the same question

# if AnswerType == "Entity":
#     answer = obj['EntityName']
# elif AnswerType == "Value":
#     answer = obj['AnswerArgument']

import json
import os
from WebQSP_base import WebQSPDataset

path = WebQSPDataset.get_dataset_path('WebQSP.test.json')

config = {
    'path': path,
}

print(f'Loading {config["path"]}')
with open(config['path']) as fd:
    raw_dataset = json.load(fd)

dataset = WebQSPDataset(raw_dataset, config)

# DATASET_START_FROM = int(os.environ.get('DATASET_START_FROM', 0))
# if DATASET_START_FROM:
#     print(f'Loading dataset from {DATASET_START_FROM}')
#     dataset = dataset[DATASET_START_FROM:]
#     dataset.config['start_from'] = DATASET_START_FROM