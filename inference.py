import json

import numpy as np
from transformers import BertTokenizer
from transformers import AutoTokenizer
from model import UTC
import torch
from template import UTCTemplate
# import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm



dir = '/home/yangjx/project1/utc/rrsong/utc-base-intent'
# ckpt_file = '/home/yangjx/project1/utc/rrsong/checkpoint/step-0300.ckpt'
tokenizer = BertTokenizer.from_pretrained(dir)
model = UTC.from_pretrained(dir)

utc_template = UTCTemplate(tokenizer, max_length=512)

def infer_correct():
    for name in ['relabel_intent_test']: #['dev', 'test1', 'test2', 'test3', 'test4', 'test5']:
        # 将文件内容读取为列表
        with open(f'/home/yangjx/project1/utc/data-dept/{name}.txt', 'r', encoding='utf-8') as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines]  # 去掉每行末尾的换行符
            # print(lines[0])

        # print(lines[1])
        correct_num = 0
        query = []
        choices = []
        label = []
        predict = []
        predict_prob = []

        for sample in tqdm(lines):
            example = json.loads(sample)
            true_label = example['labels'][0]
            query.append(example['text_a'])
            choices.append(example['choices'])
            label.append(true_label)

            ids = utc_template(example)
            # print('ids',ids)
            input_ids = torch.tensor(ids['input_ids']).unsqueeze(0)
            token_type_ids = torch.tensor(ids['token_type_ids']).unsqueeze(0)
            attention_mask = torch.tensor(ids['attention_mask']).unsqueeze(0)
            position_ids = torch.tensor(ids['position_ids']).unsqueeze(0)
            omask_positions  = torch.tensor(ids['omask_positions']).unsqueeze(0)
            cls_positions = torch.tensor(ids['cls_positions']).unsqueeze(0)
            labels = torch.tensor(ids['labels']).unsqueeze(1)

            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                omask_positions=omask_positions,
                cls_positions=cls_positions
                # labels=labels  # 可以选择传入 labels 来计算损失
            )
            # print(outputs)
            logits = outputs['option_logits']

            # 使用 softmax 函数将 logits 转换为概率
            probabilities = F.softmax(logits, dim=-1)

            # probabilities 现在包含了每个类别的概率
            # print(probabilities.tolist()[0])

            max_value, pre_label = torch.max(probabilities, dim=1)

            predict.append(pre_label.item())
            predict_prob.append(probabilities.tolist()[0])

            # true_label = example['labels']
            # print(true_label, label_tag.item())

            if true_label == pre_label.item():
                correct_num += 1
            # print(correct_num)

        print(correct_num/len(lines))




data = pd.read_csv('/home/yangjx/project1/utc/data-intent/relabel_intent_test.txt', sep='\t', header=None)
# print(data)
lines = data[0]#[:100]

correct_num = 0
query = []
choices = []
label = []
predict = []
predict_prob = []

for sample in tqdm(lines):
    example = json.loads(sample)
    true_label = example['labels'][0]
    query.append(example['text_a'])
    choices.append(example['choices'])
    label.append(example['choices'][true_label])

    ids = utc_template(example)
    # print('ids',ids)
    input_ids = torch.tensor(ids['input_ids']).unsqueeze(0)
    token_type_ids = torch.tensor(ids['token_type_ids']).unsqueeze(0)
    attention_mask = torch.tensor(ids['attention_mask']).unsqueeze(0)
    position_ids = torch.tensor(ids['position_ids']).unsqueeze(0)
    omask_positions = torch.tensor(ids['omask_positions']).unsqueeze(0)
    cls_positions = torch.tensor(ids['cls_positions']).unsqueeze(0)
    labels = torch.tensor(ids['labels']).unsqueeze(1)

    outputs = model(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        omask_positions=omask_positions,
        cls_positions=cls_positions
        # labels=labels  # 可以选择传入 labels 来计算损失
    )
    # print(outputs)
    logits = outputs['option_logits']

    # 使用 softmax 函数将 logits 转换为概率
    probabilities = F.softmax(logits, dim=-1)

    # probabilities 现在包含了每个类别的概率
    # print(probabilities.tolist()[0])

    max_value, pre_label = torch.max(probabilities, dim=1)
    # print(pre_label.item())

    predict.append(example['choices'][pre_label.item()])
    predict_prob.append(probabilities.tolist()[0])

    # true_label = example['labels']
    # print(true_label, label_tag.item())

    if true_label == pre_label.item():
        correct_num += 1
        print('对')
    else:
        print('错')

# pre_analyse = pd.DataFrame({'query': query, 'choices': choices, 'label': label, 'predict': predict,
#                             'predict_prob': predict_prob})
# pre_analyse.to_csv('/home/yangjx/project1/utc/data-dept/5wmodelpre_analyse.csv', index=False)
print(correct_num, lines)
print(correct_num / len(lines))