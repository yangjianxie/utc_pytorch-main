import json
import pandas as pd
import requests
from transformers import BertTokenizer
from model import UTC
import torch
from template import UTCTemplate
import torch.nn.functional as F

class UTCPredictor:

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = UTC.from_pretrained(model_dir)
        self.utc_template = UTCTemplate(self.tokenizer, max_length=512)

        # 检查是否有可用的 GPU，如果有，则将模型转移到 GPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model.to(self.device)  # 将模型移动到 GPU
        else:
            self.device = torch.device("cpu")  # 使用 CPU

    def text_classify(self, text, choices):
        sample = {
            "text_a": text,
            "text_b": "",
            "choices": choices
        }
        ids = self.utc_template(sample)
        # 创建输入字典
        inputs = {
            'input_ids': torch.tensor(ids['input_ids']).unsqueeze(0).to(self.device),
            'token_type_ids': torch.tensor(ids['token_type_ids']).unsqueeze(0).to(self.device),
            'attention_mask': torch.tensor(ids['attention_mask']).unsqueeze(0).to(self.device),
            'position_ids': torch.tensor(ids['position_ids']).unsqueeze(0).to(self.device),
            'omask_positions': torch.tensor(ids['omask_positions']).unsqueeze(0).to(self.device),
            'cls_positions': torch.tensor(ids['cls_positions']).unsqueeze(0).to(self.device)
        }

        outputs = self.model(**inputs)
        # print(outputs)
        logits = outputs['option_logits']
        probabilities = F.softmax(logits, dim=-1)
        max_value, pre_label = torch.max(probabilities, dim=1)

        return (pre_label.item(), probabilities.tolist()[0])


if __name__ == '__main__':
    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = UTCPredictor('rrsong/utc-base')
    text = '[意图识别]user:你好\nassistant:有什么可以帮您?\nuser:我肚子有点疼\nassistant:还有别的症状吗?\nuser:没有了。我要看什么科?'
    choices = ['科室推荐','疾病症状查询', '疾病诊断查询', '医生查询', '疾病检查查询', '疾病并发症查询', '医院科室评价', '医院查询', '疾病饮食查询', '疾病护理查询', '医院评价查询']
    index, prob = model.text_classify(text, choices)
    print(index, prob)


    # data = pd.read_csv("/Users/yjx/PycharmProjects/utc/jd-data-intent/jd_intent_data.txt", delimiter="\t", header=None)
    # all_labels = data.loc[:, 0].unique().tolist()
    # url = 'https://nocoai-vector-rest-svc.nlp.nocodetech.cn/m3encode'
    # data = {
    #     "texts": all_labels
    # }
    #
    # response = requests.post(url, json=data)
    # print(response.json())






'''
    dir = '/home/yangjx/project1/utc/rrsong/utc-base-jd-intent'
    utc_predictor = UTCPredictor(model_dir=dir)

    # text = '[意图识别]user:你好\nassistant:有什么可以帮您?\nuser:我肚子有点疼\nassistant:还有别的症状吗?\nuser:没有了。我要看什么科?'
    # choices = ['科室推荐' ,'疾病症状查询', '疾病诊断查询', '医生查询', '疾病检查查询', '疾病并发症查询', '医院科室评价', '医院查询', '疾病饮食查询', '疾病护理查询', '医院评价查询']
    # index, prob = utc_predictor.text_classify(text, choices)
    # print(index, prob)

    # /home/yangjx/project1/utc/data-dept/0923dept/relabel_dept_test.txt
    with open('/home/yangjx/project1/utc/jd-data-intent/jd_intent_test.txt', 'r') as file:
        num = 0
        for line in file:
            line = json.loads(line)
            text = line['text_a']
            choices = line['choices']
            index, prob = utc_predictor.text_classify(text, choices)
            label = choices[line['labels'][0]]
            pre_label = choices[index]
            if pre_label != label:
                num += 1
                print('text:', text)
                print('choices:', choices)
                print('label:', label)
                print('pre_label:', pre_label)
                print('prob:', prob)
                print('max_prob:', max(prob))
                print('*'*100)

        print('num:', num)
        
'''