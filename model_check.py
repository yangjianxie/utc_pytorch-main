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
# from lightning.fabric import Fabric

from transformers import ErniePreTrainedModel, ErnieModel, ErnieConfig

model1 = UTC.from_pretrained("/home/yangjx/project1/utc/rrsong/utc-base")
print(model1.state_dict().keys())  # 打印模型参数键
a = model1.state_dict().keys()

model2 = UTC.from_pretrained("/home/yangjx/project1/utc/rrsong/utc-base-finetune")
print(model2.state_dict().keys())  # 打印模型参数键
b = model2.state_dict().keys()

print(a==b)
