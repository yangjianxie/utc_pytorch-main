# utc_pytorch
Paddle通用文本分类UTC（Universal Text Classification）的pytorch实现.
### 推理
```
运行infer.py
```
### 微调
```
运行finetune.sh

数据格式：
{"text_a": "我怎么查我的就医历史？", "text_b": "", "question": "", "choices": ["院内服务功能_病历查询", "院内服务功能_检前须知", "院内服务功能_更换就诊人", "院内服务功能_就医记录", "挂号_挂号"], "labels": [3]}



