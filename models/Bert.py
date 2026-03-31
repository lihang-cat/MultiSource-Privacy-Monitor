import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class BertForSentenceClassification(nn.Module):
    def __init__(self, model_name, num_labels, dropout_prob=0.1):
        """
        num_labels: 应该是 3 (Identity, Credential, Location)
        Safe 是通过全 0 向量隐含表示的。
        """
        super(BertForSentenceClassification, self).__init__()

        # 1. 加载预训练基座
        print(f"Loading Baseline Backbone: {model_name}...")
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        # 2. 分类层
        self.dropout = nn.Dropout(dropout_prob)
        #  我们做整句分类，所以只需要一个分类头
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        # 3. 损失函数
        # BCEWithLogitsLoss 包含了 Sigmoid + BCE
        # reduction='mean' 直接计算 batch 平均损失，
        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels=None):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)


        pooled_output = outputs.pooler_output

        # 如果模型没有 pooler_output (某些 RoBERTa 版本), 可以手动取 [CLS]
        if pooled_output is None:
            pooled_output = outputs.last_hidden_state[:, 0, :]

        pooled_output = self.dropout(pooled_output)


        logits = self.classifier(pooled_output)


        if labels is not None:

            loss = self.loss_fct(logits, labels.float())
            return loss

        else:
            # 推理模式：返回概率
            return torch.sigmoid(logits)