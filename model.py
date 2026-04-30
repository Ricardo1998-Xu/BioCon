# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class FocalLoss_Weight(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss_Weight, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha_weight = self.alpha[target]
            focal_loss *= alpha_weight

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.classifier = nn.Linear(self.encoder.config.hidden_size, args.num_labels)
        self.value_head = nn.Linear(self.encoder.config.hidden_size, 1)
        # Define dropout layer, dropout_probability is taken from args.
        self.dropout = nn.Dropout(args.dropout_probability)

    def forward(self, input_ids=None, labels=None):
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        hidden_states = self.dropout(hidden_states)

        token_logits = hidden_states[:, 0, :]  # (batch_size, hidden_size)

        logits = self.classifier(token_logits)  # (batch_size, 2)
        values = self.value_head(token_logits).squeeze(-1)

        prob = nn.functional.softmax(logits, dim=-1)
        if labels is not None:
            class_weights = torch.tensor([1.0, 5.0]).to("cuda")
            # loss_fct = nn.CrossEntropyLoss(weight=class_weights)

            # loss_fct = FocalLoss()
            # loss_fct = nn.CrossEntropyLoss()
            loss_fct = FocalLoss_Weight(alpha=class_weights)
            loss = loss_fct(logits, labels)
            return loss, prob, values
        else:
            return prob, values


class CodeT5Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(CodeT5Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)
        self.value_head = nn.Linear(config.hidden_size, 1)
        self.args = args

    def get_t5_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def forward(self, source_ids=None, labels=None):
        source_ids = source_ids.view(-1, self.args.block_size)
        vec = self.get_t5_vec(source_ids)
        logits = self.classifier(vec)
        values = self.value_head(vec).squeeze(-1)
        prob = nn.functional.softmax(logits, dim=-1)

        if labels is not None:
            class_weights = torch.tensor([1.0, 5.0]).to("cuda")
            # loss_fct = nn.CrossEntropyLoss(weight=class_weights)

            # loss_fct = FocalLoss()
            # loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss_fct = FocalLoss_Weight(alpha=class_weights)
            # loss_fct = FocalLoss()
            loss = loss_fct(logits, labels)
            return loss, prob, values
        else:
            return prob, values


class CodeGen(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(CodeGen, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)
        self.value_head = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids=None, labels=None, weight=None):
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]  # [B, L, H]
        logits = self.classifier(hidden_states)  # [B, L, num_labels]
        values = self.value_head(hidden_states).squeeze(-1)  # [B, L]
        batch_size = input_ids.size(0)
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        values = values[torch.arange(batch_size, device=logits.device), sequence_lengths]
        prob = nn.functional.softmax(pooled_logits, dim=-1)

        if labels is not None:
            labels = labels.to(logits.device)
            class_weights = torch.tensor([1.0, 5.0]).to("cuda")
            # loss_fct = nn.CrossEntropyLoss(weight=weight)
            loss_fct = FocalLoss_Weight(alpha=class_weights)
            loss = loss_fct(pooled_logits.view(-1, self.args.num_labels), labels.view(-1))
            return loss, prob, values
        else:
            return prob, values
