#!/usr/bin/env python


"""Transformer Encoder Architecture-based pre-trained Language Model
implementation for Question-Answering"""


from typing import Any, List, Optional, Tuple, Union

import lightning as L
import torch
import torch.nn as nn
import torchmetrics
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer


class TransformerEncoderQuestionAnswering(L.LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler],
        lm_name: str = "bert-base-uncased",
        truncation_side: str = "right",
        max_length: int = 512,
    ):
        super(TransformerEncoderQuestionAnswering, self).__init__()
        self.save_hyperparameters(logger=False)
        self.truncation_side = truncation_side
        self.max_length = max_length

        # ================
        # Model Definition
        # ================

        self.lm_name: str = lm_name
        self.config: transformers.PretrainedConfig = (
            AutoConfig.from_pretrained(self.lm_name)
        )
        self.lm: transformers.PretrainedModel = AutoModel.from_config(
            self.config, add_pooling_layer=False
        )
        self.tokenizer: transformers.PretrainedTokenizer = (
            AutoTokenizer.from_pretrained(
                self.lm_name, truncation_side=self.truncation_side
            )
        )
        self.qa_outputs = nn.Linear(
            in_features=self.config.hidden_size, out_features=2
        )
        self.sigmoid = nn.Sigmoid()

        # ==================================================
        # Training Objectives, Learning Rates and Schedulers
        # ==================================================

        self.criterion = nn.CrossEntropyLoss()

        # ================================
        # Metrics (also Logging of Losses)
        # ================================

        self.train_squad = None
        self.test_squad = None
        self.validation_squad = None
        self.validation_squad_best = torchmetrics.MaxMetric()
        self.train_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()
        self.validation_loss = torchmetrics.MeanMetric()

    def process(self, q: Union[str, List[str]], c: Union[str, List[str]]):
        if isinstance(q, str) and isinstance(c, str):
            return self.tokenizer(
                q,
                c,
                truncation="only_second",
                padding="longest",
                max_length=self.max_length,
                return_tensors="pt",
            )

        return self.tokenizer(
            [[a, b] for a, b in zip(q, c)],
            truncation="only_second",
            padding="longest",
            max_length=self.max_length,
            return_tensors="pt",
        )

    def forward(self, q: Union[str, List[str]], c: Union[str, List[str]]):
        tokenized = self.process(q, c)
        tokenized.to(self.device)
        outputs = self.lm(**tokenized)
        hidden_states = outputs.last_hidden_state
        qa_outputs = self.qa_outputs(hidden_states)
        start_logits, end_logits = qa_outputs.split(1, dim=-1)
        return start_logits, end_logits

    def model_step(self, batch: Any):
        q, a, c, idx = batch
        start_logits, end_logits = self.forward(q, c)

    def training_step(self, *args, **kwargs):
        pass

    def validation_step(self, *args, **kwargs):
        pass

    def test_step(self, *args, **kwargs):
        pass

    def predict_step(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        pass

    # ===========================================
    #                   Hooks
    # ===========================================
    # NOTE: Cannot find these hooks in the source code of LightningModule

    def on_train_start(self, *args, **kwargs):
        pass

    def on_train_epoch_end(self, *args, **kwargs):
        pass

    def on_val_epoch_end(self, *args, **kwargs):
        pass

    def on_test_epoch_end(self, *args, **kwargs):
        pass
