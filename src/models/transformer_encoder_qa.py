#!/usr/bin/env python


"""Transformer Encoder Architecture-based pre-trained Language Model
implementation for Question-Answering"""


from typing import Optional

import lightning as L
import torch
import torch.nn as nn
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
        self.lm: transformers.PretrainedModel = AutoModel.from_pretrained(
            self.lm_name
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

    def forward(self, *args, **kwargs):
        pass

    def model_step(self, *args, **kwargs):
        pass

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
