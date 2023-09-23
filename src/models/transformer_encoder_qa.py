#!/usr/bin/env python


"""Transformer Encoder Architecture-based pre-trained Language Model
implementation for Question-Answering"""


import os
from typing import Any, Dict, List, Optional
from typing import SupportsFloat as Numeric
from typing import Tuple, Union

import lightning as L
import torch
import torch.nn as nn
import torchmetrics
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer

SINGLE_PRED_TYPE = Dict[str, str]
PREDS_TYPE = Union[SINGLE_PRED_TYPE, List[SINGLE_PRED_TYPE]]
SINGLE_TARGET_TYPE = Dict[
    str, Union[str, Dict[str, Union[List[str], List[int]]]]
]
TARGETS_TYPE = Union[SINGLE_TARGET_TYPE, List[SINGLE_TARGET_TYPE]]


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

        self.criterion = nn.CrossEntropyLoss(reduction="sum")

        # ================================
        # Metrics (also Logging of Losses)
        # ================================

        self.train_squad = torchmetrics.text.SQuAD()
        self.test_squad = torchmetrics.text.SQuAD()
        self.validation_squad = torchmetrics.text.SQuAD()
        self.validation_squad_f1_best = torchmetrics.MaxMetric()
        self.validation_squad_exact_match_best = torchmetrics.MaxMetric()

        self.train_start_loss = torchmetrics.MeanMetric()
        self.train_end_loss = torchmetrics.MeanMetric()
        self.train_loss = torchmetrics.MeanMetric()

        self.validation_start_loss = torchmetrics.MeanMetric()
        self.validation_end_loss = torchmetrics.MeanMetric()
        self.validation_loss = torchmetrics.MeanMetric()

        self.test_start_loss = torchmetrics.MeanMetric()
        self.test_end_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()

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

    def convert_to_compatible_tokens(
        self, a: List[str], c: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert provided answers to model-compatible start and end tokens.
        Used in loss computation"""
        context_ids_list = self.tokenizer(
            c, return_tensors="pt", truncation=True, padding="longest"
        ).input_ids.tolist()
        answer_ids_list = self.tokenizer(
            a, return_tensors="pt", truncation=True, padding="longest"
        ).input_ids.tolist()
        answer_ids_list = [
            item[
                1 : item.index(
                    self.tokenizer.encode(self.tokenizer.sep_token)[1]
                )
            ]
            for item in answer_ids_list
        ]
        start_idx = [
            [
                idx
                for idx in range(
                    len(context_ids_list[i]) - len(answer_ids_list[i]) + 1
                )
                if context_ids_list[i][idx : idx + len(answer_ids_list[i])]
                == answer_ids_list[i]
            ][0]
            for i in range(len(answer_ids_list))
        ]
        end_idx = [
            [start_idx[i] + len(answer_ids_list[i])]
            for i in range(len(start_idx))
        ]

        start_idx = torch.tensor(start_idx)
        end_idx = torch.tensor(end_idx)

        if len(start_idx.size()) == 1:
            start_idx = start_idx.unsqueeze(dim=1)
        if len(end_idx.size()) == 1:
            end_idx = end_idx.unsqueeze(dim=1)

        return start_idx, end_idx

    def convert_targets_to_SQuAD_format(
        self, a: List[str], a_start: torch.Tensor, c: List[str], idx: List[str]
    ) -> TARGETS_TYPE:
        """Convert target data into `torchmetrics.text.SQuAD` compatible
        format"""
        targets = []
        for ans, a_start_idx, context, id in zip(a, a_start, c, idx):
            targets.append(
                {
                    "answers": {
                        "answer_start": [a_start_idx.item()],
                        "text": [ans],
                    },
                    "id": id,
                }
            )
        return targets

    def get_predicted_texts(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        context: str,
    ) -> str:
        """Convert start and end logits to predicted text from a given
        context"""
        start_position = start_logits.argmax()
        end_position = end_logits.argmax()
        context_encoded = self.tokenizer(
            context, return_tensors="pt", truncation=True, padding="longest"
        ).input_ids.squeeze()
        text = self.tokenizer.decode(
            context_encoded[start_position:end_position]
        )
        return text

    def convert_predictions_to_SQuAD_format(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        context: List[str],
        idx: List[str],
    ) -> PREDS_TYPE:
        """Get prediction texts and convert to `torchmetrics.text.SQuAD`
        compatible format"""
        predictions = []
        for s, e, c, id in zip(start_logits, end_logits, context, idx):
            text = self.get_predicted_texts(s, e, c)
            predictions.append({"prediction_text": text, "id": id})
        return predictions

    def log_stats(
        self,
        name: str,
        start_loss: torch.Tensor,
        end_loss: torch.Tensor,
        loss: torch.Tensor,
        squad: Dict[Numeric, Numeric],
    ) -> None:
        self.log(
            os.path.join(name, "start_loss"),
            start_loss,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            os.path.join(name, "end_loss"),
            end_loss,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            os.path.join(name, "loss"), loss, on_step=False, on_epoch=True
        )
        self.log(
            os.path.join(name, "f1"), squad["f1"], on_step=False, on_epoch=True
        )
        self.log(
            os.path.join(name, "exact_match"),
            squad["exact_match"],
            on_step=False,
            on_epoch=True,
        )

    def forward(
        self, q: Union[str, List[str]], c: Union[str, List[str]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokenized = self.process(q, c)
        tokenized.to(self.device)
        outputs = self.lm(**tokenized)
        hidden_states = outputs.last_hidden_state
        qa_outputs = self.qa_outputs(hidden_states)
        start_logits, end_logits = qa_outputs.split(1, dim=-1)
        return start_logits, end_logits

    def model_step(
        self,
        batch: Tuple[List[str], List[str], torch.Tensor, List[str], List[str]],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        q, a, a_start, c, idx = batch
        start_logits, end_logits = self.forward(q, c)
        start_positions, end_positions = self.convert_to_compatible_tokens(
            a, c
        )
        ignore_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignore_index)
        end_positions = end_positions.clamp(0, ignore_index)
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        start_loss = criterion(start_logits, start_positions)
        end_loss = criterion(end_logits, end_positions)
        loss = (start_loss + end_loss) / 2.0
        return (
            start_loss,
            end_loss,
            loss,
            start_logits,
            end_logits,
            start_positions,
            end_positions,
        )

    def training_step(
        self,
        batch: Tuple[List[str], List[str], torch.Tensor, List[str], List[str]],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        q, a, a_start, c, idx = batch
        (
            start_loss,
            end_loss,
            loss,
            start_logits,
            end_logits,
            start_positions,
            end_positions,
        ) = self.model_step(batch)

        self.train_start_loss(start_loss)
        self.train_end_loss(end_loss)
        self.train_loss(loss)
        targets_SQuAD: TARGETS_TYPE = self.convert_targets_to_SQuAD_format(
            a, a_start, c, idx
        )
        predicted_texts_SQuAD: PREDS_TYPE = (
            self.convert_predictions_to_SQuAD_format(
                start_logits, end_logits, c, idx
            )
        )
        self.train_squad(predicted_texts_SQuAD, targets_SQuAD)
        self.log_stats(
            "train",
            self.train_start_loss,
            self.train_end_loss,
            self.train_loss,
            self.train_squad,
        )

        return {
            "loss": loss,
            "start_logits": start_logits,
            "end_logits": end_logits,
        }

    def validation_step(
        self,
        batch: Tuple[List[str], List[str], torch.Tensor, List[str], List[str]],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        q, a, a_start, c, idx = batch
        (
            start_loss,
            end_loss,
            loss,
            start_logits,
            end_logits,
            start_positions,
            end_positions,
        ) = self.model_step(batch)

        self.validation_start_loss(start_loss)
        self.validation_end_loss(end_loss)
        self.validation_loss(loss)
        targets_SQuAD: TARGETS_TYPE = self.convert_targets_to_SQuAD_format(
            a, a_start, c, idx
        )
        predicted_texts_SQuAD: PREDS_TYPE = (
            self.convert_predictions_to_SQuAD_format(
                start_logits, end_logits, c, idx
            )
        )
        self.validation_squad(predicted_texts_SQuAD, targets_SQuAD)
        self.log_stats(
            "validation",
            self.validation_start_loss,
            self.validation_end_loss,
            self.validation_loss,
            self.validation_squad,
        )

        return {
            "loss": loss,
            "start_logits": start_logits,
            "end_logits": end_logits,
        }

    def test_step(
        self,
        batch: Tuple[List[str], List[str], torch.Tensor, List[str], List[str]],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        q, a, a_start, c, idx = batch
        (
            start_loss,
            end_loss,
            loss,
            start_logits,
            end_logits,
            start_positions,
            end_positions,
        ) = self.model_step(batch)

        self.test_start_loss(start_loss)
        self.test_end_loss(end_loss)
        self.test_loss(loss)
        targets_SQuAD: TARGETS_TYPE = self.convert_targets_to_SQuAD_format(
            a, a_start, c, idx
        )
        predicted_texts_SQuAD: PREDS_TYPE = (
            self.convert_predictions_to_SQuAD_format(
                start_logits, end_logits, c, idx
            )
        )
        self.test_squad(predicted_texts_SQuAD, targets_SQuAD)
        self.log_stats(
            "test",
            self.test_start_loss,
            self.test_end_loss,
            self.test_loss,
            self.test_squad,
        )

        return {
            "loss": loss,
            "start_logits": start_logits,
            "end_logits": end_logits,
        }

    def predict_step(self, q: Union[str, List[str]], c: Union[str, List[str]]):
        start_logits, end_logits = self(q, c)
        predicted = []
        for s, e, context in zip(start_logits, end_logits, c):
            predicted.append(self.get_predicted_texts(s, e, context))

        return predicted

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    # ===========================================
    #                   Hooks
    # ===========================================

    def on_train_start(self, *args, **kwargs):
        pass

    def on_train_epoch_end(self, *args, **kwargs):
        pass

    def on_validation_epoch_end(self, *args, **kwargs):
        val_squad = self.validation_squad.compute()
        val_f1 = val_squad["f1"]
        val_exact_match = val_squad["exact_match"]
        self.validation_squad_f1_best(val_f1)
        self.validation_squad_exact_match_best(val_exact_match)
        self.log(
            "val/best_f1",
            self.validation_squad_f1_best.compute(),
            prog_bar=True,
        )
        self.log(
            "val/best_exact_match",
            self.validation_squad_exact_match_best.compute(),
            prog_bar=True,
        )

    def on_test_epoch_end(self, *args, **kwargs):
        pass
