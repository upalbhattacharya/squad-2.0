#!/usr/bin/env python

"""LightningDataModule for SQuAD-2.0 dataset"""

import os
from typing import Optional

import hydra
import lightning as L
import pyrootutils
from omegaconf import DictConfig
from torch.utils.data import DataLoader


pyrootutils.setup_root(
    search_from=__file__,
    indicator=".project-root",
    pythonpath=True,
    project_root_env_var=True,
)

from src.datamodules.squad_dataset import SQuADDataset


class SQuADDataModule(L.LightningDataModule):

    """LightningDataModule for SQuAD-2.0 dataset"""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super(SQuADDataModule, self).__init__()

        self.save_hyperparameters(logger=False)

        self.train_dataset: Optional[SQuADDataset] = None
        self.validation_dataset: Optional[SQuADDataset] = None
        self.test_dataset: Optional[SQuADDataset] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            if not self.train_dataset:
                self.train_dataset = SQuADDataset(
                    df_path=os.path.join(self.hparams.data_dir, "train.csv")
                )

            if not self.validation_dataset:
                self.validation_dataset = SQuADDataset(
                    df_path=os.path.join(
                        self.hparams.data_dir, "validation.csv"
                    )
                )

        if stage == "test":
            if not self.test_dataset:
                self.test_dataset = SQuADDataset(
                    df_path=os.path.join(self.hparams.data_dir, "test.csv")
                )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validation_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self):
        pass


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    dataloader = hydra.utils.instantiate(cfg)

    dataloader.prepare_data()
    dataloader.setup(stage="fit")
    train_loader = dataloader.train_dataloader()

    for q, a, a_start, c, idx in iter(train_loader):
        print(len(q), type(q))
        print(len(a), type(a))
        print(len(a_start), type(a_start))
        print(len(c), type(c))
        print(len(idx), type(idx))
        break


if __name__ == "__main__":
    main()
