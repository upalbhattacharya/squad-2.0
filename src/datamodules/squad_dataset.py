#!/usr/bin/env python

"""Dataset for SQuAD-2.0"""

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset


class SQuADDataset(Dataset):
    def __init__(self, df_path: str):
        super(SQuADDataset, self).__init__()
        self.df = pd.read_csv(df_path, index_col=False)

    def __getitem__(self, idx: int):
        q, a, a_start, c, id = self.df[
            ["question", "answer", "answer_start", "context", "idx"]
        ].iloc[idx]
        return q, a, a_start, c, id

    def __len__(self):
        return len(self.df)


@hydra.main(version_base=None, config_path=".", config_name="df_test")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    dataset = SQuADDataset(cfg.df_path)
    for q, a, a_start, c, idx in iter(dataset):
        print(q)
        print(a)
        print(a_start)
        print(c)
        print(idx)
        break


if __name__ == "__main__":
    main()
