import torch.nn as nn
import torch
import numpy as np
from torch import tensor


class HistoricalAverage:
    def __init__(self, TIMESTEP_IN, TIMESTEP_OUT, *args, **kwargs):
        super().__init__()
        self.TIMESTEP_IN = TIMESTEP_IN
        self.TIMESTEP_OUT = TIMESTEP_OUT

    def ha(self, x):
        History = []
        test = x[:, i:i + self.TIMESTEP_OUT, :, :, :]
        for i in range(0, self.TIMESTEP_IN, self.TIMESTEP_OUT):
            History.append(x[:, i:i + self.TIMESTEP_OUT, :, :, :])
        History = np.stack(History)
        result = np.mean(History, axis=0)
        return result

    @staticmethod
    def add_model_specific_args(parent_parser):
        return parent_parser
