from dataclasses import dataclass
from typing import List

import torch

from coralai.reportable import Reportable


@dataclass
class CoralaiConfig(Reportable):
    kernel: torch.Tensor
    sense_chs: List[str]
    act_chs: List[str]
    torch_device: torch.device

    report_prefix: str = "coralai_config_snap"