# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Semantic dataset.
"""

import json
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
from PIL import Image
from rich.progress import track

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs, Semantics
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from nerfstudio.data.utils.data_utils import get_depth_image_from_path, get_semantics_and_mask_tensors_from_path
from nerfstudio.model_components import losses
from nerfstudio.utils.rich_utils import CONSOLE


class SemanticDataset(DepthDataset):
    """Dataset that returns images, semantics, masks and depths.

    Args:
        dataparser_outputs: description of where and how to read input images.
    """

    exclude_batch_keys_from_device = DepthDataset.exclude_batch_keys_from_device + ["mask", "semantics"]

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        assert "semantics" in dataparser_outputs.metadata.keys() and isinstance(self.metadata["semantics"], Semantics)
        self.semantics = self.metadata["semantics"]
        self.mask_indices = torch.tensor(
            [self.semantics.classes.index(mask_class) for mask_class in self.semantics.mask_classes]
        ).view(1, 1, -1)

    def get_metadata(self, data: Dict) -> Dict:
        metadata = super().get_metadata(data)  # Get depth data from parent class
        
        # Add semantic data
        filepath = self.semantics.filenames[data["image_idx"]]
        semantic_label, mask = get_semantics_and_mask_tensors_from_path(
            filepath=filepath, mask_indices=self.mask_indices, scale_factor=self.scale_factor
        )
        if "mask" in data.keys():
            mask = mask & data["mask"]
            
        metadata.update({
            "mask": mask,
            "semantics": semantic_label
        })
        
        return metadata
