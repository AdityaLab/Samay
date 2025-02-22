import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from uni2ts.distribution import MixtureOutput
from typing import List, Dict, Any, Union

# for finetune class
import lightning as L
from torch.distributions import Distribution
from torch import nn
from samay.models.uni2ts.loss.packed import (
    PackedDistributionLoss,
    PackedLoss,
    PackedNLLLoss,
    PackedPointLoss,
)
from samay.models.uni2ts.module.norm import RMSNorm
from samay.models.uni2ts.module.position import (
    BinaryAttentionBias,
    LearnedEmbedding,
    LearnedProjection,
)
from samay.models.uni2ts.module.ts_embed import MultiInSizeLinear, MultiOutSizeLinear
from samay.models.uni2ts.optim import SchedulerType, get_scheduler
from samay.models.uni2ts.transform import (
    AddObservedMask,
    AddTimeIndex,
    AddVariateIndex,
    DefaultPatchSizeConstraints,
    DummyValueImputation,
    EvalCrop,
    EvalMaskedPrediction,
    EvalPad,
    ExtendMask,
    FixedPatchSizeConstraints,
    FlatPackCollection,
    FlatPackFields,
    GetPatchSize,
    ImputeTimeSeries,
    MaskedPrediction,
    PackFields,
    PatchCrop,
    Patchify,
    SelectFields,
    SequencifyField,
    Transformation,
)

from samay.models.uni2ts.model.moirai.module import MoiraiModule

# ------------------- HELPER FUNCTIONS -------------------
def convert_module_kwargs(module_kwargs):
    """Convert module_kwargs to ingestible dictionary format by instantiating necessary objects and removing _target_ fields."""
    
    # Extract necessary fields
    module_args = {k: v for k, v in module_kwargs.items() if k != "_target_"}
    
    # Convert patch_sizes string to tuple if needed
    if isinstance(module_args.get("patch_sizes"), str) and module_args["patch_sizes"].startswith("${as_tuple:"):
        module_args["patch_sizes"] = tuple(map(int, re.findall(r"\d+",module_args["patch_sizes"])))# Extract numbers
    
    # Handle distr_output instantiation
    if "distr_output" in module_args and isinstance(module_args["distr_output"], dict):
        distr_config = module_args["distr_output"]
        
        if "_target_" in distr_config and distr_config["_target_"] == "uni2ts.distribution.MixtureOutput":
            # Instantiate component distributions
            components = []
            for comp in distr_config.get("components", []):
                if "_target_" in comp:
                    comp_class = globals().get(comp["_target_"].split(".")[-1])  # Get class by name
                    if comp_class:
                        components.append(comp_class())  # Instantiate class
            
            # Instantiate MixtureOutput with components
            module_args["distr_output"] = MixtureOutput(components=components)
    
    return module_args

# ------------------- CUSTOM TORCH DATASET -------------------
class MoiraiTorch(Dataset):
    def __init__(self,data:list[dict]):
        """Wraps the data in a torch Dataset object.

        Args:
            data (list[dict]): The input data you want to wrap in a torch Dataset object.
        """
        super().__init__() # Call parent class constructor
        self.data = data

        # Test data usually is split into label and input parts which come as a tuple
        if isinstance(self.data, list) and isinstance(self.data[0], tuple):
            self.input = [d[0] for d in self.data]
            self.label = [d[1] for d in self.data]
    
    def __len__(self):
        """Returns the length of the data.

        Returns:
            int: how many samples are in the data.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """Returns the sample at the given index. If the data is a list it just
        returns the sample at index idx of the list. If the data is a dictionary,
        then it returns a dictionary with the same keys as the original dictionary
        but with the values collected from index idx of each field.

        Args:
            idx (int): Index indicating the sample to be returned.

        Returns:
            Union[dict, tuple(dict)]: The sample at the index idx cleaned in proper format.
        """
        if isinstance(self.data, list):
            row = self.data[idx]
            
            # In case of test data the sample is a tuple of (input, label)
            if isinstance(row, tuple):
                mod_row = []
                for i in range(len(row)):
                    # convert period to timestamp
                    if isinstance(row[i]["start"], pd.Period):
                        mod_row.append({"start": torch.tensor(row[i]["start"].to_timestamp().timestamp(), dtype=torch.float32),
                                        "target": torch.tensor(row[i]["target"], dtype=torch.float32),
                                        "freq": row[i]["start"].freqstr,
                                        "item_id": row[i]["item_id"]
                                        })
                    else:
                        mod_row.append(row[i])
                return tuple(mod_row)
            
            # Train or val data
            else:
                if isinstance(row["start"], pd.Period):  # Replace with actual column name
                    return {"start": torch.tensor(row["start"].to_timestamp().timestamp(), dtype=torch.float32),  # Convert to float timestamp
                    "target": torch.tensor(row["target"], dtype=torch.float),
                    "freq": row["start"].freqstr,
                    "item_id": row["item_id"]
                    }
            
                return row
        
        # In case of dictionary data
        elif isinstance(self.data, dict):
            row = {k: v[idx] for k, v in self.data.items()}
            if isinstance(row["start"], pd.Period):
                row["start"] = torch.tensor([x.to_timestamp().timestamp() for x in iter(self.data["start"])], dtype=torch.float32)

            return row