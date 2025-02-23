import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from uni2ts.distribution import MixtureOutput
from typing import Callable, List, Dict, Any, Union, Type, Optional

# For custom transforms
import numpy as np
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
                    row["start"] = torch.tensor(row["start"].to_timestamp().timestamp(), dtype=torch.float32),  # Convert to float timestamp
                return row
        
        # In case of dictionary data
        elif isinstance(self.data, dict):
            row = {k: v[idx] for k, v in self.data.items()}
            if isinstance(row["start"], pd.Period):
                row["start"] = torch.tensor([x.to_timestamp().timestamp() for x in iter(self.data["start"])], dtype=torch.float32)

            return row

# ------------------- CUSTOM TRANSFORMS -------------------
class CausalMeanNaNFix:
    """
    Replaces each missing value with the average of all values up to that point.

    - If the first values are missing, they are replaced by the closest non-missing value.
    - Uses a cumulative mean approach to maintain causality.

    Parameters
    ----------
    imputation_value : float, optional
        Default value to replace missing values when all values are NaN.
    """

    def __init__(self, imputation_value: float = 0.0):
        self.imputation_value = imputation_value

    def __call__(self, values: np.ndarray) -> np.ndarray:
        """Apply causal mean imputation."""
        if len(values) == 1 or np.isnan(values).all():
            return np.full_like(values, self.imputation_value)  # Replace all NaNs with default value

        mask = np.isnan(values)  # Identify missing values

        # Step 1: Replace initial NaNs using forward fill (last observed value)
        values = self.forward_fill(values)

        # Step 2: Compute cumulative mean for non-NaN values
        cumsum = np.cumsum(np.concatenate(([0.0], values[:-1])))  # Shifted cumulative sum
        indices = np.arange(1, len(values) + 1)  # Indices for division
        causal_mean = cumsum / indices  # Compute causal mean

        # Step 3: Replace NaNs with the computed causal mean
        values[mask] = causal_mean[mask]

        return values

    @staticmethod
    def forward_fill(values: np.ndarray) -> np.ndarray:
        """Replaces leading NaNs with the first non-NaN value (forward fill)."""
        if np.isnan(values[0]):
            first_valid = np.where(~np.isnan(values))[0]
            if len(first_valid) > 0:
                values[: first_valid[0]] = values[first_valid[0]]  # Fill leading NaNs

        return values

class AsNumpy:
    """
    Converts the value of a field into a NumPy array.

    Parameters
    ----------
    expected_ndim : int
        Expected number of dimensions. Throws an exception if the number of
        dimensions does not match.
    dtype : np.dtype
        NumPy dtype to use.
    """

    def __init__(self, field: str, expected_ndim: int, dtype: Type = np.float32) -> None:
        self.field = field
        self.expected_ndim = expected_ndim
        self.dtype = dtype

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies transformation to the input data dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Input dictionary containing the field to be transformed.

        Returns
        -------
        Dict[str, Any]
            The updated dictionary with the field converted to a NumPy array.
        """
        value = np.asarray(data[self.field], dtype=self.dtype)

        # Validate number of dimensions
        if value.ndim != self.expected_ndim:
            raise ValueError(
                f'Input for field "{self.field}" does not have the required dimension '
                f"(field: {self.field}, ndim observed: {value.ndim}, expected ndim: {self.expected_ndim})"
            )

        data[self.field] = value
        return data

class ArrExpandDims:
    """
    Expands dimensions of a NumPy array at the specified axis.

    Parameters
    ----------
    field : str
        Field in the dictionary to modify.
    axis : Optional[int]
        Axis along which to expand dimensions (same as `np.expand_dims`).
        If `None`, the function does nothing.
    """

    def __init__(self, field: str, axis: Optional[int] = None) -> None:
        self.field = field
        self.axis = axis

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies transformation to the input data dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Input dictionary containing the field to be transformed.

        Returns
        -------
        Dict[str, Any]
            The updated dictionary with the field modified.
        """
        if self.axis is not None:
            data[self.field] = np.expand_dims(data[self.field], axis=self.axis)
        return data

class AddObservedValues:
    """
    Replaces missing values (NaNs) in a NumPy array using a specified imputation method
    and adds an "observed" indicator: 1 for observed values, 0 for missing values.

    Parameters
    ----------
    target_field : str
        Field for which missing values will be replaced.
    output_field : str
        Field name to use for the indicator.
    imputation_method : Optional[Callable[[np.ndarray], np.ndarray]]
        A function or callable class that takes a NumPy array and returns an imputed version.
        If set to None, no imputation is performed, and only the indicator is added.
    dtype : Type
        NumPy dtype to use for the indicator.
    """

    def __init__(
        self,
        target_field: str,
        output_field: str,
        imputation_method: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        dtype: Type = np.float32,
    ) -> None:
        self.target_field = target_field
        self.output_field = output_field
        self.imputation_method = imputation_method
        self.dtype = dtype

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Applies imputation and adds an observed values mask to the data dictionary."""
        value = data[self.target_field]
        nan_entries = np.isnan(value)

        # Apply imputation if a method is provided (whether function or callable class)
        if self.imputation_method is not None and nan_entries.any():
            # If the imputation method is a class, instantiate it
            if isinstance(self.imputation_method, type) and issubclass(self.imputation_method, object):
                imputation_instance = self.imputation_method()  # Instantiate if it's a class
                value = imputation_instance(value.copy())  # Call the instance
            else:
                # If it's a function, call it directly
                value = self.imputation_method(value.copy())  # Apply the function

            data[self.target_field] = value  # Update the original data field

        # Create the observed indicator (1 = observed, 0 = missing)
        data[self.output_field] = (~nan_entries).astype(self.dtype, copy=False)

        return data