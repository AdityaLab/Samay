import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from uni2ts.distribution import MixtureOutput
from typing import Callable, List, Dict, Any, Union, Type, Optional
from pandas._libs.tslibs.period import Period

# For custom transforms
import numpy as np
# for finetune class
import lightning as L

# ------------------- HELPER FUNCTIONS -------------------
def filter_dict(d:dict, keys:List[str], ignore_missing:bool=False):
    """Filters the dictionary d to only include the keys in the list keys.

    Args:
        d (dict): The source dictionary to filter.
        keys (List[str]): The keys to keep in the dictionary.
        ignore_missing (bool, optional): If a given key is not in d then do we ignore or not. Defaults to False.

    Returns:
        _type_: _description_
    """
    result = {}

    for key in keys:
        try:
            result[key] = d[key]
        except KeyError:
            if not ignore_missing:
                raise

    return result

def handle_distr_output(distr:dict):
    """Converts the distr_output dictionary to a DistributionOutput object."""
    if "_target_" in distr:
        return str(distr["_target_"].split(".")[-1]) + "(" + ",".join([]) +")"
    return None

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

def custom_train_instance_split(ts:np.ndarray,allow_empty_interval:bool=False, axis: int=-1, min_past: int=0, min_future: int=0):
    """Get the interval to consider for a given mode based on the prediction length (>= min_future) and the start of past values.
    Always selects the last time point for splitting i.e. the forecast point for the time series.
    (Based on PredictionSampler() from Gluonts)

    Args:
        ts (np.ndarray): Time series data.
        allow_empty_interval (bool, optional): If True, the sampled part containes empty intervals. Defaults to False.
        axis (int, optional): The dimension in which we want to sample. Defaults to -1.
        min_past (int, optional): Start of past values. Defaults to 0.
        min_future (int, optional): Min number of future. Defaults to 0.

    Returns:
        np.array: sampled indices
    """
    s, f = min_past, ts.shape[axis] - min_future
    assert allow_empty_interval or s <= f
    return np.array([f]) if s <= f else np.array([], dtype=int) 


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
                    mod_row.append({})
                    for k,v in row[i].items():
                        if k == "start" or k=="forecast_start":
                            v1 = v
                            if isinstance(v1, pd.Timestamp):
                                v1 = v1.timestamp() # converts to float (Unix timestamp)
                            elif isinstance(v1, Period):
                                v1 = v1.to_timestamp().timestamp()
                            mod_row[i][k] = v if isinstance(v, torch.Tensor) else torch.tensor(v1, dtype=torch.float32)
                        else:
                            mod_row[i][k] = v
                return tuple(mod_row)
            
            # Train or val data
            else:
                mod_row = {}
                for k,v in row.items():
                    if k == "start" or k=="forecast_start":
                        v1 = v
                        if isinstance(v1, pd.Timestamp):
                            v1 = v1.timestamp() # converts to float (Unix timestamp)
                        elif isinstance(v1, Period):
                            v1 = v1.to_timestamp().timestamp()
                        mod_row[k] = v if isinstance(v, torch.Tensor) else torch.tensor(v1, dtype=torch.float32)
                    else:
                        mod_row[k] = v
                return mod_row
        
        # In case of dictionary data
        elif isinstance(self.data, dict):
            row, mod_row = {k: v[idx] for k, v in self.data.items()}, {}
            for k,v in row.items():
                if k == "start" or k=="forecast_start":
                    v1 = v
                    if isinstance(v1, pd.Timestamp):
                        v1 = v1.timestamp() # converts to float (Unix timestamp)
                    elif isinstance(v1, Period):
                        v1 = v1.to_timestamp().timestamp()
                    mod_row[k] = v if isinstance(v, torch.Tensor) else torch.tensor(v1, dtype=torch.float32)
                else:
                    mod_row[k] = v
            return mod_row

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
            # raise ValueError(
            #     f'Input for field "{self.field}" does not have the required dimension '
            #     f"(field: {self.field}, ndim observed: {value.ndim}, expected ndim: {self.expected_ndim})"
            # )
            if self.expected_ndim == 1:
                value = value[..., None]

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