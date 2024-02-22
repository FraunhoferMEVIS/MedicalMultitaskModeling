from typing import Dict, Any, Sequence, Union
import numpy as np
import albumentations as A

# A step metric dict can be aggregated by concatenating the arrays for each key
# using `utils.flatten_list_of_dicts`
StepMetricDict = Dict[str, np.ndarray]
# A step feedback dict can be processed by trainers to give training feedback during a loop
StepFeedbackDict = Dict[str, Any]

TransformsSeqType = Sequence[Union[A.BasicTransform, A.BaseCompose]]
