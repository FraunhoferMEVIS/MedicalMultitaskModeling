from typing import List, Protocol


class EncoderModel(Protocol):
    def get_feature_pyramid_channels(self) -> List[int]:
        """
        Returns the channels of the feature pyramid, starting with the channels of the input.
        """
        raise NotImplementedError

    def get_strides(self) -> List[int]:
        """
        Return the strides of the feature pyramid
        """
        raise NotImplementedError
