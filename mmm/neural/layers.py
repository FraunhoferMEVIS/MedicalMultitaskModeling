import torch
import torch.nn as nn
import torch.nn.functional as F


class StateLessBatchnorm(nn.BatchNorm2d):
    """
    Adaption of original batchnorm which should work without tracking stats.
    Architecture that use it need to provide full batches for inference.
    """

    def __init__(self, num_features, eps=0.00001, momentum=0.1, affine=True, device=None):
        track_running_stats = False
        dtype = None
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)

    @staticmethod
    def from_batchnorm(batchnorm: nn.BatchNorm2d):
        return StateLessBatchnorm(batchnorm.num_features, batchnorm.eps, batchnorm.momentum, batchnorm.affine)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # self._check_input_dim(input)

        # if self.training and self.track_running_stats:
        #     # TODO: if statement only here to tell the jit to skip emitting this when it is None
        #     if self.num_batches_tracked is not None:  # type: ignore[has-type]
        #         self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        #         if self.momentum is None:  # use cumulative moving average
        #             exponential_average_factor = 1.0 / float(self.num_batches_tracked)
        #         else:  # use exponential moving average
        #             exponential_average_factor = self.momentum
        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return F.batch_norm(
            input,
            None,  # No tracked mean
            None,  # No Tracked variance
            self.weight,
            self.bias,
            bn_training,
            self.momentum,
            self.eps,
        )
