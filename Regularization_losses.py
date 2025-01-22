import torch
import torch.nn as nn


class NormalizedCompactnessNormLoss(nn.Module):
    def __init__(self):
        super(NormalizedCompactnessNormLoss, self).__init__()

    def forward(self, features_tuple):
        # Ensure the input is a tuple with exactly 4 tensors
        assert isinstance(features_tuple, tuple) and len(features_tuple) == 4, "Input must be a tuple of four tensors."

        # Calculate the normalized norm for each tensor
        normalized_norms = [torch.norm(f) / f.numel() for f in features_tuple]

        # Compute the mean of these normalized norms
        mean_normalized_norm = torch.mean(torch.stack(normalized_norms))

        # Return the mean normalized norm as the loss
        return mean_normalized_norm