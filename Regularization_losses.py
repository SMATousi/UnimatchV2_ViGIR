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


import torch
import torch.nn as nn

class GradientPenaltyLoss(nn.Module):
    def __init__(self, entropy_qz=None):
        super(GradientPenaltyLoss, self).__init__()
        self.entropy_qz = entropy_qz

    def forward(self, embeddings, y_pred):
        # Initialize total loss to zero
        total_loss = 0.0

        # Iterate over each embedding in the tuple
        for embedding in embeddings:
            # Ensure that each embedding requires gradient
            if not embedding.requires_grad:
                raise ValueError("Each embedding must require gradients.")

            # Compute squared prediction error
            pred_loss = torch.square(y_pred)

            # Calculate gradients of pred_loss with respect to this embedding
            grad_pred_loss = torch.autograd.grad(outputs=pred_loss, inputs=embedding,
                                                 grad_outputs=torch.ones_like(pred_loss),
                                                 create_graph=True, allow_unused=True)[0]

            # Handle the case where gradients are unused (i.e., None)
            if grad_pred_loss is None:
                grad_pred_loss = torch.zeros_like(embedding)

            # Normalize the gradients
            norm = torch.norm(grad_pred_loss, p=2, dim=-1, keepdim=True) + 1e-8
            normalized_grad = grad_pred_loss / norm
            grad_squared = torch.square(normalized_grad)
            
            # Apply entropy weighting if provided
            if self.entropy_qz is not None:
                weighted_grad_squared = self.entropy_qz * grad_squared
            else:
                weighted_grad_squared = grad_squared
            
            # Sum the loss over all embeddings
            total_loss += torch.mean(weighted_grad_squared)

        # Average the loss over the number of embeddings to normalize scale
        loss = total_loss / len(embeddings)
        
        return loss
