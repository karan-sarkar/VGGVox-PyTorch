from typing import Tuple
import torch
import torch.nn as nn

def compute_backbone_output_shape(backbone: nn.Module) -> Tuple[int]:
    """
    Compute the dimension of the feature space defined by a feature extractor.
    Args:
        backbone: feature extractor

    Returns:
        shape of the feature vector computed by the feature extractor for an instance

    """
    input_images = torch.ones((4, 1, 512, 300))
    output = backbone(input_images)
    final_shape = tuple(output.shape[1:])
    # print(final_shape)
    return final_shape