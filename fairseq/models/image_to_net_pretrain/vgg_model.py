import torch.nn as nn
from .img_module import VGG_FeatureExtractor, BidirectionalLSTM
import torch
class Model(nn.Module):

    def __init__(self, input_channel, output_channel, hidden_size, num_class = None):
        super(Model, self).__init__()
        """ FeatureExtraction """
        self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        self.SequenceModeling_output = hidden_size

        """ Conv1d """
        self.conv1 = nn.Conv1d(in_channels=output_channel, out_channels=output_channel, kernel_size=4,stride = 4)

        """ Prediction """
        if num_class is not None:
            self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)


    def forward(self, input, text = None):
        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)
        
        contextual_feature = contextual_feature.permute(0, 2, 1)
        contextual_feature = self.conv1(contextual_feature)
        contextual_feature = contextual_feature.permute(0, 2, 1)
        """ Prediction stage """
        # prediction = self.Prediction(contextual_feature.contiguous())
        src_lengths = torch.ones(contextual_feature.size(0), device=contextual_feature.device) *contextual_feature.size(1)
        padding_mask = torch.zeros((contextual_feature.size(0), contextual_feature.size(1)), device=contextual_feature.device)

        return contextual_feature,padding_mask
    
    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_cfg=None,
        args=None,
    ):
        """
        Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """
        
        model_state_dict = self.state_dict()
        
        initialized_keys = []
        for key in state_dict:
            if key in model_state_dict:  
                model_state_dict[key] = state_dict[key].to(model_state_dict[key].device)
                initialized_keys.append(key)
            
        return super().load_state_dict(model_state_dict)
