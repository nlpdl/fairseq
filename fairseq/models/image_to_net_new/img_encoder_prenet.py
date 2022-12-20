import torch.nn as nn
from fairseq.models.image_to_net_new.img_module import ResNet_FeatureExtractor
import torch
class ImageFeatureExtraction(nn.Module):

    def __init__(self, input_channel, output_channel):
        super(ImageFeatureExtraction, self).__init__()
        """ FeatureExtraction """
        self.FeatureExtraction = ResNet_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1


    def forward(self, input):
        """ Feature extraction stage """
        input = input.permute(0,3,1,2)
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)
        src_lengths = torch.ones(visual_feature.size(0), device=visual_feature.device) * visual_feature.size(1)
        padding_mask = torch.zeros((visual_feature.size(0), visual_feature.size(1)), device=visual_feature.device)
        # visual_feature = visual_feature.transpose(0,1)
        return visual_feature,padding_mask
        # return {
        #     "encoder_out": [visual_feature],  # T x B x C
        #     "encoder_padding_mask": [padding_mask],  # B x T
        #     "src_lengths": [src_lengths], # B
        # }
