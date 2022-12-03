import torch.nn as nn
from .img_module import VGG_FeatureExtractor, BidirectionalLSTM
import torch
from fairseq.models import FairseqEncoder
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self,x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class CnnEncoderBase(nn.Module):
    """
    Transformer encoder consisting of *cfg.encoder.layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, embed_dim, layers = [3, 4, 23, 3]):
        super(CnnEncoderBase,self).__init__()
        self.embed_dim = embed_dim
        self.register_buffer("version", torch.Tensor([3]))
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, layers[0])
        self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(1, stride=1)
        self.avgpool = nn.AvgPool2d(2, stride=2)
        self.linear = nn.Linear(2048,self.embed_dim)

        # self.grid_embedding = PositionEmbeddingSine(cfg.decoder.embed_dim // 2, normalize=True)
        # self.embed_positions = (
        #     PositionalEmbedding(
        #         cfg.max_source_positions,
        #         2048,
        #         0,
        #         learned=cfg.encoder.learned_pos,
        #     )
        # )
        self.embed_positions = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.xavier_normal_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                # print(self.training)
        # self.dnn = DnCNN()

        # self.craft = CRAFT()#检测网络
        # if self.training:#如果是训练，就加载预训练模型
        #     self.craft.load_state_dict(copyStateDict(torch.load(cfg.pre_model_path, map_location='cuda')))
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)



    def get_pos_embedding(self, grids):
        bs = grids.shape[0]
        grid_embed = self.grid_embedding(grids)#这里不需要reshape，因为我们的cnn已经处理好格式了
        return grid_embed

    def forward(
        self,
        img_source,
    ):
        return self.forward_scriptable(img_source)


    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(self, img_source):
        # print('https://github.com/Yang-Liu1082/FDN.git',self.inplanes)
        # img_source = img_source.permute(0,3,1,2)
        # res_img = img_source - img_mask_source#已有图片减去背景图片
        # x = self.dnn(res_img)
        # res_x = x

        # polys_list = get_boxes(self.craft,img_source)
        # batch_img_tensor_list,batch_img_white_tensor_list = batch_process_box(polys_list,img_source,img_white_source)
        # batch_img_tensor_list = shared_dnn(batch_img_tensor_list,self.dnn)




        #Resnet101
        x = self.conv1(img_source) # B x C x H x W
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x) # (B, C, H, W)


        #变形
        bz, channel, height, width = x.shape
        time_step = height * width
        # x = x.view(bz, channel, time_step).permute(2,0,1).contiguous() # T x B x C
        x = x.view(bz, channel, time_step).permute(0,2,1).contiguous()

        x = self.linear(x)
        
        src_lengths = torch.ones(bz, device=x.device) * time_step
        padding_mask = torch.zeros((bz, time_step), device=x.device)


        return x,padding_mask