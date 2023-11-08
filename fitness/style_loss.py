# TAKEN FROM: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
import logging
import torch
import torch.nn.functional as F
from torchvision.models import vgg19, vgg16, VGG16_Weights, VGG19_Weights
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torch import nn
from cppn.normalization import Normalization

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    r"""Compute Gram matrix for batch of features.

    Args:
        x: Tensor. Shape :math:`(N, C, H, W)`.

    Returns:
        Gram matrix for given input
    """
    B, C, H, W = x.size()
    gram = []
    for i in range(B):
        features = x[i].view(C, H * W)

        # Add fake channel dimension
        G = torch.mm(features, features.t()).unsqueeze(0)
        G = G.div(C * H * W) # norm
        gram.append(G)

    return torch.stack(gram)


class StyleLoss(nn.Module):
    def __init__(self, model, device, target, weight=1):
        super().__init__()
        assert isinstance(model, torch.nn.Module) or model in ["vgg16", "vgg19", "mobilenet_v3_small"]
        self.device = device
        self.cnn = model
        self.weight = weight
        self.setup(target)
        
    def setup(self, target):
        if self.cnn == "mobilenet_v3_small":
            extractor = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).features
        if self.cnn == "vgg19":
            extractor = vgg19(weights=VGG19_Weights.DEFAULT).features
        elif self.cnn == "vgg16":
            extractor = vgg16(weights=VGG16_Weights.DEFAULT).features
        else:
            extractor = self.cnn
        
        extractor = extractor.to(self.device).eval()
        extractor.requires_grad = False
        
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        normalization = Normalization(self.device, cnn_normalization_mean, cnn_normalization_std)

        self.weight = self.weight
        self.target = target # just for assertion
        self.device = self.device
    
        style_losses = []

        model = nn.Sequential(normalization)

        i = 0  
        for layer in extractor.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in style_layers_default:
                # add style loss:
                target_feature = model(target).detach()
                style_loss = StyleLossValue(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], StyleLossValue):
                break
        self.model = model[:(i + 1)]
        self.losses = style_losses
    
    def forward(self, input, target):
        if not (self.target.shape == target.shape) or not torch.isclose(self.target, target, rtol=1e-3, atol=1e-3).all():
            # logging.warning("StyleLoss: target image changed")
            self.setup(target)
        self.model(input)
        style_score = torch.zeros(input.shape[0]).to(input)
        for sl in self.losses:
            style_score += sl.loss
        return torch.sqrt(torch.sqrt(style_score * self.weight)) 

class StyleLossValue(nn.Module):

    def __init__(self, target_feature):
        super(StyleLossValue, self).__init__()
        self.target_G = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        if G.shape[0] != self.target_G.shape[0]:
            if len(self.target_G.shape) == 2:
                self.target_G = self.target_G.unsqueeze(0) # add batch dimension for broadcasting
            self.loss = (G - self.target_G).pow(2).mean(dim=(1,2,))
        else:
            self.loss = F.mse_loss(G, self.target_G)
        return input
