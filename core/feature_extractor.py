import torch
from torchvision import models

class FeatureExtractor(object):
    """Extract features of images from dataset at a specified layer of a specified model

    Args:
        model_name (str): name of the model you want to use for extracting features (default: resnet101).
            Supported models: vgg16, vgg19, resnet50, resnet101, resnet152.

        layer (int): reversed index of the layer that you want to extract the feature from.
    """
    def __init__(self, model_name='resnet101', layer=3):
        if model_name.lower() == 'vgg16':
            orig_model = models.vgg16(pretrained=True)
        elif model_name.lower() == 'vgg19':
            orig_model = models.vgg19(pretrained=True)
        elif model_name.lower() == 'resnet50':
            orig_model = models.resnet50(pretrained=True)
        elif model_name.lower() == 'resnet152':
            orig_model = models.resnet152(pretrained=True)
        else:
            orig_model = models.resnet101(pretrained=True)

        self.model = torch.nn.Sequential(*list(orig_model.children())[:-layer])

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.cuda()
        self.model.eval()

    def __call__(self, images):
        images = torch.autograd.Variable(images.cuda())
        features = self.model(images)
        
        return features.permute(0, 2, 3, 1)
