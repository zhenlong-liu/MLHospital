import torchvision
import torch.nn as nn
from .model_custom import TexasClassifier, PurchaseClassifier
from .resnet import resnet20

from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    wide_resnet50_2,
    vgg11,
    vgg13,
    vgg16,
    vgg19,
    vit_b_16,
    vit_b_32,
    vit_l_16,
    vit_l_32,
)

def get_model(name="resnet18", num_classes=10, dropout=None, pretrained = False):
    print("==> Building model...")
    
    if(name == "TexasClassifier"):
        model= TexasClassifier(num_classes = num_classes, droprate = dropout)
    elif(name == "PurchaseClassifier"):
        model= PurchaseClassifier(num_classes = num_classes, droprate = dropout)
    else:
        # backbone
        model = get_model_backbone(name, pretrained=pretrained)
        # classification task
        model = modify_classifier(name, model, num_classes, dropout)
    return model
    


def get_model_backbone(name, pretrained=False):
    if "resnet" in name.lower():
        if "18" in name.lower():
            model = resnet18(weights=pretrained)
        elif "20" in name.lower():
            model = resnet20(weights=pretrained)
        elif "34" in name.lower():
            model = resnet34(weights=pretrained)
        elif "50" in name.lower():
            if("wide" in name.lower()):
                model = wide_resnet50_2(eights=pretrained)
            else:
                model = resnet50(weights=pretrained)
        elif "101" in name.lower():
            model = resnet101(weights=pretrained)
        elif "152" in name.lower():
            model = resnet152(weights=pretrained)
            
    elif "vgg" in name.lower():
        if "11" in name.lower():
            model = vgg11(weights=pretrained)
        elif "13" in name.lower():
            model = vgg13(weights=pretrained)
        elif "16" in name.lower():
            model = vgg16(weights=pretrained)
        elif "19" in name.lower():
            model = vgg19(weights=pretrained)
        
    elif "vit" in name.lower():
        if "base" in name.lower():
            if "16" in name.lower():
                model = vit_b_16(weights="IMAGENET1K_V1")
            elif "32" in name.lower():
                model = vit_b_32(weights="IMAGENET1K_V1")
        elif "large" in name.lower():
            if "16" in name.lower():
                model = vit_l_16(weights="IMAGENET1K_V1")
            elif "32" in name.lower():
                model = vit_l_32(weights="IMAGENET1K_V1")
    return model

def modify_classifier(name, model, num_classes, dropout=None):
    if "resnet" in name.lower():
        if(dropout is None):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            model.fc = nn.Sequential(
                nn.Linear(model.fc.in_features, num_classes),
                nn.Dropout(dropout)
            )
    elif "vgg" in name.lower():
        if(dropout is None):
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features.in_features, num_classes)
        else:
            model.classifier[-1] = nn.Sequential(
                nn.Linear(model.classifier[-1].in_features, num_classes),
                nn.Dropout(dropout)
            )
    elif "vit" in name.lower():
        if(dropout is None):
            model.heads[-1] = nn.Linear(model.heads[-1].in_features, num_classes)
        else:
            model.heads[-1] = nn.Sequential(
                nn.Linear(model.heads[-1].in_features, num_classes),
                nn.Dropout(dropout)
            )
    else:
        print("We don't have this model yet.")

    return model