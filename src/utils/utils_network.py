"""
Mask R-CNN utils.
"""
# =============================================================================
# Imports
# =============================================================================
import pandas as pd

import torch
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


DATA_CHARACTERISTICS = {
    "mean": torch.Tensor([0.2133, 0.2256, 0.1655]),
    "std": torch.Tensor([0.1672, 0.1721, 0.1526])
}

# =============================================================================
def set_pandas_display_options():
    """
    Sets display options for pandas dataframe
    """
    display = pd.options.display
    display.max_columns = 10
    display.max_rows = 10
    display.max_colwidth = 199
    display.width = 1000

    
def get_model_maskrcnn(num_classes, BOX_DETECTIONS_PER_IMG=200):
    """
    Returns Mask R-CNN model with a custom head.
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT", box_detections_per_img=BOX_DETECTIONS_PER_IMG)

    # replace pre-trained head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    # change normalization (transform layer)
    grcnn = torchvision.models.detection.transform.GeneralizedRCNNTransform(min_size=800, max_size=1333, 
                                                                            image_mean=DATA_CHARACTERISTICS["mean"],
                                                                            image_std=DATA_CHARACTERISTICS["std"])
    model.transform = grcnn
    
    return model


def get_model_maskrcnn_backbone(backbone, num_classes, BOX_DETECTIONS_PER_IMG=200):
    backbone  = resnet_fpn_backbone(backbone, pretrained=True, trainable_layers=None)
    print(backbone.eval())
    model = MaskRCNN(backbone=backbone, num_classes=2, box_detections_per_img=BOX_DETECTIONS_PER_IMG)

    # replace pre-trained head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    # change normalization (transform layer)
    grcnn = torchvision.models.detection.transform.GeneralizedRCNNTransform(min_size=800, max_size=1333, 
                                                                            image_mean=DATA_CHARACTERISTICS["mean"],
                                                                            image_std=DATA_CHARACTERISTICS["std"])
    model.transform = grcnn

    return model


def prediction(model, img):
    """
    Returns prediction of model for given image.
    """
    img = img.cuda() / 255
    prediction = model([img])

    prediction[0]["boxes"] = torch.Tensor(prediction[0]["boxes"].cpu().detach().numpy())
    prediction[0]["scores"] = torch.Tensor(prediction[0]["scores"].cpu().detach().numpy())
    prediction[0]["labels"] = torch.Tensor(prediction[0]["labels"].cpu().detach().numpy())
    prediction[0]["masks"] = torch.Tensor(prediction[0]["masks"].cpu().detach().numpy())

    return prediction[0]


