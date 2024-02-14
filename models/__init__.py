# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build
#from cornet import get_cornet_model
from .nsd_utils import roi_maps

def build_model(args):
    return build(args)
