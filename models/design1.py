from monai.networks.nets.classifier import Classifier, Discriminator, Critic
import torch.nn as nn
import torch
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
import torch.nn.functional as F
from vit_pytorch.vit_3d import ViT