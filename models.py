import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torchvision_models
from config import config

class ConvBlock(nn.Module):
    """Basic Convolutional Block with BatchNorm and LeakyReLU."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class FractureUNetPlusPlus(nn.Module):
    """
    U-Net++ Nested Architecture for Fine-Grained Bone Fracture Segmentation.
    Nested skip pathways capture fine trabecular boundary details.
    """
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = ConvBlock(in_channels, nb_filter[0])
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4])

        self.conv0_1 = ConvBlock(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = ConvBlock(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = ConvBlock(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = ConvBlock(nb_filter[3] + nb_filter[4], nb_filter[3])

        self.conv0_2 = ConvBlock(nb_filter[0]*2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = ConvBlock(nb_filter[1]*2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = ConvBlock(nb_filter[2]*2 + nb_filter[3], nb_filter[2])

        self.conv0_3 = ConvBlock(nb_filter[0]*3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = ConvBlock(nb_filter[1]*3 + nb_filter[2], nb_filter[1])

        self.conv0_4 = ConvBlock(nb_filter[0]*4 + nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return self.sigmoid(output)

class FractureMultiTaskNet(nn.Module):
    """
    Multi-Task Deep Learning Model:
    1. U-Net++ Segmenter for spatial fracture masking.
    2. ResNet Backboned Classifier for 6-class Fracture Typing and binary detection.
    Supports ImageNet transfer learning (pretrained=True) or from-scratch random initialization (pretrained=False).
    """
    def __init__(self, num_classes=config.NUM_CLASSES, pretrained=True):
        super().__init__()
        self.pretrained = pretrained
        self.segmenter = FractureUNetPlusPlus(in_channels=3, out_channels=1)
        
        # Classification Backbone (Transfer Learning vs Ablation Scratch)
        if pretrained:
            self.backbone = torchvision_models.resnet34(weights=torchvision_models.ResNet34_Weights.DEFAULT)
        else:
            self.backbone = torchvision_models.resnet34(weights=None)
            
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Dual Heads
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.detector = nn.Linear(num_ftrs, 1) # Binary presence score

    def forward(self, x):
        mask_pred = self.segmenter(x)
        features = self.backbone(x)
        class_logits = self.classifier(features)
        det_logits = self.detector(features)  # Raw logits for numerically stable BCEWithLogitsLoss
        return {
            'mask': mask_pred,
            'logits': class_logits,
            'det_logits': det_logits,
            'detection': torch.sigmoid(det_logits)  # Probability for inference and downstream evaluation
        }

class GradCAMExplainer:
    """
    Visual Explainability Engine supporting both Standard Grad-CAM and Grad-CAM++.
    Grad-CAM++ (Chattopadhay et al., 2018) calculates higher-order partial derivative weights
    to capture multi-instance fracture focal points and micro-crack regions.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        
        # Target layer for ResNet layer4
        target_layer = self.model.backbone.layer4[-1]
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None, method="gradcam++"):
        self.model.zero_grad()
        output = self.model(input_tensor)
        logits = output['logits']
        
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()
            
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

        grads = self.gradients.detach().cpu().numpy()[0] # (C, H, W)
        acts = self.activations.detach().cpu().numpy()[0] # (C, H, W)

        if method.lower() == "gradcam++":
            # True Grad-CAM++ formulation:
            # alpha_k = grads^2 / (2 * grads^2 + sum(acts * grads^3) + eps)
            # w_k = sum_ij(alpha_k * relu(grads))
            grads_power_2 = grads ** 2
            grads_power_3 = grads ** 3
            sum_acts = np.sum(acts, axis=(1, 2), keepdims=True)
            eps = 1e-7

            denom = 2.0 * grads_power_2 + sum_acts * grads_power_3 + eps
            denom = np.where(denom == 0.0, eps, denom)
            alphas = grads_power_2 / denom

            rel_grads = np.maximum(grads, 0)
            weights = np.sum(alphas * rel_grads, axis=(1, 2))
        else:
            # Standard Grad-CAM: Global average pooling over spatial gradients
            weights = np.mean(grads, axis=(1, 2))

        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i, :, :]

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam
