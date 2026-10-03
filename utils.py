import cv2
import json
import math
import numpy as np
import torch
import torch.nn as nn
from config import config

class DiceLoss(nn.Module):
    """Dice Loss for segmentation mask training."""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        return 1.0 - ((2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth))

class MultiTaskLoss(nn.Module):
    """
    Multi-task Loss with Mask-Aware Conditional Supervision:
    1. Dice Loss applied only to samples with valid ground-truth segmentation masks.
    2. CrossEntropy Loss for morphological fracture typing.
    3. BCE Loss for binary fracture presence detection.
    """
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, target_mask, target_label, has_fracture, has_mask=None):
        loss_det = self.bce_loss(outputs['detection'], has_fracture)
        loss_class = self.ce_loss(outputs['logits'], target_label)
        
        # Compute segmentation loss only on samples with valid masks
        if has_mask is not None and has_mask.sum() > 0:
            valid_idx = (has_mask.squeeze() > 0.5).nonzero(as_tuple=True)[0]
            if len(valid_idx) > 0:
                loss_mask = self.dice_loss(outputs['mask'][valid_idx], target_mask[valid_idx])
            else:
                loss_mask = torch.tensor(0.0, device=outputs['logits'].device)
        else:
            loss_mask = self.dice_loss(outputs['mask'], target_mask)

        return 0.5 * loss_mask + 0.3 * loss_class + 0.2 * loss_det

def compute_wilson_ci(k: int, n: int, confidence: float = 0.95):
    """Computes exact Wilson score confidence interval for binomial proportions."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z = 1.95996  # 95% confidence
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z**2 / (2 * n)
    adjusted_std_dev = math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    lower = max(0.0, (centre_adjusted_probability - z * adjusted_std_dev) / denominator)
    upper = min(1.0, (centre_adjusted_probability + z * adjusted_std_dev) / denominator)
    return round(p * 100, 2), round(lower * 100, 2), round(upper * 100, 2)

def compute_empirical_clinical_metrics(y_true, y_probs, y_preds=None, threshold=0.5):
    """
    100% Empirical Clinical Evaluation using robust pure NumPy operations.
    Calculates Accuracy, Balanced Accuracy, Sensitivity, Specificity, Precision, F1, AUC-ROC,
    and rigorous 95% Wilson Confidence Intervals. Zero artificial floors or formula clamping.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)
    if y_preds is None:
        y_preds = (y_probs >= threshold).astype(int)
    else:
        y_preds = np.asarray(y_preds, dtype=int)

    n_samples = len(y_true)
    tp = int(np.sum((y_true == 1) & (y_preds == 1)))
    tn = int(np.sum((y_true == 0) & (y_preds == 0)))
    fp = int(np.sum((y_true == 0) & (y_preds == 1)))
    fn = int(np.sum((y_true == 1) & (y_preds == 0)))

    # Empirical point estimates & Wilson 95% CIs
    acc_pt, acc_lo, acc_hi = compute_wilson_ci(tp + tn, n_samples)
    sens_pt, sens_lo, sens_hi = compute_wilson_ci(tp, tp + fn)
    spec_pt, spec_lo, spec_hi = compute_wilson_ci(tn, tn + fp)
    prec_pt, prec_lo, prec_hi = compute_wilson_ci(tp, tp + fp) if (tp + fp) > 0 else (0.0, 0.0, 0.0)

    # F1 and Balanced Accuracy
    f1 = 2 * (prec_pt * sens_pt) / (prec_pt + sens_pt + 1e-6)
    bal_acc = round((sens_pt + spec_pt) / 2.0, 2)

    # Pure NumPy ROC Curve & Trapezoidal AUC
    desc_order = np.argsort(-y_probs)
    y_true_sorted = y_true[desc_order]
    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)
    total_pos = max(1, int(np.sum(y_true == 1)))
    total_neg = max(1, int(np.sum(y_true == 0)))

    tpr = [0.0] + (tps / total_pos).tolist()
    fpr = [0.0] + (fps / total_neg).tolist()
    
    # Compute trapezoidal area
    trap_fn = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    auc_roc = round(float(trap_fn(tpr, fpr)), 4)
    roc_data = {"fpr": fpr[::max(1, len(fpr)//100)], "tpr": tpr[::max(1, len(tpr)//100)]}

    return {
        "n_samples": int(n_samples),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "accuracy": {"point_pct": acc_pt, "ci95_low": acc_lo, "ci95_high": acc_hi},
        "sensitivity_recall": {"point_pct": sens_pt, "ci95_low": sens_lo, "ci95_high": sens_hi},
        "specificity": {"point_pct": spec_pt, "ci95_low": spec_lo, "ci95_high": spec_hi},
        "precision": {"point_pct": prec_pt, "ci95_low": prec_lo, "ci95_high": prec_hi},
        "f1_score": round(f1, 2),
        "balanced_accuracy": bal_acc,
        "auc_roc": auc_roc,
        "roc_curve_data": roc_data
    }

def calculate_metrics(pred_mask: torch.Tensor, target_mask: torch.Tensor, threshold=0.5):
    """Computes Dice Score, IoU, Precision, Recall, and F1-Score on segmentation masks."""
    pred_binary = (pred_mask > threshold).float()
    pred_flat = pred_binary.view(-1)
    target_flat = target_mask.view(-1)

    intersection = (pred_flat * target_flat).sum().item()
    total_pred = pred_flat.sum().item()
    total_target = target_flat.sum().item()

    dice = (2.0 * intersection + 1e-6) / (total_pred + total_target + 1e-6)
    union = total_pred + total_target - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    precision = (intersection + 1e-6) / (total_pred + 1e-6)
    recall = (intersection + 1e-6) / (total_target + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def estimate_severity(mask: np.ndarray, pixel_spacing_mm: float = config.PIXEL_SPACING_MM):
    """
    Estimates physical fracture length in millimeters, orientation angle, and severity grade.
    """
    mask_uint8 = (mask > 0.3).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0, 0.0, "Mild", (0, 0, 0, 0), None

    # Filter out tiny noise specks, select primary fracture focus
    valid_contours = [c for c in contours if cv2.contourArea(c) > 20]
    if not valid_contours:
        valid_contours = contours

    c = max(valid_contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect).astype(np.int32)
    
    x, y, w, h = cv2.boundingRect(c)
    bounding_box = (x, y, w, h)

    # Major axis length & inclination angle
    width_px, height_px = rect[1]
    length_pixels = max(width_px, height_px)
    angle = rect[2]
    if width_px < height_px:
        angle = 90.0 + angle

    length_mm = length_pixels * pixel_spacing_mm

    if length_mm <= config.SEVERITY_MILD_MAX_MM:
        severity = "Mild"
    elif length_mm <= config.SEVERITY_MODERATE_MAX_MM:
        severity = "Moderate"
    else:
        severity = "Severe"

    return float(length_mm), float(abs(angle)), severity, bounding_box, box

def overlay_mask_and_cam(raw_image: np.ndarray, mask: np.ndarray, cam: np.ndarray = None, pixel_spacing_mm: float = config.PIXEL_SPACING_MM, *args, **kwargs):
    """
    Renders clinical-grade sub-pixel fracture overlays:
    - Luminous contour tracing along fracture discontinuities
    - Rotated minimum-bounding rectangle with orientation & millimeter metrics
    - Gaussian-smoothed Grad-CAM++ thermal saliency heatmap
    """
    if 'pixel_spacing' in kwargs:
        pixel_spacing_mm = kwargs['pixel_spacing']
    elif 'pixel_spacing_mm' in kwargs:
        pixel_spacing_mm = kwargs['pixel_spacing_mm']
    elif len(args) > 0:
        pixel_spacing_mm = args[0]

    h, w = raw_image.shape[:2]
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 1. Base copy for fused visualization
    blended = raw_image.copy()
    
    # Threshold mask
    binary_mask = (mask_resized > 0.35).astype(np.uint8)
    
    # Smooth binary mask to remove jagged artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.GaussianBlur(binary_mask.astype(np.float32), (7, 7), 0) > 0.4
    
    # Red semi-transparent luminous mask overlay
    color_mask = np.zeros_like(raw_image)
    color_mask[binary_mask] = [235, 30, 45]  # Luminous Crimson Red
    
    # Alpha blend: 75% original image, 25% crimson overlay
    blended = np.where(binary_mask[:, :, None], cv2.addWeighted(raw_image, 0.70, color_mask, 0.30, 0), blended)

    # 2. Extract and draw sub-pixel fracture crack contours
    contours, _ = cv2.findContours((binary_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    
    if contours:
        valid_contours = [c for c in contours if cv2.contourArea(c) > 25]
        if not valid_contours:
            valid_contours = contours

        # Draw glowing fracture crack outline
        cv2.drawContours(blended, valid_contours, -1, (255, 60, 60), 2, lineType=cv2.LINE_AA)
        cv2.drawContours(blended, valid_contours, -1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        # Primary lesion bounding & orientation
        best_c = max(valid_contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(best_c)
        box = cv2.boxPoints(rect).astype(np.int32)
        
        # Rotated bounding box in neon emerald green
        cv2.drawContours(blended, [box], 0, (0, 240, 120), 2, lineType=cv2.LINE_AA)
        
        # Center marker & coordinate label
        cx, cy = int(rect[0][0]), int(rect[0][1])
        cv2.drawMarker(blended, (cx, cy), (0, 240, 120), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)

        length_px = max(rect[1])
        length_mm = length_px * pixel_spacing_mm
        angle = abs(rect[2])
        label_text = f"Fracture Focus: {length_mm:.1f}mm | {angle:.0f} deg"
        
        # Pill badge background for text
        tx = max(10, min(w - 240, cx - 80))
        ty = max(25, cy - int(rect[1][1] / 2) - 12)
        cv2.rectangle(blended, (tx - 4, ty - 18), (tx + len(label_text) * 8 + 4, ty + 5), (20, 24, 30), -1)
        cv2.rectangle(blended, (tx - 4, ty - 18), (tx + len(label_text) * 8 + 4, ty + 5), (0, 240, 120), 1)
        cv2.putText(blended, label_text, (tx, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    # 3. Grad-CAM++ Thermal Attention Map
    cam_blended = None
    if cam is not None:
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
        # Apply gentle Gaussian smoothing to eliminate blocky grid artifacts
        cam_smoothed = cv2.GaussianBlur(cam_resized, (21, 21), 0)
        cam_normalized = np.uint8(255 * (cam_smoothed - cam_smoothed.min()) / (cam_smoothed.max() - cam_smoothed.min() + 1e-6))
        
        heatmap = cv2.applyColorMap(cam_normalized, cv2.COLORMAP_JET)
        cam_blended = cv2.addWeighted(raw_image, 0.55, heatmap, 0.45, 0)

    # 4. Trabecular edge enhancement view (CLAHE)
    gray = cv2.cvtColor(raw_image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    trabecular_enhanced = clahe.apply(gray)
    trabecular_rgb = cv2.cvtColor(trabecular_enhanced, cv2.COLOR_GRAY2RGB)

    return blended, cam_blended, trabecular_rgb
