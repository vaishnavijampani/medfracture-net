import os
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class SystemConfig:
    """Centralized System Configuration for Bone Fracture AI Pipeline."""
    
    # Project Paths
    PROJECT_NAME: str = "BoneFracture_AI_System"
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    CHECKPOINT_DIR: str = os.path.join(BASE_DIR, "checkpoints")
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "outputs")
    
    # Model Hyperparameters
    IMAGE_SIZE: Tuple[int, int] = (512, 512)
    NUM_CLASSES: int = 6  # Transverse, Oblique, Spiral, Comminuted, Hairline, Greenstick
    CLASS_NAMES: List[str] = field(default_factory=lambda: [
        "Transverse", "Oblique", "Spiral", "Comminuted", "Hairline", "Greenstick"
    ])
    
    # Training Parameters
    BATCH_SIZE: int = 8
    NUM_WORKERS: int = 2
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-4
    NUM_EPOCHS: int = 30
    SEED: int = 42
    DEVICE: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    
    # Image Preprocessing & Filtering
    CLAHE_CLIP_LIMIT: float = 3.0
    CLAHE_GRID_SIZE: Tuple[int, int] = (8, 8)
    PIXEL_SPACING_MM: float = 0.15  # Millimeters per pixel for physical length calculation
    
    # Clinical Severity Thresholds (Length in mm)
    SEVERITY_MILD_MAX_MM: float = 5.0
    SEVERITY_MODERATE_MAX_MM: float = 15.0

config = SystemConfig()

# Create directories if they do not exist
for path in [config.DATA_DIR, config.CHECKPOINT_DIR, config.OUTPUT_DIR]:
    os.makedirs(path, exist_ok=True)
