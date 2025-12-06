"""
Phase 3 Configuration
Centralized configuration for all adversarial attack analysis notebooks.

Usage in notebooks:
    import sys
    sys.path.insert(0, '/content/drive/MyDrive/Colab Notebooks')  # or your path
    from config import CONFIG, ATTACK_INFO, load_features, stage_zip, etc.
"""

import os
import sys
import pickle
import zipfile
from pathlib import Path

# =============================================================================
# Environment Detection
# =============================================================================

IN_COLAB = 'google.colab' in sys.modules

# =============================================================================
# Path Configuration
# =============================================================================

if IN_COLAB:
    DRIVE_ROOT = "/content/drive/MyDrive"
    DATA_ROOT = f"{DRIVE_ROOT}/Colab Notebooks/data"
    LOCAL_STAGING = "/content/staged_data"
else:
    # Update this for your local machine
    DATA_ROOT = "/Users/tyreecruse/Desktop/CS230/Project/Data"
    LOCAL_STAGING = DATA_ROOT

# Derived paths
ANALYSIS_ROOT = f"{DATA_ROOT}/analysis"
TRAINING_ROOT = f"{DATA_ROOT}/training/results from training"

# =============================================================================
# Main Configuration Dictionary
# =============================================================================

CONFIG = {
    # Input Paths
    "clip_features_dir": f"{ANALYSIS_ROOT}/features",
    "yolo_features_dir": f"{ANALYSIS_ROOT}/yolo_features",
    "zips_dir": f"{ANALYSIS_ROOT}/zips",
    "model_weights": f"{TRAINING_ROOT}/weights/best.pt",
    
    # Output Paths
    "results_dir": f"{ANALYSIS_ROOT}/results",
    "figures_dir": f"{ANALYSIS_ROOT}/figures",
    "staging_dir": LOCAL_STAGING,
    
    # Attack Configurations
    "fgsm_attacks": [
        "fgsm_030",  # ε = 0.030
        "fgsm_045",  # ε = 0.045
        "fgsm_060",  # ε = 0.060
        "fgsm_075",  # ε = 0.075
        "fgsm_090",  # ε = 0.090
        "fgsm_105",  # ε = 0.105
    ],
    "gaussian_attacks": [
        "gaussian_010",  # σ = 0.010
        "gaussian_050",  # σ = 0.050
        "gaussian_150",  # σ = 0.150
        "gaussian_200",  # σ = 0.200
        "gaussian_250",  # σ = 0.250
    ],
    "patch_attacks": ["patches"],
    
    # YOLOv8 Layer Config
    "yolo_extract_layers": [9, 15, 18, 21],
    
    # Analysis Settings
    "sigma_thresholds": [1, 2, 3, 4, 5],
    "default_sigma": 3,
    "alpha": 0.001,
    "random_seed": 42,
    
    # Detection settings
    "conf_threshold": 0.25,
    "iou_threshold": 0.5,
}

# Add combined attack list
CONFIG["all_attacks"] = (
    CONFIG["fgsm_attacks"] + 
    CONFIG["gaussian_attacks"] + 
    CONFIG["patch_attacks"]
)

# =============================================================================
# Attack Metadata
# =============================================================================

ATTACK_INFO = {
    "fgsm_030": {"type": "FGSM", "strength": 0.030},
    "fgsm_045": {"type": "FGSM", "strength": 0.045},
    "fgsm_060": {"type": "FGSM", "strength": 0.060},
    "fgsm_075": {"type": "FGSM", "strength": 0.075},
    "fgsm_090": {"type": "FGSM", "strength": 0.090},
    "fgsm_105": {"type": "FGSM", "strength": 0.105},
    "gaussian_010": {"type": "Gaussian", "strength": 0.010},
    "gaussian_050": {"type": "Gaussian", "strength": 0.050},
    "gaussian_150": {"type": "Gaussian", "strength": 0.150},
    "gaussian_200": {"type": "Gaussian", "strength": 0.200},
    "gaussian_250": {"type": "Gaussian", "strength": 0.250},
    "patches": {"type": "Patch", "strength": None},
}

# =============================================================================
# Utility Functions
# =============================================================================

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def get_clip_feature_path(attack_name):
    """Get path to CLIP feature file."""
    return f"{CONFIG['clip_features_dir']}/{attack_name}_features.pkl"


def get_yolo_feature_path(attack_name):
    """Get path to YOLOv8 feature file."""
    return f"{CONFIG['yolo_features_dir']}/{attack_name}_yolo_features.pkl"


def get_zip_path(attack_name):
    """Get path to image zip file."""
    return f"{CONFIG['zips_dir']}/{attack_name}.zip"


def load_features(attack_name, feature_type='clip'):
    """
    Load pre-extracted features.
    
    Args:
        attack_name: 'clean', 'fgsm_045', etc.
        feature_type: 'clip' or 'yolo'
    
    Returns:
        features: (N, D) numpy array
    """
    if feature_type == "clip":
        path = get_clip_feature_path(attack_name)
    else:
        path = get_yolo_feature_path(attack_name)
    
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    return data['features']


def stage_zip(attack_name, force=False):
    """
    Unzip dataset to staging area for faster I/O.
    
    Args:
        attack_name: 'clean', 'fgsm_045', etc.
        force: Re-extract even if already staged
    
    Returns:
        Path to staged directory
    """
    zip_path = get_zip_path(attack_name)
    staged_path = f"{CONFIG['staging_dir']}/{attack_name}"
    
    if os.path.exists(staged_path) and not force:
        print(f"✓ {attack_name} already staged")
        return staged_path
    
    print(f"📦 Staging {attack_name}...")
    ensure_dir(CONFIG['staging_dir'])
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(CONFIG['staging_dir'])
    
    print(f"✓ Extracted to {staged_path}")
    return staged_path


def parse_attack_name(attack_name):
    """
    Parse attack name into type and strength.
    
    Returns:
        (attack_type, strength) tuple
    """
    if attack_name in ATTACK_INFO:
        info = ATTACK_INFO[attack_name]
        return info['type'], info['strength']
    elif attack_name == 'clean':
        return 'Clean', None
    else:
        return 'Unknown', None


def print_config():
    """Print configuration summary."""
    print("=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"\nEnvironment: {'Google Colab' if IN_COLAB else 'Local'}")
    print(f"\nPaths:")
    print(f"  Data root:       {DATA_ROOT}")
    print(f"  CLIP features:   {CONFIG['clip_features_dir']}")
    print(f"  YOLOv8 features: {CONFIG['yolo_features_dir']}")
    print(f"  Zips:            {CONFIG['zips_dir']}")
    print(f"  Model:           {CONFIG['model_weights']}")
    print(f"  Results:         {CONFIG['results_dir']}")
    print(f"  Figures:         {CONFIG['figures_dir']}")
    print(f"\nAttacks:")
    print(f"  FGSM:     {len(CONFIG['fgsm_attacks'])} attacks")
    print(f"  Gaussian: {len(CONFIG['gaussian_attacks'])} attacks")
    print(f"  Patches:  {len(CONFIG['patch_attacks'])} attack")
    print(f"  Total:    {len(CONFIG['all_attacks'])} attacks")
    print(f"\nSettings:")
    print(f"  YOLOv8 layers:    {CONFIG['yolo_extract_layers']}")
    print(f"  Sigma thresholds: {CONFIG['sigma_thresholds']}")
    print(f"  Random seed:      {CONFIG['random_seed']}")
    print("=" * 60)


# =============================================================================
# Auto-create output directories
# =============================================================================

ensure_dir(CONFIG['results_dir'])
ensure_dir(CONFIG['figures_dir'])
ensure_dir(CONFIG['yolo_features_dir'])

if __name__ == "__main__":
    print_config()
