#!/usr/bin/env python3
"""
Test script to verify the standalone data collection pipeline is properly configured.
"""

import sys
import os
from pathlib import Path

def test_directory_structure():
    """Test that all required directories and files exist."""
    print("=" * 60)
    print("Testing Directory Structure")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    required_items = {
        'directories': [
            'assets',
            'assets/intri_result',
            'assets/tf_cali_result',
            'assets/cali_width_result',
            'config',
            'utils',
            'vitamin_b_data_collection_pipeline',
            'vitamin_b_data_collection_pipeline/utils',
        ],
        'files': [
            'README.md',
            'requirements.txt',
            'run_data_collection_pipeline.py',
            'config/VB_task_config.yaml',
            'vitamin_b_data_collection_pipeline/01_crop_img.py',
            'vitamin_b_data_collection_pipeline/04_get_aruco_pos.py',
            'vitamin_b_data_collection_pipeline/05_get_width.py',
            'vitamin_b_data_collection_pipeline/07_generate_dataset_plan.py',
            'vitamin_b_data_collection_pipeline/08_generate_replay_buffer.py',
            'assets/tf_cali_result/quest_2_ee_left_hand.npy',
            'assets/tf_cali_result/quest_2_ee_right_hand.npy',
        ]
    }
    
    all_pass = True
    
    # Check directories
    print("\nChecking directories:")
    for dir_path in required_items['directories']:
        full_path = base_dir / dir_path
        if full_path.exists() and full_path.is_dir():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} - NOT FOUND")
            all_pass = False
    
    # Check files
    print("\nChecking files:")
    for file_path in required_items['files']:
        full_path = base_dir / file_path
        if full_path.exists() and full_path.is_file():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - NOT FOUND")
            all_pass = False
    
    return all_pass

def test_imports():
    """Test that all required Python modules can be imported."""
    print("\n" + "=" * 60)
    print("Testing Python Imports")
    print("=" * 60)
    
    imports_to_test = [
        ('numpy', 'numpy'),
        ('scipy', 'scipy.interpolate'),
        ('opencv', 'cv2'),
        ('omegaconf', 'omegaconf'),
        ('zarr', 'zarr'),
        ('numcodecs', 'numcodecs'),
        ('transforms3d', 'transforms3d'),
        ('pyquaternion', 'pyquaternion'),
        ('click', 'click'),
        ('tqdm', 'tqdm'),
        ('pandas', 'pandas'),
        ('av', 'av'),
        ('pillow', 'PIL'),
    ]
    
    all_pass = True
    print("\nChecking Python packages:")
    for name, module in imports_to_test:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name} - NOT INSTALLED ({e})")
            all_pass = False
    
    return all_pass

def test_local_utils():
    """Test that local utility modules can be imported."""
    print("\n" + "=" * 60)
    print("Testing Local Utils Imports")
    print("=" * 60)
    
    # Add parent directory to path
    base_dir = Path(__file__).parent
    pipeline_dir = base_dir / 'vitamin_b_data_collection_pipeline'
    
    original_dir = os.getcwd()
    os.chdir(pipeline_dir)
    
    try:
        sys.path.insert(0, str(base_dir))
        
        local_utils = [
            ('utils.camera_device', True),  # (module, optional)
            ('utils.cv_util', False),
            ('utils.pose_util', False),
            ('utils.replay_buffer', False),
            ('utils.config_utils', False),
        ]
        
        all_pass = True
        print("\nChecking local utils:")
        for util_module, optional in local_utils:
            try:
                # Try importing from pipeline utils
                if '.' in util_module:
                    parent_module = util_module.split('.')[0]
                    __import__(util_module)
                    print(f"  ✓ {util_module}")
                else:
                    __import__(util_module)
                    print(f"  ✓ {util_module}")
            except ImportError as e:
                if optional:
                    print(f"  ⚠ {util_module} - OPTIONAL (needed only for live data collection)")
                else:
                    print(f"  ✗ {util_module} - IMPORT FAILED ({e})")
                    all_pass = False
        
        return all_pass
    finally:
        os.chdir(original_dir)

def test_config():
    """Test that configuration file is valid."""
    print("\n" + "=" * 60)
    print("Testing Configuration")
    print("=" * 60)
    
    try:
        from omegaconf import OmegaConf
        base_dir = Path(__file__).parent
        config_path = base_dir / 'config' / 'VB_task_config.yaml'
        
        cfg = OmegaConf.load(config_path)
        
        print("\nConfiguration loaded successfully:")
        print(f"  Task name: {cfg.task.name}")
        print(f"  Task type: {cfg.task.type}")
        print(f"  Output dir: {cfg.recorder.output}")
        print(f"  Visual resolution: {cfg.output_train_data.visual_out_res}")
        print(f"  Tactile resolution: {cfg.output_train_data.tactile_out_res}")
        
        # Check for absolute paths
        absolute_paths = []
        
        def check_paths(d, prefix=''):
            for key, value in d.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, str) and value.startswith('/home/'):
                    absolute_paths.append((full_key, value))
                elif isinstance(value, dict):
                    check_paths(value, full_key)
        
        check_paths(OmegaConf.to_container(cfg))
        
        if absolute_paths:
            print("\n  ⚠ Warning: Found absolute paths in config:")
            for key, path in absolute_paths:
                print(f"    {key}: {path}")
        else:
            print("  ✓ No absolute paths found in config")
        
        return True
    except Exception as e:
        print(f"  ✗ Configuration test failed: {e}")
        return False

def main():
    print("\n" + "=" * 60)
    print("ViTaMIn-B Data Collection Pipeline - Standalone Test")
    print("=" * 60)
    
    results = {
        'Directory Structure': test_directory_structure(),
        'Python Imports': test_imports(),
        'Local Utils': test_local_utils(),
        'Configuration': test_config(),
    }
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Pipeline is ready to use!")
    else:
        print("✗ SOME TESTS FAILED - Please fix the issues above")
        print("\nTo install missing dependencies:")
        print("  pip install -r requirements.txt")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
