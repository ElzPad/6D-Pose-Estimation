"""
Compute ADD (Average Closest Point Distance) metrics from results.json

Uses precomputed ADD error values from results.json and compares them
against class-specific object diameters at different thresholds (0.1d, 0.2d, etc.)
"""
import json
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List

def load_models_info(models_info_path: str) -> Dict:
    """Load model information including diameter for ADD metric."""
    with open(models_info_path, 'r') as f:
        models_info = yaml.safe_load(f)
    return models_info


def load_results(results_path: str) -> List[Dict]:
    """Load prediction results from JSON file."""
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results

def compute_metrics(results_path: str, 
                   models_info_path: str,
                   thresholds: List[float] = None) -> Dict:
    """
    Compute ADD metrics at different thresholds.
    
    Args:
        results_path: Path to results.json
        models_info_path: Path to models_info.yml
        thresholds: List of thresholds in terms of object diameter (default: [0.1, 0.2, 0.3, 0.4, 0.5])
    
    Returns:
        Dictionary containing computed metrics
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Load data
    print("Loading data...")
    results = load_results(results_path)
    models_info = load_models_info(models_info_path)
    
    # Store metrics
    metrics = {
        'all_items': {f'ADD@{t:.1f}d': [] for t in thresholds},
        'per_class': {f'ADD@{t:.1f}d': {} for t in thresholds}
    }
    
    # Process each result
    print(f"Processing {len(results)} predictions...")
    successful_predictions = 0
    
    for i, result in tqdm(enumerate(results)):
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{len(results)}")
        
        obj_id = result['obj_id']
        file_name = result['file']
        
        # Parse frame_id from filename (e.g., "01_0005" -> frame_id=5)
        parts = file_name.split('_')
        frame_id = int(parts[-1])

        add_distance = float(result['error'])
        
        # Get model diameter
        diameter = models_info[obj_id]['diameter']
        
        # Evaluate at each threshold
        for threshold in thresholds:
            threshold_distance = threshold * diameter
            is_correct = add_distance <= threshold_distance
            
            threshold_key = f'ADD@{threshold:.1f}d'
            metrics['all_items'][threshold_key].append(is_correct)
            
            # Per-class metrics
            if obj_id not in metrics['per_class'][threshold_key]:
                metrics['per_class'][threshold_key][obj_id] = []
            metrics['per_class'][threshold_key][obj_id].append(is_correct)
        
        successful_predictions += 1
    
    print(f"Successfully processed {successful_predictions}/{len(results)} predictions\n")
    
    # Compute averages
    results_summary = {
        'total_predictions': len(results),
        'successful_predictions': successful_predictions,
        'all_items': {},
        'per_class': {}
    }
    
    # All items metrics
    print("=" * 60)
    print("ALL ITEMS METRICS")
    print("=" * 60)
    all_items_data = []
    for threshold_key, values in metrics['all_items'].items():
        if values:
            accuracy = np.mean(values)
            results_summary['all_items'][threshold_key] = accuracy
            all_items_data.append([threshold_key, f"{accuracy:.4f}", f"{accuracy*100:.2f}%"])
    
    # Print all items table
    print(f"{'Metric':<20} {'Accuracy':<12} {'Percentage':<12}")
    print("-" * 44)
    for row in all_items_data:
        print(f"{row[0]:<20} {row[1]:<12} {row[2]:<12}")
    
    # Per-class metrics
    print("\n" + "=" * 60)
    print("PER-CLASS METRICS")
    print("=" * 60)
    
    # Build table with objects as rows and thresholds as columns
    threshold_keys = sorted(metrics['per_class'].keys())
    
    # Collect all object IDs
    all_obj_ids = set()
    for class_dict in metrics['per_class'].values():
        all_obj_ids.update(class_dict.keys())
    all_obj_ids = sorted(all_obj_ids)
    
    # Calculate column widths
    col_width = 12
    header_line = "Object".ljust(12)
    for threshold_key in threshold_keys:
        header_line += threshold_key.ljust(col_width)
    
    print("\n" + header_line)
    print("-" * len(header_line))
    
    # Print rows
    for obj_id in all_obj_ids:
        row_line = f"Object {obj_id:02d}".ljust(12)
        for threshold_key in threshold_keys:
            values = metrics['per_class'][threshold_key].get(obj_id, [])
            if values:
                accuracy = np.mean(values)
                results_summary['per_class'].setdefault(threshold_key, {})[obj_id] = accuracy
                row_line += f"{accuracy:.4f}".ljust(col_width)
            else:
                row_line += "-".ljust(col_width)
        print(row_line)
    
    return results_summary

if __name__ == "__main__":
    # Paths
    project_root = Path(__file__).parent.parent
    results_path = project_root / "results.json"
    linemod_orig_root = project_root / "data" / "linemod" / "Linemod_preprocessed"
    models_info_path = linemod_orig_root / "models" / "models_info.yml"
    
    # Compute metrics
    metrics = compute_metrics(
        str(results_path),
        str(models_info_path),
        thresholds=[0.1, 0.2, 0.3, 0.4, 0.5]
    )