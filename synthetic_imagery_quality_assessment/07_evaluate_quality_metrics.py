import os
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import json

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent if "study1" in str(SCRIPT_DIR) else SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# -------------------------
# Configuration
# -------------------------
class Config:
    masks_dir = PROJECT_ROOT / "results" / "study1" / "masks"
    output_dir = PROJECT_ROOT / "results" / "study1" / "evaluation"
    
    income_levels = ['lic', 'mic', 'hic']

config = Config()

# -------------------------
# Metric Functions
# -------------------------
def compute_mask_metrics(pred_mask, true_mask):
    """
    Compute segmentation metrics between predicted and true masks.
    
    Original function from evaluation code.
    
    Args:
        pred_mask: Predicted damage mask (numpy array)
        true_mask: Ground truth damage mask (numpy array)
    
    Returns:
        Dictionary with IoU, Dice, Precision, Recall
    """
    pred = pred_mask.flatten().astype(int)
    true = true_mask.flatten().astype(int)
    
    return {
        'IoU': jaccard_score(true, pred, average='weighted', zero_division=0),
        'Dice': f1_score(true, pred, average='weighted', zero_division=0),
        'Precision': precision_score(true, pred, average='weighted', zero_division=0),
        'Recall': recall_score(true, pred, average='weighted', zero_division=0)
    }

def compute_ci(arr):
    """
    Compute 95% confidence interval.
    """
    mean = np.mean(arr)
    std_err = np.std(arr) / np.sqrt(len(arr))
    return (mean - 1.96 * std_err, mean + 1.96 * std_err)

# -------------------------
# Main Evaluation Function
# -------------------------
def evaluate_quality():
    """Evaluate synthetic image quality across income levels and mask types."""
    
    print("="*60)
    print("Evaluating Synthetic Image Quality")
    print("="*60)
    print(f"Masks directory: {config.masks_dir}")
    print(f"Output directory: {config.output_dir}\n")
    
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate both binary and multiclass
    for mask_type in ['binary', 'multiclass']:
        print(f"\n{'='*60}")
        print(f"Evaluating {mask_type.upper()} Masks")
        print(f"{'='*60}\n")
        
        all_results = []
        income_results = {income: [] for income in config.income_levels}
        
        # Evaluate each income level
        for income in config.income_levels:
            real_mask_dir = config.masks_dir / "real" / mask_type / income
            synthetic_mask_dir = config.masks_dir / "synthetic" / mask_type / income
            
            # Find matching mask pairs
            real_masks = {f.name: f for f in real_mask_dir.glob("*.png")}
            synthetic_masks = {f.name: f for f in synthetic_mask_dir.glob("*.png")}
            
            # Match by base filename (before mask suffix)
            pair_count = 0
            
            print(f"Processing {income.upper()}")
            
            for real_name, real_path in tqdm(real_masks.items(), desc=f"{income.upper()}"):
                # Find corresponding synthetic mask
                # Real: *_binary_mask.png or *_multiclass_mask.png
                # Synthetic: same naming
                synthetic_path = synthetic_mask_dir / real_name
                
                if not synthetic_path.exists():
                    continue
                
                try:
                    # Load masks
                    real_mask = np.array(Image.open(real_path).convert('L'))
                    synthetic_mask = np.array(Image.open(synthetic_path).convert('L'))
                    
                    # Compute metrics
                    metrics = compute_mask_metrics(synthetic_mask, real_mask)
                    
                    # Store results
                    result = {
                        'filename': real_name,
                        'income': income,
                        **metrics
                    }
                    
                    all_results.append(result)
                    income_results[income].append(metrics)
                    pair_count += 1
                    
                except Exception as e:
                    print(f"\n Error processing {real_name}: {e}")
                    continue
            
            print(f"  Evaluated {pair_count} mask pairs\n")
        
        if len(all_results) == 0:
            print(f"No results for {mask_type} masks")
            continue
        
        # Compute statistics per income level
        income_summaries = {}
        
        for income in config.income_levels:
            if len(income_results[income]) == 0:
                continue
            
            iou_vals = [m['IoU'] for m in income_results[income]]
            dice_vals = [m['Dice'] for m in income_results[income]]
            f1_vals = [m['F1'] for m in income_results[income]]
            prec_vals = [m['Precision'] for m in income_results[income]]
            rec_vals = [m['Recall'] for m in income_results[income]]
            
            income_summaries[income] = {
                'n_samples': len(income_results[income]),
                'IoU': {
                    'mean': float(np.mean(iou_vals)),
                    'median': float(np.median(iou_vals)),
                    'std': float(np.std(iou_vals)),
                    '95_ci': [float(x) for x in compute_ci(iou_vals)]
                },
                'Dice': {
                    'mean': float(np.mean(dice_vals)),
                    'median': float(np.median(dice_vals)),
                    'std': float(np.std(dice_vals)),
                    '95_ci': [float(x) for x in compute_ci(dice_vals)]
                },
                'F1': {
                    'mean': float(np.mean(f1_vals)),
                    'median': float(np.median(f1_vals)),
                    'std': float(np.std(f1_vals)),
                    '95_ci': [float(x) for x in compute_ci(f1_vals)]
                },
                'Precision': {
                    'mean': float(np.mean(prec_vals)),
                    'median': float(np.median(prec_vals)),
                    'std': float(np.std(prec_vals)),
                    '95_ci': [float(x) for x in compute_ci(prec_vals)]
                },
                'Recall': {
                    'mean': float(np.mean(rec_vals)),
                    'median': float(np.median(rec_vals)),
                    'std': float(np.std(rec_vals)),
                    '95_ci': [float(x) for x in compute_ci(rec_vals)]
                }
            }
        
        # Compute overall statistics
        all_iou = [r['IoU'] for r in all_results]
        all_dice = [r['Dice'] for r in all_results]
        all_f1 = [r['F1'] for r in all_results]
        all_prec = [r['Precision'] for r in all_results]
        all_rec = [r['Recall'] for r in all_results]
        
        overall_summary = {
            'n_samples': len(all_results),
            'IoU': {
                'mean': float(np.mean(all_iou)),
                'median': float(np.median(all_iou)),
                'std': float(np.std(all_iou)),
                '95_ci': [float(x) for x in compute_ci(all_iou)]
            },
            'Dice': {
                'mean': float(np.mean(all_dice)),
                'median': float(np.median(all_dice)),
                'std': float(np.std(all_dice)),
                '95_ci': [float(x) for x in compute_ci(all_dice)]
            },
            'F1': {
                'mean': float(np.mean(all_f1)),
                'median': float(np.median(all_f1)),
                'std': float(np.std(all_f1)),
                '95_ci': [float(x) for x in compute_ci(all_f1)]
            },
            'Precision': {
                'mean': float(np.mean(all_prec)),
                'median': float(np.median(all_prec)),
                'std': float(np.std(all_prec)),
                '95_ci': [float(x) for x in compute_ci(all_prec)]
            },
            'Recall': {
                'mean': float(np.mean(all_rec)),
                'median': float(np.median(all_rec)),
                'std': float(np.std(all_rec)),
                '95_ci': [float(x) for x in compute_ci(all_rec)]
            }
        }
        
        # Save results
        output_mask_dir = config.output_dir / mask_type
        output_mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Save per-image results
        with open(output_mask_dir / "per_image_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Save income-level summaries
        with open(output_mask_dir / "income_summaries.json", 'w') as f:
            json.dump(income_summaries, f, indent=2)
        
        # Save overall summary
        with open(output_mask_dir / "overall_summary.json", 'w') as f:
            json.dump(overall_summary, f, indent=2)
        
        # Print summary
        print(f"\n{mask_type.upper()} Results Summary:")
        print(f"{'='*60}")
        print(f"Overall (n={overall_summary['n_samples']}):")
        print(f"  IoU:       {overall_summary['IoU']['mean']:.4f} ± {overall_summary['IoU']['std']:.4f}")
        print(f"  Dice:      {overall_summary['Dice']['mean']:.4f} ± {overall_summary['Dice']['std']:.4f}")
        print(f"  Precision: {overall_summary['Precision']['mean']:.4f} ± {overall_summary['Precision']['std']:.4f}")
        print(f"  Recall:    {overall_summary['Recall']['mean']:.4f} ± {overall_summary['Recall']['std']:.4f}")
        
        print(f"\nBy Income Level:")
        for income, summary in income_summaries.items():
            print(f"\n  {income.upper()} (n={summary['n_samples']}):")
            print(f"    IoU:       {summary['IoU']['mean']:.4f} ± {summary['IoU']['std']:.4f}")
            print(f"    Dice:      {summary['Dice']['mean']:.4f} ± {summary['Dice']['std']:.4f}")
            print(f"    Precision: {summary['Precision']['mean']:.4f} ± {summary['Precision']['std']:.4f}")
            print(f"    Recall:    {summary['Recall']['mean']:.4f} ± {summary['Recall']['std']:.4f}")
        
        print(f"\nResults saved to: {output_mask_dir}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"All results saved to: {config.output_dir}")
    print("="*60 + "\n")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    evaluate_quality()
