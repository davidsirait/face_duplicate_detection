# evaluate_accuracy.py
"""
Simple accuracy evaluation script for face recognition system
"""
import sys
import json
from pathlib import Path

# Add src to path so imports work when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.inference import FaceDuplicateDetector
from monitoring.metrics_tracker import MetricsTracker


def evaluate_accuracy(test_set_path="test_set.json", config_path="./config.yaml"):
    """
    Evaluate system accuracy on test set
    
    Returns precision, recall, f1_score
    """
    # Load test set
    if not Path(test_set_path).exists():
        print(f" Test set not found: {test_set_path}")
        return None
    
    with open(test_set_path, 'r') as f:
        test_set = json.load(f)
    
    print(f" Evaluating on {len(test_set)} test images...")
    print("=" * 60)
    
    # Initialize detector
    detector = FaceDuplicateDetector(config_path)
    metrics_tracker = MetricsTracker()
    
    # Counters
    tp = fp = fn = tn = 0
    
    # Evaluate each test case
    for i, test_case in enumerate(test_set, 1):
        image_path = test_case['image_path']
        ground_truth = test_case['has_duplicate']
        
        print(f"[{i}/{len(test_set)}] Testing: {Path(image_path).name}", end=" ... ")
        
        # Check if image exists
        if not Path(image_path).exists():
            print(f" SKIP (file not found)")
            continue
        
        try:
            # Run detection
            is_duplicate, match_info, top_matches = detector.check_duplicate(image_path)
            
            # Compare with ground truth
            if ground_truth and is_duplicate:
                tp += 1
                print(" TP (correct duplicate)")
            elif not ground_truth and is_duplicate:
                fp += 1
                print(" FP (false duplicate)")
            elif ground_truth and not is_duplicate:
                fn += 1
                print(" FN (missed duplicate)")
            else:  # not ground_truth and not is_duplicate
                tn += 1
                print(" TN (correct non-duplicate)")
        
        except Exception as e:
            print(f" ERROR: {e}")
            continue
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    total = tp + fp + fn + tn
    print(f"Total tested: {total}")
    print(f"True Positives (TP):  {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Negatives (TN):  {tn}")
    print()
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    
    print(f"Precision: {precision:.3f} ({precision*100:.1f}%)")
    print(f"Recall:    {recall:.3f} ({recall*100:.1f}%)")
    print(f"F1 Score:  {f1_score:.3f} ({f1_score*100:.1f}%)")
    print(f"Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
    print("=" * 60)
    
    # Log to metrics file
    metrics_tracker.log_metric(
        "accuracy_evaluation",
        f1_score,
        {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1_score, 3),
            "accuracy": round(accuracy, 3),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "test_set_size": total
        }
    )
    
    print(f"\n✓ Metrics logged to metrics.jsonl")
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "accuracy": accuracy
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate face recognition accuracy")
    parser.add_argument("--test-set", default="test_set.json", help="Path to test set JSON")
    parser.add_argument("--config", default="./config.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    results = evaluate_accuracy(args.test_set, args.config)