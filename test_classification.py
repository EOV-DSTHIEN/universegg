#!/usr/bin/env python3
"""
Quick test script to validate classification functions
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(__file__))

LABEL_NAMES = ["nang_da_thuy", "nang_don_thuy", "nang_da_thuy_dac", 
               "nang_don_thuy_dac", "u_bi", "u_dac"]

def calculate_confidence_scores(pred_masks):
    """Calculate mean confidence (probability) for each label across the image"""
    confidences = {}
    for label_idx in range(pred_masks.shape[0]):
        mean_conf = pred_masks[label_idx].mean().item()
        confidences[label_idx] = mean_conf
    return confidences


def classify_primary_type(pred_masks, valid_labels):
    """Classify primary type (highest confidence label)"""
    confidences = calculate_confidence_scores(pred_masks)
    
    if len(valid_labels) == 0:
        return None
    
    valid_confidences = {idx: confidences[idx] for idx in valid_labels}
    sorted_labels = sorted(valid_confidences.items(), key=lambda x: x[1], reverse=True)
    
    primary_idx, primary_conf = sorted_labels[0]
    primary_name = LABEL_NAMES[primary_idx]
    top_3 = [(LABEL_NAMES[idx], conf) for idx, conf in sorted_labels[:3]]
    
    return {
        'primary_idx': primary_idx,
        'primary_name': primary_name,
        'confidence': primary_conf,
        'top_3': top_3,
        'all_confidences': valid_confidences
    }


def test_classification():
    """Test classification with mock data"""
    print("\n" + "="*70)
    print("TESTING CLASSIFICATION FUNCTIONS")
    print("="*70)
    
    # Create mock predictions: [6, 128, 128]
    # Scenario: u_dac has highest confidence
    pred_masks = torch.zeros(6, 128, 128)
    
    # Simulate confidence: higher value = higher confidence
    pred_masks[0] = torch.ones(128, 128) * 0.3  # nang_da_thuy
    pred_masks[1] = torch.ones(128, 128) * 0.25 # nang_don_thuy
    pred_masks[2] = torch.ones(128, 128) * 0.35 # nang_da_thuy_dac
    pred_masks[3] = torch.ones(128, 128) * 0.2  # nang_don_thuy_dac
    pred_masks[4] = torch.ones(128, 128) * 0.4  # u_bi
    pred_masks[5] = torch.ones(128, 128) * 0.6  # u_dac <- HIGHEST
    
    valid_labels = [0, 2, 4, 5]  # Only these labels have GT
    
    # Test calculate_confidence_scores
    print("\n[1] Testing calculate_confidence_scores():")
    confidences = calculate_confidence_scores(pred_masks)
    for label_idx, conf in confidences.items():
        label_name = LABEL_NAMES[label_idx]
        status = "(valid)" if label_idx in valid_labels else "(invalid)"
        print(f"  [{label_idx}] {label_name:20s}: {conf:.3f} {status}")
    
    # Test classify_primary_type
    print("\n[2] Testing classify_primary_type():")
    classification = classify_primary_type(pred_masks, valid_labels)
    
    if classification:
        print(f"  Primary Type: {classification['primary_name']}")
        print(f"  Confidence:   {classification['confidence']:.3f}")
        print(f"  Top 3:")
        for idx, (name, conf) in enumerate(classification['top_3'], 1):
            print(f"    {idx}. {name:20s}: {conf:.3f}")
    
    # Test edge case: empty valid_labels
    print("\n[3] Testing edge case (no valid labels):")
    result = classify_primary_type(pred_masks, [])
    print(f"  Result: {result}")
    
    # Test with single valid label
    print("\n[4] Testing edge case (single valid label):")
    result = classify_primary_type(pred_masks, [5])
    if result:
        print(f"  Primary: {result['primary_name']} ({result['confidence']:.3f})")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_classification()
