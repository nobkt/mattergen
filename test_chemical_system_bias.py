#!/usr/bin/env python3
"""
Test script to reproduce and verify the fix for the chemical system bias issue.

This test reproduces the bug where using diffusion_guidance_factor=1.0-2.0 with 
chemical_system_mode="contains" causes all structures to become oxygen-only crystals.
"""

import torch
import numpy as np
from mattergen.common.data.chemgraph import ChemGraph
from mattergen.denoiser import mask_disallowed_elements
from mattergen.property_embeddings import ChemicalSystemMultiHotEmbedding


def create_test_batch():
    """Create a test batch with chemical system ['O'] conditioning."""
    batch_size = 16
    num_atoms = 32
    
    # Create logits for atomic numbers (simulating model output)
    # Shape: (num_atoms, MAX_ATOMIC_NUM + 1)
    logits = torch.randn(num_atoms, 119)  # MAX_ATOMIC_NUM is typically 118, +1 for mask
    
    # Create batch indices (which structure each atom belongs to)
    batch_idx = torch.repeat_interleave(torch.arange(batch_size), num_atoms // batch_size)
    
    # Create chemical system conditioning for oxygen ['O']
    chemical_system = ['O'] * batch_size
    
    # Create ChemGraph
    x = ChemGraph(
        num_atoms=torch.ones((batch_size, 1), dtype=torch.int) * (num_atoms // batch_size),
        chemical_system=chemical_system
    )
    
    return logits, x, batch_idx


def test_contains_mode_bias():
    """Test that contains mode doesn't over-bias toward oxygen."""
    logits, x, batch_idx = create_test_batch()
    
    # Test with chemical_system_mode="contains"
    masked_logits = mask_disallowed_elements(
        logits=logits.clone(),
        x=x,
        batch_idx=batch_idx,
        predictions_are_zero_based=True,
        chemical_system_mode="contains"
    )
    
    # Check that the bias is reasonable
    bias_applied = masked_logits - logits
    
    # Oxygen is element 8, so index 7 in 0-based
    oxygen_bias = bias_applied[:, 7]  # Oxygen column
    carbon_bias = bias_applied[:, 5]  # Carbon column (not in chemical system)
    
    print(f"Original oxygen logits mean: {logits[:, 7].mean():.3f}")
    print(f"Masked oxygen logits mean: {masked_logits[:, 7].mean():.3f}")
    print(f"Oxygen bias applied: {oxygen_bias.mean():.3f}")
    print(f"Carbon bias applied: {carbon_bias.mean():.3f}")
    
    # The bias should be positive for oxygen but not overwhelming
    assert oxygen_bias.mean() > 0, "Oxygen should have positive bias"
    assert oxygen_bias.mean() < 2.0, "Oxygen bias should not be too strong"
    assert carbon_bias.mean() == 0.0, "Non-specified elements should have no bias"
    
    # Test probability distribution after softmax
    probs = torch.softmax(masked_logits, dim=1)
    oxygen_prob = probs[:, 7].mean()
    
    print(f"Average oxygen probability: {oxygen_prob:.3f}")
    
    # Oxygen should be more likely but not dominating
    assert 0.1 < oxygen_prob < 0.8, f"Oxygen probability {oxygen_prob:.3f} should be reasonable"


def test_exact_mode_still_works():
    """Test that exact mode still works as before."""
    logits, x, batch_idx = create_test_batch()
    
    # Test with chemical_system_mode="exact" 
    masked_logits = mask_disallowed_elements(
        logits=logits.clone(),
        x=x,
        batch_idx=batch_idx,
        predictions_are_zero_based=True,
        chemical_system_mode="exact"
    )
    
    # In exact mode, only oxygen should be allowed (others should be -inf)
    oxygen_logits = masked_logits[:, 7]  # Oxygen
    carbon_logits = masked_logits[:, 5]  # Carbon
    
    print(f"Exact mode - Oxygen logits: {oxygen_logits.mean():.3f}")
    print(f"Exact mode - Carbon logits: {carbon_logits.mean():.3f}")
    
    # Carbon should be heavily penalized (close to -inf)
    assert carbon_logits.mean() < -1e5, "Non-specified elements should be masked in exact mode"


def test_multi_element_system():
    """Test with multiple elements to ensure bias scales appropriately."""
    batch_size = 8
    num_atoms = 16
    
    logits = torch.randn(num_atoms, 119)
    batch_idx = torch.repeat_interleave(torch.arange(batch_size), num_atoms // batch_size)
    
    # Multi-element chemical system
    chemical_system = ['Li-O'] * batch_size
    
    x = ChemGraph(
        num_atoms=torch.ones((batch_size, 1), dtype=torch.int) * (num_atoms // batch_size),
        chemical_system=chemical_system
    )
    
    masked_logits = mask_disallowed_elements(
        logits=logits.clone(),
        x=x,
        batch_idx=batch_idx,
        predictions_are_zero_based=True,
        chemical_system_mode="contains"
    )
    
    bias_applied = masked_logits - logits
    
    # Li is element 3, O is element 8 (0-based: 2, 7)
    li_bias = bias_applied[:, 2].mean()
    o_bias = bias_applied[:, 7].mean()
    carbon_bias = bias_applied[:, 5].mean()  # Not in system
    
    print(f"Multi-element system:")
    print(f"  Li bias: {li_bias:.3f}")
    print(f"  O bias: {o_bias:.3f}")
    print(f"  C bias: {carbon_bias:.3f}")
    
    # Both Li and O should have positive bias, but smaller than single-element case
    assert li_bias > 0, "Li should have positive bias"
    assert o_bias > 0, "O should have positive bias"
    assert carbon_bias == 0.0, "Non-specified elements should have no bias"
    
    # With 2 elements, bias should be weaker than single element case
    # This prevents over-concentration on any single element
    assert li_bias < 0.5, "Li bias should be moderate for multi-element system"
    assert o_bias < 0.5, "O bias should be moderate for multi-element system"


if __name__ == "__main__":
    print("Testing chemical system bias fix...")
    
    print("\n1. Testing contains mode bias...")
    test_contains_mode_bias()
    
    print("\n2. Testing exact mode still works...")
    test_exact_mode_still_works()
    
    print("\n3. Testing multi-element system...")
    test_multi_element_system()
    
    print("\nAll tests passed! The fix appears to work correctly.")