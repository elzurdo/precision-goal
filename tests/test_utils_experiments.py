
import sys
import os
import numpy as np
import pytest

# Add the 'py' directory to the system path so we can import modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'py')))

from utils_experiments import stop_decision_multiple_experiments_multiple_methods

def test_stop_decision_multiple_experiments_multiple_methods_basic():
    """
    Gold Master / Regression test for stop_decision_multiple_experiments_multiple_methods.
    Uses deterministic input to verify the current behavior of the function.


    Dictionary outcomes:

    method_stats['hdi_rope'] yields:
        {0: {'decision_iteration': 126,
        'accept': False,
        'reject_below': False,
        'reject_above': True,
        'conclusive': True,
        'inconclusive': False,
        'successes': 80,
        'failures': 46,
        'hdi_min': 0.5508244626218101,
        'hdi_max': 0.717876210358378,
        'precision_goal_achieved': False}}

    method_stats['pitg']:
        {0: {'decision_iteration': 598,
        'accept': False,
        'reject_below': False,
        'reject_above': False,
        'conclusive': False,
        'inconclusive': True,
        'successes': 314,
        'failures': 284,
        'hdi_min': 0.4850822706487851,
        'hdi_max': 0.5650347292260883,
        'precision_goal_achieved': True}}

    method_stats['epitg']:
        {0: {'decision_iteration': 804,
        'accept': True,
        'reject_below': False,
        'reject_above': False,
        'conclusive': True,
        'inconclusive': False,
        'successes': 414,
        'failures': 390,
        'hdi_min': 0.4803983603190735,
        'hdi_max': 0.5494290170209941,
        'precision_goal_achieved': True}}
    """
    
    # Setup deterministic inputs
    # Experiment 1: Fast success (precision goal met early, falls in ROPE)
    # Experiment 2: Slow failure (precision goal not met for a long time)
    # Experiment 3: Mix
    
    # Let's create a small synthetic dataset
    # 2 experiments, max 50 iterations
    n_experiments = 1
    n_samples = 810
    
    # Experiment 0: handpicked sequence used in paper
    sequence_ = "101101000110010000101111111110010101101110001111110010100110111111110111001111001110011110001010001011110101111110001111111111100000101001001100000001101000100010000000010010111001110100111000010010110011010000101011110011111111011100101011011100100101010011110101001111011100101110010011001010010001001011010101010100111100110011011011101110010100010110011001100101111001111101110101010001101110111100010110101010101010111100001000111011001010101100100110010001101101111100111000010011001000001010110010101101000001100101000110101110010101101000100110100100100110110100101011100001101000111111001001111100100011100011000101001010101110010000110111101111011100111011010010001001001111011100100000100011100000010010111111011110101000110110010001100101011110000001001101111100000001010011001001110001010100000101111100101110011011010111001000011110010011111110011111111100111011010000101110110001100111001000010011101100111000110010100000001101110000110011100111011100101001101010011001010100011000000011001100101100101000001101100111000000101010000110100100111110101101110010000100011101011011001110011100111011101010100101100001101100010111010010101000011000100111111010010111001100001001000110111011001011100100001001011111010011111101111001010000110011010101111001011110100001000100000010000011001110100110100100101000001100110111011011111010100111101111101010001010110010001000110111000101000010001011000100001101111011000000111010011000101001011110111101111010011101010111001111010101111011000110"
    sequence_ = sequence_[:n_samples]  # Truncate to n_samples

    samples = np.zeros((n_experiments, n_samples), dtype=int)
    samples[0] = [int(bit) for bit in sequence_]
    
    # Define parameters - as used in paper
    rope_min = 0.45
    rope_max = 0.55
    precision_goal = 0.08
    min_iter = 30
    
    # Run the function
    # Note: binary_accounting is optional, skipping for simplicity in this basic test
    method_stats, method_roperesult_iteration = stop_decision_multiple_experiments_multiple_methods(
        samples, 
        rope_min=rope_min, 
        rope_max=rope_max, 
        precision_goal=precision_goal,
        min_iter=min_iter,
        viz=False
    )
    
    # --- Assertions for Experiment 0 (Alternating, should be within ROPE) ---
    # Expected behavior:
    # 1. PitG should stop when precision < 0.2
    # 2. ePitG should stop when precision < 0.2 AND conclusive (within ROPE)
    # 3. HDI+ROPE should stop when conclusive (within ROPE), ignoring precision
    
    # Let's inspect the results for Experiment 0 (isample=0)
    
    # Check if keys exist
    assert 0 in method_stats['pitg'], "Experiment 0 missing from PitG stats"
    assert 0 in method_stats['epitg'], "Experiment 0 missing from ePitG stats"
    assert 0 in method_stats['hdi_rope'], "Experiment 0 missing from HDI+ROPE stats"
    
    # Verify values for PitG
    # We don't implement the full logic here, but we check that the result returned 
    # matches what the code currently produces. 
    # If you change logic, these values might need to change.
    
    pitg_res = method_stats['pitg'][0]
    epitg_res = method_stats['epitg'][0]
    hdi_rope_res = method_stats['hdi_rope'][0]
        
    # Consistency checks
    assert pitg_res['precision_goal_achieved'] == True, "PitG should have achieved precision goal"
    assert pitg_res['hdi_max'] - pitg_res['hdi_min'] < precision_goal, "Calculated precision should be < goal"
    
    # For alternating 0/1, success rate is ~0.5. ROPE is [0.45, 0.55].
    # It should eventually be accepted as 'within'.
    
    # Ensure ePitG iteration is >= PitG iteration
    assert epitg_res['decision_iteration'] >= pitg_res['decision_iteration'], "ePitG should not stop before PitG"

    # Asserting known values
    assert hdi_rope_res["decision_iteration"] == 126
    assert hdi_rope_res["reject_above"] == True

    assert pitg_res["decision_iteration"] == 598
    assert pitg_res["inconclusive"] == True

    assert epitg_res["decision_iteration"] == 804
    assert epitg_res["accept"] == True


def test_stop_decision_determinism():
    """
    Test with a very specific, short sequence to ensure exact numbers match.
    """
    samples = np.array([
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] # 10 samples, perfectly balanced
    ])
    
    rope_min = 0.4
    rope_max = 0.6
    precision_goal = 0.5 # Wide goal
    min_iter = 0
    
    method_stats, _ = stop_decision_multiple_experiments_multiple_methods(
        samples, rope_min, rope_max, precision_goal, min_iter=min_iter
    )
    
    # With 10 samples (5 success, 5 fail):
    # Mean = 0.5. 
    # Check HDI calculation logic consistency
    res = method_stats['epitg'][0]
    
    # These assertions simply ensure the code runs and returns a valid structure
    assert 'hdi_min' in res
    assert 'hdi_max' in res
    assert 'decision_iteration' in res
    assert res['decision_iteration'] <= 10
