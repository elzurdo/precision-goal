
import sys
import os
import numpy as np
import pandas as pd
import pytest

# Add the 'py' directory to the system path so we can import modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'py')))

from utils_experiments import stop_decision_multiple_experiments_multiple_methods
from utils_experiments_shared import create_decision_correctness_df

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

class TestCreateDecisionCorrectnessDF:
    """Test suite for create_decision_correctness_df function."""
    
    @pytest.fixture
    def rope_params(self):
        """Standard ROPE parameters for testing."""
        return {
            'rope_min': 0.45,
            'rope_max': 0.55
        }
    
    def test_conclusive_accept_correct(self, rope_params):
        """Test conclusive accept decision when true_rate is within ROPE."""
        # True rate within ROPE -> accepting is correct
        true_rate = 0.50
        
        method_stats = {
            'pitg': {
                0: {
                    'decision_iteration': 100,
                    'accept': True,
                    'reject_below': False,
                    'reject_above': False,
                    'inconclusive': False,
                    'successes': 50,
                    'failures': 50
                }
            },
            'epitg': {
                0: {
                    'decision_iteration': 100,
                    'accept': True,
                    'reject_below': False,
                    'reject_above': False,
                    'inconclusive': False,
                    'successes': 50,
                    'failures': 50
                }
            },
            'hdi_rope': {
                0: {
                    'decision_iteration': 100,
                    'accept': True,
                    'reject_below': False,
                    'reject_above': False,
                    'inconclusive': False,
                    'successes': 50,
                    'failures': 50
                }
            }
        }
        
        df = create_decision_correctness_df(method_stats, true_rate, data_type='binomial', **rope_params)
        
        assert df.loc[0, 'pitg_decision_correct'] == True
        assert df.loc[0, 'epitg_decision_correct'] == True
        assert df.loc[0, 'hdi_rope_decision_correct'] == True
    
    def test_conclusive_accept_incorrect(self, rope_params):
        """Test conclusive accept decision when true_rate is outside ROPE (should be incorrect)."""
        # True rate below ROPE -> accepting is incorrect
        true_rate = 0.30
        
        method_stats = {
            'pitg': {
                0: {
                    'decision_iteration': 100,
                    'accept': True,
                    'reject_below': False,
                    'reject_above': False,
                    'inconclusive': False,
                    'successes': 50,
                    'failures': 50
                }
            },
            'epitg': {
                0: {
                    'decision_iteration': 100,
                    'accept': True,
                    'reject_below': False,
                    'reject_above': False,
                    'inconclusive': False,
                    'successes': 50,
                    'failures': 50
                }
            },
            'hdi_rope': {
                0: {
                    'decision_iteration': 100,
                    'accept': True,
                    'reject_below': False,
                    'reject_above': False,
                    'inconclusive': False,
                    'successes': 50,
                    'failures': 50
                }
            }
        }
        
        df = create_decision_correctness_df(method_stats, true_rate, **rope_params)
        
        assert df.loc[0, 'pitg_decision_correct'] == False
        assert df.loc[0, 'epitg_decision_correct'] == False
        assert df.loc[0, 'hdi_rope_decision_correct'] == False
    
    def test_conclusive_reject_correct(self, rope_params):
        """Test conclusive reject decision when true_rate is outside ROPE."""
        # True rate above ROPE -> rejecting is correct
        true_rate = 0.70
        
        method_stats = {
            'pitg': {
                0: {
                    'decision_iteration': 100,
                    'accept': False,
                    'reject_below': False,
                    'reject_above': True,
                    'inconclusive': False,
                    'successes': 70,
                    'failures': 30
                }
            },
            'epitg': {
                0: {
                    'decision_iteration': 100,
                    'accept': False,
                    'reject_below': True,
                    'reject_above': False,
                    'inconclusive': False,
                    'successes': 30,
                    'failures': 70
                }
            },
            'hdi_rope': {
                0: {
                    'decision_iteration': 100,
                    'accept': False,
                    'reject_below': False,
                    'reject_above': True,
                    'inconclusive': False,
                    'successes': 70,
                    'failures': 30
                }
            }
        }
        
        df = create_decision_correctness_df(method_stats, true_rate, **rope_params)
        
        # All methods rejected (correctly, since true_rate > rope_max)
        assert df.loc[0, 'pitg_decision_correct'] == True
        assert df.loc[0, 'epitg_decision_correct'] == True
        assert df.loc[0, 'hdi_rope_decision_correct'] == True
    
    def test_conclusive_reject_incorrect(self, rope_params):
        """Test conclusive reject decision when true_rate is within ROPE (should be incorrect)."""
        # True rate within ROPE -> rejecting is incorrect
        true_rate = 0.50
        
        method_stats = {
            'pitg': {
                0: {
                    'decision_iteration': 100,
                    'accept': False,
                    'reject_below': False,
                    'reject_above': True,
                    'inconclusive': False,
                    'successes': 70,
                    'failures': 30
                }
            },
            'epitg': {
                0: {
                    'decision_iteration': 100,
                    'accept': False,
                    'reject_below': True,
                    'reject_above': False,
                    'inconclusive': False,
                    'successes': 30,
                    'failures': 70
                }
            },
            'hdi_rope': {
                0: {
                    'decision_iteration': 100,
                    'accept': False,
                    'reject_below': False,
                    'reject_above': True,
                    'inconclusive': False,
                    'successes': 70,
                    'failures': 30
                }
            }
        }
        
        df = create_decision_correctness_df(method_stats, true_rate, **rope_params)
        
        # All methods rejected (incorrectly, since true_rate is within ROPE)
        assert df.loc[0, 'pitg_decision_correct'] == False
        assert df.loc[0, 'epitg_decision_correct'] == False
        assert df.loc[0, 'hdi_rope_decision_correct'] == False
    
    def test_inconclusive_heuristic_accept_correct(self, rope_params):
        """Test inconclusive case where success_rate heuristic leads to correct accept."""
        # True rate within ROPE, success_rate also within ROPE -> correct
        true_rate = 0.50
        
        method_stats = {
            'pitg': {
                0: {
                    'decision_iteration': 100,
                    'accept': False,
                    'reject_below': False,
                    'reject_above': False,
                    'inconclusive': True,
                    'successes': 48,  # 48/100 = 0.48 (within ROPE)
                    'failures': 52
                }
            },
            'epitg': {
                0: {
                    'decision_iteration': 100,
                    'accept': False,
                    'reject_below': False,
                    'reject_above': False,
                    'inconclusive': True,
                    'successes': 52,  # 52/100 = 0.52 (within ROPE)
                    'failures': 48
                }
            },
            'hdi_rope': {
                0: {
                    'decision_iteration': 100,
                    'accept': False,
                    'reject_below': False,
                    'reject_above': False,
                    'inconclusive': True,
                    'successes': 50,  # 50/100 = 0.50 (within ROPE)
                    'failures': 50
                }
            }
        }
        
        df = create_decision_correctness_df(method_stats, true_rate, **rope_params)
        
        # All success rates within ROPE, true_rate within ROPE -> correct
        assert df.loc[0, 'pitg_decision_correct'] == True
        assert df.loc[0, 'epitg_decision_correct'] == True
        assert df.loc[0, 'hdi_rope_decision_correct'] == True
    
    def test_inconclusive_heuristic_reject_correct(self, rope_params):
        """Test inconclusive case where success_rate heuristic leads to correct reject."""
        # True rate outside ROPE, success_rate also outside ROPE -> correct
        true_rate = 0.70
        
        method_stats = {
            'pitg': {
                0: {
                    'decision_iteration': 100,
                    'accept': False,
                    'reject_below': False,
                    'reject_above': False,
                    'inconclusive': True,
                    'successes': 65,  # 65/100 = 0.65 (above ROPE)
                    'failures': 35
                }
            },
            'epitg': {
                0: {
                    'decision_iteration': 200,
                    'accept': False,
                    'reject_below': False,
                    'reject_above': False,
                    'inconclusive': True,
                    'successes': 120,  # 120/200 = 0.60 (above ROPE)
                    'failures': 80
                }
            },
            'hdi_rope': {
                0: {
                    'decision_iteration': 150,
                    'accept': False,
                    'reject_below': False,
                    'reject_above': False,
                    'inconclusive': True,
                    'successes': 25,  # 25/150 = 0.167 (below ROPE)
                    'failures': 125
                }
            }
        }
        
        df = create_decision_correctness_df(method_stats, true_rate, **rope_params)
        
        # Success rates outside ROPE, true_rate outside ROPE -> correct
        assert df.loc[0, 'pitg_decision_correct'] == True
        assert df.loc[0, 'epitg_decision_correct'] == True
        assert df.loc[0, 'hdi_rope_decision_correct'] == True
    
    def test_inconclusive_heuristic_incorrect(self, rope_params):
        """Test inconclusive case where success_rate heuristic leads to incorrect decision."""
        # True rate within ROPE, but success_rate outside ROPE -> incorrect
        true_rate = 0.50
        
        method_stats = {
            'pitg': {
                0: {
                    'decision_iteration': 100,
                    'accept': False,
                    'reject_below': False,
                    'reject_above': False,
                    'inconclusive': True,
                    'successes': 70,  # 70/100 = 0.70 (above ROPE, but true is within)
                    'failures': 30
                }
            },
            'epitg': {
                0: {
                    'decision_iteration': 100,
                    'accept': False,
                    'reject_below': False,
                    'reject_above': False,
                    'inconclusive': True,
                    'successes': 30,  # 30/100 = 0.30 (below ROPE, but true is within)
                    'failures': 70
                }
            },
            'hdi_rope': {
                0: {
                    'decision_iteration': 100,
                    'accept': False,
                    'reject_below': False,
                    'reject_above': False,
                    'inconclusive': True,
                    'successes': 70,  # 70/100 = 0.70 (above ROPE, but true is within)
                    'failures': 30
                }
            }
        }
        
        df = create_decision_correctness_df(method_stats, true_rate, **rope_params)
        
        # All heuristic decisions wrong (success_rate outside ROPE, but true_rate within)
        assert df.loc[0, 'pitg_decision_correct'] == False
        assert df.loc[0, 'epitg_decision_correct'] == False
        assert df.loc[0, 'hdi_rope_decision_correct'] == False
    
    def test_multiple_experiments(self, rope_params):
        """Test with multiple experiments showing mixed results."""
        true_rate = 0.50
        
        method_stats = {
            'pitg': {
                0: {  # Conclusive correct
                    'decision_iteration': 100,
                    'accept': True,
                    'reject_below': False,
                    'reject_above': False,
                    'inconclusive': False,
                    'successes': 50,
                    'failures': 50
                },
                1: {  # Conclusive incorrect
                    'decision_iteration': 100,
                    'accept': False,
                    'reject_below': False,
                    'reject_above': True,
                    'inconclusive': False,
                    'successes': 70,
                    'failures': 30
                },
                2: {  # Inconclusive correct
                    'decision_iteration': 100,
                    'accept': False,
                    'reject_below': False,
                    'reject_above': False,
                    'inconclusive': True,
                    'successes': 48,
                    'failures': 52
                },
                3: {  # Inconclusive incorrect
                    'decision_iteration': 100,
                    'accept': False,
                    'reject_below': False,
                    'reject_above': False,
                    'inconclusive': True,
                    'successes': 70,
                    'failures': 30
                }
            },
            'epitg': {
                0: {'decision_iteration': 100, 'accept': True, 'reject_below': False, 
                    'reject_above': False, 'inconclusive': False, 'successes': 50, 'failures': 50},
                1: {'decision_iteration': 100, 'accept': False, 'reject_below': False, 
                    'reject_above': True, 'inconclusive': False, 'successes': 70, 'failures': 30},
                2: {'decision_iteration': 100, 'accept': False, 'reject_below': False, 
                    'reject_above': False, 'inconclusive': True, 'successes': 48, 'failures': 52},
                3: {'decision_iteration': 100, 'accept': False, 'reject_below': False, 
                    'reject_above': False, 'inconclusive': True, 'successes': 70, 'failures': 30}
            },
            'hdi_rope': {
                0: {'decision_iteration': 100, 'accept': True, 'reject_below': False, 
                    'reject_above': False, 'inconclusive': False, 'successes': 50, 'failures': 50},
                1: {'decision_iteration': 100, 'accept': False, 'reject_below': False, 
                    'reject_above': True, 'inconclusive': False, 'successes': 70, 'failures': 30},
                2: {'decision_iteration': 100, 'accept': False, 'reject_below': False, 
                    'reject_above': False, 'inconclusive': True, 'successes': 48, 'failures': 52},
                3: {'decision_iteration': 100, 'accept': False, 'reject_below': False, 
                    'reject_above': False, 'inconclusive': True, 'successes': 70, 'failures': 30}
            }
        }
        
        df = create_decision_correctness_df(method_stats, true_rate, **rope_params)
        
        assert len(df) == 4
        
        # Experiment 0: correct
        assert df.loc[0, 'pitg_decision_correct'] == True
        assert df.loc[0, 'epitg_decision_correct'] == True
        assert df.loc[0, 'hdi_rope_decision_correct'] == True
        
        # Experiment 1: incorrect
        assert df.loc[1, 'pitg_decision_correct'] == False
        assert df.loc[1, 'epitg_decision_correct'] == False
        assert df.loc[1, 'hdi_rope_decision_correct'] == False
        
        # Experiment 2: correct (via heuristic)
        assert df.loc[2, 'pitg_decision_correct'] == True
        assert df.loc[2, 'epitg_decision_correct'] == True
        assert df.loc[2, 'hdi_rope_decision_correct'] == True
        
        # Experiment 3: incorrect (via heuristic)
        assert df.loc[3, 'pitg_decision_correct'] == False
        assert df.loc[3, 'epitg_decision_correct'] == False
        assert df.loc[3, 'hdi_rope_decision_correct'] == False
    
    def test_rope_boundary_cases(self):
        """Test edge cases at ROPE boundaries."""
        rope_min = 0.45
        rope_max = 0.55
        
        # Test true_rate exactly at rope_min (should be accepted)
        true_rate = 0.45
        
        method_stats = {
            'pitg': {
                0: {
                    'decision_iteration': 100,
                    'accept': True,
                    'reject_below': False,
                    'reject_above': False,
                    'inconclusive': False,
                    'successes': 50,
                    'failures': 50
                }
            },
            'epitg': {
                0: {
                    'decision_iteration': 100,
                    'accept': True,
                    'reject_below': False,
                    'reject_above': False,
                    'inconclusive': False,
                    'successes': 50,
                    'failures': 50
                }
            },
            'hdi_rope': {
                0: {
                    'decision_iteration': 100,
                    'accept': True,
                    'reject_below': False,
                    'reject_above': False,
                    'inconclusive': False,
                    'successes': 50,
                    'failures': 50
                }
            }
        }
        
        df = create_decision_correctness_df(method_stats, true_rate, rope_min, rope_max)
        assert df.loc[0, 'pitg_decision_correct'] == True
        
        # Test true_rate exactly at rope_max (should be accepted)
        true_rate = 0.55
        df = create_decision_correctness_df(method_stats, true_rate, rope_min, rope_max)
        assert df.loc[0, 'pitg_decision_correct'] == True
        
        # Test true_rate just below rope_min (should be rejected)
        true_rate = 0.44999
        df = create_decision_correctness_df(method_stats, true_rate, rope_min, rope_max)
        assert df.loc[0, 'pitg_decision_correct'] == False  # accepts when should reject
        
        # Test true_rate just above rope_max (should be rejected)
        true_rate = 0.55001
        df = create_decision_correctness_df(method_stats, true_rate, rope_min, rope_max)
        assert df.loc[0, 'pitg_decision_correct'] == False  # accepts when should reject
    
    def test_dataframe_structure(self, rope_params):
        """Test that output DataFrame has correct structure and columns."""
        true_rate = 0.50
        
        method_stats = {
            'pitg': {
                0: {
                    'decision_iteration': 100,
                    'accept': True,
                    'reject_below': False,
                    'reject_above': False,
                    'inconclusive': False,
                    'successes': 50,
                    'failures': 50
                }
            },
            'epitg': {
                0: {
                    'decision_iteration': 100,
                    'accept': True,
                    'reject_below': False,
                    'reject_above': False,
                    'inconclusive': False,
                    'successes': 50,
                    'failures': 50
                }
            },
            'hdi_rope': {
                0: {
                    'decision_iteration': 100,
                    'accept': True,
                    'reject_below': False,
                    'reject_above': False,
                    'inconclusive': False,
                    'successes': 50,
                    'failures': 50
                }
            }
        }
        
        df = create_decision_correctness_df(method_stats, true_rate, **rope_params)
        
        # Check index
        assert df.index.name == "experiment_idx"
        
        # Check columns for each method
        for method in ['pitg', 'epitg', 'hdi_rope']:
            assert f'{method}_decision_iteration' in df.columns
            assert f'{method}_accept' in df.columns
            assert f'{method}_reject_below' in df.columns
            assert f'{method}_reject_above' in df.columns
            assert f'{method}_inconclusive' in df.columns
            assert f'{method}_param_value' in df.columns
            assert f'{method}_decision_correct' in df.columns
        
        # TODO: explore this in notebook and resolve
        # Check data types
        # assert df['pitg_decision_correct'].dtype == bool
        #assert df['pitg_success_rate'].dtype == float
