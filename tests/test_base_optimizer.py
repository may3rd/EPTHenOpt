# tests/test_base_optimizer.py
import pytest
import numpy as np
from EPTHenOpt.hen_models import Stream, Utility, CostParameters, HENProblem
from EPTHenOpt.ga_helpers import GeneticAlgorithmHEN # Using a concrete subclass

# --- Constants for Minimal Test Problem ---
# EMAT for the test problem
TEST_EMAT = 10.0
# Pinch deviation penalty factor for tests
TEST_PINCH_PENALTY_FACTOR = 100.0
# Utility cost factor
TEST_UTILITY_COST_FACTOR = 1.0

# Adaptive penalty parameters for testing
TEST_INITIAL_PENALTY = 100.0
TEST_FINAL_PENALTY = 10000.0
TEST_TOTAL_GENERATIONS = 100 # For testing adaptive penalty progression

# SWS parameters for testing
TEST_SWS_MAX_ITER = 50
TEST_SWS_CONV_TOL = 0.001


@pytest.fixture
def minimal_hen_problem():
    """
    Creates a minimal HENProblem instance for testing.
    1 Hot Stream, 1 Cold Stream, 1 Stage.
    """
    hot_stream1 = Stream(id_val="H1", Tin=200, Tout_target=100, CP=10, stream_type='hot')
    cold_stream1 = Stream(id_val="C1", Tin=50, Tout_target=150, CP=12, stream_type='cold')

    # Using generic utilities to ensure they are available if needed by logic
    hot_utility = [Utility(id_val="HU1", Tin=250, Tout=249, cost_per_energy_unit=80, U=0.8,
                           fix_cost=1000, area_cost_coeff=100, area_cost_exp=0.6, utility_type='hot_utility')]
    cold_utility = [Utility(id_val="CU1", Tin=20, Tout=30, cost_per_energy_unit=10, U=0.8,
                           fix_cost=800, area_cost_coeff=90, area_cost_exp=0.6, utility_type='cold_utility')]

    cost_params = CostParameters(
        exch_fixed=500, exch_area_coeff=80, exch_area_exp=0.6,
        heater_fixed=600, heater_area_coeff=90, heater_area_exp=0.6,
        cooler_fixed=400, cooler_area_coeff=70, cooler_area_exp=0.6,
        EMAT=TEST_EMAT, U_overall=0.5 # Default U if not specified in matches_U_cost
    )
    
    problem = HENProblem(
        hot_streams=[hot_stream1],
        cold_streams=[cold_stream1],
        hot_utility=hot_utility,
        cold_utility=cold_utility,
        cost_params=cost_params,
        num_stages=1
    )
    return problem

@pytest.fixture
def ga_optimizer_for_testing(minimal_hen_problem):
    """
    Creates a GeneticAlgorithmHEN instance with the minimal problem
    and test-specific penalty parameters.
    """
    optimizer = GeneticAlgorithmHEN(
        problem=minimal_hen_problem,
        population_size=10, # Small, not relevant for single fitness call
        generations=TEST_TOTAL_GENERATIONS,
        crossover_prob=0.8, # Not relevant
        mutation_prob_Z=0.1, # Not relevant
        mutation_prob_R=0.1, # Not relevant
        elitism_count=1, # Not relevant
        tournament_size=3, # Not relevant
        # Pass the adaptive penalty params to BaseOptimizer's __init__
        initial_penalty=TEST_INITIAL_PENALTY,
        final_penalty=TEST_FINAL_PENALTY,
        # Pass other relevant params from BaseOptimizer
        utility_cost_factor=TEST_UTILITY_COST_FACTOR,
        pinch_deviation_penalty_factor=TEST_PINCH_PENALTY_FACTOR,
        sws_max_iter=TEST_SWS_MAX_ITER,
        sws_conv_tol=TEST_SWS_CONV_TOL
    )
    return optimizer

# --- Test Chromosomes for 1H, 1C, 1S problem ---
# Chromosome length = 1 (Z_000) + 1 (R_hot_000) + 1 (R_cold_000) = 3
# Z_ijk: H_idx, C_idx, S_idx
# R_hot_splits: H_idx, S_idx, C_idx_target
# R_cold_splits: C_idx, S_idx, H_idx_source

# Chromosome that attempts a match between H1 and C1 in Stage 1
# Z = [1] (H1-C1 match in S1)
# R_hot = [1.0] (H1 in S1 fully goes to C1 branch)
# R_cold = [1.0] (C1 in S1 fully comes from H1 branch)
FEASIBLE_CHROMOSOME = np.array([1, 1.0, 1.0])

# Chromosome that will likely cause EMAT violation if match occurs
# Making cold stream outlet very high and hot stream inlet low for the match
# This is tricky to force directly via chromosome; EMAT check is post-SWS.
# We'll rely on a known bad configuration if possible, or check penalty application.
# For now, let's use the feasible chromosome and check EMAT under different conditions.

# Chromosome with no match (Z=0), forcing utility usage
NO_MATCH_CHROMOSOME = np.array([0, 0.0, 0.0])


def test_calculate_fitness_feasible_solution(ga_optimizer_for_testing):
    """
    Tests fitness calculation for a generally feasible chromosome.
    Expects finite TAC and minimal or zero penalties.
    """
    optimizer = ga_optimizer_for_testing
    optimizer.current_generation = 0 # Start of optimization

    costs, details = optimizer._calculate_fitness(FEASIBLE_CHROMOSOME)

    assert costs is not None
    assert isinstance(costs, dict)
    assert details is not None
    assert isinstance(details, list)

    assert np.isfinite(costs['TAC_GA_optimizing']), "TAC_GA_optimizing should be finite for a feasible solution"
    assert np.isfinite(costs['TAC_true_report']), "TAC_true_report should be finite"
    
    # Check that major penalties are zero or very small
    # These thresholds might need adjustment based on the exact problem and solution
    assert costs.get('penalty_EMAT_etc', 0) < TEST_INITIAL_PENALTY * 0.1, "EMAT penalty should be low for feasible solution"
    assert costs.get('penalty_unmet_targets', 0) < TEST_INITIAL_PENALTY * 0.1, "Unmet targets penalty should be low"
    assert costs.get('penalty_SWS_non_convergence', 0) == 0, "SWS should converge"


def test_calculate_fitness_no_match_solution(ga_optimizer_for_testing):
    """
    Tests fitness calculation for a chromosome with no process-process matches.
    Expects higher utility costs.
    """
    optimizer = ga_optimizer_for_testing
    optimizer.current_generation = 0

    costs, details = optimizer._calculate_fitness(NO_MATCH_CHROMOSOME)

    assert np.isfinite(costs['TAC_GA_optimizing'])
    assert costs.get('op_cost_hot_utility', 0) > 0, "Should have hot utility cost if no match"
    assert costs.get('op_cost_cold_utility', 0) > 0, "Should have cold utility cost if no match"
    assert costs.get('penalty_SWS_non_convergence', 0) == 0, "SWS should converge even with no matches"


def test_sws_non_convergence_early_exit(ga_optimizer_for_testing, minimal_hen_problem):
    """
    Tests that if SWS fails to converge, a penalty is returned and other calculations are skipped.
    We force non-convergence by setting sws_max_iter to 0.
    """
    # Create a new optimizer instance with sws_max_iter = 0 for this test
    optimizer_sws_fail = GeneticAlgorithmHEN(
        problem=minimal_hen_problem,
        population_size=10, generations=TEST_TOTAL_GENERATIONS,
        crossover_prob=0.8, mutation_prob_Z=0.1, mutation_prob_R=0.1,
        initial_penalty=TEST_INITIAL_PENALTY, final_penalty=TEST_FINAL_PENALTY,
        utility_cost_factor=TEST_UTILITY_COST_FACTOR,
        pinch_deviation_penalty_factor=TEST_PINCH_PENALTY_FACTOR,
        sws_max_iter=0, # Force SWS non-convergence
        sws_conv_tol=TEST_SWS_CONV_TOL
    )
    optimizer_sws_fail.current_generation = 10 # Any generation

    costs, details = optimizer_sws_fail._calculate_fitness(FEASIBLE_CHROMOSOME)

    assert not np.isfinite(costs['TAC_true_report']), "True TAC should be inf on SWS failure"
    assert 'penalty_SWS_non_convergence' in costs
    assert costs['penalty_SWS_non_convergence'] > 0, "SWS non-convergence penalty should be applied"
    
    # Check that other cost components that come after SWS are not substantially calculated or are zero
    assert costs.get('capital_process_exchangers', 0) == 0, "No capital cost if SWS failed"
    assert costs.get('op_cost_hot_utility', 0) == 0, "No op cost if SWS failed"
    assert not details, "Details list should be empty on SWS failure"
    
    # The TAC_GA_optimizing should be based on the adaptive penalty
    expected_sws_penalty_base = TEST_INITIAL_PENALTY + (TEST_FINAL_PENALTY - TEST_INITIAL_PENALTY) * (optimizer_sws_fail.current_generation / optimizer_sws_fail.generations)
    assert costs['TAC_GA_optimizing'] == pytest.approx(expected_sws_penalty_base * 1e3)


def test_adaptive_penalty_initial_generation(ga_optimizer_for_testing):
    """
    Tests that penalties use the initial_penalty factor at generation 0.
    To ensure a penalty is triggered, we need a chromosome that reliably causes one.
    Let's try to force an EMAT violation by manipulating stream temperatures directly
    in a custom HENProblem for this specific test, as it's hard with just chromosome.
    Alternatively, we can check the penalty_SWS_non_convergence value as it uses the adaptive factor.
    """
    optimizer = ga_optimizer_for_testing
    optimizer.current_generation = 0
    
    # Re-using the SWS failure mechanism as it cleanly shows adaptive penalty usage
    optimizer.sws_max_iter = 0 # Force SWS non-convergence
    
    costs, _ = optimizer._calculate_fitness(FEASIBLE_CHROMOSOME)
    
    # SWS non-convergence penalty is adaptive_penalty_factor * 1e3
    # At gen 0, adaptive_penalty_factor should be TEST_INITIAL_PENALTY
    assert 'penalty_SWS_non_convergence' in costs
    assert costs['penalty_SWS_non_convergence'] == pytest.approx(TEST_INITIAL_PENALTY * 1e3)
    assert costs['TAC_GA_optimizing'] == pytest.approx(TEST_INITIAL_PENALTY * 1e3)
    
    # Reset sws_max_iter for other tests if optimizer fixture is shared and mutable
    optimizer.sws_max_iter = TEST_SWS_MAX_ITER


def test_adaptive_penalty_final_generation(ga_optimizer_for_testing):
    """
    Tests that penalties use the final_penalty factor at the last generation.
    """
    optimizer = ga_optimizer_for_testing
    optimizer.current_generation = optimizer.generations # Last generation
    
    optimizer.sws_max_iter = 0 # Force SWS non-convergence
    
    costs, _ = optimizer._calculate_fitness(FEASIBLE_CHROMOSOME)
    
    assert 'penalty_SWS_non_convergence' in costs
    assert costs['penalty_SWS_non_convergence'] == pytest.approx(TEST_FINAL_PENALTY * 1e3)
    assert costs['TAC_GA_optimizing'] == pytest.approx(TEST_FINAL_PENALTY * 1e3)

    optimizer.sws_max_iter = TEST_SWS_MAX_ITER


def test_adaptive_penalty_intermediate_generation(ga_optimizer_for_testing):
    """
    Tests that penalties are interpolated correctly at an intermediate generation.
    """
    optimizer = ga_optimizer_for_testing
    optimizer.current_generation = TEST_TOTAL_GENERATIONS // 2 # Mid-point
    
    optimizer.sws_max_iter = 0 # Force SWS non-convergence
    
    costs, _ = optimizer._calculate_fitness(FEASIBLE_CHROMOSOME)
    
    expected_ratio = (TEST_TOTAL_GENERATIONS // 2) / TEST_TOTAL_GENERATIONS
    expected_adaptive_factor = TEST_INITIAL_PENALTY + (TEST_FINAL_PENALTY - TEST_INITIAL_PENALTY) * expected_ratio
    
    assert 'penalty_SWS_non_convergence' in costs
    assert costs['penalty_SWS_non_convergence'] == pytest.approx(expected_adaptive_factor * 1e3)
    assert costs['TAC_GA_optimizing'] == pytest.approx(expected_adaptive_factor * 1e3)
    
    optimizer.sws_max_iter = TEST_SWS_MAX_ITER

# More tests could be added for:
# - Pinch deviation penalty application
# - Forbidden/Required match penalties
# - Specific EMAT violation setup (might require a more tailored HENProblem fixture)
# - Different chromosome structures (e.g., more streams/stages)
