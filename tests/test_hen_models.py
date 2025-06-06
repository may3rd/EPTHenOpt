# tests/test_hen_models.py
import pytest
import numpy as np
from EPTHenOpt.hen_models import Stream, Utility, CostParameters, HENProblem

# --- Constants for a 2H, 2C, 2S Test Problem ---
# (This section is unchanged and correct)
TEST_EMAT_MODELS = 10.0
DEFAULT_U_OVERALL = 0.75
DEFAULT_EXCH_FIXED_COST = 1000
DEFAULT_EXCH_AREA_COEFF = 100
DEFAULT_EXCH_AREA_EXP = 0.65

@pytest.fixture
def problem_2h_2c_2s():
    # (This fixture is unchanged and correct)
    hot_streams = [
        Stream(id_val="H1", Tin=200, Tout_target=100, CP=10, stream_type='hot', h_coeff=0.8),
        Stream(id_val="H2", Tin=180, Tout_target=80, CP=15, stream_type='hot', h_coeff=0.8)
    ]
    cold_streams = [
        Stream(id_val="C1", Tin=50, Tout_target=150, CP=12, stream_type='cold', h_coeff=0.8),
        Stream(id_val="C2", Tin=70, Tout_target=130, CP=8, stream_type='cold', h_coeff=0.8)
    ]
    hot_utility = [Utility(id_val="HU1", Tin=250, Tout=249, cost_per_energy_unit=80, U=0.8, utility_type='hot_utility')]
    cold_utility = [Utility(id_val="CU1", Tin=20, Tout=30, cost_per_energy_unit=10, U=0.8, utility_type='cold_utility')]
    cost_params = CostParameters(
        EMAT=TEST_EMAT_MODELS, U_overall=DEFAULT_U_OVERALL, exch_fixed=DEFAULT_EXCH_FIXED_COST,
        exch_area_coeff=DEFAULT_EXCH_AREA_COEFF, exch_area_exp=DEFAULT_EXCH_AREA_EXP
    )
    problem = HENProblem(
        hot_streams=hot_streams, cold_streams=cold_streams, hot_utility=hot_utility,
        cold_utility=cold_utility, cost_params=cost_params, num_stages=2
    )
    return problem

def test_decode_chromosome_dimensions(problem_2h_2c_2s):
    # (This test is unchanged and correct)
    problem = problem_2h_2c_2s
    NH, NC, ST = problem.NH, problem.NC, problem.num_stages
    len_Z, len_R_hot, len_R_cold = NH*NC*ST, NH*ST*NC, NC*ST*NH
    chromosome_length = len_Z + len_R_hot + len_R_cold
    dummy_chromosome = np.random.rand(chromosome_length)
    dummy_chromosome[:len_Z] = np.random.randint(0, 2, size=len_Z)
    Z_ijk, R_hot_splits, R_cold_splits = problem._decode_chromosome(dummy_chromosome)
    assert Z_ijk.shape == (NH, NC, ST)
    assert R_hot_splits.shape == (NH, ST, NC)
    assert R_cold_splits.shape == (NC, ST, NH)
    assert Z_ijk.dtype == int

@pytest.fixture
def pinch_test_problem():
    # --- FIX ---
    # Added h_coeff to Stream and Utility objects to prevent TypeError.
    hot_streams = [Stream(id_val="H1", Tin=200, Tout_target=100, CP=10, h_coeff=0.8)]
    cold_streams = [Stream(id_val="C1", Tin=50, Tout_target=150, CP=10, h_coeff=0.8)]
    cost_params = CostParameters(EMAT=10.0, U_overall=None) # Set U_overall to None to trigger h_coeff logic
    
    problem = HENProblem(
        hot_streams=hot_streams,
        cold_streams=cold_streams,
        hot_utility=[Utility(id_val="HU1", Tin=250, Tout=249, cost_per_energy_unit=1, h_coeff=0.8)],
        cold_utility=[Utility(id_val="CU1", Tin=20, Tout=30, cost_per_energy_unit=1, h_coeff=0.8)],
        cost_params=cost_params,
        num_stages=1
    )
    return problem

def test_calculate_pinch_targets_simple_match(pinch_test_problem):
    # (This test is unchanged and correct)
    problem = pinch_test_problem
    assert problem.Q_H_min_pinch == pytest.approx(0.0)
    assert problem.Q_C_min_pinch == pytest.approx(0.0)
    assert problem.T_pinch_hot_actual == pytest.approx(200.0)
    assert problem.T_pinch_cold_actual == pytest.approx(190.0)

@pytest.fixture
def pinch_test_problem_hot_utility_needed():
    # --- FIX ---
    # Added h_coeff to Stream and Utility objects to prevent TypeError.
    hot_streams = [Stream(id_val="H1", Tin=150, Tout_target=70, CP=10, h_coeff=0.8)]
    cold_streams = [Stream(id_val="C1", Tin=50, Tout_target=100, CP=10, h_coeff=0.8)]
    cost_params = CostParameters(EMAT=10.0, U_overall=None) # Set U_overall to None to trigger h_coeff logic
    problem = HENProblem(hot_streams, cold_streams, 
                         [Utility(id_val="HU1", Tin=250, Tout=249, cost_per_energy_unit=1, h_coeff=0.8)], 
                         [Utility(id_val="CU1", Tin=20, Tout=30, cost_per_energy_unit=1, h_coeff=0.8)], 
                         cost_params, 1)
    return problem

def test_calculate_pinch_targets_utility_needed(pinch_test_problem_hot_utility_needed):
    # (This test is unchanged and correct)
    problem = pinch_test_problem_hot_utility_needed
    assert problem.Q_H_min_pinch == pytest.approx(0.0)
    assert problem.Q_C_min_pinch == pytest.approx(300.0)
    assert problem.T_pinch_hot_actual == pytest.approx(150.0)
    assert problem.T_pinch_cold_actual == pytest.approx(140.0)

def test_cost_matrix_initialization_defaults(problem_2h_2c_2s):
    # (This test is unchanged and correct)
    problem = problem_2h_2c_2s
    expected_U_default = np.full((problem.NH, problem.NC), DEFAULT_U_OVERALL)
    assert np.array_equal(problem.U_matrix_process, expected_U_default)
    expected_fixed_cost_default = np.full((problem.NH, problem.NC), DEFAULT_EXCH_FIXED_COST)
    assert np.array_equal(problem.fixed_cost_process_exchangers, expected_fixed_cost_default)

def test_cost_matrix_initialization_with_matches_U_cost(problem_2h_2c_2s):
    # (This test is unchanged and correct)
    problem_definition = problem_2h_2c_2s
    specific_U_H1C2, specific_fixed_H1C2 = 1.2, 1500
    specific_coeff_H1C2, specific_exp_H1C2 = 120, 0.7
    matches_U_data = [
        {'hot': 'H1', 'cold': 'C2', 'U': specific_U_H1C2, 'fix_cost': specific_fixed_H1C2, 
         'area_cost_coeff': specific_coeff_H1C2, 'area_cost_exp': specific_exp_H1C2}
    ]
    problem_with_matches = HENProblem(
        hot_streams=problem_definition.hot_streams, cold_streams=problem_definition.cold_streams,
        hot_utility=problem_definition.hot_utility, cold_utility=problem_definition.cold_utility,
        cost_params=problem_definition.cost_params, num_stages=problem_definition.num_stages,
        matches_U_cost=matches_U_data
    )
    assert problem_with_matches.U_matrix_process[0,1] == specific_U_H1C2
    assert problem_with_matches.fixed_cost_process_exchangers[0,1] == specific_fixed_H1C2
