# tests/test_utils.py
import pytest
import csv
import os
import numpy as np # Needed for test_calculate_lmtd_standard_case
from pathlib import Path
from EPTHenOpt.utils import calculate_lmtd, find_stream_index_by_id, load_data_from_csv
from EPTHenOpt.hen_models import Stream

# --- Constants ---
MIN_LMTD_FROM_UTILS = 1e-6

# --- Tests for calculate_lmtd ---

def test_calculate_lmtd_standard_case():
    """ Test LMTD calculation for a standard counter-current heat exchanger. """
    Th_in, Th_out = 150.0, 100.0
    Tc_in, Tc_out = 40.0, 80.0
    delta_T1 = Th_in - Tc_out  # 70
    delta_T2 = Th_out - Tc_in  # 60
    
    # --- FIX ---
    # Calculate the expected value first, then use pytest.approx in the assert.
    expected_lmtd = (delta_T1 - delta_T2) / np.log(delta_T1 / delta_T2)
    assert calculate_lmtd(Th_in, Th_out, Tc_in, Tc_out) == pytest.approx(expected_lmtd)

# (Other calculate_lmtd tests are correct and unchanged)
def test_calculate_lmtd_delta_t1_equals_delta_t2():
    assert calculate_lmtd(100.0, 80.0, 30.0, 50.0) == pytest.approx(50.0)

def test_calculate_lmtd_delta_t1_very_small():
    assert calculate_lmtd(80.0 + MIN_LMTD_FROM_UTILS/2, 60.0, 20.0, 80.0) == MIN_LMTD_FROM_UTILS

def test_calculate_lmtd_delta_t2_very_small():
    assert calculate_lmtd(100.0, 40.0 + MIN_LMTD_FROM_UTILS/2, 40.0, 80.0) == MIN_LMTD_FROM_UTILS

def test_calculate_lmtd_one_delta_zero():
    assert calculate_lmtd(100, 80, 50, 100) == MIN_LMTD_FROM_UTILS

def test_calculate_lmtd_negative_delta_t_implies_crossing():
    assert calculate_lmtd(70.0, 50.0, 30.0, 80.0) == MIN_LMTD_FROM_UTILS

# --- Tests for find_stream_index_by_id ---
# (This section is correct and unchanged)
@pytest.fixture
def sample_streams_list():
    return [Stream(id_val="H1"), Stream(id_val="C1"), Stream(id_val="H20"), Stream(id_val="ColdStream_X")]

def test_find_stream_index_existing_id(sample_streams_list):
    assert find_stream_index_by_id(sample_streams_list, "C1") == 1

def test_find_stream_index_non_existing_id(sample_streams_list):
    assert find_stream_index_by_id(sample_streams_list, "NonExistent") == -1

def test_find_stream_index_empty_list():
    assert find_stream_index_by_id([], "H1") == -1

# --- Tests for load_data_from_csv ---
# (This section is correct and unchanged)
def create_csv_string(headers, rows_of_dicts):
    output = ",".join(headers) + "\n"
    for row_dict in rows_of_dicts:
        row_values = [str(row_dict.get(h, "")) for h in headers]
        output += ",".join(row_values) + "\n"
    return output

@pytest.fixture
def create_dummy_csv_files(tmp_path):
    files_created = {}
    def _creator(
        streams_data=None, utils_data=None, matches_U_data=None, 
        forbidden_data=None, required_data=None
    ):
        if streams_data:
            streams_file = tmp_path / "streams.csv"
            streams_file.write_text(create_csv_string(["Name", "Type", "TIN_spec", "TOUT_spec", "Fcp"], streams_data))
            files_created['streams'] = streams_file
        if utils_data:
            utils_file = tmp_path / "utilities.csv"
            utils_file.write_text(create_csv_string(["Name", "Type", "TIN_utility", "TOUT_utility", "Unit_Cost_Energy", "U_overall", "Fixed_Cost_Unit", "Area_Cost_Coeff", "Area_Cost_Exp"], utils_data))
            files_created['utilities'] = utils_file
        if matches_U_data:
            matches_U_file = tmp_path / "matches_U.csv"
            matches_U_file.write_text(create_csv_string(["Hot_Stream", "Cold_Stream", "U_overall", "Fixed_Cost_Unit", "Area_Cost_Coeff", "Area_Cost_Exp"], matches_U_data))
            files_created['matches_U'] = matches_U_file
        return files_created
    return _creator

def test_load_data_successful_minimum(create_dummy_csv_files):
    streams_data = [{"Name": "H1", "Type": "hot", "TIN_spec": 200, "TOUT_spec": 100, "Fcp": 10}]
    utils_data = [{"Name": "Steam", "Type": "hot_utility", "TIN_utility": 250, "TOUT_utility": 249, "Unit_Cost_Energy": 0.02, "U_overall": 1.2, "Fixed_Cost_Unit": 1000, "Area_Cost_Coeff": 80, "Area_Cost_Exp": 0.6}]
    paths = create_dummy_csv_files(streams_data=streams_data, utils_data=utils_data)
    hs, cs, hu, cu, mu, fm, rm = load_data_from_csv(str(paths['streams']), str(paths['utilities']))
    assert len(hs) == 1 and hs[0]['Name'] == "H1"
    assert len(hu) == 1 and hu[0]['Name'] == "Steam"

# --- FIX ---
# Removed the unnecessary and problematic inject_tmp_path_into_fixture_creator fixture
# No other changes needed in this section. The existing load_data tests are fine.

