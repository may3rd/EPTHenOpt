# EPTHenOpt/utils.py
"""Utility functions for the EPTHenOpt package.

This module provides a collection of helper functions used across the package.
Responsibilities include:
- Loading problem data from CSV files.
- Calculating the Log Mean Temperature Difference (LMTD).
- Formatting and displaying final optimization results.
- Exporting results to structured files.

"""
import csv
import math
import numpy as np # Added for np.argwhere
import copy      # Added for copy.deepcopy
from pathlib import Path
import json

MIN_LMTD = 1e-6

OBJ_KEY_OPTIMIZING = "TAC_GA_optimizing"
OBJ_KEY_REPORT = "TAC_true_report"
OBJ_KEY_CO2 = "total_co2_emissions"

def calculate_lmtd(Th_in, Th_out, Tc_in, Tc_out):
    """Calculates the Log Mean Temperature Difference (LMTD).

    This function includes robust checks for temperature crosses and avoids
    division-by-zero errors when temperature differences are equal.

    Parameters
    ----------
    Th_in : float
        Inlet temperature of the hot stream, [K].
    Th_out : float
        Outlet temperature of the hot stream, [K].
    Tc_in : float
        Inlet temperature of the cold stream, [K].
    Tc_out : float
        Outlet temperature of the cold stream, [K].

    Returns
    -------
    float
        The calculated LMTD value, or a minimum value (`MIN_LMTD`)
        if the temperature profile is invalid or results in a non-positive LMTD.

    Notes
    -----
    The function handles two main edge cases:
    1.  Temperature cross: If `delta_T1` or `delta_T2` is non-positive, it
        indicates an invalid heat exchange, and a minimal LMTD is returned.
    2.  Equal temperature differences: If `delta_T1` and `delta_T2` are nearly
        identical, the LMTD is simply that value, avoiding a division by zero
        in the standard formula.

    Examples
    --------
    >>> calculate_lmtd(Th_in=400, Th_out=350, Tc_in=300, Tc_out=320)
    63.8263363403332
    >>> calculate_lmtd(Th_in=400, Th_out=350, Tc_in=300, Tc_out=350)
    50.0

    """
    delta_T1 = Th_in - Tc_out
    delta_T2 = Th_out - Tc_in

    # Check for invalid temperature configurations (temperature cross or zero delta)
    # which would lead to log(negative) or log(zero).
    if delta_T1 <= MIN_LMTD or delta_T2 <= MIN_LMTD:
        return MIN_LMTD

    # Check if temperature differences are nearly equal to avoid division by zero
    # from log(1), which is a common cause for the warning.
    if abs(delta_T1 - delta_T2) < 1e-9:
        # If they are equal, LMTD is simply that value (arithmetic mean)
        return delta_T1 
        
    try:
        # Standard LMTD formula calculation
        lmtd_val = (delta_T1 - delta_T2) / math.log(delta_T1 / delta_T2)
        
        # Final check to ensure the result is a valid, positive number
        if not math.isfinite(lmtd_val) or lmtd_val < MIN_LMTD:
            return MIN_LMTD
        return lmtd_val
    except (ValueError, ZeroDivisionError):
        # Catch any other unexpected math errors
        return MIN_LMTD


def find_stream_index_by_id(streams_list, stream_id_to_find):
    """Finds the index of a stream object in a list by its ID.

    Parameters
    ----------
    streams_list : list[Stream]
        A list of Stream objects to search through.
    stream_id_to_find : str
        The string ID of the stream to find.

    Returns
    -------
    int
        The index of the stream object if found, otherwise -1.
    
    Examples
    --------
    >>> from types import SimpleNamespace
    >>> streams = [SimpleNamespace(id='H1'), SimpleNamespace(id='H2')]
    >>> find_stream_index_by_id(streams, 'H2')
    1
    >>> find_stream_index_by_id(streams, 'C1')
    -1

    """
    for index, stream_obj in enumerate(streams_list):
        if stream_obj.id == stream_id_to_find:
            return index
    return -1

def load_data_from_csv(streams_filepath, utilities_filepath, matches_U_filepath=None, forbidden_matches_filepath=None, required_matches_filepath=None): # Corrected parameter names
    """Loads all problem data from specified CSV files.

    This function reads stream data, utility data, and optional constraint
    data from CSV files and returns them as lists of dictionaries.

    Parameters
    ----------
    streams_filepath : str
        Path to the streams CSV file.
    utilities_filepath : str
        Path to the utilities CSV file.
    matches_U_filepath : str, optional
        Path to a CSV file specifying costs and U-values for specific
        stream matches. Defaults to None.
    forbidden_matches_filepath : str, optional
        Path to a CSV file specifying forbidden matches between streams or
        between streams and utilities. Defaults to None.
    required_matches_filepath : str, optional
        Path to a CSV file specifying required matches between streams.
        Defaults to None.

    Returns
    -------
    tuple
        A tuple containing six elements:
        (loaded_hot_streams, loaded_cold_streams, loaded_hot_utilities,
        loaded_cold_utilities, loaded_matches_U, loaded_forbidden_matches,
        loaded_required_matches)
        Each element is a list of dictionaries, where each dictionary
        represents a row from the corresponding CSV file.

    """
    loaded_hot_streams = []
    loaded_cold_streams = []
    loaded_hot_utilities = []
    loaded_cold_utilities = []
    loaded_matches_U = []
    loaded_forbidden_matches = []
    loaded_required_matches = []
    
    try:
        with open(streams_filepath, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row_idx, row in enumerate(reader):
                try:
                    stream_data = {
                        'Name': row['Name'],
                        'Type': row['Type'].lower(),
                        'TIN_spec': float(row['TIN_spec']),
                        'TOUT_spec': float(row['TOUT_spec']),
                        'Fcp': float(row['Fcp'])
                    }
                    # Optionally add h_coeff if it's in your CSV
                    # 'h_coeff': float(row.get('h_coeff', 0)) 
                    if stream_data['Type'] == 'hot': loaded_hot_streams.append(stream_data)
                    elif stream_data['Type'] == 'cold': loaded_cold_streams.append(stream_data)
                    else: print(f"Warning: Unknown stream type '{row['Type']}' for stream '{row['Name']}'. Skipping.")
                except KeyError as e:
                    print(f"Error: Missing column {e} in streams.csv at row {row_idx+1}.")
                    return None,None,None,None,None,None,None
                except ValueError as e:
                    print(f"Error: Could not convert value to float in streams.csv at row {row_idx+1} (column affected by {e}).")
                    return None,None,None,None,None,None,None
    except FileNotFoundError:
        print(f"Error: Streams file not found at {streams_filepath}")
        return None,None,None,None,None,None,None
    except Exception as e:
        print(f"Error reading streams CSV: {e}")
        return None,None,None,None,None,None,None
    
    if matches_U_filepath:
        try:
            with open(matches_U_filepath, mode='r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row_idx, row in enumerate(reader):
                    try:
                        match_U_cost = {
                            'hot': row['Hot_Stream'], 
                            'cold': row['Cold_Stream'], 
                            'U': float(row['U_overall']), 
                            'fix_cost': float(row['Fixed_Cost_Unit']), 
                            'area_cost_coeff': float(row['Area_Cost_Coeff']), 
                            'area_cost_exp': float(row['Area_Cost_Exp'])
                        }
                        loaded_matches_U.append(match_U_cost)
                    except KeyError as e:
                        print(f"Error: Missing column {e} in {matches_U_filepath} at row {row_idx+1}.")
                    except ValueError as e:
                        print(f"Error: Could not convert value to float in {matches_U_filepath} at row {row_idx+1} (column affected by {e}).")
        except FileNotFoundError:
            print(f"Warning: Matches U cost file not found at {matches_U_filepath}. Proceeding without specific match costs.") # Changed to Warning
            loaded_matches_U = None # Explicitly set to None if not found but path given
        except Exception as e:
            print(f"Error reading {matches_U_filepath} CSV: {e}")
            loaded_matches_U = None
    else:
        loaded_matches_U = None # If no path is given
        
    try:
        with open(utilities_filepath, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row_idx, row in enumerate(reader):
                try:
                    util_data = {
                        'Name': row['Name'],
                        'Type': row['Type'].lower(),
                        'TIN_utility': float(row['TIN_utility']),
                        'TOUT_utility': float(row['TOUT_utility']),
                        'Unit_Cost_Energy': float(row['Unit_Cost_Energy']),
                        'U_overall': float(row['U_overall']),
                        'Fixed_Cost_Unit': float(row['Fixed_Cost_Unit']),
                        'Area_Cost_Coeff': float(row['Area_Cost_Coeff']),
                        'Area_Cost_Exp': float(row['Area_Cost_Exp'])
                        # 'h_coeff': float(row.get('h_coeff', 0)) # Optional
                    }
                    if util_data['Type'] == 'hot_utility': loaded_hot_utilities.append(util_data)
                    elif util_data['Type'] == 'cold_utility': loaded_cold_utilities.append(util_data)
                    else: print(f"Warning: Unknown utility type '{row['Type']}' for utility '{row['Name']}'. Skipping.")
                except KeyError as e:
                    print(f"Error: Missing column {e} in utilities.csv at row {row_idx+1}.")
                    return None,None,None,None,None,None,None
                except ValueError as e:
                    print(f"Error: Could not convert value to float in utilities.csv at row {row_idx+1} (column affected by {e}).")
                    return None,None,None,None,None,None,None
    except FileNotFoundError:
        print(f"Error: Utilities file not found at {utilities_filepath}")
        return None,None,None,None,None,None,None
    except Exception as e:
        print(f"Error reading utilities CSV: {e}")
        return None,None,None,None,None,None,None

    if not loaded_hot_utilities and any(s['Type'] == 'cold' for s in loaded_cold_streams):
        print("Warning: No hot utilities loaded. Cold streams might not meet targets without them or a default.")
    if not loaded_cold_utilities and any(s['Type'] == 'hot' for s in loaded_hot_streams):
        print("Warning: No cold utilities loaded. Hot streams might not meet targets without them or a default.")
    
    if forbidden_matches_filepath:
        try:
            with open(forbidden_matches_filepath, mode='r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row_idx, row in enumerate(reader):
                    try:
                        forbidden_match = {'hot': row['Hot_Stream'], 'cold': row['Cold_Stream_Or_Utility']}
                        loaded_forbidden_matches.append(forbidden_match)
                    except KeyError as e:
                        print(f"Error: Missing column {e} in {forbidden_matches_filepath} at row {row_idx+1}.")
                if loaded_forbidden_matches: print(f"Loaded {len(loaded_forbidden_matches)} forbidden matches from {forbidden_matches_filepath}")
        except FileNotFoundError:
            print(f"Warning: Forbidden matches file not found at {forbidden_matches_filepath}. No forbidden matches will be applied from this file.")
        except Exception as e:
            print(f"Error reading {forbidden_matches_filepath} CSV: {e}")
        
    if required_matches_filepath:
        try:
            with open(required_matches_filepath, mode='r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row_idx, row in enumerate(reader):
                    try:
                        required_match = {'hot': row['Hot_Stream'], 'cold': row['Cold_Stream'], 'min_Q_total': float(row['Min_Q_Total'])}
                        loaded_required_matches.append(required_match)
                    except KeyError as e:
                        print(f"Error: Missing column {e} in {required_matches_filepath} at row {row_idx+1}.")
                    except ValueError as e:
                         print(f"Error: Could not convert Min_Q_Total to float in {required_matches_filepath} at row {row_idx+1} for '{row.get('Min_Q_Total')}'.")
                if loaded_required_matches: print(f"Loaded {len(loaded_required_matches)} required matches from {required_matches_filepath}")
        except FileNotFoundError:
            print(f"Warning: Required matches file not found at {required_matches_filepath}. No required matches will be enforced from this file.")
        except Exception as e:
            print(f"Error reading {required_matches_filepath} CSV: {e}")
            
    return loaded_hot_streams, loaded_cold_streams, loaded_hot_utilities, loaded_cold_utilities, loaded_matches_U, loaded_forbidden_matches, loaded_required_matches


def export_results(results_data, hen_problem, output_dir):
    """Exports the optimization results to structured files (JSON and CSV).

    This function creates an output directory and saves three files:
    - `summary.json`: A JSON file with the final cost breakdown.
    - `network_structure.csv`: A CSV detailing each heat exchanger in the network.
    - `stream_results.csv`: A CSV showing the final state of each stream.

    Parameters
    ----------
    results_data : dict
        The dictionary containing the best run's results, including 'costs'
        and 'details' keys.
    hen_problem : HENProblem
        The HEN problem instance used for the optimization.
    output_dir : str
        The directory where result files will be saved.

    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nExporting results to directory: {output_path.resolve()}")

    # --- 1. Export Summary Costs to JSON ---
    summary_costs = results_data.get('costs', {})
    summary_path = output_path / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary_costs, f, indent=4)
    print(f"  - Saved summary.json")

    details = results_data.get('details', [])
    if not details:
        print("  - No detailed structure to export.")
        return

    # --- 2. Export Network Structure to CSV ---
    structure_path = output_path / 'network_structure.csv'
    structure_headers = [
        "Unit_Type", "Hot_Stream", "Cold_Stream", "Stage", "Heat_Duty_kW",
        "Area_m2", "LMTD_K", "Hot_In_K", "Hot_Out_K", "Hot_Stream_FCp", "Hot_Stream_Split_Ratio",
        "Cold_In_K", "Cold_Out_K", "Cold_Stream_FCp", "Cold_Stream_Split_Ratio"
    ]
    structure_rows = []
    final_stream_temps = {stream.id: stream.Tin for stream in hen_problem.hot_streams + hen_problem.cold_streams}

    for detail in details:
        row = {}
        Q_val = detail.get('Q', 0.0)
        
        # Initialize values to avoid errors
        row['Hot_Stream_FCp'] = 'N/A'
        row['Hot_Stream_Split_Ratio'] = 'N/A'
        row['Cold_Stream_FCp'] = 'N/A'
        row['Cold_Stream_Split_Ratio'] = 'N/A'

        if detail.get('type') == 'heater':
            row['Unit_Type'] = 'Heater'
            cold_stream = hen_problem.cold_streams[detail['C_idx']]
            row['Hot_Stream'] = detail.get('Util_ID', 'N/A')
            row['Cold_Stream'] = cold_stream.id
            row['Stage'] = 'N/A'
            row['Heat_Duty_kW'] = Q_val
            row['Area_m2'] = detail.get('Area', 0)
            row['Hot_In_K'] = detail.get('util_Tin', 0)
            row['Hot_Out_K'] = detail.get('util_Tout', 0)
            row['Cold_In_K'] = detail.get('Tc_in', 0)
            row['Cold_Out_K'] = detail.get('Tc_out', 0)
            row['LMTD_K'] = calculate_lmtd(row['Hot_In_K'], row['Hot_Out_K'], row['Cold_In_K'], row['Cold_Out_K'])
            final_stream_temps[cold_stream.id] = row['Cold_Out_K']
        elif detail.get('type') == 'cooler':
            row['Unit_Type'] = 'Cooler'
            hot_stream = hen_problem.hot_streams[detail['H_idx']]
            row['Hot_Stream'] = hot_stream.id
            row['Cold_Stream'] = detail.get('Util_ID', 'N/A')
            row['Stage'] = 'N/A'
            row['Heat_Duty_kW'] = Q_val
            row['Area_m2'] = detail.get('Area', 0)
            row['Hot_In_K'] = detail.get('Th_in', 0)
            row['Hot_Out_K'] = detail.get('Th_out', 0)
            row['Cold_In_K'] = detail.get('util_Tin', 0)
            row['Cold_Out_K'] = detail.get('util_Tout', 0)
            row['LMTD_K'] = calculate_lmtd(row['Hot_In_K'], row['Hot_Out_K'], row['Cold_In_K'], row['Cold_Out_K'])
            final_stream_temps[hot_stream.id] = row['Hot_Out_K']
        else: # Process Exchanger
            row['Unit_Type'] = 'Process_Exchanger'
            hot_stream = hen_problem.hot_streams[detail['H']]
            cold_stream = hen_problem.cold_streams[detail['C']]
            row['Hot_Stream'] = hot_stream.id
            row['Cold_Stream'] = cold_stream.id
            row['Stage'] = detail.get('k', -1) + 1
            row['Heat_Duty_kW'] = Q_val
            row['Area_m2'] = detail.get('Area', 0)
            row['Hot_In_K'] = detail.get('Th_in', 0)
            row['Hot_Out_K'] = detail.get('Th_out', 0)
            row['Cold_In_K'] = detail.get('Tc_in', 0)
            row['Cold_Out_K'] = detail.get('Tc_out', 0)
            row['LMTD_K'] = calculate_lmtd(row['Hot_In_K'], row['Hot_Out_K'], row['Cold_In_K'], row['Cold_Out_K'])

            # --- ADDED: Calculate FCp and Split Ratio ---
            if abs(row['Hot_Out_K'] - row['Hot_In_K']) > 1e-6:
                hot_fcp = Q_val / abs(row['Hot_Out_K'] - row['Hot_In_K'])
                row['Hot_Stream_FCp'] = hot_fcp
                if hot_stream.CP > 1e-6:
                    row['Hot_Stream_Split_Ratio'] = hot_fcp / hot_stream.CP

            if abs(row['Cold_Out_K'] - row['Cold_In_K']) > 1e-6:
                cold_fcp = Q_val / abs(row['Cold_Out_K'] - row['Cold_In_K'])
                row['Cold_Stream_FCp'] = cold_fcp
                if cold_stream.CP > 1e-6:
                    row['Cold_Stream_Split_Ratio'] = cold_fcp / cold_stream.CP

        structure_rows.append(row)

    with open(structure_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=structure_headers)
        writer.writeheader()
        writer.writerows(structure_rows)
    print(f"  - Saved network_structure.csv")

    # --- 3. Export Final Stream States to CSV ---
    stream_results_path = output_path / 'stream_results.csv'
    stream_headers = [
        "Stream_ID", "Stream_Type", "Initial_Tin_K", "Target_Tout_K",
        "Final_Tout_K", "Is_Target_Met"
    ]
    stream_rows = []
    final_outlet_Th_after_utility = summary_costs.get('final_outlet_Th_after_utility', [])
    final_outlet_Tc_after_utility = summary_costs.get('final_outlet_Tc_after_utility', [])
    
    def append_streams(streams_list, final_outlet_after_utility, stream_rows):
        for i, stream in enumerate(streams_list):
            stream_rows.append({
                "Stream_ID": stream.id,
                "Stream_Type": stream.type,
                "Initial_Tin_K": stream.Tin,
                "Target_Tout_K": stream.Tout_target,
                "Final_Tout_K": final_outlet_after_utility[i],
                "Is_Target_Met": abs(stream.Tout_target - final_outlet_after_utility[i]) < 1e-6,
            })
        return stream_rows
    
    stream_rows = append_streams(hen_problem.hot_streams, final_outlet_Th_after_utility, stream_rows)
    stream_rows = append_streams(hen_problem.cold_streams, final_outlet_Tc_after_utility, stream_rows)

    with open(stream_results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=stream_headers)
        writer.writeheader()
        writer.writerows(stream_rows)
    print(f"  - Saved stream_results.csv")

def export_multi_objective_results(pareto_front, output_dir):
    """Exports the Pareto front from a multi-objective run to a CSV file.

    Parameters
    ----------
    pareto_front : list[dict]
        A list of solution dictionaries, where each dictionary represents a
        point on the Pareto front.
    output_dir : str
        The directory where `pareto_front_results.csv` will be saved.

    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pareto_path = output_path / 'pareto_front_results.csv'
    
    headers = [
        "Solution_ID", "TAC_true_report", "Total_CO2_Emissions_kg_per_hr", 
        "Total_Capital_Cost", "Total_Operating_Cost", "Hot_Utility_kW", "Cold_Utility_kW"
    ]
    
    rows = []
    for i, solution in enumerate(pareto_front):
        costs = solution['costs']
        rows.append({
            "Solution_ID": i + 1,
            "TAC_true_report": costs.get('TAC_true_report'),
            "Total_CO2_Emissions_kg_per_hr": costs.get('total_co2_emissions'),
            "Total_Capital_Cost": costs.get('total_capital_cost'),
            "Total_Operating_Cost": costs.get('total_operating_cost'),
            "Hot_Utility_kW": costs.get('Q_hot_consumed_kW_actual'),
            "Cold_Utility_kW": costs.get('Q_cold_consumed_kW_actual'),
        })
        
    with open(pareto_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nExported Pareto front solutions to: {pareto_path.resolve()}")

def display_optimization_results(all_run_results, hen_problem_instance, model_name, output_dir=None, objective='single'):
    """Summarizes and displays the final optimization results to the console.

    This function processes results from one or more optimization runs. For
    single-objective runs, it identifies the best overall solution, prints a
    detailed cost breakdown, and shows the resulting network structure. For
    multi-objective runs, it summarizes the Pareto front. It also triggers
    the export of results to files if an output directory is provided.

    Parameters
    ----------
    all_run_results : list[dict]
        A list of dictionaries, where each contains results from one run.
        Expected keys: 'seed', 'costs', 'chromosome', 'details'.
    hen_problem_instance : HENProblem
        The HENProblem instance used for the optimization, needed to decode
        chromosomes and access stream data.
    model_name : str
        The name of the optimization model used (e.g., 'GA', 'TLBO').
    output_dir : str, optional
        Directory to save results files. If provided, `export_results` or
        `export_multi_objective_results` will be called. Defaults to None.
    objective : str, optional
        The optimization objective type ('single' or 'multi').
        Defaults to 'single'.

    """
    if objective == 'multi':
        print(f"\n--- Pareto Front Summary for NSGA-II ---")
        print(f"Found {len(all_run_results)} optimal trade-off solutions.")
        if output_dir:
            export_multi_objective_results(all_run_results, output_dir)
    else:
        display_problem_summary(hen_problem_instance)
        print(f"Pinch Analysis (EMAT={hen_problem_instance.cost_params.EMAT}K): Q_H_min={hen_problem_instance.Q_H_min_pinch:.2f}kW, Q_C_min={hen_problem_instance.Q_C_min_pinch:.2f}kW")
        if hen_problem_instance.T_pinch_hot_actual is not None: print(f"  T_Pinch_Hot={hen_problem_instance.T_pinch_hot_actual:.2f}K, T_Pinch_Cold={hen_problem_instance.T_pinch_cold_actual:.2f}K")

        print(f"--- Summary of Multiple {model_name} Runs ---")
        if not all_run_results:
            print("No results to summarize.")
            return

        best_overall_objective_val = float('inf') # Use a generic term, as it's model's objective
        best_run_final_info = None   
        
        for run_result in all_run_results:
            if not run_result or 'costs' not in run_result or run_result['costs'] is None:
                print(f"Skipping invalid run result: {run_result}")
                continue

            # The key used for optimization (e.g., OBJ_KEY_OPTIMIZING or a similar objective for TLBO)
            objective_val = run_result['costs'].get(OBJ_KEY_OPTIMIZING, float('inf')) 
            true_tac_for_display = run_result['costs'].get('TAC_true_report', float('inf'))

            objective_val_str = f"{objective_val:.2f}" if objective_val != float('inf') else "Inf"
            true_tac_str = f"{true_tac_for_display:.2f}" if true_tac_for_display != float('inf') else "Inf"
            
            print(f"Run with Seed {run_result.get('seed', 'N/A')}: True TAC = {true_tac_str} (Optimized Obj. = {objective_val_str})")
            
            if objective_val < best_overall_objective_val : 
                best_overall_objective_val = objective_val
                best_run_final_info = copy.deepcopy(run_result)

        if best_run_final_info and 'costs' in best_run_final_info and \
        best_run_final_info['costs'] is not None and \
        best_run_final_info['costs'].get(OBJ_KEY_OPTIMIZING, float('inf')) != float('inf'): # Check against the optimizing TAC
            
            overall_best_true_tac_val = best_run_final_info['costs'].get('TAC_true_report', float('inf'))
            # overall_best_ga_tac_val = best_run_final_info['costs'].get(OBJ_KEY_OPTIMIZING, float('inf')) # Already have this in best_overall_objective_val

            true_tac_overall_str = f"{overall_best_true_tac_val:.2f}" if overall_best_true_tac_val != float('inf') else "Inf"
            optimized_obj_overall_str = f"{best_overall_objective_val:.2f}" if best_overall_objective_val != float('inf') else "Inf"

            print(f"\nBest True TAC found across all runs (corresponding to best Optimized Objective): {true_tac_overall_str}")
            print(f"  (This solution had an Optimized Objective of: {optimized_obj_overall_str})")
            print(f"  Best solution from Seed: {best_run_final_info.get('seed', 'N/A')}")
            
            costs_to_print = best_run_final_info['costs']
            print("\nCost Breakdown for the Best Overall Solution:")
            print(f"  Optimized Objective : {costs_to_print.get(OBJ_KEY_OPTIMIZING,0):.2f}")
            print(f"  True TAC            : {costs_to_print.get('TAC_true_report', 0):.2f}")
            print(f"  CapEx (Process Ex.) : {costs_to_print.get('capital_process_exchangers',0):.2f}")
            print(f"  CapEx (Heaters)     : {costs_to_print.get('capital_heaters',0):.2f}")
            print(f"  CapEx (Coolers)     : {costs_to_print.get('capital_coolers',0):.2f}")
            print(f"  OpEx (Hot Utility)  : {costs_to_print.get('op_cost_hot_utility',0):.2f}")
            print(f"  OpEx (Cold Utility) : {costs_to_print.get('op_cost_cold_utility',0):.2f}")
            
            penalty_keys = [k for k in costs_to_print if "penalty" in k.lower() and costs_to_print.get(k, 0) > 1e-6]
            if penalty_keys:
                print("  Penalties Applied (in Optimized Objective):")
                for pk in penalty_keys:
                    print(f"    {pk.replace('_', ' ').title()}: {costs_to_print[pk]:.2f}")
            else:
                print("  No significant penalties applied in the optimized objective for the best solution.")


            print("\nStructure of the Best Overall Solution:")
            full_chromosome_best = best_run_final_info.get('chromosome')
            details_overall = best_run_final_info.get('details')

            if full_chromosome_best is not None and details_overall is not None and hen_problem_instance:
                Z_overall_best, _, _ = hen_problem_instance._decode_chromosome(full_chromosome_best)
                
                if Z_overall_best is not None:
                    active_matches = np.argwhere(Z_overall_best == 1)
                    if active_matches.size == 0:
                        print("  No active process-process matches found in the best solution.")
                
                total_Q_recovered, total_area_process_exch = 0.0, 0.0
                Q_hot_util_op_val, Q_cold_util_op_val = 0.0, 0.0
                
                print("\n  Process Heat Exchangers:")
                process_exchangers_count = 0
                for detail in details_overall:
                    if 'H' in detail and 'C' in detail and 'type' not in detail: # Process Exchanger
                        hot_stream_obj = hen_problem_instance.hot_streams[detail['H']]
                        cold_stream_obj = hen_problem_instance.cold_streams[detail['C']]
                        hot_name = hot_stream_obj.id
                        cold_name = cold_stream_obj.id
                        
                        Q_val = detail.get('Q',0.0)
                        Th_in_val = detail.get('Th_in',0.0)
                        Th_out_val = detail.get('Th_out',0.0)
                        Tc_in_val = detail.get('Tc_in',0.0)
                        Tc_out_val = detail.get('Tc_out',0.0)
                        Area_val = detail.get('Area',0.0)
                        U = detail.get('U',0.0)
                        lmtd = detail.get('lmtd',0.0)
                        cost = detail.get('cost',0.0)

                        if Q_val < 1e-6: continue 

                        hot_CFp, cold_CFp = 0.0, 0.0
                        hot_Split_ratio, cold_Split_ratio = 0.0, 0.0

                        if abs(Th_in_val - Th_out_val) > 1e-6: hot_CFp = Q_val / abs(Th_in_val - Th_out_val)
                        if hot_stream_obj.CP > 1e-6: hot_Split_ratio = hot_CFp / hot_stream_obj.CP
                        
                        if abs(Tc_in_val - Tc_out_val) > 1e-6: cold_CFp = Q_val / abs(Tc_in_val - Tc_out_val)
                        if cold_stream_obj.CP > 1e-6: cold_Split_ratio = cold_CFp / cold_stream_obj.CP

                        print(f"  {process_exchangers_count+1:2d}  {hot_name}-{cold_name} (Stage {detail.get('k',0)+1}): Q={Q_val:.2f} kW, Area={Area_val:.2f} m^2, U={U:.2f}, LMTD={lmtd:.2f}, Cost=${cost:.2f}")
                        print(f"      {hot_name}: FlowCp_branch={hot_CFp:.2f} (SplitFrac={hot_Split_ratio:.3f}), Tin={Th_in_val:.1f} K, Tout={Th_out_val:.1f} K")
                        print(f"      {cold_name}: FlowCp_branch={cold_CFp:.2f} (SplitFrac={cold_Split_ratio:.3f}), Tin={Tc_in_val:.1f} K, Tout={Tc_out_val:.1f} K\n")
                        
                        process_exchangers_count += 1
                        total_Q_recovered += Q_val
                        total_area_process_exch += Area_val
                if process_exchangers_count == 0:
                    print("    None.")
                    
                print(f"  Total process exchangers: {process_exchangers_count}")
                print(f"  Total Q Recovered (Process Exchangers): {total_Q_recovered:.2f} kW")
                print(f"  Total Area (Process Exchangers): {total_area_process_exch:.2f} m^2")
                
                print("\n  Utility Units:")
                heaters_count, coolers_count = 0, 0
                
                # --- HEATER PRINTING LOGIC ---
                for detail in details_overall:
                    if detail.get('type') == 'heater':
                        heaters_count += 1
                        cold_stream_obj = hen_problem_instance.cold_streams[detail['C_idx']]
                        # Use the correct keys from the 'detail' dictionary
                        q_val = detail.get('Q', 0)
                        area_val = detail.get('Area', 0) # Use 'Area' (capital A)
                        tc_in_val = detail.get('Tc_in', 0)
                        tc_out_val = detail.get('Tc_out', 0)
                        cap_cost = detail.get('cap', 0)
                        op_cost = detail.get('op', 0)
                        
                        print(f"   {heaters_count:2d} Heater for {cold_stream_obj.id}: Q={q_val:.2f} kW, Area={area_val:.2f} m^2")
                        print(f"       Process Stream: Tin={tc_in_val:.1f} K, Tout={tc_out_val:.1f} K")
                        print(f"       Costs: Capital=${cap_cost:.2f}, Operating=${op_cost:.2f}\n")
                        
                        Q_hot_util_op_val += q_val
                
                if heaters_count == 0:
                    print("    No Heaters.\n")

                # --- COOLER PRINTING LOGIC ---
                for detail in details_overall:
                    if detail.get('type') == 'cooler':
                        coolers_count += 1
                        hot_stream_obj = hen_problem_instance.hot_streams[detail['H_idx']]
                        # Use the correct keys from the 'detail' dictionary
                        q_val = detail.get('Q', 0)
                        area_val = detail.get('Area', 0) # Use 'Area' (capital A)
                        th_in_val = detail.get('Th_in', 0)
                        th_out_val = detail.get('Th_out', 0)
                        cap_cost = detail.get('cap', 0)
                        op_cost = detail.get('op', 0)
                        
                        print(f"   {coolers_count:2d} Cooler for {hot_stream_obj.id}: Q={q_val:.2f} kW, Area={area_val:.2f} m^2")
                        print(f"       Process Stream: Tin={th_in_val:.1f} K, Tout={th_out_val:.1f} K")
                        print(f"       Costs: Capital=${cap_cost:.2f}, Operating=${op_cost:.2f}\n")
                        
                        Q_cold_util_op_val += q_val
                
                if coolers_count == 0:
                    print("    No Coolers.")
                
                if Q_hot_util_op_val > 1e-6 or Q_cold_util_op_val > 1e-6:
                    print(f"\n  Utility Duty Summary:")
                    if Q_hot_util_op_val > 1e-6:
                        print(f"    Total Hot Utility: {heaters_count} Exchanger{'s' if coolers_count > 1 else ''}, total duty: {Q_hot_util_op_val:.2f} kW")
                    else:
                        print(f"    No Hot Utility required.")
                    if Q_cold_util_op_val > 1e-6:
                        print(f"    Total Cold Utility {coolers_count} Exchanger{'s' if coolers_count > 1 else ''}:, total duty: {Q_cold_util_op_val:.2f} kW")
                    else: print(f"    No Cold Utility required.")
                else:
                    print(f"\n  Neither Hot or Cold Utility Required by the best solution.")
            else:
                print("  Best solution chromosome, details, or HEN problem instance are missing. Cannot print detailed structure.")
        else:
            print(f"\nNo valid (finite optimized objective) best solution found across all runs for {model_name}.")
        
        # --- After printing to console, call the export function ---
        if best_run_final_info and output_dir:
            export_results(best_run_final_info, hen_problem_instance, output_dir)

def display_problem_summary(hen_problem):
    """Prints a summary of the HEN problem definition to the console."""
    print("\n" + "="*80)
    print("HEN Problem Summary".center(80))
    print("="*80)
    
    print(f"\nHot Streams ({hen_problem.NH} total):")
    if hen_problem.hot_streams:
        max_len = max(len(hs.id) for hs in hen_problem.hot_streams)
        for hs in hen_problem.hot_streams:
            print(f"  - ID: {hs.id:<{max_len}} | Tin: {hs.Tin:<7.1f}K | Tout: {hs.Tout_target:<7.1f}K | CP: {hs.CP:<7.2f} kW/K")
    else:
        print("  None")
        
    print(f"\nCold Streams ({hen_problem.NC} total):")
    if hen_problem.cold_streams:
        max_len = max(len(cs.id) for cs in hen_problem.cold_streams)
        for cs in hen_problem.cold_streams:
            print(f"  - ID: {cs.id:<{max_len}} | Tin: {cs.Tin:<7.1f}K | Tout: {cs.Tout_target:<7.1f}K | CP: {cs.CP:<7.2f} kW/K")
    else:
        print("  None")
        
    print(f"\nHot Utilities ({hen_problem.NHU} total):")
    if hen_problem.hot_utility:
        max_len = max(len(hu.id) for hu in hen_problem.hot_utility)
        for hu in hen_problem.hot_utility:
            print(f"  - ID: {hu.id:<{max_len}} | Tin: {hu.Tin:<7.1f}K | Tout: {hu.Tout:<7.1f}K | Cost: {hu.cost} $/kW")
    else:
        print("  None")

    print(f"\nCold Utilities ({hen_problem.NCU} total):")
    if hen_problem.cold_utility:
        max_len = max(len(cu.id) for cu in hen_problem.cold_utility)
        for cu in hen_problem.cold_utility:
            print(f"  - ID: {cu.id:<{max_len}} | Tin: {cu.Tin:<7.1f}K | Tout: {cu.Tout:<7.1f}K | Cost: {cu.cost} $/kW")
    else:
        print("  None")
    
    print(f"\nNumber of stage: {hen_problem.num_stages}")
    
    print("\n" + "="*80 + "\n")

def display_help():
    """Prints the command-line interface help message and exits."""
    help_string = """usage: run_problem.py [-h] [--streams_file STREAMS_FILE] [--utilities_file UTILITIES_FILE]
                              [--matches_U_file MATCHES_U_FILE]
                              [--forbidden_matches_file FORBIDDEN_MATCHES_FILE]
                              [--required_matches_file REQUIRED_MATCHES_FILE]
                              [--EMAT_setting EMAT_SETTING] [--model {GA,TLBO}]
                              [--population_size POPULATION_SIZE] [--epochs EPOCHS]
                              [--generations_per_epoch GENERATIONS_PER_EPOCH]
                              [--number_of_workers NUMBER_OF_WORKERS] [--num_stages NUM_STAGES]
                              [--default_U_overall DEFAULT_U_OVERALL]
                              [--default_exch_fixed_cost DEFAULT_EXCH_FIXED_COST]
                              [--default_exch_area_coeff DEFAULT_EXCH_AREA_COEFF]
                              [--default_exch_area_exp DEFAULT_EXCH_AREA_EXP]
                              [--ga_crossover_prob GA_CROSSOVER_PROB]
                              [--ga_mutation_prob_Z_setting GA_MUTATION_PROB_Z_SETTING]
                              [--ga_mutation_prob_R_setting GA_MUTATION_PROB_R_SETTING]
                              [--ga_r_mutation_std_dev_factor_setting GA_R_MUTATION_STD_DEV_FACTOR_SETTING]
                              [--ga_elitism_frac GA_ELITISM_FRAC]
                              [--ga_tournament_size GA_TOURNAMENT_SIZE]
                              [--tlbo_teaching_factor {0,1,2}]
                              [--utility_cost_factor UTILITY_COST_FACTOR]
                              [--pinch_dev_penalty_factor PINCH_DEV_PENALTY_FACTOR]
                              [--sws_max_iter SWS_MAX_ITER] [--sws_conv_tol SWS_CONV_TOL]
                              [--initial_penalty INITIAL_PENALTY] [--final_penalty FINAL_PENALTY]

Run Heat Exchanger Network (HEN) Synthesis Optimization using EPTHenOpt.

options:
  -h, --help            show this help message and exit

File Path Arguments:
  --streams_file STREAMS_FILE
                        Path to the streams CSV file. (default: streams.csv)
  --utilities_file UTILITIES_FILE
                        Path to the utilities CSV file. (default: utilities.csv)
  --matches_U_file MATCHES_U_FILE
                        Optional: CSV for specific match costs/U-values. (default: None)
  --forbidden_matches_file FORBIDDEN_MATCHES_FILE
                        Optional: CSV for forbidden matches. (default: None)
  --required_matches_file REQUIRED_MATCHES_FILE
                        Optional: CSV for required matches. (default: None)

Core Optimization Arguments:
  --model {GA,TLBO}     Optimization model to use. (default: GA)
  --population_size POPULATION_SIZE
                        Population size for the optimizer. (default: 200)
  --epochs EPOCHS       Number of epochs for optimizer runs. (default: 10)
  --generations_per_epoch GENERATIONS_PER_EPOCH
                        Number of generations to run per epoch. (default: 20)
  --number_of_workers NUMBER_OF_WORKERS
                        Number of parallel workers (<=1 for sequential). (default: 1)
  --num_stages NUM_STAGES
                        HEN superstructure stages (0 to auto-calculate). (default: 0)

Problem & Cost Parameters:
  --EMAT_setting EMAT_SETTING
                        Minimum Approach Temperature (EMAT) in K. (default: 3.0)
  --default_U_overall DEFAULT_U_OVERALL
                        Default overall heat transfer coefficient (U). (default: 0.5)
  --default_exch_fixed_cost DEFAULT_EXCH_FIXED_COST
                        Default fixed cost for exchangers. (default: 0.0)
  --default_exch_area_coeff DEFAULT_EXCH_AREA_COEFF
                        Default area coefficient for exchangers. (default: 1000.0)
  --default_exch_area_exp DEFAULT_EXCH_AREA_EXP
                        Default area exponent for exchangers. (default: 0.6)

GA Specific Parameters:
  --ga_crossover_prob GA_CROSSOVER_PROB
                        Crossover probability. (default: 0.85)
  --ga_mutation_prob_Z_setting GA_MUTATION_PROB_Z_SETTING
                        Mutation probability for Z matrix. (default: 0.1)
  --ga_mutation_prob_R_setting GA_MUTATION_PROB_R_SETTING
                        Mutation probability for R matrices. (default: 0.1)
  --ga_r_mutation_std_dev_factor_setting GA_R_MUTATION_STD_DEV_FACTOR_SETTING
                        Std dev factor for R mutation. (default: 0.1)
  --ga_elitism_frac GA_ELITISM_FRAC
                        Fraction of population for elitism. (default: 0.1)
  --ga_tournament_size GA_TOURNAMENT_SIZE
                        Tournament size for selection. (default: 3)

TLBO Specific Parameters:
  --tlbo_teaching_factor {0,1,2}
                        TLBO Teaching Factor (TF). 0 for random (1 or 2). (default: 0)

Fitness & Penalty Parameters:
  --utility_cost_factor UTILITY_COST_FACTOR
                        Factor on utility op-ex in TAC. (default: 1.0)
  --pinch_dev_penalty_factor PINCH_DEV_PENALTY_FACTOR
                        Penalty for deviation from pinch targets. (default: 150.0)
  --sws_max_iter SWS_MAX_ITER
                        Max iterations for SWS calculation. (default: 300)
  --sws_conv_tol SWS_CONV_TOL
                        Convergence tolerance for SWS. (default: 1e-05)
  --initial_penalty INITIAL_PENALTY
                        Initial adaptive penalty factor. (default: 1000.0)
  --final_penalty FINAL_PENALTY
                        Final adaptive penalty factor. (default: 10000000.0)
"""
    print(help_string)
    exit(0)