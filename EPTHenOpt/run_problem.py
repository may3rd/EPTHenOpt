# EPTHenOpt/run_problem.py
"""
Main executable script for running HEN optimizations with EPTHenOpt.

This script serves as the command-line interface (CLI) for the package. It
handles parsing user arguments, loading problem data from CSV files, setting
up the optimizer (GA or TLBO), and orchestrating the optimization run either
sequentially or in parallel. Finally, it displays the results.
"""
import argparse
import sys
import json
from pathlib import Path
import time
from tqdm import tqdm

# Use relative imports to prevent installation conflicts
from .hen_models import Stream, Utility, CostParameters, HENProblem
from .ga_helpers import GeneticAlgorithmHEN
from .tlbo_helpers import TeachingLearningBasedOptimizationHEN
from .pso_helpers import ParticleSwarmOptimizationHEN
from .sa_helpers import SimulatedAnnealingHEN
from .aco_helpers import AntColonyOptimizationHEN
from .nsga2_helpers import NSGAIIHEN
from .utils import (
    load_data_from_csv, display_optimization_results,
    display_problem_summary, display_help,
    OBJ_KEY_OPTIMIZING, OBJ_KEY_REPORT, OBJ_KEY_CO2, TRUE_TAC_KEY
)
from .cores import run_parallel_with_migration

SOLVER_MAP = {
    'GA': GeneticAlgorithmHEN,
    'TLBO': TeachingLearningBasedOptimizationHEN,
    'PSO': ParticleSwarmOptimizationHEN,
    'SA': SimulatedAnnealingHEN,
    'ACO': AntColonyOptimizationHEN
}

# --- Main Execution Function ---
def main(args):
    """
    Main function to run HEN synthesis, configured by command-line arguments.
    """
    if args.objective == 'multi':
        model_display_name = 'NSGA-II'
    else:
        model_display_name = args.model.upper()
    print(f"HEN Synthesis using {model_display_name} with EPTHenOpt")

    loaded_hs_data, loaded_cs_data, loaded_hu_data, loaded_cu_data, \
    loaded_matches_U, loaded_forbidden, loaded_required = load_data_from_csv(
        args.streams_file, args.utilities_file, args.matches_U_file,
        args.forbidden_matches_file, args.required_matches_file
    )

    if not (loaded_hs_data or loaded_cs_data):
        print("Error: No hot or cold stream data loaded. Exiting.")
        exit(1)

    # Map CSV dictionary keys to the class constructor parameters
    hot_streams = [
        Stream(
            id_val=s['Name'], Tin=s['TIN_spec'], Tout_target=s['TOUT_spec'],
            CP=s['Fcp'], stream_type='hot',
            h_coeff=s['h_coeff'] if 'h_coeff' in s else None
        ) for s in loaded_hs_data
    ] if loaded_hs_data else []

    cold_streams = [
        Stream(
            id_val=s['Name'], Tin=s['TIN_spec'], Tout_target=s['TOUT_spec'],
            CP=s['Fcp'], stream_type='cold',
            h_coeff=s['h_coeff'] if 'h_coeff' in s else None
        ) for s in loaded_cs_data
    ] if loaded_cs_data else []

    hot_utilities = [
        Utility(
            id_val=u['Name'], Tin=u['TIN_utility'], Tout=u['TOUT_utility'],
            U=u['U_overall'], cost_per_energy_unit=u['Unit_Cost_Energy'],
            fix_cost=u['Fixed_Cost_Unit'], area_cost_coeff=u['Area_Cost_Coeff'],
            area_cost_exp=u['Area_Cost_Exp'], utility_type='hot_utility',
            co2_factor=float(u.get('co2_factor', args.default_co2_hot_utility or 0.0)),
            h_coeff=u['h_coeff'] if 'h_coeff' in u else None
        ) for u in loaded_hu_data
    ] if loaded_hu_data else []

    cold_utilities = [
        Utility(
            id_val=u['Name'], Tin=u['TIN_utility'], Tout=u['TOUT_utility'],
            U=u['U_overall'], cost_per_energy_unit=u['Unit_Cost_Energy'],
            fix_cost=u['Fixed_Cost_Unit'], area_cost_coeff=u['Area_Cost_Coeff'],
            area_cost_exp=u['Area_Cost_Exp'], utility_type='cold_utility',
            co2_factor=float(u.get('co2_factor', args.default_co2_cold_utility or 0.0)),
            h_coeff=u['h_coeff'] if 'h_coeff' in u else None
        ) for u in loaded_cu_data
    ] if loaded_cu_data else []

    if not hot_utilities and cold_streams:
        hot_utilities.append(Utility(id_val="DefaultHU", Tin=500, Tout=499,
                                     U=1.0, cost_per_energy_unit=0.02,
                                     fix_cost=1000, area_cost_coeff=80,
                                     area_cost_exp=0.6, utility_type='hot_utility',
                                     co2_factor=args.default_co2_hot_utility or 0.0))
        print("Warning: No hot utilities loaded from file, using a default hot utility.")

    if not cold_utilities and hot_streams:
        cold_utilities.append(Utility(id_val="DefaultCU", Tin=20, Tout=30,
                                      U=1.0, cost_per_energy_unit=0.005,
                                      fix_cost=500, area_cost_coeff=70,
                                      area_cost_exp=0.65, utility_type='cold_utility',
                                      co2_factor=args.default_co2_cold_utility or 0.0))
        print("Warning: No cold utilities loaded from file, using a default cold utility.")

    cost_params = CostParameters(
        exch_fixed=args.exchanger_fixed_cost,
        exch_area_coeff=args.exchanger_area_coeff,
        exch_area_exp=args.exchanger_area_exp,
        heater_fixed=args.heater_fixed_cost,
        heater_area_coeff=args.heater_area_coeff,
        heater_area_exp=args.heater_area_exp,
        cooler_fixed=args.cooler_fixed_cost,
        cooler_area_coeff=args.cooler_area_coeff,
        cooler_area_exp=args.cooler_area_exp,
        U_overall=args.default_U_overall,
        EMAT=args.EMAT_setting,
        )
    
    num_stages = args.num_stages if args.num_stages > 0 else max(1, len(hot_streams), len(cold_streams))

    hen_problem = HENProblem(
        hot_streams=hot_streams, cold_streams=cold_streams,
        hot_utility=hot_utilities, cold_utility=cold_utilities,
        cost_params=cost_params, num_stages=num_stages,
        matches_U_cost=loaded_matches_U, forbidden_matches=loaded_forbidden,
        required_matches=loaded_required, no_split=args.no_split,
        min_Q_limit=args.mininum_Q_limit
    )

    display_problem_summary(hen_problem)

    print("\n" + "="*50)
    print("Optimization Run Configuration".center(50))
    print("="*50)
    print(f"  - Optimization Model: {args.model}")
    print(f"  - Parallel Workers: {args.number_of_workers}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Generations per Epoch: {args.generations_per_epoch}")
    print(f"  - Total Generations: {args.epochs * args.generations_per_epoch}")
    print(f"  - Population Size: {args.population_size}")

    if args.model.upper() == 'GA':
        print("\n  Genetic Algorithm (GA) Parameters:")
        print(f"    - Crossover Probability: {args.ga_crossover_prob}")
        print(f"    - Mutation Prob (Z): {args.ga_mutation_prob_Z_setting}")
        print(f"    - Mutation Prob (R): {args.ga_mutation_prob_R_setting}")
        print(f"    - Elitism Fraction: {args.ga_elitism_frac}")
        print(f"    - Tournament Size: {args.ga_tournament_size}")
    elif args.model.upper() == 'TLBO':
        print("\n  Teaching-Learning-Based Optimization (TLBO) Parameters:")
        tf_display = 'Random (1 or 2)' if args.tlbo_teaching_factor == 0 else str(args.tlbo_teaching_factor)
        print(f"    - Teaching Factor (TF): {tf_display}")

    print("="*50 + "\n")

    common_opt_params = {"utility_cost_factor": args.utility_cost_factor, "pinch_deviation_penalty_factor": args.pinch_dev_penalty_factor, "sws_max_iter": args.sws_max_iter, "sws_conv_tol": args.sws_conv_tol}
    total_gens = args.epochs * args.generations_per_epoch
    processed_results = []
    
    start_time = time.time()

    if args.objective == 'multi':
        print("\nRunning in Multi-Objective mode (NSGA-II)...")
        # NSGA-II uses GA-like parameters for its variation operators
        solver_params = {
            **common_opt_params,
            "generations": total_gens,
            "crossover_prob": args.ga_crossover_prob,
            "mutation_prob_Z": args.ga_mutation_prob_Z_setting,
            "mutation_prob_R": args.ga_mutation_prob_R_setting,
            "initial_penalty": args.initial_penalty,
            "final_penalty": args.final_penalty
        }
        solver = NSGAIIHEN(problem=hen_problem, population_size=args.population_size, **solver_params)
        solver.run()
        processed_results = solver.best_front  # The result is the entire front
    else:
        # --- Single-Objective Path ---
        model_opt_specific_params = {}
        if args.model.upper() == 'GA':
            model_opt_specific_params = { "crossover_prob": args.ga_crossover_prob, "mutation_prob_Z": args.ga_mutation_prob_Z_setting, "mutation_prob_R": args.ga_mutation_prob_R_setting, "r_mutation_std_dev_factor": args.ga_r_mutation_std_dev_factor_setting, "elitism_count": int(args.ga_elitism_frac * args.population_size), "tournament_size": args.ga_tournament_size }
        elif args.model.upper() == 'TLBO':
            model_opt_specific_params = { "tlbo_teaching_factor": args.tlbo_teaching_factor }
        elif args.model.upper() == 'PSO':
            model_opt_specific_params = { "inertia_weight": args.pso_inertia_weight, "cognitive_coeff": args.pso_cognitive_coeff, "social_coeff": args.pso_social_coeff }
        elif args.model.upper() == 'SA':
            model_opt_specific_params = { "initial_temp": args.sa_initial_temp, "final_temp": args.sa_final_temp, "cooling_rate": args.sa_cooling_rate }
        elif args.model.upper() == 'ACO':
            model_opt_specific_params = { "evaporation_rate": args.aco_evaporation_rate, "pheromone_influence": args.aco_pheromone_influence, "pheromone_deposit_amount": args.aco_pheromone_deposit_amount }
         
        solver_params = { **common_opt_params, **model_opt_specific_params, "initial_penalty": args.initial_penalty, "final_penalty": args.final_penalty, "generations": total_gens }

        if args.number_of_workers <= 1:
            print("\nRunning in sequential mode (1 worker)...")
            solver_class = SOLVER_MAP.get(args.model.upper())
            if not solver_class:
                raise ValueError(f"Unknown model: {args.model}")
            
            solver = solver_class(problem=hen_problem, population_size=args.population_size, **solver_params)

            for epoch in tqdm(range(args.epochs), desc="Epochs Progress"):
                print(f"\n--- Starting Epoch {epoch + 1}/{args.epochs} ---")
                solver.run_epoch(
                    generations_in_epoch=args.generations_per_epoch,
                    current_gen_offset=epoch * args.generations_per_epoch,
                    run_id="sequential"
                )
                # You can update the progress bar's description with the lates cost
                best_costs = solver.best_costs_overall_dict.get(OBJ_KEY_REPORT, float('inf'))
                tqdm.write(f"Epoch {epoch+1}/{args.epochs} complete. Current Best Obj.: {best_costs:.2f}")

            run_results = [(0, solver.best_chromosome_overall, solver.best_costs_overall_dict, solver.best_details_overall)]
        else:
            print(f"\nRunning in parallel mode with {args.number_of_workers} workers...")
            run_results = run_parallel_with_migration( problem=hen_problem, model_name=args.model, population_size=args.population_size, epochs=args.epochs, generations_per_epoch=args.generations_per_epoch, common_params=common_opt_params, model_specific_params=model_opt_specific_params, num_workers=args.number_of_workers, initial_penalty=args.initial_penalty, final_penalty=args.final_penalty, generations_total=total_gens)

        processed_results = []
        if run_results:
            for res_item in run_results:
                if isinstance(res_item, Exception): print(f"Error from worker: {res_item}"); continue
                if res_item and len(res_item) == 4:
                    worker_id, best_chromo, best_costs, best_details = res_item
                    if best_costs: processed_results.append({'seed': f"worker_{worker_id}", 'costs': best_costs, 'chromosome': best_chromo, 'details': best_details})
                    else: print(f"Worker {worker_id} returned invalid costs.")
                else: print(f"Received malformed result from a worker: {res_item}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nOptimization completed in {elapsed_time:.2f} seconds.")
    display_optimization_results(processed_results, hen_problem, args.model, args.output_dir)


def cli():
    """Command-line interface function."""
    if '--help' in sys.argv or '-h' in sys.argv:
        display_help()
        exit(0)

    parser = argparse.ArgumentParser(
        description="Run Heat Exchanger Network (HEN) Synthesis Optimization using EPTHenOpt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    
    parser.add_argument('--config_file', type=str, default='config.json', help="Path to the JSON configuration file.")
    temp_args, _ = parser.parse_known_args()
    
    config_defaults = {}
    config_path = Path(temp_args.config_file)
    if config_path.is_file():
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            # Flatten the nested JSON structure for argparse
            for section, params in config_data.items():
                if section == 'problem_cost_parameters':
                    for key, value in params.items():
                        if isinstance(value, dict): # For exchanger, heater, cooler
                            for sub_key, sub_value in value.items():
                                config_defaults[f"{key}_{sub_key}"] = sub_value
                        else:
                            config_defaults[key] = value
                else:
                    config_defaults.update(params)

    # --- Add all other arguments ---
    file_group = parser.add_argument_group('File Path Arguments')
    file_group.add_argument('--streams_file', type=str)
    file_group.add_argument('--utilities_file', type=str)
    file_group.add_argument('--matches_U_file', type=str)
    file_group.add_argument('--forbidden_matches_file', type=str)
    file_group.add_argument('--required_matches_file', type=str)
    
    core_group = parser.add_argument_group('Core Optimization Arguments')
    core_group.add_argument('--model', type=str, choices=['GA', 'TLBO', 'PSO', 'SA', 'ACO'])
    core_group.add_argument('--population_size', type=int)
    core_group.add_argument('--epochs', type=int)
    core_group.add_argument('--generations_per_epoch', type=int)
    core_group.add_argument('--number_of_workers', type=int)
    core_group.add_argument('--num_stages', type=int)
    core_group.add_argument('--no_split', action='store_true', default=False, help="Enforce no stream splitting in the network design.")
    core_group.add_argument('--verbose', action='store_true')
    core_group.add_argument('--output_dir', type=str, default=None, help="Directory to save structured output files (e.g., CSV, JSON).")

    problem_group = parser.add_argument_group('Problem & Cost Parameters')
    problem_group.add_argument('--EMAT_setting', type=float)
    problem_group.add_argument('--default_U_overall', type=float)
    problem_group.add_argument('--mininum_Q_limit', type=float)
    # Separated cost parameters
    problem_group.add_argument('--exchanger_fixed_cost', type=float)
    problem_group.add_argument('--exchanger_area_coeff', type=float)
    problem_group.add_argument('--exchanger_area_exp', type=float)
    problem_group.add_argument('--heater_fixed_cost', type=float)
    problem_group.add_argument('--heater_area_coeff', type=float)
    problem_group.add_argument('--heater_area_exp', type=float)
    problem_group.add_argument('--cooler_fixed_cost', type=float)
    problem_group.add_argument('--cooler_area_coeff', type=float)
    problem_group.add_argument('--cooler_area_exp', type=float)

    ga_group = parser.add_argument_group('GA Specific Parameters')
    ga_group.add_argument('--ga_crossover_prob', type=float)
    ga_group.add_argument('--ga_mutation_prob_Z_setting', type=float)
    ga_group.add_argument('--ga_mutation_prob_R_setting', type=float)
    ga_group.add_argument('--ga_r_mutation_std_dev_factor_setting', type=float)
    ga_group.add_argument('--ga_elitism_frac', type=float)
    ga_group.add_argument('--ga_tournament_size', type=int)

    tlbo_group = parser.add_argument_group('TLBO Specific Parameters')
    tlbo_group.add_argument('--tlbo_teaching_factor', type=int, choices=[0, 1, 2])
    
    aco_group = parser.add_argument_group('ACO Specific Parameters')
    aco_group.add_argument('--aco_evaporation_rate', type=float, help="Pheromone evaporation rate (rho).")
    aco_group.add_argument('--aco_pheromone_influence', type=float, help="Influence of pheromone (alpha).")
    aco_group.add_argument('--aco_pheromone_deposit_amount', type=float, help="Pheromone deposit scaling factor (Q).")
    
    penalty_group = parser.add_argument_group('Fitness & Penalty Parameters')
    penalty_group.add_argument('--utility_cost_factor', type=float)
    penalty_group.add_argument('--pinch_dev_penalty_factor', type=float)
    penalty_group.add_argument('--sws_max_iter', type=int)
    penalty_group.add_argument('--sws_conv_tol', type=float)
    penalty_group.add_argument('--initial_penalty', type=float)
    penalty_group.add_argument('--final_penalty', type=float)
    
    env_group = parser.add_argument_group('Environmental Parameters')
    env_group.add_argument('--objective', type=str, default='single', choices=['single', 'multi'], help="Set optimization objective: single (TAC only) or multi (TAC and CO2).")
    env_group.add_argument('--default_co2_hot_utility', type=float, help="Default CO2 factor for hot utilities (kg/kWh).")
    env_group.add_argument('--default_co2_cold_utility', type=float, help="Default CO2 factor for cold utilities (kg/kWh).")

    parser.set_defaults(**config_defaults)
    
    parsed_args = parser.parse_args()
    
    main(parsed_args)


if __name__ == "__main__":
    cli()
