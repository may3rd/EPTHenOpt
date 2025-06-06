import argparse
import sys # ADDED: To check for command line arguments

# Updated import to use the new package name EPTHenOpt
from EPTHenOpt import (
    Stream, Utility, CostParameters, HENProblem,
    GeneticAlgorithmHEN, TeachingLearningBasedOptimizationHEN,
    load_data_from_csv, display_optimization_results,
    run_parallel_with_migration, display_problem_summary, 
    display_help # Import the new function
)

# --- Constants for Default Values ---
# (Constants remain the same)
DEFAULT_STREAMS_FILE = "streams.csv"
DEFAULT_UTILITIES_FILE = "utilities.csv"
DEFAULT_MATCHES_U_FILE = None
DEFAULT_FORBIDDEN_MATCHES_FILE = None
DEFAULT_REQUIRED_MATCHES_FILE = None
DEFAULT_EMAT = 3.0
DEFAULT_MODEL = 'GA'
DEFAULT_POP_SIZE = 200
DEFAULT_EPOCHS = 10
DEFAULT_GEN_PER_EPOCH = 20
DEFAULT_WORKERS = 1
DEFAULT_GA_CROSSOVER_PROB = 0.85
DEFAULT_GA_MUT_Z_PROB = 0.1
DEFAULT_GA_MUT_R_PROB = 0.1
DEFAULT_GA_R_MUT_STD_FACTOR = 0.1
DEFAULT_GA_ELITISM_FRAC = 0.1
DEFAULT_TLBO_TEACHING_FACTOR = 0
DEFAULT_UTILITY_COST_FACTOR = 1.0
DEFAULT_PINCH_DEV_PENALTY = 150.0
DEFAULT_SWS_MAX_ITER = 300
DEFAULT_SWS_CONV_TOL = 1e-5
DEFAULT_INITIAL_PENALTY = 1e3
DEFAULT_FINAL_PENALTY = 1e7


# --- Main Execution Function ---
def main(args):
    """
    Main function to run HEN synthesis, configured by command-line arguments.
    """
    # (The body of main() remains unchanged)
    print(f"HEN Synthesis using {args.model} with EPTHenOpt")
    
    loaded_hs_data, loaded_cs_data, loaded_hu_data, loaded_cu_data, \
    loaded_matches_U, loaded_forbidden, loaded_required = load_data_from_csv(
        args.streams_file, args.utilities_file, args.matches_U_file,
        args.forbidden_matches_file, args.required_matches_file
    )
    
    if not (loaded_hs_data or loaded_cs_data):
        print("Error: No hot or cold stream data loaded. Exiting.")
        exit(1)

    hot_streams = [Stream(id_val=s['Name'], Tin=float(s['TIN_spec']), Tout_target=float(s['TOUT_spec']), CP=float(s['Fcp']), stream_type='hot', h_coeff=float(s.get('h_coeff',0))) for s in loaded_hs_data] if loaded_hs_data else []
    cold_streams = [Stream(id_val=s['Name'], Tin=float(s['TIN_spec']), Tout_target=float(s['TOUT_spec']), CP=float(s['Fcp']), stream_type='cold', h_coeff=float(s.get('h_coeff',0))) for s in loaded_cs_data] if loaded_cs_data else []
    
    hot_utilities = [Utility(id_val=u['Name'], Tin=float(u['TIN_utility']), Tout=float(u['TOUT_utility']), U=float(u['U_overall']), cost_per_energy_unit=float(u['Unit_Cost_Energy']), fix_cost=float(u['Fixed_Cost_Unit']), area_cost_coeff=float(u['Area_Cost_Coeff']), area_cost_exp=float(u['Area_Cost_Exp']), utility_type='hot_utility', h_coeff=float(u.get('h_coeff',0))) for u in loaded_hu_data] if loaded_hu_data else []
    if not hot_utilities and cold_streams:
        hot_utilities.append(Utility(id_val="DefaultHU", Tin=500, Tout=499, U=1.0, cost_per_energy_unit=0.02, fix_cost=1000, area_cost_coeff=80, area_cost_exp=0.6, utility_type='hot_utility'))
        print("Warning: No hot utilities loaded from file, using a default hot utility.")

    cold_utilities = [Utility(id_val=u['Name'], Tin=float(u['TIN_utility']), Tout=float(u['TOUT_utility']), U=float(u['U_overall']), cost_per_energy_unit=float(u['Unit_Cost_Energy']), fix_cost=float(u['Fixed_Cost_Unit']), area_cost_coeff=float(u['Area_Cost_Coeff']), area_cost_exp=float(u['Area_Cost_Exp']), utility_type='cold_utility', h_coeff=float(u.get('h_coeff',0))) for u in loaded_cu_data] if loaded_cu_data else []
    if not cold_utilities and hot_streams:
        cold_utilities.append(Utility(id_val="DefaultCU", Tin=20, Tout=30, U=1.0, cost_per_energy_unit=0.005, fix_cost=500, area_cost_coeff=70, area_cost_exp=0.65, utility_type='cold_utility'))
        print("Warning: No cold utilities loaded from file, using a default cold utility.")

    cost_params = CostParameters(EMAT=args.EMAT_setting, U_overall=args.default_U_overall, exch_fixed=args.default_exch_fixed_cost, exch_area_coeff=args.default_exch_area_coeff, exch_area_exp=args.default_exch_area_exp, heater_fixed=args.default_exch_fixed_cost, heater_area_coeff=args.default_exch_area_coeff, heater_area_exp=args.default_exch_area_exp, cooler_fixed=args.default_exch_fixed_cost, cooler_area_coeff=args.default_exch_area_coeff, cooler_area_exp=args.default_exch_area_exp)
    num_stages = args.num_stages if args.num_stages > 0 else max(1, len(hot_streams), len(cold_streams))
    
    hen_problem = HENProblem(hot_streams=hot_streams, cold_streams=cold_streams, hot_utility=hot_utilities, cold_utility=cold_utilities, cost_params=cost_params, num_stages=num_stages, matches_U_cost=loaded_matches_U, forbidden_matches=loaded_forbidden, required_matches=loaded_required)

    # --- ADDED: Display Run Configuration ---
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
    model_opt_specific_params = {}
    if args.model.upper() == 'GA': model_opt_specific_params = { "crossover_prob": args.ga_crossover_prob, "mutation_prob_Z": args.ga_mutation_prob_Z_setting, "mutation_prob_R": args.ga_mutation_prob_R_setting, "r_mutation_std_dev_factor": args.ga_r_mutation_std_dev_factor_setting, "elitism_count": int(args.ga_elitism_frac * args.population_size), "tournament_size": args.ga_tournament_size }
    elif args.model.upper() == 'TLBO': model_opt_specific_params = { "tlbo_teaching_factor": args.tlbo_teaching_factor }
    
    total_gens = args.epochs * args.generations_per_epoch
    if args.number_of_workers <= 1:
        print("Running in sequential mode (1 worker)...")
        solver_params = { **common_opt_params, **model_opt_specific_params, "initial_penalty": args.initial_penalty, "final_penalty": args.final_penalty, "generations": total_gens }
        solver = GeneticAlgorithmHEN(problem=hen_problem, population_size=args.population_size, **solver_params) if args.model.upper() == 'GA' else TeachingLearningBasedOptimizationHEN(problem=hen_problem, population_size=args.population_size, **solver_params)
        solver.run()
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
    display_optimization_results(processed_results, hen_problem, args.model)

if __name__ == "__main__":
    # --- MODIFIED: Check for the custom help flag before setting up the full parser ---
    # This is the correct way to handle a custom help message. It bypasses argparse entirely.
    if '--help' in sys.argv or '-h' in sys.argv:
        display_help()
        exit(0)

    # --- Argument Parser Setup ---
    # Set `add_help=False` to prevent the automatic creation of the -h/--help argument,
    # which avoids the conflict.
    parser = argparse.ArgumentParser(
        description="Run Heat Exchanger Network (HEN) Synthesis Optimization using EPTHenOpt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False # KEY CHANGE: Prevents the conflict
    )

    # --- REMOVED: The conflicting argument definition for --help/-h has been removed. ---
    
    # (The rest of the argparse setup remains the same)
    file_group = parser.add_argument_group('File Path Arguments')
    file_group.add_argument('--streams_file', type=str, default=DEFAULT_STREAMS_FILE, help="Path to the streams CSV file.")
    file_group.add_argument('--utilities_file', type=str, default=DEFAULT_UTILITIES_FILE, help="Path to the utilities CSV file.")
    file_group.add_argument('--matches_U_file', type=str, default=DEFAULT_MATCHES_U_FILE, help="Optional: CSV for specific match costs/U-values.")
    file_group.add_argument('--forbidden_matches_file', type=str, default=DEFAULT_FORBIDDEN_MATCHES_FILE, help="Optional: CSV for forbidden matches.")
    file_group.add_argument('--required_matches_file', type=str, default=DEFAULT_REQUIRED_MATCHES_FILE, help="Optional: CSV for required matches.")
    
    core_group = parser.add_argument_group('Core Optimization Arguments')
    core_group.add_argument('--model', type=str, default=DEFAULT_MODEL, choices=['GA', 'TLBO'], help="Optimization model to use.")
    core_group.add_argument('--population_size', type=int, default=DEFAULT_POP_SIZE, help="Population size for the optimizer.")
    core_group.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help="Number of epochs for optimizer runs.")
    core_group.add_argument('--generations_per_epoch', type=int, default=DEFAULT_GEN_PER_EPOCH, help="Number of generations to run per epoch.")
    core_group.add_argument('--number_of_workers', type=int, default=DEFAULT_WORKERS, help="Number of parallel workers (<=1 for sequential).")
    core_group.add_argument('--num_stages', type=int, default=0, help="HEN superstructure stages (0 to auto-calculate).")

    problem_group = parser.add_argument_group('Problem & Cost Parameters')
    problem_group.add_argument('--EMAT_setting', type=float, default=DEFAULT_EMAT, help="Minimum Approach Temperature (EMAT) in K.")
    problem_group.add_argument('--default_U_overall', type=float, default=0.5, help="Default overall heat transfer coefficient (U).")
    problem_group.add_argument('--default_exch_fixed_cost', type=float, default=0.0, help="Default fixed cost for exchangers.")
    problem_group.add_argument('--default_exch_area_coeff', type=float, default=1000.0, help="Default area coefficient for exchangers.")
    problem_group.add_argument('--default_exch_area_exp', type=float, default=0.6, help="Default area exponent for exchangers.")

    ga_group = parser.add_argument_group('GA Specific Parameters')
    ga_group.add_argument('--ga_crossover_prob', type=float, default=DEFAULT_GA_CROSSOVER_PROB, help="Crossover probability.")
    ga_group.add_argument('--ga_mutation_prob_Z_setting', type=float, default=DEFAULT_GA_MUT_Z_PROB, help="Mutation probability for Z matrix.")
    ga_group.add_argument('--ga_mutation_prob_R_setting', type=float, default=DEFAULT_GA_MUT_R_PROB, help="Mutation probability for R matrices.")
    ga_group.add_argument('--ga_r_mutation_std_dev_factor_setting', type=float, default=DEFAULT_GA_R_MUT_STD_FACTOR, help="Std dev factor for R mutation.")
    ga_group.add_argument('--ga_elitism_frac', type=float, default=DEFAULT_GA_ELITISM_FRAC, help="Fraction of population for elitism.")
    ga_group.add_argument('--ga_tournament_size', type=int, default=3, help="Tournament size for selection.")

    tlbo_group = parser.add_argument_group('TLBO Specific Parameters')
    tlbo_group.add_argument('--tlbo_teaching_factor', type=int, default=DEFAULT_TLBO_TEACHING_FACTOR, choices=[0, 1, 2], help="TLBO Teaching Factor (TF). 0 for random (1 or 2).")
    
    penalty_group = parser.add_argument_group('Fitness & Penalty Parameters')
    penalty_group.add_argument('--utility_cost_factor', type=float, default=DEFAULT_UTILITY_COST_FACTOR, help="Factor on utility op-ex in TAC.")
    penalty_group.add_argument('--pinch_dev_penalty_factor', type=float, default=DEFAULT_PINCH_DEV_PENALTY, help="Penalty for deviation from pinch targets.")
    penalty_group.add_argument('--sws_max_iter', type=int, default=DEFAULT_SWS_MAX_ITER, help="Max iterations for SWS calculation.")
    penalty_group.add_argument('--sws_conv_tol', type=float, default=DEFAULT_SWS_CONV_TOL, help="Convergence tolerance for SWS.")
    penalty_group.add_argument('--initial_penalty', type=float, default=DEFAULT_INITIAL_PENALTY, help="Initial adaptive penalty factor.")
    penalty_group.add_argument('--final_penalty', type=float, default=DEFAULT_FINAL_PENALTY, help="Final adaptive penalty factor.")

    parsed_args = parser.parse_args()
    
    # --- REMOVED: The redundant check at the end is no longer needed. ---
    
    main(parsed_args)