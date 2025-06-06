# EPTHenOpt/cores.py
"""
Core parallel processing module for the EPTHenOpt package.

This module contains the functions responsible for setting up and managing
the multiprocessing environment, including the optimization worker function
and the logic for inter-process communication (migration).
"""
import time
import multiprocessing
import queue

# Import the specific optimizer classes that the worker will need to instantiate.
from .ga_helpers import GeneticAlgorithmHEN
from .tlbo_helpers import TeachingLearningBasedOptimizationHEN

def optimization_worker(
    worker_id, model_name, problem, population_size, epochs, generations_per_epoch,
    common_params, model_specific_params, migration_queue, results_queue
):
    """
    A worker process that runs GA or TLBO with migration.
    This function is intended to be the target of a multiprocessing.Process.
    """
    print(f"Worker {worker_id}: Starting with model {model_name}.")
    solver_params = {**common_params, **model_specific_params}
    
    solver = None
    try:
        if model_name.upper() == 'GA':
            solver = GeneticAlgorithmHEN(
                problem=problem, population_size=population_size,
                random_seed=int(time.time()) + worker_id, **solver_params)
        elif model_name.upper() == 'TLBO':
            solver = TeachingLearningBasedOptimizationHEN(
                problem=problem, population_size=population_size,
                random_seed=int(time.time()) + worker_id, **solver_params)
        else:
            raise ValueError(f"Unknown model type: {model_name}")

        for epoch in range(epochs):
            solver.run_epoch(
                generations_in_epoch=generations_per_epoch,
                current_gen_offset=epoch * generations_per_epoch, 
                run_id=str(worker_id))
            
            best_costs = solver.best_costs_overall_dict
            if best_costs and best_costs.get('TAC_GA_optimizing') != float('inf'):
                current_best_obj = best_costs.get('TAC_GA_optimizing', float('inf'))
                print(f"Worker {worker_id}: Epoch {epoch+1}/{epochs} complete. Current Best Obj.: {current_best_obj:.2f}")

            best_chromosome = solver.get_best_chromosome()
            if best_chromosome is not None:
                try:
                    migration_queue.put(best_chromosome, block=False, timeout=0.1)
                    incoming_chromosome = migration_queue.get(block=True, timeout=0.5)
                    solver.inject_chromosome(incoming_chromosome)
                    print(f"Worker {worker_id}: Migration successful at epoch {epoch+1}.")
                except (queue.Full, queue.Empty):
                    print(f"Worker {worker_id}: Migration skipped at epoch {epoch+1} (queue busy/empty).")
                    pass 
                except Exception as e_mig:
                    print(f"Worker {worker_id}: Minor error during migration: {e_mig}")

        results_queue.put((
            worker_id, solver.best_chromosome_overall,
            solver.best_costs_overall_dict, solver.best_details_overall))
    except Exception as e_worker:
        print(f"FATAL ERROR in worker {worker_id}: {e_worker}")
        results_queue.put((worker_id, None, {"TAC_GA_optimizing": float('inf'), "TAC_true_report": float('inf'), "error": str(e_worker)}, None))
    finally:
        print(f"Worker {worker_id}: Finished.")

def run_parallel_with_migration(
    problem, model_name, population_size, epochs, generations_per_epoch,
    common_params, model_specific_params, num_workers, initial_penalty,
    final_penalty, generations_total
):
    """
    Manages the pool of optimizer workers and their communication.
    """
    manager = multiprocessing.Manager()
    migration_queue = manager.Queue(maxsize=num_workers if num_workers > 0 else 1)
    results_queue = manager.Queue()
    processes = []
    for i in range(num_workers):
        worker_common_params = {
            **common_params,
            "initial_penalty": initial_penalty,
            "final_penalty": final_penalty,
            "generations": generations_total # Total generations for adaptive penalty scaling
        }
        p = multiprocessing.Process(
            target=optimization_worker,
            args=(
                i, model_name, problem, population_size, epochs,
                generations_per_epoch, worker_common_params, model_specific_params,
                migration_queue, results_queue
            )
        )
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    all_results = []
    while not results_queue.empty():
        try:
            all_results.append(results_queue.get_nowait())
        except queue.Empty:
            break
    return all_results
