# EPTHenOpt/cores.py
"""Core parallel processing module for the EPTHenOpt package.

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
from .pso_helpers import ParticleSwarmOptimizationHEN
from .sa_helpers import SimulatedAnnealingHEN
from .aco_helpers import AntColonyOptimizationHEN

from .base_optimizer import OBJ_KEY_OPTIMIZING, OBJ_KEY_REPORT

def optimization_worker(
    worker_id, model_name, problem, population_size, epochs, generations_per_epoch,
    common_params, model_specific_params, migration_queue, results_queue
):
    """A worker process that runs a metaheuristic optimization with migration.

    This function is intended to be the target of a ``multiprocessing.Process``.
    It initializes a specific optimizer (e.g., GA, TLBO), runs it for a
    series of epochs, and periodically attempts to exchange its best solution
    with other workers via a shared migration queue.

    Parameters
    ----------
    worker_id : int
        A unique identifier for the worker process.
    model_name : str
        The name of the optimization model to use (e.g., 'GA', 'TLBO').
    problem : HENProblem
        The heat exchanger network problem instance to be solved.
    population_size : int
        The size of the population for the optimizer.
    epochs : int
        The total number of epochs the worker will run.
    generations_per_epoch : int
        The number of generations to run within each epoch before migration.
    common_params : dict
        A dictionary of parameters common to all optimizers.
    model_specific_params : dict
        A dictionary of parameters specific to the chosen optimizer model.
    migration_queue : multiprocessing.Queue
        A shared queue used to send and receive chromosomes for migration
        between workers.
    results_queue : multiprocessing.Queue
        A shared queue where the final best result from this worker is placed.

    Notes
    -----
    The worker catches all exceptions to prevent a single worker failure from
    crashing the entire parallel run. Failed workers will place an error
    message in the results queue.

    Migration is attempted at the end of each epoch. If the queue is full (cannot
    put) or empty (cannot get), the migration step is skipped for that epoch to
    avoid deadlocks.

    """
    print(f"Worker {worker_id}: Starting with model {model_name}.")
    solver_params = {**common_params, **model_specific_params}
    
    solver = None
    try:
        if model_name.upper() == 'GA':
            solver = GeneticAlgorithmHEN(problem=problem, population_size=population_size, random_seed=int(time.time()) + worker_id, **solver_params)
        elif model_name.upper() == 'TLBO':
            solver = TeachingLearningBasedOptimizationHEN(problem=problem, population_size=population_size, random_seed=int(time.time()) + worker_id, **solver_params)
        elif model_name.upper() == 'PSO':
            solver = ParticleSwarmOptimizationHEN(problem=problem, population_size=population_size, random_seed=int(time.time()) + worker_id, **solver_params)
        elif model_name.upper() == 'SA':
            solver = SimulatedAnnealingHEN(problem=problem, population_size=population_size, random_seed=int(time.time()) + worker_id, **solver_params)
        elif model_name.upper() == 'ACO': # Add the new choice
            solver = AntColonyOptimizationHEN(problem=problem, population_size=population_size, random_seed=int(time.time()) + worker_id, **solver_params)
        else:
            raise ValueError(f"Unknown model type: {model_name}")

        for epoch in range(epochs):
            solver.run_epoch(
                generations_in_epoch=generations_per_epoch,
                current_gen_offset=epoch * generations_per_epoch, 
                run_id=str(worker_id))
            
            best_costs = solver.best_costs_overall_dict
            if best_costs and best_costs.get(OBJ_KEY_OPTIMIZING) != float('inf'):
                current_best_obj = best_costs.get(OBJ_KEY_OPTIMIZING, float('inf'))
                current_true_best_obj = best_costs.get(OBJ_KEY_REPORT, float('inf'))
                print(f"Worker {worker_id}: Epoch {epoch+1}/{epochs} complete. Current Best Obj.: {current_best_obj:.2f}, TAC: {current_true_best_obj:.2f}")

            best_chromosome = solver.get_best_chromosome()
            
            if best_chromosome is not None:
                try:
                    # Try to send our best solution
                    # migration_queue.put(best_chromosome, block=False, timeout=0.1)
                    migration_queue.put_nowait(best_chromosome)
                    
                    # Try to receive a solution from another worker
                    # incoming_chromosome = migration_queue.get(block=True, timeout=0.5)
                    incoming_chromosome = migration_queue.get_nowait()
                    
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
        results_queue.put((worker_id, None, {OBJ_KEY_OPTIMIZING: float('inf'), OBJ_KEY_REPORT: float('inf'), "error": str(e_worker)}, None))
    finally:
        print(f"Worker {worker_id}: Finished.")

def run_parallel_with_migration(
    problem, model_name, population_size, epochs, generations_per_epoch,
    common_params, model_specific_params, num_workers, initial_penalty,
    final_penalty, generations_total
):
    """Manages the pool of optimizer workers and their communication.

    This function sets up the multiprocessing environment, creates the shared
    queues for migration and results, and starts the worker processes. After
    all workers have completed, it collects and returns the results.

    Parameters
    ----------
    problem : HENProblem
        The heat exchanger network problem instance.
    model_name : str
        The name of the optimization model to use.
    population_size : int
        The population size for each worker's optimizer.
    epochs : int
        The total number of epochs for each worker to run.
    generations_per_epoch : int
        The number of generations per epoch.
    common_params : dict
        A dictionary of parameters common to all optimizers.
    model_specific_params : dict
        A dictionary of parameters specific to the chosen optimizer.
    num_workers : int
        The number of parallel worker processes to spawn.
    initial_penalty : float
        The initial penalty factor for the adaptive penalty function.
    final_penalty : float
        The final penalty factor for the adaptive penalty function.
    generations_total : int
        The total number of generations, used for scaling the adaptive penalty.

    Returns
    -------
    list
        A list of results from all workers. Each item is a tuple:
        (worker_id, best_chromosome, best_costs_dict, best_details_list).

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
