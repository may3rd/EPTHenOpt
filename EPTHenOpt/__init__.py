# EPTHenOpt/__init__.py

"""
EPTHenOpt: A Python package for Heat Exchanger Network (HEN) Synthesis and Optimization.

This package provides tools for modeling HEN problems and solving them using
metaheuristic algorithms like Genetic Algorithm (GA) and Teaching-Learning-Based
Optimization (TLBO).
"""
import json
from types import SimpleNamespace
from pathlib import Path

# Import key classes from data model modules
from .hen_models import (
    Stream, Utility, CostParameters, HENProblem
)

# Import the main optimizer classes from their respective algorithm modules
from .ga_helpers import GeneticAlgorithmHEN
from .tlbo_helpers import TeachingLearningBasedOptimizationHEN
from .pso_helpers import ParticleSwarmOptimizationHEN
from .sa_helpers import SimulatedAnnealingHEN
from .aco_helpers import AntColonyOptimizationHEN
from .nsga2_helpers import NSGAIIHEN

# Import the base optimizer class for users who might want to extend the package
from .base_optimizer import BaseOptimizer

# Import the most useful utility functions
from .utils import (
    load_data_from_csv,
    calculate_lmtd,
    find_stream_index_by_id,
    display_optimization_results,
    display_problem_summary,
    display_help,
    OBJ_KEY_OPTIMIZING,
    OBJ_KEY_REPORT,
    OBJ_KEY_CO2
)

# Import the core parallel execution function
from .cores import run_parallel_with_migration

# Define the public API of the package using __all__.
__all__ = [
    # Models
    'Stream', 'Utility', 'CostParameters', 'HENProblem',

    # Optimizers
    'GeneticAlgorithmHEN', 'TeachingLearningBasedOptimizationHEN',
    'ParticleSwarmOptimizationHEN', 'SimulatedAnnealingHEN',
    'AntColonyOptimizationHEN','NSGAIIHEN',

    # Base Class
    'BaseOptimizer',

    # Utilities
    'load_data_from_csv', 'calculate_lmtd', 'find_stream_index_by_id',
    'display_optimization_results', 'display_problem_summary', 'display_help',
    'OBJ_KEY_OPTIMIZING', 'OBJ_KEY_REPORT', 'OBJ_KEY_CO2',

    # Core Execution
    'run_parallel_with_migration', 'run'
]

__version__ = "0.8.0" # Version bump for new feature
__author__ = "Maetee Lorprajuksiri (26008353@pttgcgroup.com) E-PT-PX Department, GC Maintenance and Engineering Co. Ltd."

def run(config_file='config.json', **kwargs):
    """
    High-level programmatic API to run a HEN optimization.

    This function provides a simple way to configure and run an optimization
    by passing parameters as keyword arguments or via a config file.

    Args:
        config_file (str, optional): Path to the JSON configuration file.
                                     Defaults to 'config.json'.
        **kwargs: Keyword arguments corresponding to the command-line options.
                  These will override any values from the config file.
    """
    # Import inside the function to avoid circular dependencies
    from .run_problem import main as run_problem_main

    # Load defaults from the config file if it exists
    defaults = {}
    config_path = Path(config_file)
    if config_path.is_file():
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            for section, params in config_data.items():
                defaults.update(params)
    else:
        print(f"Warning: Config file '{config_file}' not found. Using internal defaults.")

    # Update defaults with any user-provided keyword arguments
    defaults.update(kwargs)
    args = SimpleNamespace(**defaults)

    # Call the main logic from run_problem.py
    run_problem_main(args)
