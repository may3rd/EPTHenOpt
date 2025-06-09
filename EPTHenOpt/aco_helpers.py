# EPTHenOpt/aco_helpers.py
"""
Ant Colony Optimization (ACO) helpers for the EPTHenOpt package.

This module provides the `AntColonyOptimizationHEN` class, which
implements the ACO algorithm. It uses a constructive approach where ants
build network structures based on probabilistic decisions guided by
pheromone trails.
"""
import random
import copy
import numpy as np

from .base_optimizer import BaseOptimizer
from .utils import OBJ_KEY_OPTIMIZING, OBJ_KEY_REPORT, OBJ_KEY_CO2

class AntColonyOptimizationHEN(BaseOptimizer):
    """
    Implements the Ant Colony Optimization algorithm for HEN synthesis.
    """
    def __init__(self, problem, population_size, generations,
                 evaporation_rate=0.1, pheromone_influence=1.0,
                 pheromone_deposit_amount=100.0, **kwargs):
        super().__init__(problem=problem, population_size=population_size, generations=generations, **kwargs)
        
        self.rho = evaporation_rate  # Pheromone evaporation rate
        self.alpha = pheromone_influence  # Influence of pheromone
        self.Q = pheromone_deposit_amount # Pheromone deposit scaling factor

        # Initialize pheromone matrix. One value for each possible match (Z_ijk).
        pheromone_shape = (self.problem.NH, self.problem.NC, self.problem.num_stages)
        self.pheromones = np.ones(pheromone_shape)

        # To store the solutions built by ants in the current generation
        self.ant_solutions = []

    def _construct_solution_for_ant(self):
        """Constructs a full chromosome (Z and R parts) for one ant."""
        z_part_flat = np.zeros(self.len_Z)
        pheromone_flat = self.pheromones.flatten()

        # Probabilistically build the Z matrix based on pheromones
        # Normalize pheromones to create probabilities
        pheromone_probs = (pheromone_flat ** self.alpha) / np.sum(pheromone_flat ** self.alpha)
        
        # Decide which matches to include. We can set a budget or use the probability directly.
        # A simple approach is to use the probabilities to turn matches on.
        for i in range(self.len_Z):
            if random.random() < pheromone_probs[i] * self.len_Z * 0.1: # Heuristic scaling
                 z_part_flat[i] = 1

        # Generate the continuous R parts randomly
        r_hot_part = np.random.uniform(0.01, 1.0, size=self.len_R_hot_splits)
        r_cold_part = np.random.uniform(0.01, 1.0, size=self.len_R_cold_splits)
        
        return np.concatenate((z_part_flat, r_hot_part, r_cold_part))

    def _update_pheromones(self):
        """Evaporate and deposit new pheromones."""
        # Evaporation
        self.pheromones *= (1 - self.rho)

        if not self.ant_solutions:
            return

        # Sort ants by fitness (lower is better)
        self.ant_solutions.sort(key=lambda x: x['fitness'])
        
        # Deposit pheromones from the best ants
        num_best_ants = max(1, int(self.population_size * 0.1)) # Top 10% of ants
        best_ants = self.ant_solutions[:num_best_ants]

        for ant in best_ants:
            z_matrix = ant['chromosome'][:self.len_Z].reshape(self.pheromones.shape)
            # The better the solution, the more pheromone it deposits
            deposit_amount = self.Q / ant['fitness']
            self.pheromones += z_matrix * deposit_amount

        # Also deposit pheromone from the best-ever solution
        if self.best_chromosome_overall is not None:
             z_best_overall = self.best_chromosome_overall[:self.len_Z].reshape(self.pheromones.shape)
             best_fitness = self.best_costs_overall_dict.get(OBJ_KEY_OPTIMIZING, float('inf'))
             if best_fitness != float('inf'):
                 deposit_amount = (self.Q * 2) / best_fitness # Elite ants deposit more
                 self.pheromones += z_best_overall * deposit_amount


    def evolve_one_generation(self, gen_num=0, run_id_for_print=""):
        self.current_generation = gen_num
        self.ant_solutions = []

        # Each ant constructs a solution
        for _ in range(self.population_size):
            ant_chromosome = self._construct_solution_for_ant()
            costs, details = self._calculate_fitness(ant_chromosome)
            fitness = costs.get(OBJ_KEY_OPTIMIZING, float('inf'))
            
            self.ant_solutions.append({'chromosome': ant_chromosome, 'fitness': fitness})

            # Check if this is the new best-so-far solution
            if fitness < self.best_costs_overall_dict[OBJ_KEY_OPTIMIZING]:
                self.best_costs_overall_dict = copy.deepcopy(costs)
                self.best_chromosome_overall = ant_chromosome.copy()
                self.best_details_overall = details
        
        # Update pheromone trails based on the quality of solutions found
        self._update_pheromones()

        if self.verbose:
            print_prefix = f"Run {run_id_for_print} - ACO - " if run_id_for_print else "ACO - "
            overall_best_true_str = f"{self.best_costs_overall_dict['TAC_true_report']:.2f}" if self.best_costs_overall_dict.get('TAC_true_report') != float('inf') else "Inf"
            print(f"{print_prefix}Gen {gen_num+1:03d} | Best True TAC (Overall): {overall_best_true_str} | ACO Obj: {self.best_costs_overall_dict[OBJ_KEY_OPTIMIZING]:.2f}")

    def inject_chromosome(self, chromosome):
        """ACO does not have a persistent population, so injection can be handled
           by strongly boosting the pheromone trail for the injected solution."""
        if self.pheromones is not None and chromosome is not None:
            z_injected = chromosome[:self.len_Z].reshape(self.pheromones.shape)
            # Use a high deposit amount to make this path very attractive
            self.pheromones += z_injected * (self.Q * 5)
