# EPTHenOpt/sa_helpers.py
"""
Simulated Annealing (SA) helpers for the EPTHenOpt package.

This module provides the `SimulatedAnnealingHEN` class. Since SA is a
single-solution algorithm, it is adapted to the population-based framework
by treating each member of the "population" as an independent SA trajectory.
"""
import random
import copy
import numpy as np
import math

from .base_optimizer import BaseOptimizer
from .utils import OBJ_KEY_OPTIMIZING, OBJ_KEY_REPORT, OBJ_KEY_CO2

class SimulatedAnnealingHEN(BaseOptimizer):
    """
    Implements the Simulated Annealing algorithm for HEN synthesis.
    """
    def __init__(self, problem, population_size, generations,
                 initial_temp=10000.0, final_temp=0.1, cooling_rate=0.95,
                 **kwargs):
        # SA is not a population-based method, but we use the 'population'
        # to run multiple independent SA chains in parallel.
        super().__init__(problem=problem, population_size=population_size, generations=generations, **kwargs)
        
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.current_temp = initial_temp
        
        # Store current state for each independent run
        self.current_solutions = self.population
        self.current_fitnesses = []

        # Initial evaluation
        self._evaluate_initial_solutions()

    def _evaluate_initial_solutions(self):
        self.current_fitnesses = []
        for solution in self.current_solutions:
            costs, details = self._calculate_fitness(solution)
            self.current_fitnesses.append(costs.get(OBJ_KEY_OPTIMIZING, float('inf')))
            # Update overall best if this initial solution is good
            if costs.get(OBJ_KEY_OPTIMIZING, float('inf')) < self.best_costs_overall_dict[OBJ_KEY_OPTIMIZING]:
                self.best_costs_overall_dict = copy.deepcopy(costs)
                self.best_chromosome_overall = solution.copy()
                self.best_details_overall = details

    def evolve_one_generation(self, gen_num=0, run_id_for_print=""):
        self.current_generation = gen_num
        
        new_solutions = []
        new_fitnesses = []

        for i in range(self.population_size):
            current_sol = self.current_solutions[i]
            current_fit = self.current_fitnesses[i]
            
            # Generate a neighbor solution
            neighbor_sol = self._get_neighbor(current_sol)
            neighbor_costs, neighbor_details = self._calculate_fitness(neighbor_sol)
            neighbor_fit = neighbor_costs.get(OBJ_KEY_OPTIMIZING, float('inf'))
            
            # Metropolis acceptance criterion
            if neighbor_fit < current_fit:
                new_solutions.append(neighbor_sol)
                new_fitnesses.append(neighbor_fit)
            else:
                delta_e = neighbor_fit - current_fit
                if self.current_temp > 1e-6 and random.random() < math.exp(-delta_e / self.current_temp):
                    new_solutions.append(neighbor_sol)
                    new_fitnesses.append(neighbor_fit)
                else:
                    new_solutions.append(current_sol)
                    new_fitnesses.append(current_fit)

            # Update overall best if a new best is found
            if new_fitnesses[-1] < self.best_costs_overall_dict[OBJ_KEY_OPTIMIZING]:
                self.best_costs_overall_dict = copy.deepcopy(neighbor_costs)
                self.best_chromosome_overall = new_solutions[-1].copy()
                self.best_details_overall = neighbor_details

        self.current_solutions = new_solutions
        self.current_fitnesses = new_fitnesses
        
        # Update temperature (cooling schedule)
        self.current_temp *= self.cooling_rate
        if self.current_temp < self.final_temp:
            self.current_temp = self.final_temp

        if self.verbose:
            print_prefix = f"Run {run_id_for_print} - SA - " if run_id_for_print else "SA - "
            overall_best_true_str = f"{self.best_costs_overall_dict['TAC_true_report']:.2f}" if self.best_costs_overall_dict.get('TAC_true_report') != float('inf') else "Inf"
            print(f"{print_prefix}Gen {gen_num+1:03d} | Best True TAC (Overall): {overall_best_true_str} | Temp: {self.current_temp:.2f}")

    def _get_neighbor(self, chromosome):
        """Creates a neighbor by applying a small mutation."""
        neighbor = chromosome.copy()
        # Mutate a small number of genes
        num_mutations = max(1, int(len(chromosome) * 0.05)) # Mutate 5% of genes
        for _ in range(num_mutations):
            idx = random.randint(0, len(chromosome) - 1)
            if idx < self.len_Z: # Discrete part
                neighbor[idx] = 1 - neighbor[idx]
            else: # Continuous part
                noise = np.random.normal(0, 0.1)
                neighbor[idx] = max(1e-6, neighbor[idx] + noise)
        return neighbor

    def inject_chromosome(self, chromosome):
        """Replaces a random solution with the injected one."""
        if self.current_solutions:
            idx_to_replace = random.randint(0, self.population_size - 1)
            self.current_solutions[idx_to_replace] = chromosome.copy()
            costs, _ = self._calculate_fitness(chromosome)
            self.current_fitnesses[idx_to_replace] = costs.get(OBJ_KEY_OPTIMIZING, float('inf'))
