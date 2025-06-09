# EPTHenOpt/tlbo_helpers.py
"""
Teaching-Learning-Based Optimization (TLBO) helpers for the EPTHenOpt package.

This module provides the `TeachingLearningBasedOptimizationHEN` class, which
implements the TLBO algorithm for HEN synthesis. It includes the logic for
the 'Teacher Phase' and 'Learner Phase' adapted to the HEN problem domain.
"""
import copy
import numpy as np
import random

from .base_optimizer import BaseOptimizer
from .utils import OBJ_KEY_OPTIMIZING, OBJ_KEY_REPORT, OBJ_KEY_CO2

class TeachingLearningBasedOptimizationHEN(BaseOptimizer):
    def __init__(self,
                 problem,
                 population_size,
                 generations, # Total generations for a full run
                 random_seed=None,
                 utility_cost_factor=1.0,
                 pinch_deviation_penalty_factor=0.0,
                 sws_max_iter=50,
                 sws_conv_tol=0.001,
                 tlbo_teaching_factor=0, # ADDED: New tuning parameter
                 **kwargs): # Catch-all for any other base params

        # Call the BaseOptimizer's __init__
        super().__init__(problem=problem,
                         population_size=population_size,
                         generations=generations,
                         random_seed=random_seed,
                         utility_cost_factor=utility_cost_factor,
                         pinch_deviation_penalty_factor=pinch_deviation_penalty_factor,
                         sws_max_iter=sws_max_iter,
                         sws_conv_tol=sws_conv_tol,
                         **kwargs) # Pass along any other kwargs

        # ADDED: Store the teaching factor
        self.tlbo_teaching_factor = tlbo_teaching_factor
        
        # TLBO specific: Ensure fitnesses are evaluated for the initial population
        if self.population and not self.fitnesses:
            self._evaluate_initial_population()

    def _evaluate_initial_population(self):
        """Helper to evaluate fitness for the initial population."""
        self.fitnesses = []
        self.details_list = []
        for chromosome in self.population:
            try:
                fitness, details = self._calculate_fitness(chromosome)
                self.fitnesses.append(fitness)
                self.details_list.append(details)
            except Exception as e:
                error_costs = {OBJ_KEY_OPTIMIZING: float('inf'), OBJ_KEY_REPORT: float('inf')}
                self.fitnesses.append(error_costs)
                self.details_list.append([])
        
        if self.fitnesses:
            best_idx_initial = np.argmin([f[OBJ_KEY_OPTIMIZING] for f in self.fitnesses])
            if self.fitnesses[best_idx_initial][OBJ_KEY_OPTIMIZING] < self.best_costs_overall_dict[OBJ_KEY_OPTIMIZING]:
                self.best_costs_overall_dict = copy.deepcopy(self.fitnesses[best_idx_initial])
                self.best_chromosome_overall = self.population[best_idx_initial].copy()
                self.best_details_overall = copy.deepcopy(self.details_list[best_idx_initial])


    def evolve_one_generation(self, gen_num=0, run_id_for_print=""):
        """Performs a single generation of the TLBO algorithm."""
        self.current_generation = gen_num
        
        if len(self.fitnesses) != len(self.population):
            self._evaluate_initial_population()
            if not self.fitnesses : return

        # --- Teacher Phase ---
        current_best_idx = np.argmin([f[OBJ_KEY_OPTIMIZING] for f in self.fitnesses])
        teacher_chromosome = self.population[current_best_idx].copy()
        
        mean_solution_vector = np.mean(np.array(self.population), axis=0)

        new_population_after_teacher_phase = []
        new_fitnesses_after_teacher_phase = []
        new_details_after_teacher_phase = []

        for i in range(self.population_size):
            learner_chromosome = self.population[i]
            
            # MODIFIED: Use the teaching factor parameter
            if self.tlbo_teaching_factor == 0:
                TF = random.randint(1, 2) # Original random behavior
            else:
                TF = self.tlbo_teaching_factor # Use fixed value (1 or 2)
                
            r = random.random()

            modified_learner = learner_chromosome + r * (teacher_chromosome - TF * mean_solution_vector)
            
            modified_learner[:self.len_Z] = np.round(np.clip(modified_learner[:self.len_Z], 0, 1))
            modified_learner[self.len_Z:] = np.clip(modified_learner[self.len_Z:], 1e-6, None)

            try:
                mod_fitness, mod_details = self._calculate_fitness(modified_learner)
            except Exception as e:
                mod_fitness = {OBJ_KEY_OPTIMIZING: float('inf'), OBJ_KEY_REPORT: float('inf')}
                mod_details = []

            if mod_fitness[OBJ_KEY_OPTIMIZING] < self.fitnesses[i][OBJ_KEY_OPTIMIZING]:
                new_population_after_teacher_phase.append(modified_learner)
                new_fitnesses_after_teacher_phase.append(mod_fitness)
                new_details_after_teacher_phase.append(mod_details)
            else:
                new_population_after_teacher_phase.append(learner_chromosome)
                new_fitnesses_after_teacher_phase.append(self.fitnesses[i])
                new_details_after_teacher_phase.append(self.details_list[i])
        
        self.population = new_population_after_teacher_phase
        self.fitnesses = new_fitnesses_after_teacher_phase
        self.details_list = new_details_after_teacher_phase

        # --- Learner Phase ---
        # (Learner phase remains unchanged)
        new_population_after_learner_phase = []
        new_fitnesses_after_learner_phase = []
        new_details_after_learner_phase = []

        for i in range(self.population_size):
            current_learner = self.population[i]
            current_fitness = self.fitnesses[i]
            current_details = self.details_list[i]
            
            j = random.choice([k for k in range(self.population_size) if k != i])
            other_learner = self.population[j]
            other_fitness = self.fitnesses[j]
            
            r = random.random()
            if current_fitness[OBJ_KEY_OPTIMIZING] < other_fitness[OBJ_KEY_OPTIMIZING]:
                 modified_learner_lp = current_learner + r * (current_learner - other_learner)
            else:
                 modified_learner_lp = current_learner + r * (other_learner - current_learner)
            
            modified_learner_lp[:self.len_Z] = np.round(np.clip(modified_learner_lp[:self.len_Z], 0, 1))
            modified_learner_lp[self.len_Z:] = np.clip(modified_learner_lp[self.len_Z:], 1e-6, None)

            try:
                mod_lp_fitness, mod_lp_details = self._calculate_fitness(modified_learner_lp)
            except Exception as e:
                mod_lp_fitness = {OBJ_KEY_OPTIMIZING: float('inf'), OBJ_KEY_REPORT: float('inf')}
                mod_lp_details = []

            if mod_lp_fitness[OBJ_KEY_OPTIMIZING] < current_fitness[OBJ_KEY_OPTIMIZING]:
                new_population_after_learner_phase.append(modified_learner_lp)
                new_fitnesses_after_learner_phase.append(mod_lp_fitness)
                new_details_after_learner_phase.append(mod_lp_details)
            else:
                new_population_after_learner_phase.append(current_learner)
                new_fitnesses_after_learner_phase.append(current_fitness)
                new_details_after_learner_phase.append(current_details)

        self.population = new_population_after_learner_phase
        self.fitnesses = new_fitnesses_after_learner_phase
        self.details_list = new_details_after_learner_phase
        
        if self.fitnesses:
            best_idx_this_gen = np.argmin([f[OBJ_KEY_OPTIMIZING] for f in self.fitnesses])
            if self.fitnesses[best_idx_this_gen][OBJ_KEY_OPTIMIZING] < self.best_costs_overall_dict[OBJ_KEY_OPTIMIZING]:
                self.best_costs_overall_dict = copy.deepcopy(self.fitnesses[best_idx_this_gen])
                self.best_chromosome_overall = self.population[best_idx_this_gen].copy()
                self.best_details_overall = copy.deepcopy(self.details_list[best_idx_this_gen])
    
        if self.verbose:
            print_prefix = f"Run {run_id_for_print} - PSO - " if run_id_for_print else "PSO - "
            overall_best_true_str = f"{self.best_costs_overall_dict['TAC_true_report']:.2f}" if self.best_costs_overall_dict.get('TAC_true_report') != float('inf') else "Inf"
            best_obj_str = f"{self.best_costs_overall_dict[OBJ_KEY_OPTIMIZING]:.2f}" if self.best_costs_overall_dict.get(OBJ_KEY_OPTIMIZING) != float('inf') else "Inf"
            print(f"{print_prefix}Gen {gen_num+1:03d} | Best True TAC (Overall): {overall_best_true_str} | TLBO Obj: {best_obj_str}")

    def inject_chromosome(self, chromosome):
        """Injects an external chromosome into the population, replacing the worst member."""
        if self.population and self.fitnesses:
            try:
                new_fitness, new_details = self._calculate_fitness(chromosome)
            except Exception as e:
                new_fitness = {OBJ_KEY_OPTIMIZING: float('inf'), OBJ_KEY_REPORT: float('inf')}
                new_details = []

            worst_idx = np.argmax([f[OBJ_KEY_OPTIMIZING] for f in self.fitnesses])
            self.population[worst_idx] = chromosome.copy()
            self.fitnesses[worst_idx] = new_fitness
            self.details_list[worst_idx] = new_details

