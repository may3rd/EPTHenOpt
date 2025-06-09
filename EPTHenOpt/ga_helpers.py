# gth/ga_helpers.py
"""
Genetic Algorithm (GA) helpers module for the EPTHenOpt package.

This module provides the `GeneticAlgorithmHEN` class, which implements the
Genetic Algorithm for solving Heat Exchanger Network (HEN) synthesis problems.
It includes mechanisms for selection, crossover, and mutation tailored to the
HEN chromosome structure.
"""
import random
import copy
import numpy as np

from .base_optimizer import BaseOptimizer
from .utils import OBJ_KEY_OPTIMIZING, OBJ_KEY_REPORT # Import the new base class

class GAVariationMixin:
    """
    A Mixin class providing standard genetic variation operators (crossover and mutation)
    for HEN chromosome structures.
    """

class GeneticAlgorithmHEN(BaseOptimizer, GAVariationMixin):
    def __init__(self, 
                 problem,
                 population_size,
                 generations, # Total generations for a full run
                 crossover_prob,
                 mutation_prob_Z,
                 mutation_prob_R,
                 elitism_count=1, # Default changed as per prior discussions
                 tournament_size=3,
                 random_seed=None,
                 utility_cost_factor=1.0,
                 pinch_deviation_penalty_factor=0.0,
                 r_mutation_std_dev_factor=0.1,
                 sws_max_iter=50,
                 sws_conv_tol=0.001,
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

        # GA-specific attributes
        self.crossover_prob = crossover_prob
        self.mutation_prob_Z = mutation_prob_Z
        self.mutation_prob_R = mutation_prob_R
        self.elitism_count = elitism_count 
        self.r_mutation_std_dev_factor = r_mutation_std_dev_factor
        self.tournament_size = tournament_size

        # Note: _initialize_population() is called in BaseOptimizer's __init__
        # If GA needs specific initial fitness eval, it would go here or override _initialize_population

    # _calculate_fitness, _decode_chromosome, _create_random_full_chromosome, _initialize_population
    # are now inherited from BaseOptimizer.

    def _crossover(self, parent1_chromo, parent2_chromo):
        """Performs single-point crossover on two parent chromosomes."""
        offspring1 = parent1_chromo.copy()
        offspring2 = parent2_chromo.copy()
        if random.random() < self.crossover_prob:
            size = len(parent1_chromo)
            if size > 1:
                cx_pt = random.randint(1, size - 1)
                offspring1 = np.concatenate((parent1_chromo[:cx_pt], parent2_chromo[cx_pt:]))
                offspring2 = np.concatenate((parent2_chromo[:cx_pt], parent1_chromo[cx_pt:]))
        return offspring1, offspring2

    def _mutate_continuous_gene(self, value):
        """Helper function to apply Gaussian mutation to a single continuous gene."""
        # This method relies on r_mutation_std_dev_factor being present in the main class
        std_dev = max(1e-3, abs(value * self.r_mutation_std_dev_factor))
        noise = random.gauss(0, std_dev)
        return max(1e-6, value + noise)

    def _mutation(self, chromosome):
        """
        Applies bit-flip mutation to the discrete Z-part and Gaussian mutation
        to the continuous R-parts of the chromosome.
        """
        mutated_chromosome = chromosome.copy()
        # Mutate the discrete (Z) part
        for i in range(self.len_Z):
            if random.random() < self.mutation_prob_Z:
                mutated_chromosome[i] = 1 - mutated_chromosome[i]
        
        # Mutate the continuous (R) parts
        for i in range(self.len_Z, self.chromosome_length):
            if random.random() < self.mutation_prob_R:
                mutated_chromosome[i] = self._mutate_continuous_gene(mutated_chromosome[i])
                
        return mutated_chromosome
    
    def _selection(self, current_population_evaluations):
        """
        Performs parent selection using k-way tournament selection.
        This method is more robust against fitness scaling issues and non-finite costs.
        """
        selected_indices = []
        population_size = len(current_population_evaluations)

        # Robust check for an empty population to prevent errors.
        if population_size == 0:
            return []

        # We need to select a full mating pool to generate the next generation's offspring.
        for _ in range(population_size):
            # 1. Randomly select k individuals for the tournament without replacement.
            #    Ensures a fair competition among distinct contenders.
            try:
                contender_indices = random.sample(range(population_size), self.tournament_size)
            except ValueError:
                # Fallback if tournament_size > population_size
                contender_indices = random.sample(range(population_size), population_size)

            # 2. Determine the winner of the tournament.
            tournament_winner_idx = -1
            best_fitness_in_tournament = float('inf')

            for idx in contender_indices:
                # Safely get the fitness, defaulting to infinity if absent.
                fitness = current_population_evaluations[idx]['costs'].get(OBJ_KEY_OPTIMIZING, float('inf'))
                
                # The contender with the lowest (best) fitness wins.
                # This naturally handles non-finite values.
                if fitness < best_fitness_in_tournament:
                    best_fitness_in_tournament = fitness
                    tournament_winner_idx = idx
            
            # 3. Handle the edge case where all contenders have non-finite fitness.
            #    In this scenario, no winner would be chosen. We fall back to picking randomly.
            if tournament_winner_idx == -1:
                tournament_winner_idx = random.choice(contender_indices)
            
            selected_indices.append(tournament_winner_idx)

        return selected_indices

    def evolve_one_generation(self, gen_num=0, run_id_for_print=""):
        """Performs a single generation of the genetic algorithm."""
        self.current_generation = gen_num

        current_population_evaluations = []
        for chromo in self.population:
            try:
                costs_dict, details = self._calculate_fitness(chromo) # Inherited
                current_population_evaluations.append({'chromosome': chromo, 'costs': costs_dict, 'details': details})
            except Exception as e:
                # print(f"Error calculating fitness for a chromosome: {e}")
                error_costs = {OBJ_KEY_OPTIMIZING: float('inf'), OBJ_KEY_REPORT: float('inf')}
                current_population_evaluations.append({'chromosome': chromo, 'costs': error_costs, 'details': []})

        if not current_population_evaluations:
            # print("Warning: No evaluations in current population. Re-initializing.")
            self._initialize_population() # Inherited
            # Potentially evaluate fitnesses here again if needed immediately
            return # Skip the rest of generation evolution

        current_population_evaluations.sort(key=lambda x: x['costs'][OBJ_KEY_OPTIMIZING])
        
        best_ga_tac_this_gen = current_population_evaluations[0]['costs'][OBJ_KEY_OPTIMIZING]
        if best_ga_tac_this_gen < self.best_costs_overall_dict[OBJ_KEY_OPTIMIZING]:
            self.best_costs_overall_dict = copy.deepcopy(current_population_evaluations[0]['costs'])
            self.best_chromosome_overall = current_population_evaluations[0]['chromosome'].copy()
            self.best_details_overall = copy.deepcopy(current_population_evaluations[0]['details'])
        
        new_population = []
        for i in range(min(self.elitism_count, len(current_population_evaluations))):
            new_population.append(current_population_evaluations[i]['chromosome'].copy())
        
        selected_parent_indices = self._selection(current_population_evaluations)
        
        num_offspring_to_generate = self.population_size - len(new_population)
        children_generated = 0
        idx_for_selection = 0
        
        if not selected_parent_indices : # Fallback if selection fails
             while children_generated < num_offspring_to_generate:
                new_population.append(self._create_random_full_chromosome())
                children_generated +=1
        else:
            while children_generated < num_offspring_to_generate:
                parent1_idx = selected_parent_indices[idx_for_selection % len(selected_parent_indices)]
                idx_for_selection += 1
                parent2_idx = selected_parent_indices[idx_for_selection % len(selected_parent_indices)]
                idx_for_selection += 1
                
                parent1 = current_population_evaluations[parent1_idx]['chromosome']
                parent2 = current_population_evaluations[parent2_idx]['chromosome']

                offspring1, offspring2 = self._crossover(parent1, parent2)
                
                if children_generated < num_offspring_to_generate:
                    new_population.append(self._mutation(offspring1))
                    children_generated += 1
                if children_generated < num_offspring_to_generate: # Check again
                    new_population.append(self._mutation(offspring2))
                    children_generated += 1
        
        self.population = new_population
        # Ensure population size is maintained, padding with random if necessary
        while len(self.population) < self.population_size:
            self.population.append(self._create_random_full_chromosome())
        if len(self.population) > self.population_size:
            self.population = self.population[:self.population_size]
        
        if self.verbose:
            print_prefix = f"Run {run_id_for_print} - GA - " if run_id_for_print else "GA - "
            overall_best_true_str = f"{self.best_costs_overall_dict['TAC_true_report']:.2f}" if self.best_costs_overall_dict.get('TAC_true_report') != float('inf') else "Inf"
            print(f"{print_prefix}Gen {gen_num+1:03d} | Best True TAC (Overall): {overall_best_true_str} | GA Obj: {best_ga_tac_this_gen:.2f}")



    def inject_chromosome(self, chromosome):
        """Injects an external chromosome into the population, replacing the worst member if elitism is active."""
        if self.population:
            # To effectively replace the worst, we would typically evaluate all, sort, and replace.
            # For simplicity with elitism, if the new chromosome is better than the worst elite,
            # it might not get in. A common strategy is to just add it and let selection sort it out,
            # or replace a random non-elite, or the actual worst one.
            # Current GA sorts by TAC_GA_optimizing, so last one after sort is worst.
            # This requires re-evaluating and re-sorting if we want to be precise.
            # A simpler approach for injection: add and re-sort or replace the absolute worst.
            
            # For now, using the strategy: replace the last element (likely worst after sorting in evolve_one_generation)
            self.population[-1] = chromosome.copy()
            # Optionally, re-evaluate the new chromosome's fitness immediately if needed by GA's flow,
            # though evolve_one_generation will do it at the start of the next generation.
