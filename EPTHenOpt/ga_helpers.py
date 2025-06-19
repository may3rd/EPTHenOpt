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
from .utils import OBJ_KEY_OPTIMIZING, OBJ_KEY_REPORT, TRUE_TAC_KEY # Import the new base class

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
    
    def _repair_chromosome(self, chromosome):
        """
        Ensures chromosome validity.
        1. If no_split is True, it enforces that each stream has at most one
        match per stage by randomly selecting one and disabling others.
        2. It forces R (split ratio) values to zero for any non-existent
        match (where Z=0).
        """
        repaired_chromosome = chromosome.copy()
        Z_ijk, R_hot, R_cold = self._decode_chromosome(repaired_chromosome)

        # --- Part 1: Repair Z matrix for no-split constraint (if active) ---
        if self.problem.no_split:
            # ### Nothing to be done since no-split caes chromosome hoas on R
            return repaired_chromosome
        
            # Check and repair hot streams
            for i in range(self.problem.NH):
                for k in range(self.problem.num_stages):
                    active_matches_indices = np.where(Z_ijk[i, :, k] == 1)[0]
                    if len(active_matches_indices) > 1:
                        # An illegal split exists for hot stream i in stage k.
                        # Randomly choose one match to keep.
                        chosen_index_to_keep = random.choice(active_matches_indices)
                        
                        # Turn off all other matches for this stream in this stage.
                        for j_idx in active_matches_indices:
                            if j_idx != chosen_index_to_keep:
                                Z_ijk[i, j_idx, k] = 0

            # Check and repair cold streams
            for j in range(self.problem.NC):
                for k in range(self.problem.num_stages):
                    active_matches_indices = np.where(Z_ijk[:, j, k] == 1)[0]
                    if len(active_matches_indices) > 1:
                        # An illegal split exists for cold stream j in stage k.
                        chosen_index_to_keep = random.choice(active_matches_indices)
                        
                        # Turn off all other matches for this stream in this stage.
                        for i_idx in active_matches_indices:
                            if i_idx != chosen_index_to_keep:
                                Z_ijk[i_idx, j, k] = 0

        # --- Part 2: Repair R matrices based on the (potentially modified) Z matrix ---
        # This part runs regardless of the no_split setting.
        
        # Create a boolean mask of where Z is zero
        z_is_zero_mask = (Z_ijk == 0)

        # Use this mask to find the corresponding locations in R and set them to zero.
        # We need to reshape/broadcast the mask to fit the R matrices' shapes.
        
        # For R_hot (NH, ST, NC), we need to permute the Z mask from (NH, NC, ST)
        z_mask_for_R_hot = np.transpose(z_is_zero_mask, (0, 2, 1))
        R_hot[z_mask_for_R_hot] = 0.0

        # For R_cold (NC, ST, NH), we also need to permute the Z mask
        z_mask_for_R_cold = np.transpose(z_is_zero_mask, (1, 2, 0))
        R_cold[z_mask_for_R_cold] = 0.0

        # --- Re-assemble the repaired chromosome ---
        # Flatten all parts and concatenate them back together.
        repaired_z_flat = Z_ijk.flatten()
        repaired_r_hot_flat = R_hot.flatten() # type: ignore
        repaired_r_cold_flat = R_cold.flatten() # type: ignore
        
        final_repaired_chromosome = np.concatenate((repaired_z_flat, repaired_r_hot_flat, repaired_r_cold_flat))
        
        return final_repaired_chromosome

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
        
        if self.problem.no_split:
            return mutated_chromosome
        
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
                
                # -- repair offspring
                offspring1 = self._repair_chromosome(offspring1)
                offspring2 = self._repair_chromosome(offspring2)
                
                if children_generated < num_offspring_to_generate:
                    new_population.append(self._repair_chromosome(self._mutation(offspring1)))
                    children_generated += 1
                if children_generated < num_offspring_to_generate: # Check again
                    new_population.append(self._repair_chromosome(self._mutation(offspring2)))
                    children_generated += 1
        
        self.population = new_population
        # Ensure population size is maintained, padding with random if necessary
        while len(self.population) < self.population_size:
            self.population.append(self._create_random_full_chromosome())
        if len(self.population) > self.population_size:
            self.population = self.population[:self.population_size]
        
        if self.verbose:
            print_prefix = f"Run {run_id_for_print} - GA - " if run_id_for_print else "GA - "
            overall_best_true_str = f"{self.best_costs_overall_dict[TRUE_TAC_KEY]:.2f}" if self.best_costs_overall_dict.get(TRUE_TAC_KEY) != float('inf') else "Inf"
            print(f"{print_prefix}Gen {gen_num+1:03d} | Best True TAC (Overall): {overall_best_true_str} | GA Obj: {best_ga_tac_this_gen:.2f}")



    def inject_chromosome(self, chromosome):
        """
        ### REFINED ###
        Injects an external chromosome from migration into the population
        by replacing the current worst member.
        """
        if not self.population or not self.fitnesses:
            # If population/fitness not ready, just replace a random member
            if self.population:
                self.population[random.randint(0, self.population_size - 1)] = chromosome.copy()
            return

        # Find the index of the worst individual based on the optimizing fitness key
        worst_fitness = -1
        worst_idx = -1
        for i, fitness_dict in enumerate(self.fitnesses):
            current_fitness = fitness_dict.get(OBJ_KEY_OPTIMIZING, float('inf'))
            if current_fitness > worst_fitness:
                worst_fitness = current_fitness
                worst_idx = i

        # Replace the worst individual if one was found
        if worst_idx != -1:
            self.population[worst_idx] = chromosome.copy()
            # Invalidate its old fitness so it gets re-evaluated in the next generation
            self.fitnesses[worst_idx] = {OBJ_KEY_OPTIMIZING: float('inf')} 
            print(f"  -> Injected chromosome replaced member {worst_idx} with fitness {worst_fitness:.2f}")
