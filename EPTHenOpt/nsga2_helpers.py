# EPTHenOpt/nsga2_helpers.py
"""
NSGA-II (Non-dominated Sorting Genetic Algorithm II) for multi-objective
HEN optimization in the EPTHenOpt package.
"""
import random
import copy
import numpy as np

from .base_optimizer import BaseOptimizer
from .utils import OBJ_KEY_OPTIMIZING, OBJ_KEY_REPORT, OBJ_KEY_CO2, TRUE_TAC_KEY

class NSGAIIHEN(BaseOptimizer):
    """
    Implements the NSGA-II algorithm for multi-objective HEN synthesis,
    optimizing for Total Annualized Cost (TAC) and CO2 Emissions.
    """
    def __init__(self, problem, population_size, generations, crossover_prob, mutation_prob_Z, mutation_prob_R, **kwargs):
        super().__init__(problem=problem, population_size=population_size, generations=generations, **kwargs)
        self.crossover_prob = crossover_prob
        self.mutation_prob_Z = mutation_prob_Z
        self.mutation_prob_R = mutation_prob_R

    def _fast_non_dominated_sort(self, population_with_fitness):
        fronts = [[]]
        for individual_p in population_with_fitness:
            individual_p['domination_count'] = 0
            individual_p['dominated_solutions'] = []
            for individual_q in population_with_fitness:
                if self._dominates(individual_p, individual_q):
                    individual_p['dominated_solutions'].append(individual_q)
                elif self._dominates(individual_q, individual_p):
                    individual_p['domination_count'] += 1
            if individual_p['domination_count'] == 0:
                individual_p['rank'] = 0
                fronts[0].append(individual_p)
        
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for individual_p in fronts[i]:
                for individual_q in individual_p['dominated_solutions']:
                    individual_q['domination_count'] -= 1
                    if individual_q['domination_count'] == 0:
                        individual_q['rank'] = i + 1
                        next_front.append(individual_q)
            i += 1
            if next_front:
                fronts.append(next_front)
        return fronts

    def _dominates(self, p, q):
        # Checks if solution p dominates solution q
        p_costs = p['costs']
        q_costs = q['costs']
        # Objectives to minimize
        obj_p = np.array([p_costs.get(TRUE_TAC_KEY, float('inf')), p_costs.get(OBJ_KEY_CO2, float('inf'))])
        obj_q = np.array([q_costs.get(TRUE_TAC_KEY, float('inf')), q_costs.get(OBJ_KEY_CO2, float('inf'))])
        
        return np.all(obj_p <= obj_q) and np.any(obj_p < obj_q)

    def _calculate_crowding_distance(self, front):
        if not front:
            return
        
        num_solutions = len(front)
        for p in front:
            p['crowding_distance'] = 0
            
        num_objectives = 2 # TAC and CO2
        obj_keys = [TRUE_TAC_KEY, OBJ_KEY_CO2]

        for m in range(num_objectives):
            key = obj_keys[m]
            # Sort by objective value
            front.sort(key=lambda p: p['costs'].get(key, float('inf')))
            
            # Assign infinite distance to boundary solutions
            front[0]['crowding_distance'] = float('inf')
            front[-1]['crowding_distance'] = float('inf')

            f_max = front[-1]['costs'].get(key, float('-inf'))
            f_min = front[0]['costs'].get(key, float('inf'))

            if f_max == f_min:
                continue

            for i in range(1, num_solutions - 1):
                front[i]['crowding_distance'] += (front[i+1]['costs'].get(key, 0) - front[i-1]['costs'].get(key, 0)) / (f_max - f_min)

    def _selection(self, population):
        # Binary tournament selection based on rank and crowding distance
        p1 = random.choice(population)
        p2 = random.choice(population)
        
        if p1['rank'] < p2['rank']:
            return p1
        elif p2['rank'] < p1['rank']:
            return p2
        else: # Same rank, use crowding distance
            if p1['crowding_distance'] > p2['crowding_distance']:
                return p1
            else:
                return p2

    def evolve_one_generation(self, gen_num=0, run_id_for_print=""):
        self.current_generation = gen_num

        # Create combined population
        combined_population = self.population + self.offspring_population if hasattr(self, 'offspring_population') else self.population
        
        # Evaluate all individuals in the combined population
        evaluated_population = []
        for chromo in combined_population:
            costs, details = self._calculate_fitness(chromo)
            evaluated_population.append({'chromosome': chromo, 'costs': costs, 'details': details})

        # Sort into fronts
        fronts = self._fast_non_dominated_sort(evaluated_population)
        
        # Build the next population
        new_population = []
        for front in fronts:
            self._calculate_crowding_distance(front)
            if len(new_population) + len(front) > self.population_size:
                # Sort the last front by crowding distance and add the best
                front.sort(key=lambda p: p['crowding_distance'], reverse=True)
                num_to_add = self.population_size - len(new_population)
                new_population.extend(front[:num_to_add])
                break
            new_population.extend(front)

        self.population = [p['chromosome'] for p in new_population]
        
        # Create offspring for the next generation
        self.offspring_population = []
        for _ in range(self.population_size // 2):
            parent1 = self._selection(new_population)
            parent2 = self._selection(new_population)
            # Simple crossover and mutation from GA
            offspring1_chromo, offspring2_chromo = self._crossover(parent1['chromosome'], parent2['chromosome'])
            self.offspring_population.append(self._mutation(offspring1_chromo))
            self.offspring_population.append(self._mutation(offspring2_chromo))
            
        # For reporting purposes, store the current best front
        self.best_front = fronts[0]

        if self.verbose:
            print(f"NSGA-II Gen {gen_num+1:03d} | Pareto Front Size: {len(self.best_front)}")

    # Re-using GA crossover and mutation
    def _crossover(self, parent1_chromo, parent2_chromo):
        offspring1 = parent1_chromo.copy()
        offspring2 = parent2_chromo.copy()
        if random.random() < self.crossover_prob:
            size = len(parent1_chromo)
            if size > 1:
                cx_pt = random.randint(1, size - 1)
                offspring1 = np.concatenate((parent1_chromo[:cx_pt], parent2_chromo[cx_pt:]))
                offspring2 = np.concatenate((parent2_chromo[:cx_pt], parent1_chromo[cx_pt:]))
        return offspring1, offspring2

    def _mutation(self, chromosome):
        mutated_chromosome = chromosome.copy()
        len_Z = self.len_Z
        for i in range(len_Z):
            if random.random() < self.mutation_prob_Z:
                mutated_chromosome[i] = 1 - mutated_chromosome[i]
        
        for i in range(len_Z, self.chromosome_length):
            if random.random() < self.mutation_prob_R:
                std_dev = max(1e-3, abs(chromosome[i] * 0.1))
                mutated_chromosome[i] += random.gauss(0, std_dev)
                mutated_chromosome[i] = max(1e-6, mutated_chromosome[i])
        return mutated_chromosome

