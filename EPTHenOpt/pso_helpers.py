# EPTHenOpt/pso_helpers.py
"""
Particle Swarm Optimization (PSO) helpers for the EPTHenOpt package.

This module provides the `ParticleSwarmOptimizationHEN` class, which
implements the PSO algorithm for HEN synthesis.
"""
import random
import copy
import numpy as np

from .base_optimizer import BaseOptimizer
from .utils import OBJ_KEY_OPTIMIZING, OBJ_KEY_REPORT, OBJ_KEY_CO2, TRUE_TAC_KEY

from typing import Any

class Particle:
    """Represents a single particle in the PSO swarm."""
    def __init__(self, chromosome, problem_bounds):
        self.position = chromosome
        self.velocity = np.random.uniform(-1, 1, len(chromosome))
        self.pbest_position = chromosome.copy()
        self.pbest_fitness = float('inf')
        self.pbest_details: Any = None
        self.bounds = problem_bounds

    def update_velocity(self, gbest_position, w, c1, c2):
        """Update particle velocity based on pbest and gbest."""
        r1 = random.random()
        r2 = random.random()
        cognitive_velocity = c1 * r1 * (self.pbest_position - self.position)
        social_velocity = c2 * r2 * (gbest_position - self.position)
        self.velocity = w * self.velocity + cognitive_velocity + social_velocity

    def update_position(self):
        """Update particle position and apply bounds."""
        self.position += self.velocity
        
        # Clip positions to be within bounds [0, 1] for Z part, [min, max] for R parts
        len_Z = self.bounds['len_Z']
        self.position[:len_Z] = np.clip(self.position[:len_Z], 0, 1)
        self.position[len_Z:] = np.clip(self.position[len_Z:], 1e-6, np.inf)

class ParticleSwarmOptimizationHEN(BaseOptimizer):
    """
    Implements the Particle Swarm Optimization algorithm for HEN synthesis.
    """
    def __init__(self, problem, population_size, generations,
                 inertia_weight=0.5, cognitive_coeff=1.5, social_coeff=1.5,
                 **kwargs):
        super().__init__(problem=problem, population_size=population_size, generations=generations, **kwargs)
        
        self.inertia_weight = inertia_weight
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        
        self.gbest_position = None
        self.gbest_fitness = float('inf')
        
        # Initialize particles instead of a simple population list
        problem_bounds = {'len_Z': self.len_Z}
        self.swarm = [Particle(self._create_random_full_chromosome(), problem_bounds) for _ in range(self.population_size)]
        
        # Initial evaluation
        self._evaluate_swarm()

    def _evaluate_swarm(self):
        for particle in self.swarm:
            # Convert continuous Z part to binary for fitness evaluation
            eval_chromosome = particle.position.copy()
            eval_chromosome[:self.len_Z] = (eval_chromosome[:self.len_Z] > 0.5).astype(int)
            
            costs, details = self._calculate_fitness(eval_chromosome)
            current_fitness = costs.get(OBJ_KEY_OPTIMIZING, float('inf'))
            
            if current_fitness < particle.pbest_fitness:
                particle.pbest_fitness = current_fitness
                particle.pbest_position = particle.position.copy()
                particle.pbest_details = details

            if current_fitness < self.gbest_fitness:
                self.gbest_fitness = current_fitness
                self.gbest_position = particle.position.copy()
                self.best_chromosome_overall = eval_chromosome.copy()
                self.best_costs_overall_dict = costs
                self.best_details_overall = details

    def evolve_one_generation(self, gen_num=0, run_id_for_print=""):
        self.current_generation = gen_num
        
        for particle in self.swarm:
            particle.update_velocity(self.gbest_position, self.inertia_weight, self.cognitive_coeff, self.social_coeff)
            particle.update_position()
        
        self._evaluate_swarm()

        if self.verbose:
            print_prefix = f"Run {run_id_for_print} - PSO - " if run_id_for_print else "PSO - "
            overall_best_true_str = f"{self.best_costs_overall_dict[TRUE_TAC_KEY]:.2f}" if self.best_costs_overall_dict.get(TRUE_TAC_KEY) != float('inf') else "Inf"
            print(f"{print_prefix}Gen {gen_num+1:03d} | Best True TAC (Overall): {overall_best_true_str} | PSO Obj: {self.gbest_fitness:.2f}")

    def inject_chromosome(self, chromosome):
        """Replaces the worst particle with the injected chromosome."""
        if not self.swarm: return
        
        # Find worst particle
        worst_particle_idx = -1
        max_pbest_fitness = -1
        for i, p in enumerate(self.swarm):
            if p.pbest_fitness > max_pbest_fitness:
                max_pbest_fitness = p.pbest_fitness
                worst_particle_idx = i

        if worst_particle_idx != -1:
            problem_bounds = {'len_Z': self.len_Z}
            self.swarm[worst_particle_idx] = Particle(chromosome, problem_bounds)
            # Re-evaluate the new particle immediately
            eval_chromosome = chromosome.copy()
            eval_chromosome[:self.len_Z] = (eval_chromosome[:self.len_Z] > 0.5).astype(int)
            costs, details = self._calculate_fitness(eval_chromosome)
            new_fitness = costs.get(OBJ_KEY_OPTIMIZING, float('inf'))
            self.swarm[worst_particle_idx].pbest_fitness = new_fitness
            self.swarm[worst_particle_idx].pbest_details = details

