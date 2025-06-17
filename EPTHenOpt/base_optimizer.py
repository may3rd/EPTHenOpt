# gth/base_optimizer.py
"""
Base optimizer module for the EPTHenOpt package.

This module defines the `BaseOptimizer` class, which serves as the foundation
for all optimization algorithms in the package. It encapsulates the shared
logic for problem handling, population management, and fitness calculation,
including the complex Stage-Wise Superstructure (SWS) and cost evaluation.
"""
import numpy as np
import random

from .hen_models import HENProblem
from .utils import calculate_lmtd, OBJ_KEY_OPTIMIZING, OBJ_KEY_REPORT, OBJ_KEY_CO2, TRUE_TAC_KEY

class BaseOptimizer:
    def __init__(self, 
                 problem: HENProblem,
                 population_size: int,
                 generations: int,
                 random_seed=None,
                 utility_cost_factor=1.0,
                 pinch_deviation_penalty_factor=0.0,
                 sws_max_iter=50,
                 sws_conv_tol=0.001,
                 initial_penalty=1e3,
                 final_penalty=1e7,
                 **kwargs):

        self.verbose = kwargs.get('verbose', False)
        
        self.problem: HENProblem = problem
        self.population_size = population_size
        self.generations = generations
        self.random_seed = random_seed
        
        # Common parameters for fitness calculation
        self.utility_cost_factor = utility_cost_factor
        self.pinch_deviation_penalty_factor = pinch_deviation_penalty_factor
        self.sws_max_iter = sws_max_iter
        self.sws_conv_tol = sws_conv_tol
        
        self.initial_penalty = initial_penalty
        self.final_penalty = final_penalty

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # Define chromosome segment lengths
        self.len_Z = self.problem.NH * self.problem.NC * self.problem.num_stages
        self.len_R_hot_splits = self.problem.NH * self.problem.num_stages * self.problem.NC
        self.len_R_cold_splits = self.problem.NC * self.problem.num_stages * self.problem.NH
        self.chromosome_length = self.len_Z + self.len_R_hot_splits + self.len_R_cold_splits
        
        # --- State variables ---
        self.population = []
        self.fitnesses = [] 
        self.details_list = []
        self.current_generation = 0

        # Best-so-far tracking
        self.best_chromosome_overall = None
        self.best_costs_overall_dict = {OBJ_KEY_OPTIMIZING: float('inf'), OBJ_KEY_REPORT: float('inf')}
        self.best_details_overall = None
        
        # --- Pre-compute constant NumPy arrays for performance ---
        self._hs_Tin = np.array([s.Tin for s in self.problem.hot_streams])
        self._hs_Tout_target = np.array([s.Tout_target for s in self.problem.hot_streams])
        self._hs_CP = np.array([s.CP for s in self.problem.hot_streams])

        self._cs_Tin = np.array([s.Tin for s in self.problem.cold_streams])
        self._cs_Tout_target = np.array([s.Tout_target for s in self.problem.cold_streams])
        self._cs_CP = np.array([s.CP for s in self.problem.cold_streams])
        
        # This initial array can also be pre-computed
        self._initial_T_mix_H = np.array([[s.Tin for _ in range(self.problem.num_stages)] for s in self.problem.hot_streams])
        self._initial_T_mix_C = np.array([[s.Tin for _ in range(self.problem.num_stages)] for s in self.problem.cold_streams])

        self._initialize_population()

    def _initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            self.population.append(self._create_random_full_chromosome())

    def _create_random_full_chromosome(self):
        # --- call the function defined in HEN_model class.
        return self.problem._create_random_full_chromosome()

    def _decode_chromosome(self, chromosome):
        # --- call the function defined in HEN_model class.
        return self.problem._decode_chromosome(chromosome)

    # --- REFACTORED FITNESS CALCULATION ---

    def _calculate_fitness(self, chromosome):
        """
        Orchestrates the multi-step process of evaluating a chromosome's fitness (cost).
        This method acts as a high-level story, delegating specific tasks to helper methods.
        """
        Z_ijk, R_hot, R_cold = self._decode_chromosome(chromosome)
        adaptive_penalty = self._get_adaptive_penalty()

        # 1. Check for hard constraint violations that warrant a "death penalty"
        if self.problem.no_split and self._check_no_split_violation(Z_ijk):
            return self._create_death_penalty_costs("no_split", adaptive_penalty), []

        # 2. Calculate heat recovery network performance using SWS
        FH_ijk, FC_ijk = self._calculate_split_fractions(Z_ijk, R_hot, R_cold)
        sws_results = self._perform_sws(Z_ijk, FH_ijk, FC_ijk)
        if not sws_results['converged']:
            return self._create_death_penalty_costs("SWS_non_convergence", adaptive_penalty), []

        # 3. Calculate costs and penalties for the heat recovery section
        proc_ex_costs, proc_ex_details = self._calculate_process_exchanger_costs(
            Z_ijk, sws_results, FH_ijk, FC_ijk, adaptive_penalty
        )
        
        # 4. Assign and cost utilities (heaters and coolers)
        util_costs, util_details, final_temps = self._assign_utilities_and_cost(
            sws_results['final_Th_after_recovery'], 
            sws_results['final_Tc_after_recovery'], 
            adaptive_penalty
        )
        all_exchanger_details = proc_ex_details + util_details

        # 5. Calculate all other penalties based on the final network configuration
        final_penalties = self._calculate_final_penalties(
            Z_ijk, sws_results['Q_ijk'], final_temps, 
            util_costs['Q_hot_consumed'], util_costs['Q_cold_consumed'], 
            adaptive_penalty
        )

        # 6. Aggregate all costs and penalties into a final dictionary
        all_costs = {**proc_ex_costs, **util_costs}
        all_penalties = {**final_penalties, "penalty_EMAT_etc": proc_ex_costs['penalty_EMAT_etc'], "penalty_unmet_targets": util_costs['penalty_unmet_targets']}

        final_cost_dict = self._aggregate_costs(all_costs, all_penalties, final_temps)
        
        return final_cost_dict, all_exchanger_details

    def _get_adaptive_penalty(self):
        """Calculates the penalty factor, which scales with the generation number."""
        gen_ratio = min(1.0, self.current_generation / self.generations if self.generations > 0 else 1.0)
        return self.initial_penalty + (self.final_penalty - self.initial_penalty) * gen_ratio

    def _create_death_penalty_costs(self, reason, adaptive_penalty):
        """Returns a cost dictionary with a massive penalty for fatal errors."""
        penalty_value = adaptive_penalty * 1e6
        return {
            OBJ_KEY_OPTIMIZING: penalty_value,
            OBJ_KEY_REPORT: float('inf'),
            f"penalty_{reason}": penalty_value,
        }

    def _check_no_split_violation(self, Z_ijk):
        """Checks if the no-split constraint is violated."""
        NH, NC, ST = self.problem.NH, self.problem.NC, self.problem.num_stages
        for i in range(NH):
            for k in range(ST):
                if np.sum(Z_ijk[i, :, k]) > 1:
                    return True
        for j in range(NC):
            for k in range(ST):
                if np.sum(Z_ijk[:, j, k]) > 1:
                    return True
        return False

    def _calculate_split_fractions(self, Z_ijk, R_hot_splits, R_cold_splits):
        """Calculates the actual split fractions FH_ijk and FC_ijk."""
        NH, NC, ST = self.problem.NH, self.problem.NC, self.problem.num_stages
        FH_ijk = np.zeros((NH, NC, ST))
        FC_ijk = np.zeros((NH, NC, ST))
        
        epsilon = 1e-9 # A small number to prevent division by zero

        for k in range(ST):
            # Hot stream splits
            for i in range(NH):
                active_indices = np.where(Z_ijk[i, :, k] == 1)[0]
                num_active = len(active_indices)
                if num_active == 1:
                    FH_ijk[i, active_indices[0], k] = 1.0
                elif num_active > 1:
                    raw_r = R_hot_splits[i, k, active_indices]
                    sum_r = np.sum(raw_r)
                    FH_ijk[i, active_indices, k] = raw_r / (sum_r + epsilon)
            
            # Cold stream splits
            for j in range(NC):
                active_indices = np.where(Z_ijk[:, j, k] == 1)[0]
                num_active = len(active_indices)
                if num_active == 1:
                    FC_ijk[active_indices[0], j, k] = 1.0
                elif num_active > 1:
                    raw_r = R_cold_splits[j, k, active_indices]
                    sum_r = np.sum(raw_r)
                    FC_ijk[active_indices, j, k] = raw_r / (sum_r + epsilon)
                    
        return FH_ijk, FC_ijk
    # In base_optimizer.py

    def _perform_sws(self, Z_ijk, FH_ijk, FC_ijk):
        """Performs the Stage Wise Superstructure (SWS) calculation.

        This is a highly optimized hybrid implementation. It pre-allocates
        working arrays to minimize memory overhead inside the loops, while
        vectorizing the core arithmetic for maximum performance.
        """
        # 1. Initialization and Pre-computation
        # ---------------------------------------
        NH, NC, ST = self.problem.NH, self.problem.NC, self.problem.num_stages
        EMAT = self.problem.cost_params.EMAT

        if NH == 0 or NC == 0:
            return {"converged": True, "Q_ijk": np.zeros((NH, NC, ST)), 
                    "T_mix_H": np.empty((NH, ST)), "T_mix_C": np.empty((NC, ST)),
                    "final_Th_after_recovery": self._hs_Tin,
                    "final_Tc_after_recovery": self._cs_Tin}

        T_mix_H = self._initial_T_mix_H.copy()
        T_mix_C = self._initial_T_mix_C.copy()
        Q_ijk = np.zeros((NH, NC, ST))
        
        # --- Performance Critical: Pre-allocate working matrices ONCE ---
        # These will be updated in-place inside the loop to avoid re-allocation.
        q_limits = np.empty((4, NH, NC))
        CPH_b_matrix = np.empty((NH, NC))
        CPC_b_matrix = np.empty((NH, NC))
        
        # --- NEW: Pre-calculate a static feasibility mask for all stages ---
        # This mask checks for fundamental thermodynamic impossibility.
        # It assumes the worst case: coldest possible Th_in (target) and
        # hottest possible Tc_in (target). A match is impossible if hs.Tout_target > cs.Tout_target
        # This is a heuristic but effective pre-filter. A more precise check is still done inside.
        static_feasibility_mask = self._hs_Tout_target[:, np.newaxis] <= self._cs_Tout_target[np.newaxis, :]

        # Combine with superstructure mask. A match can only happen if Z_ijk is ever 1.
        # This prevents calculations for pairs that never match in any stage.
        combined_static_mask = Z_ijk.any(axis=2) * static_feasibility_mask

        # 2. SWS Iteration Loop
        # -----------------------
        for sws_iter in range(self.sws_max_iter):
            T_mix_H_prev = T_mix_H.copy()
            T_mix_C_prev = T_mix_C.copy()

            # 2a. Hot Pass (calculates Q_ijk for all stages)
            # -----------------------------------------------
            for k in range(ST):
                # Skip stage if no matches are possible at all
                if not Z_ijk[:, :, k].any():
                    Q_ijk[:, :, k] = 0
                    T_mix_H[:, k] = T_mix_H_prev[:, k-1] if k > 0 else self._hs_Tin
                    continue
                
                TinH_stage_k = T_mix_H_prev[:, k - 1] if k > 0 else self._hs_Tin
                Tcin_stage_k = T_mix_C_prev[:, k + 1] if k < ST - 1 else self._cs_Tin
                
                # --- NEW: Dynamic Feasibility Check for the current temperatures ---
                # This is the precise check you requested.
                dynamic_feasibility_mask = (TinH_stage_k[:, np.newaxis] > Tcin_stage_k[np.newaxis, :] + EMAT)
                
                # Final mask for this stage's calculation
                active_mask = combined_static_mask * dynamic_feasibility_mask

                # Update working matrices in-place where the mask is True
                np.multiply(self._hs_CP[:, np.newaxis], FH_ijk[:, :, k], out=CPH_b_matrix, where=active_mask)
                np.multiply(self._cs_CP[np.newaxis, :], FC_ijk[:, :, k], out=CPC_b_matrix, where=active_mask)
                
                # Calculate Q limits, writing results directly into the pre-allocated array `q_limits`
                np.multiply(CPH_b_matrix, (TinH_stage_k[:, np.newaxis] - self._hs_Tout_target[:, np.newaxis]), out=q_limits[0], where=active_mask)
                np.multiply(CPH_b_matrix, (TinH_stage_k[:, np.newaxis] - (Tcin_stage_k[np.newaxis, :] + EMAT)), out=q_limits[1], where=active_mask)
                np.multiply(CPC_b_matrix, (self._cs_Tout_target[np.newaxis, :] - Tcin_stage_k[np.newaxis, :]), out=q_limits[2], where=active_mask)
                np.multiply(CPC_b_matrix, ((TinH_stage_k[:, np.newaxis] - EMAT) - Tcin_stage_k[np.newaxis, :]), out=q_limits[3], where=active_mask)

                # Find the minimum limit and apply masks
                Q_m = np.min(q_limits, axis=0)
                # Set Q to 0 if it's below the minimum limit, then apply other masks.
                Q_m_limited = np.where(Q_m >= self.problem.min_Q_limit, Q_m, 0)
                Q_ijk[:, :, k] = Q_m_limited * active_mask

                # Update hot stream mixer temperatures
                Q_total_from_hot_at_k = np.sum(Q_ijk[:, :, k], axis=1)
                with np.errstate(divide='ignore', invalid='ignore'):
                    delta_T = Q_total_from_hot_at_k / self._hs_CP
                    T_mix_H[:, k] = TinH_stage_k - np.nan_to_num(delta_T)

            # 2b. Cold Pass
            # -------------
            for k in range(ST - 1, -1, -1):
                TinC_stage_k = T_mix_C_prev[:, k + 1] if k < ST - 1 else self._cs_Tin
                Q_total_to_cold_at_k = np.sum(Q_ijk[:, :, k], axis=0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    delta_T = Q_total_to_cold_at_k / self._cs_CP
                    T_mix_C[:, k] = TinC_stage_k + np.nan_to_num(delta_T)

            # 2c. Convergence Check
            # -----------------------
            delta_H = np.max(np.abs(T_mix_H - T_mix_H_prev)) if NH > 0 else 0
            delta_C = np.max(np.abs(T_mix_C - T_mix_C_prev)) if NC > 0 else 0

            if delta_H < self.sws_conv_tol and delta_C < self.sws_conv_tol and sws_iter > 0:
                return {"converged": True, "Q_ijk": Q_ijk, "T_mix_H": T_mix_H, "T_mix_C": T_mix_C,
                        "final_Th_after_recovery": T_mix_H[:, -1],
                        "final_Tc_after_recovery": T_mix_C[:, 0]}

        return {"converged": False, "Q_ijk": np.zeros((NH, NC, ST)),
                    "T_mix_H": np.empty((NH, ST)), "T_mix_C": np.empty((NC, ST)),
                    "final_Th_after_recovery": self._hs_Tin,
                    "final_Tc_after_recovery": self._cs_Tin}

    def _calculate_process_exchanger_costs(self, Z_ijk, sws_results, FH_ijk, FC_ijk, adaptive_penalty):
        """Calculates capital cost and EMAT penalty for process exchangers."""
        capital_cost = 0.0
        penalty_EMAT = 0.0
        details = []
        NH, NC, ST = self.problem.NH, self.problem.NC, self.problem.num_stages
        EMAT = self.problem.cost_params.EMAT
        
        Q_ijk, T_mix_H, T_mix_C = sws_results['Q_ijk'], sws_results['T_mix_H'], sws_results['T_mix_C']

        for k in range(ST):
            for i in range(NH):
                for j in range(NC):
                    if Z_ijk[i, j, k] == 1 and Q_ijk[i, j, k] > 1e-6:
                        hs, cs = self.problem.hot_streams[i], self.problem.cold_streams[j]
                        CPH_b = hs.CP * FH_ijk[i,j,k]
                        CPC_b = cs.CP * FC_ijk[i,j,k]
                        
                        if CPH_b < 1e-9 or CPC_b < 1e-9: continue
                        
                        Q = Q_ijk[i, j, k]
                        
                        Th_in = T_mix_H[i, k-1] if k > 0 else hs.Tin
                        Tc_in = T_mix_C[j, k+1] if k < ST-1 else cs.Tin
                        Th_out = Th_in - Q / CPH_b
                        Tc_out = Tc_in + Q / CPC_b

                        # EMAT penalty check
                        if (Th_in - Tc_out) < EMAT - 1e-3: penalty_EMAT += adaptive_penalty * (EMAT - (Th_in - Tc_out))
                        if (Th_out - Tc_in) < EMAT - 1e-3: penalty_EMAT += adaptive_penalty * (EMAT - (Th_out - Tc_in))

                        lmtd = calculate_lmtd(Th_in, Th_out, Tc_in, Tc_out)
                        U = self.problem.U_matrix_process[i, j]
                        
                        area = Q / (U * lmtd) if U * lmtd > 1e-9 else 1e9
                        if area < 0: area = 1e9
                        
                        cost = self.problem.fixed_cost_process_exchangers[i,j] + \
                               self.problem.area_cost_process_coeff[i,j] * (area ** self.problem.area_cost_process_exp[i,j])
                        
                        capital_cost += cost
                        details.append({'H': i, 'C': j, 'k': k, 'Q': Q, 'Area': area, 'U': U, 'lmtd': lmtd, 'cost': cost,
                                        'Th_in': Th_in, 'Th_out': Th_out, 'Tc_in': Tc_in, 'Tc_out': Tc_out})

        costs = {"capital_process_exchangers": capital_cost, "penalty_EMAT_etc": penalty_EMAT}
        return costs, details

    def _assign_utilities_and_cost(self, final_Th_before, final_Tc_before, adaptive_penalty):
        """Assigns and costs the best available utility for each stream needing one."""
        NH, NC = self.problem.NH, self.problem.NC
        EMAT = self.problem.cost_params.EMAT
        final_Th, final_Tc = final_Th_before.copy(), final_Tc_before.copy()
        
        costs = {
            "capital_heaters": 0.0, "capital_coolers": 0.0,
            "op_cost_hot_utility": 0.0, "op_cost_cold_utility": 0.0,
            "Q_hot_consumed": 0.0, "Q_cold_consumed": 0.0,
            OBJ_KEY_CO2: 0.0, "penalty_unmet_targets": 0.0
        }
        details_list = []

        # Coolers for Hot Streams
        for i in range(NH):
            hs = self.problem.hot_streams[i]
            Q_needed = hs.CP * (final_Th[i] - hs.Tout_target)
            if Q_needed > self.problem.min_Q_limit and hs.CP > 1e-9:
                best_cu_choice = {'cost': float('inf')}
                for cu in self.problem.cold_utility:
                    is_forbidden = any(f.get('hot') == hs.id and f.get('cold') == cu.id for f in self.problem.forbidden_matches or [])
                    if is_forbidden: continue

                    Th_in, Th_out = final_Th[i], hs.Tout_target
                    Tc_in, Tc_out = cu.Tin, cu.Tout if cu.Tout is not None else cu.Tin + 5
                    
                    if Th_in < Tc_out + EMAT - 1e-3 or Th_out < Tc_in + EMAT - 1e-3: continue # EMAT violation

                    lmtd = calculate_lmtd(Th_in, Th_out, Tc_in, Tc_out)
                    area = Q_needed / (cu.U * lmtd) if cu.U * lmtd > 1e-9 else 1e9
                    if area < 0: area = 1e9

                    cap_cost = cu.fix_cost + cu.area_cost_coeff * (area ** cu.area_cost_exp)
                    op_cost = cu.cost * Q_needed
                    if cap_cost + op_cost < best_cu_choice['cost']:
                        best_cu_choice = {'cost': cap_cost + op_cost,
                                          'cap': cap_cost, 'op': op_cost, 
                                          'cu': cu, 'Area': area,
                                          'Th_in': Th_in, 'Th_out': Th_out,
                                          'util_Tin': Tc_in, 'util_Tout': Tc_out,
                                          'Q':Q_needed}
                
                if best_cu_choice['cost'] != float('inf'):
                    costs['capital_coolers'] += best_cu_choice['cap']
                    costs['op_cost_cold_utility'] += best_cu_choice['op']
                    costs['Q_cold_consumed'] += best_cu_choice['Q']
                    costs[OBJ_KEY_CO2] += best_cu_choice['Q'] * best_cu_choice['cu'].co2_factor # type: ignore
                    final_Th[i] = hs.Tout_target
                    details_list.append({'type': 'cooler', 'H_idx': i, **best_cu_choice})
                else:
                    costs['penalty_unmet_targets'] += adaptive_penalty * Q_needed

        # Heaters for Cold Streams
        for j in range(NC):
            cs = self.problem.cold_streams[j]
            Q_needed = cs.CP * (cs.Tout_target - final_Tc[j])
            if Q_needed > self.problem.min_Q_limit and cs.CP > 1e-9:
                best_hu_choice = {'cost': float('inf')}
                for hu in self.problem.hot_utility:
                    is_forbidden = any(f.get('hot') == hu.id and f.get('cold') == cs.id for f in self.problem.forbidden_matches or [])
                    if is_forbidden: continue

                    Tc_in, Tc_out = final_Tc[j], cs.Tout_target
                    Th_in, Th_out = hu.Tin, hu.Tout if hu.Tout is not None else hu.Tin - 5

                    if Th_in < Tc_out + EMAT - 1e-3 or Th_out < Tc_in + EMAT - 1e-3: continue

                    lmtd = calculate_lmtd(Th_in, Th_out, Tc_in, Tc_out)
                    area = Q_needed / (hu.U * lmtd) if hu.U * lmtd > 1e-9 else 1e9
                    if area < 0: area = 1e9

                    cap_cost = hu.fix_cost + hu.area_cost_coeff * (area ** hu.area_cost_exp)
                    op_cost = hu.cost * Q_needed
                    if cap_cost + op_cost < best_hu_choice['cost']:
                        best_hu_choice = {'cost': cap_cost + op_cost,
                                          'cap': cap_cost, 'op': op_cost,
                                          'hu': hu, 'Area': area,
                                          'Tc_in': Tc_in, 'Tc_out': Tc_out,
                                          'util_Tin': Th_in, 'util_Tout': Th_out,
                                          'Q':Q_needed}

                if best_hu_choice['cost'] != float('inf'):
                    costs['capital_heaters'] += best_hu_choice['cap']
                    costs['op_cost_hot_utility'] += best_hu_choice['op']
                    costs['Q_hot_consumed'] += best_hu_choice['Q']
                    costs[OBJ_KEY_CO2] += best_hu_choice['Q'] * best_hu_choice['hu'].co2_factor # type: ignore
                    final_Tc[j] = cs.Tout_target
                    details_list.append({'type': 'heater', 'C_idx': j, **best_hu_choice})
                else:
                    costs['penalty_unmet_targets'] += adaptive_penalty * Q_needed
        
        final_temps = {"Th": final_Th, "Tc": final_Tc}
        return costs, details_list, final_temps

    def _calculate_final_penalties(self, Z_ijk, Q_ijk, final_temps, Q_hot_consumed, Q_cold_consumed, adaptive_penalty):
        """Calculates all remaining penalties for the final network structure."""
        penalties = {
            "penalty_pinch_deviation": 0.0,
            "penalty_forbidden_matches": 0.0,
            "penalty_required_matches": 0.0
        }
        temp_tolerance = 0.001
        
        # Add to unmet target penalty if final temps are still off
        for i, hs in enumerate(self.problem.hot_streams):
            if abs(final_temps["Th"][i] - hs.Tout_target) > temp_tolerance:
                penalties["penalty_unmet_targets"] = penalties.get("penalty_unmet_targets", 0) + adaptive_penalty * abs(final_temps["Th"][i] - hs.Tout_target)
        for j, cs in enumerate(self.problem.cold_streams):
            if abs(final_temps["Tc"][j] - cs.Tout_target) > temp_tolerance:
                penalties["penalty_unmet_targets"] = penalties.get("penalty_unmet_targets", 0) + adaptive_penalty * abs(final_temps["Tc"][j] - cs.Tout_target)

        # Pinch deviation
        if hasattr(self.problem, 'Q_H_min_pinch') and self.problem.Q_H_min_pinch is not None:
            if Q_hot_consumed > self.problem.Q_H_min_pinch + 1e-3:
                penalties['penalty_pinch_deviation'] += self.pinch_deviation_penalty_factor * (Q_hot_consumed - self.problem.Q_H_min_pinch)
        if hasattr(self.problem, 'Q_C_min_pinch') and self.problem.Q_C_min_pinch is not None:
            if Q_cold_consumed > self.problem.Q_C_min_pinch + 1e-3:
                penalties['penalty_pinch_deviation'] += self.pinch_deviation_penalty_factor * (Q_cold_consumed - self.problem.Q_C_min_pinch)

        # Forbidden and Required Matches
        hot_ids = [s.id for s in self.problem.hot_streams]
        cold_ids = [s.id for s in self.problem.cold_streams]
        active_matches = {(hot_ids[i], cold_ids[j]) for i, j, k in np.argwhere(Z_ijk == 1).tolist()}

        if self.problem.forbidden_matches:
            for f in self.problem.forbidden_matches:
                if (f['hot'], f['cold']) in active_matches:
                    penalties['penalty_forbidden_matches'] += adaptive_penalty
        
        if self.problem.required_matches:
            for r in self.problem.required_matches:
                hot_idx = hot_ids.index(r['hot'])
                cold_idx = cold_ids.index(r['cold'])
                if (r['hot'], r['cold']) not in active_matches or np.sum(Q_ijk[hot_idx, cold_idx, :]) < r.get('min_Q_total', 1e-6):
                    penalties['penalty_required_matches'] += adaptive_penalty
        
        return penalties

    def _aggregate_costs(self, costs, penalties, final_temps):
        """Sums all cost and penalty components into the final dictionary."""
        total_capital = costs['capital_process_exchangers'] + costs['capital_heaters'] + costs['capital_coolers']
        total_operating = costs['op_cost_hot_utility'] + costs['op_cost_cold_utility']
        
        # Sum of only penalties that should be in the optimized objective
        penalties_sum_for_obj = sum(v for k, v in penalties.items() if k != 'penalty_unmet_targets' and v > 0)
        
        # True TAC includes unavoidable penalties (unmet targets if no utility could satisfy it)
        true_tac_report = total_capital + total_operating + penalties['penalty_unmet_targets']

        # The objective for the optimizer includes all penalties to guide the search
        tac_for_ga = total_capital + (total_operating * self.utility_cost_factor) + penalties_sum_for_obj + penalties['penalty_unmet_targets']

        return {
            OBJ_KEY_OPTIMIZING: tac_for_ga,
            OBJ_KEY_REPORT: true_tac_report,
            OBJ_KEY_CO2: costs[OBJ_KEY_CO2],
            "capital_process_exchangers": costs['capital_process_exchangers'],
            "capital_heaters": costs['capital_heaters'],
            "capital_coolers": costs['capital_coolers'],
            "op_cost_hot_utility": costs['op_cost_hot_utility'],
            "op_cost_cold_utility": costs['op_cost_cold_utility'],
            "total_capital_cost": total_capital,
            "total_operating_cost": total_operating,
            "Q_hot_consumed_kW_actual": costs['Q_hot_consumed'],
            "Q_cold_consumed_kW_actual": costs['Q_cold_consumed'],
            "penalty_total_in_GA_TAC": penalties_sum_for_obj + penalties['penalty_unmet_targets'],
            "final_outlet_Th_after_utility": final_temps["Th"].tolist(),
            "final_outlet_Tc_after_utility": final_temps["Tc"].tolist(),
            **penalties
        }

    # --- Methods to be implemented by subclasses ---
    def evolve_one_generation(self, gen_num=0, run_id_for_print=""):
        """Performs a single generation of the optimization algorithm."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def inject_chromosome(self, chromosome):
        """Injects an external chromosome into the population."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    # --- Common public methods for epoch interface ---
    def run_epoch(self, generations_in_epoch, current_gen_offset=0, run_id=""):
        """Runs the optimizer for a specified number of generations (an epoch)."""
        for gen in range(generations_in_epoch):
            self.evolve_one_generation(gen_num=current_gen_offset + gen, run_id_for_print=run_id)

    def get_best_chromosome(self):
        """Returns the best chromosome found so far."""
        return self.best_chromosome_overall
    
    def _evaluate_population(self, population):
        for chromosome in population:
            detailed_costs, exchanger_details_list = self._calculate_fitness(chromosome)
            self.fitnesses.append(detailed_costs)
            self.details_list.append(exchanger_details_list)
        return self.fitnesses, self.details_list

    def run(self, run_id_for_print=""):
        """Runs the optimizer for the total number of generations."""
        if hasattr(self, '_evaluate_population') and not self.fitnesses and self.population:
             self.fitnesses, self.details_list = self._evaluate_population(self.population)

        self.run_epoch(self.generations, run_id=run_id_for_print)
        return self.best_chromosome_overall, self.best_costs_overall_dict, self.best_details_overall