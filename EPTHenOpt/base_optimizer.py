# gth/base_optimizer.py
"""
Base optimizer module for the EPTHenOpt package.

This module defines the `BaseOptimizer` class, which serves as the foundation
for all optimization algorithms in the package. It encapsulates the shared
logic for problem handling, population management, and fitness calculation,
including the complex Sequential Workspace Synthesis (SWS) and cost evaluation.
"""
import numpy as np
import random

from .hen_models import HENProblem
from .utils import calculate_lmtd

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
        # For TLBO, fitnesses are often stored. GA might not need this at the class level.
        # We can initialize it here and let TLBO use it.
        self.fitnesses = [] 
        self.details_list = []
        self.current_generation = 0

        # Best-so-far tracking
        self.best_chromosome_overall = None
        self.best_costs_overall_dict = {"TAC_GA_optimizing": float('inf'), "TAC_true_report": float('inf')}
        self.best_details_overall = None
        
        # Initialize population (can be called by subclasses if they have specific needs first)
        # Or, call it here directly if no pre-initialization steps are needed by subclasses.
        self._initialize_population()

    def _initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            self.population.append(self._create_random_full_chromosome())
        # If fitnesses are needed for all individuals at initialization (e.g., TLBO)
        # this would be the place to call _evaluate_population or similar.
        # For now, let TLBO handle its initial fitness evaluation.

    def _create_random_full_chromosome(self):
        z_part = np.random.randint(0, 2, size=self.len_Z)
        r_hot_part = np.random.uniform(0.01, 1.0, size=self.len_R_hot_splits)
        r_cold_part = np.random.uniform(0.01, 1.0, size=self.len_R_cold_splits)
        return np.concatenate((z_part, r_hot_part, r_cold_part))

    def _decode_chromosome(self, chromosome):
        # Delegate to the problem instance, which knows its structure
        return self.problem._decode_chromosome(chromosome)

    def _calculate_fitness(self, chromosome):
        # This entire long method is moved from ga_helpers.py / tlbo_helpers.py
        # Ensure all necessary self.problem attributes are accessed correctly.
        Z_ijk, R_hot_splits, R_cold_splits = self._decode_chromosome(chromosome)
        
        # --- 1. Adaptive Penalty Calculation ---
        # The penalty factor grows from initial to final value as generations progress.
        gen_ratio = min(1.0, self.current_generation / self.generations if self.generations > 0 else 1.0)
        adaptive_penalty_factor = self.initial_penalty + (self.final_penalty - self.initial_penalty) * gen_ratio

        NH = self.problem.NH
        NC = self.problem.NC
        ST = self.problem.num_stages
        EMAT = self.problem.cost_params.EMAT

        capital_cost_process_exchangers = 0.0
        capital_cost_heaters = 0.0
        capital_cost_coolers = 0.0
        annual_hot_utility_op_cost = 0.0
        annual_cold_utility_op_cost = 0.0
        penalty_EMAT = 0.0
        penalty_unmet_targets = 0.0
        penalty_pinch_deviation = 0.0
        
        exchanger_details_list = []

        # --- 1. Determine Actual Split Fractions (FH_ijk, FC_ijk) ---
        FH_ijk = np.zeros((NH, NC, ST)) 
        FC_ijk = np.zeros((NH, NC, ST)) 

        for k_stage_split_loop in range(ST):
            for i_hot_split_loop in range(NH):
                active_cold_targets_indices = [j_cold_target for j_cold_target in range(NC) if Z_ijk[i_hot_split_loop, j_cold_target, k_stage_split_loop] == 1]
                num_active_hot_branches = len(active_cold_targets_indices)
                if num_active_hot_branches == 1: 
                    FH_ijk[i_hot_split_loop, active_cold_targets_indices[0], k_stage_split_loop] = 1.0
                elif num_active_hot_branches > 1:
                    raw_r_values = R_hot_splits[i_hot_split_loop, k_stage_split_loop, active_cold_targets_indices]
                    sum_r = np.sum(raw_r_values)
                    if sum_r > 1e-6:
                        normalized_r = raw_r_values / sum_r
                        for idx, j_cold_actual_target_idx in enumerate(active_cold_targets_indices): 
                            FH_ijk[i_hot_split_loop, j_cold_actual_target_idx, k_stage_split_loop] = normalized_r[idx]
                    elif active_cold_targets_indices: 
                        for j_cold_actual_target_idx in active_cold_targets_indices: 
                            FH_ijk[i_hot_split_loop, j_cold_actual_target_idx, k_stage_split_loop] = 1.0 / num_active_hot_branches
            
            for j_cold_split_loop in range(NC):
                active_hot_sources_indices = [i_hot_source for i_hot_source in range(NH) if Z_ijk[i_hot_source, j_cold_split_loop, k_stage_split_loop] == 1]
                num_active_cold_branches = len(active_hot_sources_indices)
                if num_active_cold_branches == 1: 
                    FC_ijk[active_hot_sources_indices[0], j_cold_split_loop, k_stage_split_loop] = 1.0
                elif num_active_cold_branches > 1:
                    raw_r_values = R_cold_splits[j_cold_split_loop, k_stage_split_loop, active_hot_sources_indices]
                    sum_r = np.sum(raw_r_values)
                    if sum_r > 1e-6:
                        normalized_r = raw_r_values / sum_r
                        for idx, i_hot_actual_source_idx in enumerate(active_hot_sources_indices): 
                            FC_ijk[i_hot_actual_source_idx, j_cold_split_loop, k_stage_split_loop] = normalized_r[idx]
                    elif active_hot_sources_indices: 
                        for i_hot_actual_source_idx in active_hot_sources_indices: 
                            FC_ijk[i_hot_actual_source_idx, j_cold_split_loop, k_stage_split_loop] = 1.0 / num_active_cold_branches

        # --- 2. SWS Temperature Iteration Loop ---
        Q_ijk_converged = np.zeros((NH, NC, ST))
        T_mix_H_outlet_current_sws = np.array([[hs.Tin for _ in range(ST)] for hs in self.problem.hot_streams])
        T_mix_C_outlet_current_sws = np.array([[cs.Tin for _ in range(ST)] for cs in self.problem.cold_streams])
        T_mix_H_outlet_prev_sws_iter = T_mix_H_outlet_current_sws.copy()
        T_mix_C_outlet_prev_sws_iter = T_mix_C_outlet_current_sws.copy()
        
        sws_converged = False

        for sws_iter_count in range(self.sws_max_iter):
            T_mix_H_for_convergence_check = T_mix_H_outlet_current_sws.copy()
            T_mix_C_for_convergence_check = T_mix_C_outlet_current_sws.copy()
            Q_ijk_this_sws_iter_pass = np.zeros((NH, NC, ST))

            for k_stage_loop in range(ST):
                TinH_overall_to_stage_k_matches = np.zeros(NH)
                for i_hot_idx in range(NH):
                    TinH_overall_to_stage_k_matches[i_hot_idx] = T_mix_H_outlet_prev_sws_iter[i_hot_idx, k_stage_loop-1] if k_stage_loop > 0 else self.problem.hot_streams[i_hot_idx].Tin
                Q_total_from_hot_stream_at_stage_k = np.zeros(NH)
                for i_hot_idx in range(NH):
                    hs = self.problem.hot_streams[i_hot_idx]
                    TinH_for_hs_branches_in_stage_k = TinH_overall_to_stage_k_matches[i_hot_idx]
                    for j_cold_idx in range(NC):
                        cs = self.problem.cold_streams[j_cold_idx]
                        if Z_ijk[i_hot_idx, j_cold_idx, k_stage_loop] == 1:
                            Tcin_for_cs_branch_in_stage_k = T_mix_C_outlet_prev_sws_iter[j_cold_idx, k_stage_loop+1] if k_stage_loop < ST-1 else cs.Tin
                            CPH_b = hs.CP * FH_ijk[i_hot_idx, j_cold_idx, k_stage_loop]
                            CPC_b = cs.CP * FC_ijk[i_hot_idx, j_cold_idx, k_stage_loop]
                            Q_m = 0 
                            if CPH_b > 1e-9 and CPC_b > 1e-9:
                                Q_H_target_limit = CPH_b * (TinH_for_hs_branches_in_stage_k - hs.Tout_target)
                                Q_H_EMAT_limit   = CPH_b * (TinH_for_hs_branches_in_stage_k - (Tcin_for_cs_branch_in_stage_k + EMAT))
                                Q_C_target_limit = CPC_b * (cs.Tout_target - Tcin_for_cs_branch_in_stage_k)
                                Q_C_EMAT_limit   = CPC_b * ((TinH_for_hs_branches_in_stage_k - EMAT) - Tcin_for_cs_branch_in_stage_k)
                                Q_m = max(0, min(Q_H_target_limit, Q_H_EMAT_limit, Q_C_target_limit, Q_C_EMAT_limit))
                            Q_ijk_this_sws_iter_pass[i_hot_idx, j_cold_idx, k_stage_loop] = Q_m
                            Q_total_from_hot_stream_at_stage_k[i_hot_idx] += Q_m
                for i_hot_mixer_idx in range(NH):
                    hs_m = self.problem.hot_streams[i_hot_mixer_idx]
                    if hs_m.CP > 1e-9:
                        T_mix_H_outlet_current_sws[i_hot_mixer_idx, k_stage_loop] = TinH_overall_to_stage_k_matches[i_hot_mixer_idx] - Q_total_from_hot_stream_at_stage_k[i_hot_mixer_idx] / hs_m.CP
                    else:
                        T_mix_H_outlet_current_sws[i_hot_mixer_idx, k_stage_loop] = TinH_overall_to_stage_k_matches[i_hot_mixer_idx]

            for k_stage_loop in range(ST - 1, -1, -1):
                TinC_overall_to_stage_k_matches = np.zeros(NC)
                for j_cs_idx in range(NC):
                    TinC_overall_to_stage_k_matches[j_cs_idx] = T_mix_C_outlet_prev_sws_iter[j_cs_idx, k_stage_loop+1] if k_stage_loop < ST-1 else self.problem.cold_streams[j_cs_idx].Tin
                Q_total_to_cold_stream_at_stage_k = np.zeros(NC)
                for j_cold_idx in range(NC):
                    for i_hot_idx in range(NH):
                        if Z_ijk[i_hot_idx, j_cold_idx, k_stage_loop] == 1:
                            Q_total_to_cold_stream_at_stage_k[j_cold_idx] += Q_ijk_this_sws_iter_pass[i_hot_idx,j_cold_idx,k_stage_loop]
                for j_cold_mixer_idx in range(NC):
                    cs_m = self.problem.cold_streams[j_cold_mixer_idx]
                    if cs_m.CP > 1e-9:
                        T_mix_C_outlet_current_sws[j_cold_mixer_idx, k_stage_loop] = TinC_overall_to_stage_k_matches[j_cold_mixer_idx] + Q_total_to_cold_stream_at_stage_k[j_cold_mixer_idx] / cs_m.CP
                    else:
                        T_mix_C_outlet_current_sws[j_cold_mixer_idx, k_stage_loop] = TinC_overall_to_stage_k_matches[j_cold_mixer_idx]

            delta_H_conv = np.max(np.abs(T_mix_H_for_convergence_check - T_mix_H_outlet_current_sws)) if NH > 0 and ST > 0 else 0
            delta_C_conv = np.max(np.abs(T_mix_C_for_convergence_check - T_mix_C_outlet_current_sws)) if NC > 0 and ST > 0 else 0
            T_mix_H_outlet_prev_sws_iter = T_mix_H_outlet_current_sws.copy()
            T_mix_C_outlet_prev_sws_iter = T_mix_C_outlet_current_sws.copy()
            Q_ijk_converged = Q_ijk_this_sws_iter_pass.copy() 
            if delta_H_conv < self.sws_conv_tol and delta_C_conv < self.sws_conv_tol and sws_iter_count > 0:
                sws_converged = True
                break
        
        # --- ADDED: Early Exit on SWS Failure ---
        # If the loop finished without converging, this chromosome is invalid.
        # Return a high penalty immediately to save computation.
        if not sws_converged:
            failed_costs = {
                "TAC_GA_optimizing": adaptive_penalty_factor * 1e3, # High penalty
                "TAC_true_report": float('inf'),
                "penalty_SWS_non_convergence": adaptive_penalty_factor * 1e3,
            }
            # Return empty details list and the failure cost dictionary
            return {k: v for k, v in failed_costs.items() if v != 0}, []

        # --- Stage 3 & 4: Exchanger Area/Cost and Utility Calculations ---
        for k_idx_final_cost_loop in range(ST):
            for i_idx_final_cost_loop in range(NH):
                hs_final = self.problem.hot_streams[i_idx_final_cost_loop]
                for j_idx_final_cost_loop in range(NC):
                    cs_final = self.problem.cold_streams[j_idx_final_cost_loop]
                    if Z_ijk[i_idx_final_cost_loop, j_idx_final_cost_loop, k_idx_final_cost_loop] == 1 and \
                       Q_ijk_converged[i_idx_final_cost_loop, j_idx_final_cost_loop, k_idx_final_cost_loop] > 1e-6:
                        Q_final_ex = Q_ijk_converged[i_idx_final_cost_loop, j_idx_final_cost_loop, k_idx_final_cost_loop]
                        Th_in_final_ex = T_mix_H_outlet_current_sws[i_idx_final_cost_loop, k_idx_final_cost_loop-1] if k_idx_final_cost_loop > 0 else hs_final.Tin
                        Tc_in_final_ex = T_mix_C_outlet_current_sws[j_idx_final_cost_loop, k_idx_final_cost_loop+1] if k_idx_final_cost_loop < ST-1 else cs_final.Tin
                        CPH_b_final_ex = hs_final.CP * FH_ijk[i_idx_final_cost_loop, j_idx_final_cost_loop, k_idx_final_cost_loop]
                        CPC_b_final_ex = cs_final.CP * FC_ijk[i_idx_final_cost_loop, j_idx_final_cost_loop, k_idx_final_cost_loop]
                        if CPH_b_final_ex < 1e-9 or CPC_b_final_ex < 1e-9: continue
                        Th_out_final_ex = Th_in_final_ex - Q_final_ex / CPH_b_final_ex
                        Tc_out_final_ex = Tc_in_final_ex + Q_final_ex / CPC_b_final_ex
                        dTa_final = Th_in_final_ex - Tc_out_final_ex
                        dTb_final = Th_out_final_ex - Tc_in_final_ex
                        if dTa_final < EMAT - 1e-3: penalty_EMAT += adaptive_penalty_factor * max(0, EMAT - dTa_final) # Ensure positive or zero
                        if dTb_final < EMAT - 1e-3: penalty_EMAT += adaptive_penalty_factor * max(0, EMAT - dTb_final)
                        lmtd_final_ex = calculate_lmtd(float(Th_in_final_ex), float(Th_out_final_ex), float(Tc_in_final_ex), float(Tc_out_final_ex))
                        U_final_ex = self.problem.U_matrix_process[i_idx_final_cost_loop, j_idx_final_cost_loop]
                        area_final_ex = 1e9
                        if U_final_ex > 1e-9 and lmtd_final_ex > 1e-9 : area_final_ex = Q_final_ex / (U_final_ex * lmtd_final_ex)
                        if area_final_ex < 0: area_final_ex = 1e9
                        CF_process_val = self.problem.fixed_cost_process_exchangers[i_idx_final_cost_loop,j_idx_final_cost_loop]
                        C_area_process_val = self.problem.area_cost_process_coeff[i_idx_final_cost_loop,j_idx_final_cost_loop]
                        B_exp_process_val = self.problem.area_cost_process_exp[i_idx_final_cost_loop,j_idx_final_cost_loop]
                        cost_ex_final = CF_process_val + C_area_process_val * (area_final_ex ** B_exp_process_val)
                        capital_cost_process_exchangers += cost_ex_final
                        exchanger_details_list.append({'H': i_idx_final_cost_loop, 'C': j_idx_final_cost_loop, 'k': k_idx_final_cost_loop, 
                                                       'Q': Q_final_ex, 'Area': area_final_ex, 
                                                       'Th_in': Th_in_final_ex, 'Th_out': Th_out_final_ex, 
                                                       'Tc_in': Tc_in_final_ex, 'Tc_out': Tc_out_final_ex})
        
        final_Th_after_sws_recovery = np.array([hs.Tin for hs in self.problem.hot_streams]) if ST == 0 else T_mix_H_outlet_current_sws[:, ST-1]
        final_Tc_after_sws_recovery = np.array([cs.Tin for cs in self.problem.cold_streams]) if ST == 0 else T_mix_C_outlet_current_sws[:, 0]
        
        target_temp_penalty_factor = 1e9 
        temp_tolerance = 0.001
        Q_hot_consumed_kW_actual = 0.0
        Q_cold_consumed_kW_actual = 0.0
        final_outlet_Th_after_utility = final_Th_after_sws_recovery.copy()
        final_outlet_Tc_after_utility = final_Tc_after_sws_recovery.copy()

        Q_cold_HS_required = np.zeros(NH)
        Q_hot_CS_required = np.zeros(NC)
        
        for i_hot_idx in range(NH):
            hs = self.problem.hot_streams[i_hot_idx]
            Q_total_recovered_for_hs = np.sum(Q_ijk_converged[i_hot_idx, :, :])
            Q_cold_HS_required[i_hot_idx] = max(0, hs.CP * (hs.Tin - hs.Tout_target) - Q_total_recovered_for_hs)

        for j_cold_idx in range(NC):
            cs = self.problem.cold_streams[j_cold_idx]
            Q_total_recovered_for_cs = np.sum(Q_ijk_converged[:, j_cold_idx, :])
            Q_hot_CS_required[j_cold_idx] = max(0, cs.CP * (cs.Tout_target - cs.Tin) - Q_total_recovered_for_cs)

        # Utility assignment logic (coolers for hot streams)
        if self.problem.cold_utility:
            for i_hot_util_loop in range(NH):
                hs_util = self.problem.hot_streams[i_hot_util_loop]
                temp_before_cu = final_Th_after_sws_recovery[i_hot_util_loop]
                # Q_cooler_needed should be calculated based on remaining duty to reach target
                Q_cooler_needed = hs_util.CP * (temp_before_cu - hs_util.Tout_target)


                if Q_cooler_needed > 1e-6 and hs_util.CP > 1e-9:
                    best_cu_obj_for_this_need = None
                    min_incremental_cost_for_this_cooler = float('inf')
                    best_cooler_capital_cost = 0; best_cooler_op_cost_for_this_Q = 0; best_cooler_details = {}

                    for cu_candidate in self.problem.cold_utility:
                        Th_in_cu = temp_before_cu; Th_out_cu = hs_util.Tout_target
                        Tc_in_cu_u = cu_candidate.Tin
                        Tc_out_cu_u = cu_candidate.Tout if cu_candidate.Tout is not None and cu_candidate.Tout > Tc_in_cu_u else Tc_in_cu_u + 5 # Ensure positive delta T for utility
                        
                        emat_ok_cu = True
                        if Th_in_cu < Tc_out_cu_u + EMAT - 1e-3: emat_ok_cu = False
                        if Th_out_cu < Tc_in_cu_u + EMAT - 1e-3: emat_ok_cu = False
                        if Th_out_cu < Th_in_cu: # Check if cooling is possible
                            lmtd_cu_u = calculate_lmtd(Th_in_cu, Th_out_cu, Tc_in_cu_u, Tc_out_cu_u)
                            U_cu_u = cu_candidate.U 
                            area_cu_u = 1e9
                            if U_cu_u > 1e-9 and lmtd_cu_u > 1e-9: area_cu_u = Q_cooler_needed / (U_cu_u * lmtd_cu_u)
                            if area_cu_u < 0: area_cu_u = 1e9
                            
                            current_cooler_capital = cu_candidate.fix_cost + cu_candidate.area_cost_coeff * (area_cu_u ** cu_candidate.area_cost_exp)
                            current_cooler_op = cu_candidate.cost * Q_cooler_needed
                            current_total_impact_cu = current_cooler_capital + current_cooler_op

                            if emat_ok_cu and current_total_impact_cu < min_incremental_cost_for_this_cooler:
                                min_incremental_cost_for_this_cooler = current_total_impact_cu
                                best_cu_obj_for_this_need = cu_candidate
                                best_cooler_capital_cost = current_cooler_capital
                                best_cooler_op_cost_for_this_Q = current_cooler_op
                                best_cooler_details = {'type': 'cooler', 'H_idx': i_hot_util_loop, 'Q': Q_cooler_needed, 
                                                       'Area': area_cu_u, 'Th_in': Th_in_cu, 'Th_out': Th_out_cu, 
                                                       'util_Tin': Tc_in_cu_u, 'util_Tout':Tc_out_cu_u, 'Util_ID': cu_candidate.id}
                        else: # Cooling not possible or EMAT violation
                            pass # Or add penalty if this utility was the only option and target not met
                    
                    if best_cu_obj_for_this_need:
                        Q_cold_consumed_kW_actual += Q_cooler_needed
                        capital_cost_coolers += best_cooler_capital_cost
                        annual_cold_utility_op_cost += best_cooler_op_cost_for_this_Q
                        exchanger_details_list.append(best_cooler_details)
                        final_outlet_Th_after_utility[i_hot_util_loop] = hs_util.Tout_target 
                    elif Q_cooler_needed > 1e-6 : # If cooling was needed but no suitable utility found
                        penalty_unmet_targets += target_temp_penalty_factor * Q_cooler_needed

        # Utility assignment logic (heaters for cold streams)
        if self.problem.hot_utility:
            for j_cold_util_loop in range(NC):
                cs_util = self.problem.cold_streams[j_cold_util_loop]
                temp_before_hu = final_Tc_after_sws_recovery[j_cold_util_loop]
                # Q_heater_val should be calculated based on remaining duty
                Q_heater_val = cs_util.CP * (cs_util.Tout_target - temp_before_hu)


                if Q_heater_val > 1e-6 and cs_util.CP > 1e-9:
                    best_hu_obj_for_this_need = None
                    min_incremental_cost_for_this_heater = float('inf')
                    best_heater_capital_cost = 0; best_heater_op_cost_for_this_Q = 0; best_heater_details = {}
                    
                    for hu_candidate in self.problem.hot_utility:
                        Tc_in_hu_u = temp_before_hu; Tc_out_hu_u = cs_util.Tout_target
                        Th_in_hu_u = hu_candidate.Tin
                        Th_out_hu_u = hu_candidate.Tout if hu_candidate.Tout is not None and hu_candidate.Tout < Th_in_hu_u else Th_in_hu_u - 5 # Ensure positive delta T
                        
                        emat_ok_hu = True
                        if Th_in_hu_u < Tc_out_hu_u + EMAT - 1e-3: emat_ok_hu = False
                        if Th_out_hu_u < Tc_in_hu_u + EMAT - 1e-3: emat_ok_hu = False

                        if Tc_out_hu_u > Tc_in_hu_u: # Check if heating is possible
                            lmtd_hu_u = calculate_lmtd(Th_in_hu_u, Th_out_hu_u, Tc_in_hu_u, Tc_out_hu_u)
                            U_hu_u = hu_candidate.U
                            area_hu_u = 1e9
                            if U_hu_u > 1e-9 and lmtd_hu_u > 1e-9: area_hu_u = Q_heater_val / (U_hu_u * lmtd_hu_u)
                            if area_hu_u < 0: area_hu_u = 1e9
                            
                            current_heater_capital = hu_candidate.fix_cost + hu_candidate.area_cost_coeff * (area_hu_u ** hu_candidate.area_cost_exp)
                            current_heater_op = hu_candidate.cost * Q_heater_val
                            current_total_impact_hu = current_heater_capital + current_heater_op

                            if emat_ok_hu and current_total_impact_hu < min_incremental_cost_for_this_heater:
                                min_incremental_cost_for_this_heater = current_total_impact_hu
                                best_hu_obj_for_this_need = hu_candidate
                                best_heater_capital_cost = current_heater_capital
                                best_heater_op_cost_for_this_Q = current_heater_op
                                best_heater_details = {'type': 'heater', 'C_idx': j_cold_util_loop, 'Q': Q_heater_val, 
                                                    'Area': area_hu_u, 'Tc_in': Tc_in_hu_u, 'Tc_out': Tc_out_hu_u,
                                                    'util_Tin':Th_in_hu_u, 'util_Tout':Th_out_hu_u, 'Util_ID': hu_candidate.id}
                        else: # Heating not possible or EMAT violation
                            pass

                    if best_hu_obj_for_this_need:
                        Q_hot_consumed_kW_actual += Q_heater_val
                        capital_cost_heaters += best_heater_capital_cost
                        annual_hot_utility_op_cost += best_heater_op_cost_for_this_Q
                        exchanger_details_list.append(best_heater_details)
                        final_outlet_Tc_after_utility[j_cold_util_loop] = cs_util.Tout_target
                    elif Q_heater_val > 1e-6: # If heating was needed but no suitable utility found
                         penalty_unmet_targets += target_temp_penalty_factor * Q_heater_val
        
        # Final Target Check
        for i_target_check in range(NH):
            hs_target = self.problem.hot_streams[i_target_check]
            if abs(final_outlet_Th_after_utility[i_target_check] - hs_target.Tout_target) > temp_tolerance:
                penalty_unmet_targets += adaptive_penalty_factor * abs(final_outlet_Th_after_utility[i_target_check] - hs_target.Tout_target)
        for j_target_check in range(NC):
            cs_target = self.problem.cold_streams[j_target_check]
            if abs(final_outlet_Tc_after_utility[j_target_check] - cs_target.Tout_target) > temp_tolerance:
                penalty_unmet_targets += adaptive_penalty_factor * abs(final_outlet_Tc_after_utility[j_target_check] - cs_target.Tout_target)

        # Pinch Deviation Penalty
        if hasattr(self.problem, 'Q_H_min_pinch') and self.problem.Q_H_min_pinch is not None:
            if Q_hot_consumed_kW_actual > self.problem.Q_H_min_pinch + 1e-3 : penalty_pinch_deviation += self.pinch_deviation_penalty_factor * (Q_hot_consumed_kW_actual - self.problem.Q_H_min_pinch)
        if hasattr(self.problem, 'Q_C_min_pinch') and self.problem.Q_C_min_pinch is not None:
            if Q_cold_consumed_kW_actual > self.problem.Q_C_min_pinch + 1e-3: penalty_pinch_deviation += self.pinch_deviation_penalty_factor * (Q_cold_consumed_kW_actual - self.problem.Q_C_min_pinch)
        
        # Forbidden and Required Match Penalties
        forbidden_matches_penalty = 0
        if self.problem.forbidden_matches: # Assuming it's a list of dicts like {'hot': 'H1', 'cold': 'C1'}
            for Z_row_idx, Z_col_idx, Z_stage_idx in np.argwhere(Z_ijk == 1):
                hot_stream_id = self.problem.hot_streams[Z_row_idx].id
                cold_stream_id = self.problem.cold_streams[Z_col_idx].id
                for forbidden in self.problem.forbidden_matches:
                    if forbidden['hot'] == hot_stream_id and forbidden['cold'] == cold_stream_id:
                        forbidden_matches_penalty += adaptive_penalty_factor # Significant penalty
                        break 
        
        required_matches_penalty = 0
        if self.problem.required_matches: # Assuming list of dicts like {'hot': 'H1', 'cold': 'C1', 'min_Q_total': 100}
            for required in self.problem.required_matches:
                match_found_and_met = False
                for Z_row_idx, Z_col_idx, Z_stage_idx in np.argwhere(Z_ijk == 1): # Iterate active matches
                    hot_stream_id = self.problem.hot_streams[Z_row_idx].id
                    cold_stream_id = self.problem.cold_streams[Z_col_idx].id
                    if required['hot'] == hot_stream_id and required['cold'] == cold_stream_id:
                        # Sum Q over all stages for this H-C pair
                        Q_total_for_pair = np.sum(Q_ijk_converged[Z_row_idx, Z_col_idx, :])
                        if Q_total_for_pair >= required.get('min_Q_total', 1e-6): # Use a default min_Q if not specified
                            match_found_and_met = True
                            break 
                if not match_found_and_met:
                    required_matches_penalty += adaptive_penalty_factor # Significant penalty if required match not met

        total_annual_capital_cost = capital_cost_process_exchangers + capital_cost_heaters + capital_cost_coolers
        total_annual_operating_cost = annual_hot_utility_op_cost + annual_cold_utility_op_cost
        
        # Ensure total_penalty_applied_to_ga is a sum of non-negative values
        penalties_sum = sum(p for p in [penalty_EMAT, penalty_unmet_targets, penalty_pinch_deviation, forbidden_matches_penalty, required_matches_penalty] if p > 0)

        TAC_for_GA = total_annual_capital_cost + (total_annual_operating_cost * self.utility_cost_factor) + penalties_sum
        true_TAC_report = total_annual_capital_cost + total_annual_operating_cost + sum(p for p in [penalty_EMAT, penalty_unmet_targets] if p > 0) # True TAC usually includes unavoidable penalties like EMAT violations

        detailed_costs = {
            "TAC_GA_optimizing": TAC_for_GA, "TAC_true_report": true_TAC_report,
            "capital_process_exchangers": capital_cost_process_exchangers, 
            "capital_heaters": capital_cost_heaters,
            "capital_coolers": capital_cost_coolers, 
            "op_cost_hot_utility": annual_hot_utility_op_cost,
            "op_cost_cold_utility": annual_cold_utility_op_cost, 
            "total_capital_cost": total_annual_capital_cost,
            "total_operating_cost": total_annual_operating_cost, 
            "penalty_EMAT_etc": penalty_EMAT, # Renamed for clarity in previous versions
            "penalty_unmet_targets": penalty_unmet_targets, 
            "penalty_pinch_deviation": penalty_pinch_deviation,
            "penalty_forbidden_matches": forbidden_matches_penalty,
            "penalty_required_matches": required_matches_penalty,
            "penalty_total_in_GA_TAC": penalties_sum,
            "Q_hot_consumed_kW_actual": Q_hot_consumed_kW_actual,
            "Q_cold_consumed_kW_actual": Q_cold_consumed_kW_actual
        }
        return detailed_costs, exchanger_details_list

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

    def run(self, run_id_for_print=""):
        """Runs the optimizer for the total number of generations."""
        # This can be called if running sequentially, or by the worker at the very end.
        # Ensure fitnesses are evaluated if not already (e.g. for TLBO)
        if hasattr(self, '_evaluate_population') and not self.fitnesses and self.population:
             self.fitnesses, self.details_list = self._evaluate_population(self.population)

        self.run_epoch(self.generations, run_id=run_id_for_print)
        return self.best_chromosome_overall, self.best_costs_overall_dict, self.best_details_overall
