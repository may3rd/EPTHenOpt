# EPTHenOpt/hen_models.py
"""Data models for Heat Exchanger Network (HEN) problems.

This module contains the core classes for defining a HEN problem, including:

*   ``Stream``: Represents hot and cold process streams.
*   ``Utility``: Represents hot and cold utilities.
*   ``CostParameters``: Encapsulates all cost-related parameters for the network.
*   ``HENProblem``: The main class that aggregates all problem data.
"""
import numpy as np

class Stream:
    def __init__(self, id_val=None, Tin=None, Tout_target=None, CP=None,
                 h_coeff=None, U=None, stream_type=None):
        self.id = id_val
        self.Tin = Tin
        self.Tout_target = Tout_target
        self.CP = CP
        self.h = h_coeff
        self.U = U
        self.type = stream_type

class Utility:
    def __init__(self, id_val=None, Tin=None, Tout=None, h_coeff=None, U=None,
                 cost_per_energy_unit=None, fix_cost=None, area_cost_coeff=None,
                 area_cost_exp=None, utility_type=None, co2_factor=0.0):
        self.id = id_val
        self.Tin = Tin
        self.Tout = Tout
        self.h = h_coeff
        self.U = U
        self.cost = cost_per_energy_unit
        self.fix_cost = fix_cost
        self.area_cost_coeff = area_cost_coeff
        self.area_cost_exp = area_cost_exp
        self.type = utility_type
        self.co2_factor = co2_factor

class CostParameters:
    def __init__(self, exch_fixed=0.0, exch_area_coeff=1000.0, exch_area_exp=0.6,
                 heater_fixed=0.0, heater_area_coeff=1200.0, heater_area_exp=0.6,
                 cooler_fixed=0.0, cooler_area_coeff=800.0, cooler_area_exp=0.6,
                 EMAT=10.0, U_overall=None):
        self.exch_fixed = exch_fixed
        self.exch_area_coeff = exch_area_coeff
        self.exch_area_exp = exch_area_exp
        self.heater_fixed = heater_fixed
        self.heater_area_coeff = heater_area_coeff
        self.heater_area_exp = heater_area_exp
        self.cooler_fixed = cooler_fixed
        self.cooler_area_coeff = cooler_area_coeff
        self.cooler_area_exp = cooler_area_exp
        self.EMAT = EMAT
        self.U_overall = U_overall

class HENProblem:
    def __init__(self, hot_streams=None, cold_streams=None, hot_utility=None, cold_utility=None,
                 cost_params=None, num_stages=1, matches_U_cost=None,
                 forbidden_matches=None, required_matches=None, annual_op_hours=8000,
                 no_split=False):
        self.hot_streams = hot_streams or []
        self.cold_streams = cold_streams or []
        self.hot_utility = hot_utility or []
        self.cold_utility = cold_utility or []
        self.cost_params: CostParameters = cost_params # type: ignore
        self.num_stages = num_stages
        self.NH = len(self.hot_streams)
        self.NC = len(self.cold_streams)
        self.NHU = len(self.hot_utility)
        self.NCU = len(self.cold_utility)
        self.matches_U_cost = matches_U_cost
        self.annual_op_hours = annual_op_hours

        self.forbidden_matches = forbidden_matches
        self.required_matches = required_matches
        self.no_split = no_split

        self.U_matrix_process = np.zeros((self.NH, self.NC))
        self.fixed_cost_process_exchangers = np.zeros((self.NH, self.NC))
        self.area_cost_process_coeff = np.zeros((self.NH, self.NC))
        self.area_cost_process_exp = np.zeros((self.NH, self.NC))

        self.U_heaters = np.zeros((self.NHU, self.NC))
        self.U_coolers = np.zeros((self.NH, self.NCU))

        if cost_params:
            self.U_matrix_process.fill(cost_params.U_overall if cost_params.U_overall is not None else 0)
            self.fixed_cost_process_exchangers.fill(cost_params.exch_fixed)
            self.area_cost_process_coeff.fill(cost_params.exch_area_coeff)
            self.area_cost_process_exp.fill(cost_params.exch_area_exp)
            self.U_heaters.fill(cost_params.U_overall if cost_params.U_overall is not None else 0)
            self.U_coolers.fill(cost_params.U_overall if cost_params.U_overall is not None else 0)

        if matches_U_cost:
            hot_stream_ids = {hs.id: idx for idx, hs in enumerate(self.hot_streams)}
            cold_stream_ids = {cs.id: idx for idx, cs in enumerate(self.cold_streams)}
            for match_spec in matches_U_cost:
                hot_id = match_spec.get('hot')
                cold_id = match_spec.get('cold')
                if hot_id in hot_stream_ids and cold_id in cold_stream_ids:
                    i, j = hot_stream_ids[hot_id], cold_stream_ids[cold_id]
                    self.U_matrix_process[i,j] = float(match_spec.get('U', self.U_matrix_process[i,j]))
                    self.fixed_cost_process_exchangers[i,j] = float(match_spec.get('fix_cost', self.fixed_cost_process_exchangers[i,j]))
                    self.area_cost_process_coeff[i,j] = float(match_spec.get('area_cost_coeff', self.area_cost_process_coeff[i,j]))
                    self.area_cost_process_exp[i,j] = float(match_spec.get('area_cost_exp', self.area_cost_process_exp[i,j]))

        if cost_params and cost_params.U_overall is None:
            for i in range(self.NH):
                for j in range(self.NC):
                    if self.U_matrix_process[i,j] == 0:
                        h_hot = self.hot_streams[i].h if self.hot_streams[i].h is not None and self.hot_streams[i].h > 1e-9 else 1e9
                        h_cold = self.cold_streams[j].h if self.cold_streams[j].h is not None and self.cold_streams[j].h > 1e-9 else 1e9
                        if (self.hot_streams[i].h is None or self.hot_streams[i].h <= 1e-9) or \
                           (self.cold_streams[j].h is None or self.cold_streams[j].h <= 1e-9):
                            self.U_matrix_process[i,j] = 1e-6
                        else:
                            self.U_matrix_process[i, j] = 1.0 / (1.0/h_hot + 1.0/h_cold)

            if self.hot_utility:
                for iu in range(self.NHU):
                    for j in range(self.NC):
                        if self.U_heaters[iu,j] == 0:
                            h_hot_util = self.hot_utility[iu].h if self.hot_utility[iu].h is not None and self.hot_utility[iu].h > 1e-9 else 1e9
                            if self.hot_utility[iu].U is not None:
                                self.U_heaters[iu,j] = self.hot_utility[iu].U
                            else:
                                h_cold_stream = self.cold_streams[j].h if self.cold_streams[j].h is not None and self.cold_streams[j].h > 1e-9 else 1e9
                                if (h_hot_util <=1e-9) or (h_cold_stream <= 1e-9):
                                    self.U_heaters[iu,j] = 1e-6
                                else:
                                    self.U_heaters[iu,j] = 1.0 / (1.0/h_hot_util + 1.0/h_cold_stream)

            if self.cold_utility:
                for ic in range(self.NCU):
                    for i in range(self.NH):
                        if self.U_coolers[i,ic] == 0:
                            h_cold_util = self.cold_utility[ic].h if self.cold_utility[ic].h is not None and self.cold_utility[ic].h > 1e-9 else 1e9
                            if self.cold_utility[ic].U is not None:
                                self.U_coolers[i,ic] = self.cold_utility[ic].U
                            else:
                                h_hot_stream = self.hot_streams[i].h if self.hot_streams[i].h is not None and self.hot_streams[i].h > 1e-9 else 1e9
                                if (h_cold_util <=1e-9) or (h_hot_stream <= 1e-9):
                                    self.U_coolers[i,ic] = 1e-6
                                else:
                                    self.U_coolers[i,ic] = 1.0 / (1.0/h_cold_util + 1.0/h_hot_stream)

        self.Q_H_min_pinch, self.Q_C_min_pinch, self.T_pinch_hot_actual, self.T_pinch_cold_actual = self._calculate_pinch_targets()

    def _calculate_pinch_targets(self):
        """
        Calculates the minimum utility requirements (Q_H_min, Q_C_min) and the
        pinch temperatures using the Problem Table Algorithm. This version has
        corrected logic for active stream summation and pinch point identification.
        """
        if not self.hot_streams and not self.cold_streams:
            return 0, 0, None, None
        if not self.cost_params or self.cost_params.EMAT is None:
            return 0, 0, None, None

        EMAT = self.cost_params.EMAT

        # 1. Get all unique temperature points
        temp_points = set()
        for hs in self.hot_streams:
            temp_points.add(hs.Tin)
            temp_points.add(hs.Tout_target)
        for cs in self.cold_streams:
            temp_points.add(cs.Tin + EMAT)
            temp_points.add(cs.Tout_target + EMAT)

        sorted_temps = sorted(list(temp_points), reverse=True)

        if len(sorted_temps) < 2:
            return 0, 0, None, None

        # 2. Build the heat cascade
        heat_cascade = [0.0]

        for i in range(len(sorted_temps) - 1):
            T_high = sorted_temps[i]
            T_low = sorted_temps[i+1]
            T_mid = (T_high + T_low) / 2.0

            sum_fcp_h_active = sum(hs.CP for hs in self.hot_streams if hs.Tin > T_mid and hs.Tout_target < T_mid)
            sum_fcp_c_active = sum(cs.CP for cs in self.cold_streams if cs.Tout_target + EMAT > T_mid and cs.Tin + EMAT < T_mid)

            delta_H_interval = (sum_fcp_h_active - sum_fcp_c_active) * (T_high - T_low)
            heat_cascade.append(heat_cascade[-1] + delta_H_interval)

        # 3. Determine minimum utilities
        q_h_min = 0.0
        min_cascade_value = min(heat_cascade)
        if min_cascade_value < 0:
            q_h_min = -min_cascade_value

        feasible_cascade = [q + q_h_min for q in heat_cascade]
        q_c_min = feasible_cascade[-1]

        # 4. Find the pinch point
        pinch_index = -1
        min_feasible_flow = min(feasible_cascade)
        for i, flow in enumerate(feasible_cascade):
            if abs(flow - min_feasible_flow) < 1e-6:
                pinch_index = i
                break

        if pinch_index != -1:
            t_pinch_hot = sorted_temps[pinch_index]
            t_pinch_cold = t_pinch_hot - EMAT
        else:
            t_pinch_hot, t_pinch_cold = None, None

        # Clean up near-zero values
        if abs(q_h_min) < 1e-6: q_h_min = 0.0
        if abs(q_c_min) < 1e-6: q_c_min = 0.0

        return q_h_min, q_c_min, t_pinch_hot, t_pinch_cold

    def _decode_chromosome(self, chromosome):
        len_Z = self.NH * self.NC * self.num_stages
        len_R_hot_splits = self.NH * self.num_stages * self.NC
        z_part_flat = chromosome[:len_Z]
        r_hot_part_flat = chromosome[len_Z : len_Z + len_R_hot_splits]
        r_cold_part_flat = chromosome[len_Z + len_R_hot_splits:]
        Z_ijk = z_part_flat.reshape((self.NH, self.NC, self.num_stages)).astype(int)
        R_hot_splits_decoded = r_hot_part_flat.reshape((self.NH, self.num_stages, self.NC))
        R_cold_splits_decoded = r_cold_part_flat.reshape((self.NC, self.num_stages, self.NH))
        return Z_ijk, R_hot_splits_decoded, R_cold_splits_decoded
