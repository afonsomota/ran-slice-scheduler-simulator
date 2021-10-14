import math
import statistics
import logging, sys
import numpy as np
import copy
from functools import reduce

import Util
from Scheduler import PassV1
from IntraScheduler import SchedulerPF

from UE import UE
from Cell import Cell

'''TODO list

 - Buffer limit
 - BER based on link-level simulations (using AWGN theoretical ber)
 - HARQ and retransmissions: soft combining
 - LCIDs (currently one buffer per UE)
 - Type 1 or type 0 allocation
 - Accept configuration files

'''


class Slice:

  def __init__(self, slice_id, type, scheduler, params, sim, target=None):
    self.id = slice_id
    self.scheduler = scheduler
    self.type = type
    self.init_params = params
    self.params = copy.deepcopy(params)
    self.violations = 0
    self.excesses = []
    self.excesses_ts = []
    self.overuses = []
    self.overuses_ts = []
    self.ues = {}
    self.ue_outputs = {}
    self.init_stage = True
    self.sim = sim
    self.last_ts = 0
    self.performance = None
    self.primary_kpi = None

    # Target
    if target is None:
      self.target = {}
    else:
      self.target = target
    self.target_metric = {}
    for metric in self.target:
      if metric == 'admission' or metric == 'primary':
        continue
      self.target_metric[metric] = {'x': [], 'y': []}
      if metric == 'system_throughput' and 'update' in params:
        new_metric = 'in_system_throughput'
        self.target_metric[new_metric] = {'x': [], 'y': []}
    if 'admission' in self.target:
      assert 'rate' in self.target['admission'] and 'capacity' in self.target['admission']
      self.admission = Util.BucketAdmission(self.target['admission']['rate'],
                                            self.target['admission']['capacity'],
                                            self.sim.tti)
    if 'primary' in self.target:
      self.primary_kpi = self.target['primary']
    if 'kpi-contract' in self.target:
      self.kpi_violations = Util.FineArray(100, aggr_fun=sum, error_fun=(lambda x: 0))
      self.kpi_violations_average = Util.FineArray(100)
      if self.target['kpi-contract'] == 'urllc':
        assert 'delay' in self.target and 'reliability' in self.target
      elif self.target['kpi-contract'] == 'embb-link':
        assert 'system_throughput' in self.target and 'throughput_average_time' in self.target
    else:
      self.kpi_violations = None
      self.kpi_violations_average = None
    if sim.conf['performance_window'] is not None and self.primary_kpi is not None:
      self.performance = Util.FineArray(sim.conf['performance_window'])


    # Update
    # TODO: update devia ser por CellSlice? ou por celula de alguma maneira!
    if 'update' in self.params:
      self.reserved_rate = {}
      for metric in self.params['update']:
        assert 'target' in self.params['update'][metric]
        #print("Update", self.id, metric)
        metric_params = {
          'type': 'direct',
          'alpha': 1,
          'update_time': 200,
          'analyze_time': 200,
          'past': 1,
          'reference_size': 1500,
          'force_reference': False,
          'ue_weight_type': 'proportional'
        }
        self.reserved_rate[metric] = {'x': [], 'y': []}
        metric_params.update(self.params['update'][metric])
        self.params['update'][metric] = metric_params
        assert metric_params['update_time'] >= metric_params['analyze_time']
        self.over_potential = Util.FineArray(metric_params['update_time'])
        uwt = self.params['update'][metric]['ue_weight_type']
        assert uwt == 'proportional' or \
               uwt == 'equal' or \
               (uwt == 'custom' and 'ue_weight_fun' in self.params['update'][metric])
        if uwt == 'equal':
          self.params['update'][metric]['ue_weight_fun'] = \
            (lambda sli, ts, update_conf, usr: 1 / len(update_conf['rbs_output']))
        # analyze_time = self.params['update'][metric]['analyze_time']
        self.params['update'][metric]['bits_output'] = {}  # Util.FineArray(analyze_time)
        self.params['update'][metric]['rbs_output'] = {}
        self.params['update'][metric]['mcs'] = {}
        self.params['update'][metric]['potential'] = {'x': [], 'y': []}

    # rb or bytes
    if 'units' in params:
      self.units = params['units']
    else:
      self.units = 'rbs'
    self.packet_count = 0
    self.total_packet_count = 0
    self.unreliable_packets = 0
    self.total_unreliable_packets = 0
    self.error_packets = 0
    self.averaged_system_throughput = 0
    self.averaged_system_throughput_in = 0
    self.averaged_input = 0
    self.last_input = 0
    self.averaged_remainder = 0
    self.ip_fraction = 1
    self.subclass_metrics = {}
    self.instances_to_average = 0

  def addUE(self, ue):
    self.ues[ue.id] = ue
    if 'update' in self.params:
      for metric in self.params['update']:
        # analyze_time = self.params['update'][metric]['analyze_time']
        self.params['update'][metric]['bits_output'][ue.id] = 0  # Util.FineArray(analyze_time)
        self.params['update'][metric]['rbs_output'][ue.id] = 0  # Util.FineArray(analyze_time)
        self.params['update'][metric]['mcs'][ue.id] = []  # Util.FineArray(analyze_time)

  def removeUE(self, ue):
    del self.ues[ue.id]
    if 'update' in self.params:
      for metric in self.params['metric']:
        del self.params['update'][metric]['bits_output'][ue.id]
        del self.params['update'][metric]['rbs_output'][ue.id]

  def updateParameters(self):
    if 'update' in self.params:
      for metric in self.params['update']:
        m_rb = self._get_reserved_rate(metric)
        assert m_rb is not None
        # regulate mAR
        update_conf = self.params['update'][metric]
        past = update_conf['past']
        alpha = update_conf['alpha']
        target = update_conf['target']
        ts = self.last_ts  # last_ts already upgraded to current ts
        time_to_store = (ts % update_conf['analyze_time'] == 0) and not self.init_stage
        time_to_update = (ts % update_conf['update_time'] == 0) and not self.init_stage

        if time_to_store:
          if update_conf['ue_weight_type'] == 'proportional':
            bits = 0
            rbs = 0
            for u in update_conf['rbs_output']:
              bits += update_conf['bits_output'][u]
              rbs += update_conf['rbs_output'][u]
              update_conf['bits_output'][u] = 0
              update_conf['rbs_output'][u] = 0
              if rbs == 0:
                bits = update_conf['reference_size'] * 8
                rbs = Util.get_required_rbs(update_conf['mcs'][u], bits / 8, self.sim.tti)
            potential = m_rb * bits / rbs
          else:
            potential = 0
            for u in update_conf['rbs_output']:
              w_u = update_conf['ue_weight_fun'](self, ts, update_conf, u)
              bits = update_conf['bits_output'][u]
              rbs = update_conf['rbs_output'][u]
              update_conf['bits_output'][u] = 0
              update_conf['rbs_output'][u] = 0
              if rbs == 0:
                bits = update_conf['reference_size'] * 8
                rbs = Util.get_required_rbs(update_conf['mcs'][u], bits / 8, self.sim.tti)
            potential += w_u * bits / rbs
            potential *= m_rb
          if ts >= 0:
            update_conf['potential']['y'].append(potential)
            update_conf['potential']['x'].append(ts * self.sim.tti * 1000)
            self.over_potential.fineAppend(ts * self.sim.tti * 1000, potential/target - 1)
          potential_ma = update_conf.get('potential_ma', potential)
          #ma_w = 2 / (past + 1)
          ma_w = 1
          update_conf['potential_ma'] = ma_w * potential + (1 - ma_w) * potential_ma

        if time_to_update:
          potential = update_conf['potential_ma']
          new_m = math.ceil(alpha * m_rb * target / potential)
          self._update_metric(metric, new_m)
          if ts >= 0:
            self.reserved_rate[metric]['x'].append(ts * self.sim.tti * 1000)
            self.reserved_rate[metric]['y'].append(new_m)

  def _update_metric(self, metric, new_m):
    pass

  def _get_reserved_rate(self, name=None):
    return None

  def updateSchedulingMetrics(self, ts, traffic, scheduled):
    tp_avg_time = self.target.get('throughput_average_time', 1)
    no_avgs = tp_avg_time / self.sim.tti / 1000
    stp_target = self.target.get('system_throughput')
    if self.primary_kpi is not None and self.primary_kpi == 'system_throughput' and self.performance is not None and ts >=0:
      self.performance.fineAppend(ts * self.sim.tti * 1000, scheduled['bytes'] * 8 / self.sim.tti / 1000)
    if stp_target is not None:
      if ts % tp_avg_time == 0 and ts != 0 and not self.init_stage:
        window_ts = (ts - tp_avg_time) * self.sim.tti * 1000  # processing values of the last window
        if self.instances_to_average == 0:
          self.averaged_system_throughput = 0
          if self.averaged_input == 0:
            te = 1
          else:
            te = 0
        else:
          # self.averaged_system_throughput /= self.instances_to_average
          # self.averaged_input /= self.instances_to_average
          if 'embb-link' == self.target.get('kpi-contract', ''):
            kpi_violations = 0
          if self.averaged_input < stp_target:
            te = self.averaged_input / self.averaged_system_throughput
            if 'embb-link' == self.target.get('kpi-contract', '') and \
                self.averaged_input < self.averaged_system_throughput - self.last_input:
              kpi_violations = 1
          else:
            te = self.averaged_system_throughput / stp_target
            if 'embb-link' == self.target.get('kpi-contract', '') and \
                self.averaged_system_throughput < stp_target:
              kpi_violations = 1
        if 'embb-link' == self.target.get('kpi-contract', ''):
          if window_ts >= 0:
            self.kpi_violations.fineAppend(window_ts, kpi_violations)
            self.kpi_violations_average.fineAppend(window_ts, kpi_violations)

        if window_ts > 0:
          self.add_target_metric('system_throughput', te, window_ts)
        # self.updateParameters()

        self.averaged_system_throughput = 0
        self.ue_outputs = {}
        self.averaged_remainder = 0
        self.averaged_input = 0
      potential = self._get_potential(ts, traffic, scheduled)
      for u in scheduled['ues']:
        if u not in self.ue_outputs:
          self.ue_outputs[u] = {'bytes': 0, 'rbs': 0}
        self.ue_outputs[u]['bytes'] += scheduled['ues'][u]['bytes']
        self.ue_outputs[u]['rbs'] += scheduled['ues'][u]['rbs']
      if scheduled['rbs'] > 0:
        self.averaged_system_throughput += scheduled['bytes'] * 8 / self.sim.tti / 1000 / no_avgs
        self.instances_to_average += 1
      self.last_input = 8 * (traffic['bytes'] - self.averaged_remainder) / self.sim.tti / 1000 / no_avgs
      self.averaged_input += 8 * (traffic['bytes'] - self.averaged_remainder) / self.sim.tti / 1000 / no_avgs
      self.averaged_remainder = traffic['bytes'] - scheduled['bytes']
    self.last_ts = ts
    if 'update' in self.params:
      for metric in self.params['update']:
        for u in scheduled['ues']:
          self.params['update'][metric]['rbs_output'][u] += scheduled['ues'][u]['rbs']
          self.params['update'][metric]['bits_output'][u] += scheduled['ues'][u]['bytes'] * 8 / self.sim.tti / 1000
          self.params['update'][metric]['mcs'][u] = scheduled['ues'][u]['mcs']
      self.updateParameters()
    self.init_stage = False

  #        if 'update' in self.params:
  #          if scheduled['rbs'] > 0:
  #            if self.params['update']['ue_weight_type'] == 'proportional':
  #              self.averaged_system_throughput_in += self._get_reserved_rate() * \
  #                                                    scheduled['bytes'] * 8 / scheduled['rbs'] / self.sim.tti / 1000
  #            else:
  #              potential_t = 0
  #              w_sum = 0
  #              for u in scheduled['ues']:
  #                if scheduled['ues'][u]['rbs'] != 0:
  #                  w_u = self.params['update']['ue_weight_fun'](self, ts, traffic, scheduled, u)
  #                  w_sum += w_u
  #                  potential_t += self._get_reserved_rate() * w_u * \
  #                                 scheduled['ues'][u]['bytes'] * 8 / scheduled['ues'][u]['rbs'] / \
  #                                 self.sim.tti / 1000
  #              self.averaged_system_throughput_in += potential_t / w_sum
  #

  def tick(self):
    if 'admission' in self.target:
      self.admission.tick()

  def set_target(self, target):
    self.target = target

  def _get_potential(self, ts, traffic, scheduled):
    return None

  def add_target_metric(self, metric, te, ts):
    self.target_metric[metric]['x'].append(ts * self.sim.tti * 1000)
    self.target_metric[metric]['y'].append(te)

  def updatePacketMetrics(self, pkt):
    dl_target = self.target.get('delay')
    ptp_target = self.target_metric.get('user_throughput')
    self.packet_count += 1
    self.total_packet_count += 1
    ts = pkt.get('sent', pkt['scheduled'])
    if 'recv' in pkt and 'sent' in pkt:
      delay = (pkt['recv'] - pkt['sent'])
      if self.primary_kpi is not None and self.primary_kpi == "delay" and self.performance is not None and pkt['sent']>=0:
        self.performance.fineAppend(pkt['sent'], delay)
      if dl_target is not None:
        te = dl_target / delay
        self.add_target_metric('delay', te, ts)
        if 'urllc' == self.target.get('kpi-contract'):
          kpi_violations = 0
        if delay > dl_target:
          self.unreliable_packets += 1
          self.total_unreliable_packets += 1
        if 'urllc' == self.target.get('kpi-contract'):
          #if self.unreliable_packets / self.packet_count > self.target.get('reliability', 0):
          if not Util.check_reliability(self.unreliable_packets, self.packet_count, self.target.get('reliability', 0)):
            kpi_violations = 1
            self.packet_count = 0
            self.unreliable_packets = 0
          if pkt['recv'] >= 0:
            self.kpi_violations.fineAppend(pkt['recv'], kpi_violations)
            self.kpi_violations_average.fineAppend(pkt['recv'], kpi_violations)
      if ptp_target is not None:
        ptp = 8 * pkt['size'] / delay
        te = ptp / ptp_target
        self.add_target_metric('user_throughput', te, ts)
        if ptp < ptp_target: self.unreliable_packets += 1
    elif 'recv' not in pkt:
      # non-delivered packet does not respect contract?
      self.error_packets += 1
      self.unreliable_packets += 1
      if dl_target: self.add_target_metric('delay', 0, ts)
      if ptp_target: self.add_target_metric('user_throughput', 0, ts)

  def getMetrics(self):
    reliability = self.total_unreliable_packets / self.total_packet_count
    packet_loss = self.error_packets / self.total_packet_count
    rl_metric = self.target.get('reliability')
    pl_metric = self.target.get('packet_loss')

    if rl_metric is not None:
      if reliability == 0:
        safe_reliability = 10 ** -10
      else:
        safe_reliability = reliability
      self.target_metric['reliability'] = 10 * np.log10(rl_metric) - 10 * np.log10(safe_reliability)

    if pl_metric is not None:
      if packet_loss == 0:
        safe_pl = 10 ** -10
      else:
        safe_pl = packet_loss
      self.target_metric['packet_loss'] = 10 * np.log10(pl_metric) - 10 * np.log10(safe_pl)

    metric_list = {
      'violations': self.violations,
      'reliability': reliability,
      'packet_loss': packet_loss,
      'excesses': {
        'x': self.excesses_ts,
        'y': self.excesses
      },
      'overuses': {
        'x': self.overuses_ts,
        'y': self.overuses
      },
      'target_metric': self.target_metric,
      # TODO additional metrics specific to subclasses
      'subclass_metrics': self.subclass_metrics,
    }
    if self.performance is not None:
      metric_list['performance'] = self.performance.get_x_and_y()
    if 'update' in self.params:
      metric_list['reserved_rate'] = self.reserved_rate
      metric_list['over_potential'] = self.over_potential.get_x_and_y()
    if 'kpi-contract' in self.target:
      metric_list['kpi_violations'] = self.kpi_violations.get_x_and_y()
      metric_list['kpi_violations_average'] = self.kpi_violations_average.get_x_and_y()
    return metric_list


class RSlice(Slice):

  def __init__(self, slice_id, scheduler, params, sim, target=None):
    self.tg = params['TG']
    self.mar = params['mAR']
    self.pool = self.tg * self.mar
    self.input = 0
    self.byte_output = 0
    self.rb_output = 0
    self.output_potential = 0
    self.byte_input = 0
    self.byte_remainder = 0
    self.last_ts = 1
    self.remainder = 0
    self.ip_output_bytes = 0
    self.ttis = 0
    super().__init__(slice_id, 'R', scheduler, params, sim, target)

  def updateMar(self, value):
    self.mar = value
    self.params['mAR'] = value

  def updateTg(self, value):
    self.tg = value
    self.params['TG'] = value

  def _get_reserved_rate(self, name=None):
    if name is None:
      return self.mar
    elif name == 'mAR':
      return self.mar

  def _update_metric(self, metric, new_m):
    if metric == 'mAR':
      self.updateMar(new_m)

  def _get_potential(self, ts, traffic, scheduled):
    if scheduled['rbs'] != 0:
      return self.mar * scheduled['bytes'] / scheduled['rbs']
    else:
      return None
      # no_ues = len(scheduled['ues'])
      # potential = 0
      # for u in scheduled['ues'].values():
      #  potential += PC_TBS[u['mcs']][self.mar - 1]/8/self.mar
      # potential = self.mar * potential / no_ues
      # return potential

  # def updateParameters(self):
  #  pass

  #     tp_metric = 'in_system_throughput'
  #     if len(self.target_metric[tp_metric]['y']) > past:
  #       mar_step = 0  # math.ceil(self.mar * self.params.get('mar_step', 0.1))
  #       tp_target_arr = self.target_metric[tp_metric]['y'][-past:]
  #       if len(tp_target_arr) != 0:
  #         tp_target = sum(tp_target_arr) / len(tp_target_arr)
  #         new_mar = math.ceil((mar_step + self.mar) / tp_target)
  #         if self.id == 0 and self.mar != new_mar:
  #           print(self.last_ts, self.mar, new_mar, tp_target)
  #         self.updateMar(new_mar)
  #         # if tp_target > (1 + up_sense):
  #         #  self.updateMar(self.mar - mar_step)
  #         # elif tp_target < (1 - low_sense):
  #         #  self.updateMar(self.mar + mar_step)

  def updateSchedulingMetrics(self, ts, traffic, scheduled):

    # if last ts of current pool
    if ts // self.tg != self.last_ts // self.tg and self.ttis > 0:
      self.ttis = 0
      tg_ts = self.tg * (ts // self.tg)
      # check violation
      if self.pool > 0 and self.input > self.tg * self.mar:
        # print(self.id,self.pool,self.input,self.tg*self.mar,flush=True)
        if self.pool >= self.mar:
          self.violations += 1
      overuse = 0
      excess = 0
      if self.pool > 0:
        excess = self.pool / self.tg
      else:
        overuse = abs(self.pool) / self.tg
      if ts >= 0:
        self.excesses.append(excess)
        self.excesses_ts.append(tg_ts)
        self.overuses.append(abs(overuse))
        self.overuses_ts.append(tg_ts)

      if 'mAR' not in self.subclass_metrics:
        self.subclass_metrics['mAR'] = {'x': [], 'y': []}
      if ts >= 0 or True:
        self.subclass_metrics['mAR']['x'].append(ts * self.sim.tti * 1000)
        self.subclass_metrics['mAR']['y'].append(self.mar)
      self.last_ts = ts
      # self.updateParameters()
      # reset pool
      self.pool = self.tg * self.mar
      self.input = traffic[self.units]
      self.byte_input = traffic['bytes']
      self.byte_output = 0
      self.rb_output = 0
      self.output_potential = 0
      self.byte_remainder = 0
      self.remainder = 0
      self.ip_output_bytes = 0
    else:
      self.input += traffic[self.units] - self.remainder
      self.byte_input += traffic['bytes'] - self.byte_remainder

    for u in traffic['ues']:
      if u in scheduled['ues']:
        self.output_potential += self.mar * scheduled['ues'][u]['bytes'] / scheduled['rbs'] if scheduled[
                                                                                                 'rbs'] != 0 else 0
    self.byte_output += scheduled['bytes']
    self.rb_output += scheduled['rbs']
    self.remainder = traffic[self.units] - scheduled[self.units]
    self.byte_remainder = traffic['bytes'] - scheduled['bytes']
    self.ttis += 1

    old_pool = self.pool
    self.pool -= scheduled[self.units]
    # set in profile fraction of the output
    if self.pool > 0:
      self.ip_fraction = 1
    elif old_pool > 0:
      self.ip_fraction = old_pool / (old_pool - self.pool)
    else:
      self.ip_fraction = 0

    if traffic[self.units] == 0:
      self.pool -= self.mar
    self.last_ts = ts

    super().updateSchedulingMetrics(ts, traffic, scheduled)


class PSlice(Slice):

  def __init__(self, slice_id, scheduler, params, sim, target=None):

    self.mpr = params['MPR']
    self.bc = params['BC']
    self.delta = params.get('delta', 5)
    self.tokens = self.bc
    self.last_ts = 0
    self.violations = 0
    self.fine_excesses = []
    self.fine_overuses = []
    self.fine_time = 100
    self.bw = sim.bw
    if 'update' in params:
      if 'byte_capacity' in params['update']:
        # TODO fine array precision by configuration parameter
        self.bc_index = Util.FineArray(100)
      if 'byte_rate' in params['update']:
        window = 100
        self.output_potential = Util.FineArray(window, aggr_fun=sum, error_fun=(lambda x: 0))
        self.input_for_potential = Util.FineArray(window, aggr_fun=sum, error_fun=(lambda x: 0))
    super().__init__(slice_id, 'P', scheduler, params, sim, target)

  def _get_potential(self, ts, traffic, scheduled):
    return None

  def _get_reserved_rate(self, name=None):
    if name is None:
      return self.mpr
    elif name == "MPR":
      return self.mpr
    elif name == "BC":
      return self.bc

  def update_mpr(self, value):
    self.mpr = value
    self.params['MPR'] = value

  def update_bc(self, value):
    self.bc = value
    self.params['BC'] = value

  def _update_metric(self, metric, new_m):
    if metric == 'MPR':
      self.update_mpr(new_m)
    elif metric == 'BC':
      self.update_bc(new_m)

  # def updateParameters(self):
  #  pass

  def get_reserved_rate(self):
    return self.mpr

  def updateSchedulingMetrics(self, ts, traffic, scheduled):

    self.tokens += self.mpr * (ts - self.last_ts)
    self.last_ts = ts

    excess = 0
    if self.tokens > self.bc:
      excess = self.tokens - self.bc
    self.fine_excesses.append(excess)
    self.tokens = min(self.tokens, self.bc)

    self.tokens -= scheduled[self.units]
    overuse = 0
    if self.tokens < 0:
      overuse = abs(self.tokens)
      self.tokens = 0
    self.fine_overuses.append(overuse)

    # if 'update' in self.params:
    #  min_tokens = self.params['update'].get('min_tokens', 0.2 * self.mpr)
    #  bc_step = 0.2 * self.bc
    #  if self.tokens < min_tokens:
    #    self.bc += bc_step

    # if 'update' in self.params and 'byte_rate' in self.params['update']:
    #  for u in traffic['ues']:
    #    fine_out = self.output_potential.getFineArray()
    #    if len(fine_out) == 0:
    #      remaining = 0
    #    else:
    #      fine_in = self.input_for_potential.getFineArray()
    #      remaining = fine_in[-1] - fine_out[-1]
    #    _, outp, _ = self.output_potential.fineAppend(ts, self.mpr * scheduled['ues'][u]['bytes'] / scheduled['rbs'])
    #    _, in4p, _ = self.input_for_potential.fineAppend(ts, traffic - remaining)
    #    mpr_thres = self.params['update'].get('mpr_threshold', 0.3)
    #    mpr_step = self.params['update'].get('mpr_step', 0.1)
    #    mpr_step = math.ceil(mpr_step * self.mpr)

    #    if outp is not None:
    #      if in4p < (1 - mpr_thres) * outp:
    #        self.mpr += mpr_step
    #      elif in4p > (1 + mpr_thres) * outp:
    #        self.mpr -= mpr_step

    # if 'update' in self.params and 'byte_capacity' in self.params['update']:
    #  total_ues = len(scheduled['ues'])
    #  cur_bc_index = 0
    #  bc_step = 0.2 * self.bc
    #  for u in scheduled['ues']:
    #    u_mcs = scheduled['ues'][u]['mcs']
    #    cur_bc_index += Util.getRequiredRBs(u_mcs, self.params['update']['byte_capacity'])
    #  self.bc_index.fineAppend(ts, cur_bc_index / total_ues)
    #  bc_period = self.params['update'].get('bc_period', 1000)
    #  if ts % bc_period == 0:
    #    bc_index = np.mean(self.bc_index.getLastValues(bc_period))
    #    bc_threshold = self.params['update'].get('bc_threshold', 0.3)
    #    if self.bc > bc_index * (1 + bc_threshold):
    #      self.bc -= bc_step
    #    elif self.bc < bc_index * (1 - bc_threshold):
    #      self.bc += bc_step

    if (ts + 1) % self.fine_time == 0:
      if len(self.fine_excesses) == 0 or len(self.fine_overuses) == 0:
        print("Empty values", self.fine_overuses, self.fine_excesses)
      if (ts >= 0):
        coarse_ts = self.fine_time * (ts // self.fine_time)
        self.excesses.append(np.mean(self.fine_excesses))
        self.excesses_ts.append(coarse_ts)
        self.overuses.append(np.mean(self.fine_overuses))
        self.overuses_ts.append(coarse_ts)
      self.fine_overuses = []
      self.fine_excesses = []

    if traffic[self.units] > scheduled[self.units] and scheduled['rbs'] != self.bw and self.tokens > 0:
      # print(traffic[self.units], scheduled[self.units], self.tokens)
      self.violations += 1

    super().updateSchedulingMetrics(ts, traffic, scheduled)


class Simulation:

  def __init__(self, in_sim_conf=None, ue_conf=None, slice_conf=None, verbose=True):
    if ue_conf is None:
      ue_conf = [
        {'slice': 0, 'traffic': 'full_buffer', 'params': {'size': 1500}, 'count': 10},
        {'slice': 1, 'traffic': 'cbr', 'params': {'interval': 0.005, 'jitter': 0.001, 'size': 100, 'drift': 0.005},
         'count': 20},
        {'slice': 2, 'traffic': 'cbr', 'params': {'interval': 0.005, 'jitter': 0.001, 'size': 100, 'drift': 0.005},
         'count': 20}
      ]
    if slice_conf is None:
      slice_conf = {
        0: {'type': 'R', 'params': {'mAR': 27, 'TG': 100}, 'scheduler': SchedulerPF()},
        1: {'type': 'P', 'params': {'MPR': 9, 'BC': 250}, 'scheduler': SchedulerPF()},
        2: {'type': 'R', 'params': {'mAR': 9, 'TG': 100}, 'scheduler': SchedulerPF()}
      }
    sim_conf = {
      'bw': 50,
      'timeout': 10,
      'tti': 0.001,
      'cell_metric_window': 1,
      'inter-scheduler': PassV1,
      'store_packets': True,
      'packet_metrics': ['delay', 'throughput', 'delivery'],
      'packet_metric_window': 100,
      'cqi_report_period': 10,
      'sinr_store_window': 100,
      'only_in_profile': False,
      'rtx_part_of_slice': True,
      'performance_window': 100,
      'warmup': 0
    }
    if in_sim_conf is not None:
      sim_conf.update(in_sim_conf)
    self.conf = sim_conf

    self.tti = sim_conf['tti']
    self.bw = sim_conf['bw']
    self.timeout = sim_conf['timeout']
    self._initSlices(slice_conf)
    self.scheduler = sim_conf['inter-scheduler'](slice_conf, sim=self)
    self._initCells(self.scheduler)
    self._initUEs(ue_conf)
    if verbose:
      level = logging.INFO
    else:
      level = logging.ERROR
    logging.basicConfig(stream=sys.stdout, level=level, format='%(message)s')
    self.logging = logging
    logging.info("Simultion init...")

  def _initSlices(self, conf):
    self.slices = {}
    for s in conf:
      # self.slices[s] = Slice(s,conf[s]['type'],conf[s]['scheduler'],conf[s]['params'])
      if conf[s]['type'] == 'P':
        self.slices[s] = PSlice(s, conf[s]['scheduler'], conf[s]['params'], self, conf[s].get('target'))
      else:
        self.slices[s] = RSlice(s, conf[s]['scheduler'], conf[s]['params'], self, conf[s].get('target'))

  def _initCells(self, scheduler, no_cells=1):
    self.cells = {}
    for i in range(0, no_cells):
      self.cells[i] = Cell(scheduler, self, self.tti, self.bw)
      for s in self.slices.values():
        self.cells[i].addSlice(s)

  def _initUEs(self, conf):
    self.ues = {}
    i = 0
    for ue in conf:
      for n in range(0, ue['count']):
        seed_conf = ue['params'].get('traffic_seed')
        if type(seed_conf) is np.ndarray:
          index = n % len(seed_conf)
          seed = seed_conf[index]
        else:
          seed = seed_conf
        generator = np.random.default_rng(seed)
        start_ts = 0
        if 'drift' in ue['params']:
          drift = ue['params']['drift'] / self.tti
          if drift != 0:
            start_ts = generator.random() * drift
        start_ts -= self.conf['warmup'] * 1000
        movement = None
        if 'movement' in ue:
          movement = ue['movement']
        sli = self.slices[ue['slice']]
        self.ues[i] = UE(i, self, sli, ue['traffic'], ue['params'], movement, start_ts=start_ts, generator=generator)
        self.slices[ue['slice']].addUE(self.ues[i])
        self.ues[i].attachClosest(self.cells)
        i += 1

  def run(self):
    no_tti = int(self.timeout / self.tti)
    frame = -int(self.conf['warmup'] / self.tti)
    logging.info("Start simulation")
    while True:
      logging.info("TTI %d", frame)
      for sli in self.slices.values():
        sli.tick()
      for u in self.ues.values():
        u.tick()
        u.triggerActivity(frame * self.tti)
      for c in self.cells:
        logging.info("Cell %d", c)
        self.cells[c].schedule()
      frame += 1
      if frame >= no_tti:
        for c in self.cells.values():
          c.deactivate()
        done = True
        for u in self.ues.values():
          u.active = False
          if u.checkActivity():
            done = False
        if done:
          break

    slice_metrics = {}
    user_metrics = {}
    for s in self.slices:
      slice_metrics[s] = {'delivery': [], 'delay': [], 'tp': [], 'retx': [], 'no_segments': []}
    for u in self.ues.values():
      metrics = u.getMetrics()
      metrics['retx'] = (u.metrics['retx']['count'], 0)
      no_segments_list = tuple(map(lambda x: x['no_segments'], u.metrics['packets'].values()))
      # metrics['no_segments'] = (statistics.mean(no_segments_list),1.96*statistics.pstdev(no_segments_list)/ \
      # math.sqrt(no_segments_list))
      metrics['no_segments'] = Util.getMeanCIPair(no_segments_list)
      u.metrics['final'] = metrics
      if not self.conf['store_packets']:
        del u.metrics['packets']
      user_metrics[u.id] = u.metrics
      for m in ['delivery', 'delay', 'tp', 'retx', 'no_segments']:
        slice_metrics[u.slice.id][m].append(metrics[m])

    for s in slice_metrics:
      for m in slice_metrics[s]:
        values = tuple(map(lambda x: x[0], slice_metrics[s][m]))
        if None in values:
          slice_metrics[s][m] = (None, None)
        else:
          if m == 'retx':
            slice_metrics[s][m] = (sum(values), 0)
          else:
            avg = statistics.mean(values)
            std = statistics.pstdev(values)
            slice_metrics[s][m] = (avg, 1.96 * std / math.sqrt(len(values)))

    for s in slice_metrics:
      slice_metrics[s]['rbs'] = {}
      slice_metrics[s]['bits'] = {}
      slice_metrics[s]['cxc'] = {}
      slice_metrics[s].update(self.slices[s].getMetrics())
      logging.info("Slice %d metrics: %s", s, slice_metrics[s])

    for s in self.slices:
      for c in self.cells:
        cell = self.cells[c]
        slice_metrics[s]['cxc'][c] = cell.metrics['x']
      slice_metrics[s]['cx'] = reduce(np.union1d, slice_metrics[s]['cxc'].values())
    for c in self.cells:
      cell = self.cells[c]
      x_mask = np.isin(slice_metrics[s]['cx'], cell.metrics['x'], assume_unique=True)
      for s in self.slices:
        for m in ['rbs', 'bits']:
          slice_metrics[s][m][c] = np.zeros(x_mask.shape)
          slice_metrics[s][m][c][x_mask] = np.array(tuple(map(lambda x: x[s], cell.metrics[m])))
      # for sched in cell.metrics['rbs']:
      #  for s in self.slices:
      #    if s in sched:
      #      slice_metrics[s]['rbs'][c].append(sched[s])
      #    else:
      #      slice_metrics[s]['rbs'][c].append(0)
      # for bits in cell.metrics['bits']:
      #  for s in self.slices:
      #    if s in bits:
      #      slice_metrics[s]['bits'][c].append(bits[s])
      #    else:
      #      slice_metrics[s]['bits'][c].append(0)
    # TODO: consider x
    # slice_metrics[s]['x']['total'] = reduce(np.union1d,slice_metrics[s]['x'].values())
    for s in self.slices:
      slice_metrics[s]['rbs']['total'] = sum(slice_metrics[s]['rbs'].values())
      slice_metrics[s]['bits']['total'] = sum(slice_metrics[s]['bits'].values())

    return user_metrics, slice_metrics

# sim = Simulation(verbose=False)
# sim.run()
