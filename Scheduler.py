import Util

import math
import copy


class NVS:
  def __init__(self, slices={}, bw=50):
    self.slices = {}
    self.beta = 0.1
    for s in slices:
      rate = None
      if slices[s]['type'] == 'P':
        rate = slices[s]['params']['MPR']
      elif slices[s]['type'] == 'R':
        rate = slices[s]['params']['mAR']

      assert rate is not None

      self.slices[s] = {
        'rate': rate,
        'units': slices[s]['params'].get('units', 'rbs'),
        'w': 0
      }

  def getMaximumW(self, slices):
    max_w = 0
    maximizer = None
    for s in slices:
      if maximizer is None or max_w > self.slices[s]['w']:
        max_w = self.slices[s]['w']
        maximizer = s
    return maximizer

  def scheduleTTI(self, traffic, available_rb):
    scheduling = {}
    eligible = set(traffic.keys())

    while len(eligible) > 0 and available_rb > 0:
      maximizer = self.getMaximumW(eligible)
      share = min(traffic[maximizer], available_rb)
      scheduling[maximizer] = share
      available_rb -= share
      eligible.remove(maximizer)

    # TODO update bytes?
    for s in self.slices:
      self.slices['w'] = (1 - self.beta) * self.slices['w'] + scheduling.get(s, 0) * self.beta

    return scheduling


class PassV1:

  def __init__(self, slices={}, sim=None):
    '''Scheduler constructor
       
       Arguments:
       
       slices - dictionary containg slice configuration {'$id': {'type':['P','R'], 'params':contract_params}'''
    self.slices = copy.deepcopy(slices)
    self.simulation = sim
    self.bw = sim.conf['bw']
    self.p_slices = set(map(lambda x: x[0], filter(lambda el: el[1]['type'] == 'P', self.slices.items())))
    self.r_slices = set(map(lambda x: x[0], filter(lambda el: el[1]['type'] == 'R', self.slices.items())))
    self.scheduling = {}
    self.rtx_scheduling = {}
    self.wasted = 0
    self.wasted_count = 0
    for s in self.p_slices:
      assert 'params' in self.slices[s] and \
             'BC' in self.slices[s]['params'] and \
             'MPR' in self.slices[s]['params']
      self.slices[s]['tokens'] = self.slices[s]['params']['BC']
      self.slices[s]['over'] = 0
    for s in self.r_slices:
      assert 'params' in self.slices[s] and \
             'mAR' in self.slices[s]['params'] and \
             'TG' in self.slices[s]['params']
      mar = self.slices[s]['params']['mAR']
      tg = self.slices[s]['params']['TG']
      self.slices[s]['pool'] = tg * mar
      if self.slices[s]['params'].get('ewma', False):
        #print("Use EWMA")
        self.slices[s]['ewma'] = mar
        self.slices[s]['alpha'] = 2 / (tg+1)
      else:
        #print("Use Fix Pool")
        self.slices[s]['ewma'] = None
    self.frameno = 0
    self.alpha = 2 / (100 + 1)
    for s in self.slices:
      if 'units' not in self.slices[s]['params']:
        self.slices[s]['params']['units'] = 'rbs'
    # self.only_in_profile = config.get('only-in-profile', False)
    self.only_in_profile = sim.conf['only_in_profile']
    self.pool_multiplier = 1.2

  def updateParams(self, slices):
    for s in slices:
      if s.type == 'P' and \
         self.slices[s.id]['params']['BC'] != s.params['BC'] and \
         self.slices[s.id]['params']['MPR'] != s.params['MPR']:
        self.slices[s.id]['params']['BC'] = s.params['BC']
        self.slices[s.id]['params']['MPR'] = s.params['MPR']
        # readjust token value
        self.slices[s.id]['tokens'] = min(s.params['BC'], self.slices[s.id]['tokens'])
      elif s.type == 'R' and \
           self.slices[s.id]['params']['TG'] != s.params['TG'] and \
           self.slices[s.id]['params']['mAR'] != s.params['mAR']:
        self.slices[s.id]['params']['TG'] = s.params['TG']
        self.slices[s.id]['params']['mAR'] = s.params['mAR']
        # Adjust pool linearly
        old_pool = self.slices[s.id]['pool']
        old_max_pool = self.slices[s.id]['params']['mAR'] * self.slices[s.id]['params']['TG']
        new_max_pool = s.params['mAR'] * s.params['params']['TG']
        self.slices[s.id]['pool'] = new_max_pool * old_pool / old_max_pool
      self.slices[s.id]['params'] = s.params

  def _tokenUpdate(self, s):
    assert s in self.p_slices
    tokens = self.slices[s]['tokens']
    mpr = self.slices[s]['params']['MPR']
    bc = self.slices[s]['params']['BC']
    self.slices[s]['tokens'] = tokens + mpr if tokens + mpr < bc else bc
    return self.slices[s]['tokens']

  def _inProfileP(self, s, slice_traffic):
    return min(slice_traffic[s]['rbs'], self.slices[s]['tokens'])

  def _pSchedule(self, s, share, l, mini_slot=None):
    if mini_slot is None:
      tokens_removed = share
    else:
      tokens_removed = share/7.0
    self.slices[s]['tokens'] -= tokens_removed
    self.scheduling[s] += share
    return l - share

  def _rSchedule(self, s, share, l):
    if self.slices[s]['ewma'] is not None:
      self.slices[s]['ewma'] += self.slices[s]['alpha'] * share
    else:
      self.slices[s]['pool'] -= share
    self.scheduling[s] += share
    return l - share

  def _rRemove(self, s, share):
    if self.slices[s]['ewma'] is not None:
      self.slices[s]['ewma'] -= self.slices[s]['alpha'] * share/7.
    else:
      self.slices[s]['pool'] += share/7.
    self.scheduling[s] -= share
    # if self.scheduling[s] == 0:
    #   del self.scheduling[s]

  def _pScheduleNSU(self, s, share, l):
    self.scheduling[s] += share
    return l - share

  def _set_p_over(self, s, share):
    if 'over' not in self.slices[s]:
      self.slices[s]['over'] = share / self.slices[s]['params']['MPR']
    else:
      self.slices[s]['over'] = (1 - self.alpha) * self.slices[s]['over'] + \
                               self.alpha * share / self.slices[s]['params']['MPR']

  def deduct_from_slice(self, s, share):
    """
    Deduct resources from the slice (usually because of Rtx)
    :param s: slice
    :param share: the share to be deducted
    :return:
    """
    if self.slices[s]['type'] == 'P':
      if self.slices[s]['tokens'] < share:
        excess = share - self.slices[s]['tokens']
        self._set_p_over(s, excess)
      self.slices[s]['tokens'] = max(self.slices[s]['tokens'] - share, 0)
    elif self.slices[s]['type'] == 'R':
      self.slices[s]['pool'] -= share
    pass

  def schedule_rtx(self, rtx_buffer, available_rbs, sli_type=None):
    temp_buf_rtxs = rtx_buffer.copy()
    for hp in temp_buf_rtxs:
      amount = hp['size']
      usr = hp['usr']
      left = available_rbs - amount
      if left < 0:
        continue
      elif sli_type is not None and hp['usr']['ue'].slice.type != sli_type:
        continue
      else:
        available_rbs -= amount
        slice_id = hp['usr']['ue'].slice.id
        ue_id = hp['usr']['ue'].id
        if slice_id not in self.rtx_scheduling:
          self.rtx_scheduling[slice_id] = {'rbs': 0, 'bytes': 0, 'ues': {}}
        self.rtx_scheduling[slice_id]['rbs'] += amount
        self.rtx_scheduling[slice_id]['bytes'] += hp['size']
        if ue_id not in self.rtx_scheduling[slice_id]['ues']:
          self.rtx_scheduling[slice_id]['ues'][ue_id] = {'rbs': 0, 'bytes': 0}
          self.rtx_scheduling[slice_id]['ues'][ue_id]['rbs'] += amount
          self.rtx_scheduling[slice_id]['ues'][ue_id]['bytes'] += hp['size']
        rtx_buffer.remove(hp)
        usr['ue'].dataTxRx(usr['sinr'], hp['mcs'], None, hp['size'], hp)
        if self.simulation.conf['rtx_part_of_slice']:
          self.deduct_from_slice(usr['ue'].slice.id, amount)
    return available_rbs

  def puncture(self, traffic, available_rb, scheduled_ue, scheduled_slices, slice_opr, mini_slot, rtx_buffer=None):

    # slice scheduling is based in old scheduling
    self.scheduling = copy.deepcopy(scheduled_slices)
    self.rtx_scheduling = {}
    for s in scheduled_slices:
      assert scheduled_slices[s] == sum(scheduled_ue[s].values()), f"{s}, {scheduled_slices[s]}, {scheduled_ue[s]}"

    slice_traffic = copy.deepcopy(traffic)

    p_sched_raw = []
    total_p_ipt = 0
    total_tokens = 0
    i = 0
    min_over = None
    min_p_over = None
    scheduled_mini_rbs = available_rb
    for s in self.p_slices:
      # self._tokenUpdate(s)
      if s in slice_traffic and slice_traffic[s]['rbs'] > 0:
        # tokens have 7 times the value
        # int of round(x,2) is to conteract when x=?,9999 when it was supposed to be ?+1
        ipt = min(slice_traffic[s]['rbs'], int(round(self.slices[s]['tokens']*7, 2)))
        # print("ipt", slice_traffic[s]['rbs'], int(round(self.slices[s]['tokens']*7, 2)))
        slice_traffic[s]['ipt'] = ipt
        slice_traffic[s]['opt'] = slice_traffic[s]['rbs'] - ipt
        total_p_ipt += ipt
        total_tokens += self.slices[s]['tokens'] / self.slices[s]['params']['delta']
        self.scheduling[s] = 0
        if min_over is None or self.slices[s]['over'] < min_over:
          min_over = self.slices[s]['over']
          min_p_over = i
        p_sched_raw.append(s)
        i += 1
    p_sched = Util.CircularIterator(p_sched_raw, min_p_over)

    # Pre-phase 1 P Slice Rtx
    if rtx_buffer is not None:
      available_rb = self.schedule_rtx(rtx_buffer, available_rb, "P")

    # Phase 1 alike
    if 0 < total_p_ipt < available_rb:
      for s in p_sched:
        available_rb = self._pSchedule(s, slice_traffic[s]['ipt'], available_rb, mini_slot=mini_slot)
    elif total_p_ipt > 0:
      rbs_to_share = available_rb
      for s in p_sched:
        token_share = int(rbs_to_share * (self.slices[s]['tokens'] / self.slices[s]['params']['delta']) / total_tokens)
        share = min(slice_traffic[s]['ipt'], token_share)
        available_rb = self._pSchedule(s, share, available_rb, mini_slot=mini_slot)
        slice_traffic[s]['ipt'] -= share
        total_p_ipt -= share
      if total_p_ipt > 0:
        rbs_to_share = available_rb
        for s in p_sched:
          share = int(rbs_to_share * slice_traffic[s]['ipt'] / total_p_ipt)
          available_rb = self._pSchedule(s, share, available_rb, mini_slot=mini_slot)

    # R slice Rtx
    if rtx_buffer is not None:
      available_rb = self.schedule_rtx(rtx_buffer, available_rb, "R")

    # Phase 3 for P slices

    scheduled_mini_rbs -= available_rb
    scheduled_old_rbs = sum([rbs for rbs in scheduled_slices.values()])
    punctured_rbs = max(0, - (self.bw - scheduled_mini_rbs - scheduled_old_rbs))
    total_slice_opr = sum(slice_opr.values())
    puncturable_opr = max(self.bw - (scheduled_old_rbs - total_slice_opr) - scheduled_mini_rbs, 0)
    new_slice_opr = slice_opr.copy()

    if not self.only_in_profile and puncturable_opr > 0:
      for s in p_sched:
        assert puncturable_opr >= 0
        share = max(min(slice_traffic[s]['opt'], puncturable_opr), 0)
        available_rb = self._pScheduleNSU(s, share, available_rb)
        new_slice_opr[s] += share
        punctured_rbs += share
        puncturable_opr -= share
        assert puncturable_opr >= 0
        self._set_p_over(s, share)
        if available_rb == 0:
          break

    # Count scheduled punctures and decide which resources to unschedule

    remaining_punctures = punctured_rbs
    scheduled_old_slices = set(scheduled_slices.keys())
    iters = 0
    punctured_ues = set()
    sched_ue = copy.deepcopy(scheduled_ue)

    # Puncture OPR
    if remaining_punctures > 0:
      for s in slice_opr:
        if slice_opr[s] == 0:
          continue
        for u in sched_ue[s]:
          scheduled_ue_old_rbs = sched_ue[s][u]
          if remaining_punctures - scheduled_ue_old_rbs < 0:
            waste = - (remaining_punctures - scheduled_ue_old_rbs)
            self.wasted_count += 1
            self.wasted += (waste - self.wasted)/self.wasted_count
          remaining_punctures = max(remaining_punctures - scheduled_ue_old_rbs, 0)
          self._rRemove(s, scheduled_ue_old_rbs)
          new_slice_opr[s] = max(0, new_slice_opr[s] - scheduled_ue_old_rbs)
          punctured_ues.add(u)
          #print(s,u,sched_ue[s][u],remaining_punctures,new_slice_opr[s],scheduled_ue_old_rbs)
          if remaining_punctures == 0 or new_slice_opr[s] == 0:
            #print("break")
            break
        for u in punctured_ues:
          if u in sched_ue[s]:
            del sched_ue[s][u]

    # Puncture with lower pool
    while remaining_punctures > 0:
      min_p = None
      min_p_slice = None
      for s in scheduled_old_slices:
        # look for the slice farther from reaching its minimum target
        if self.slices[s]['ewma'] is not None:
          if min_p_slice is None or min_p < self.slices[s]['ewma']/self.slices[s]['params']['mAR']:
            min_p = self.slices[s]['ewma']/self.slices[s]['params']['mAR']
            min_p_slice = s
        elif min_p_slice is None or min_p < self.slices[s]['pool']/self.slices[s]['params']['mAR']:
          min_p = self.slices[s]['pool']/self.slices[s]['params']['mAR']
          min_p_slice = s
      #print(min_p_slice, scheduled_old_slices)
      if min_p_slice is None:
        break
      scheduled_old_slices.remove(min_p_slice)
      for u in sched_ue[min_p_slice]:
        scheduled_ue_old_rbs = sched_ue[min_p_slice][u]
        if remaining_punctures - scheduled_ue_old_rbs < 0:
          waste = - (remaining_punctures - scheduled_ue_old_rbs)
          self.wasted_count += 1
          self.wasted += (waste - self.wasted)/self.wasted_count
        if scheduled_ue_old_rbs == 0:
          continue
        remaining_punctures = max(remaining_punctures - scheduled_ue_old_rbs, 0)
        #print(min_p_slice, self.frameno, mini_slot, "being preempted", u, remaining_punctures, scheduled_ue_old_rbs, sched_ue[min_p_slice], scheduled_slices)
        self._rRemove(min_p_slice, scheduled_ue_old_rbs)
        #new_slice_opr[s] = max(0, new_slice_opr[s] - scheduled_ue_old_rbs)
        punctured_ues.add(u)
        if remaining_punctures == 0:
          break

      iters += 1
      assert iters < 1000

    total = 0
    for s in self.scheduling:
      total += self.scheduling[s]
    assert total <= self.bw, f"Too many resources for capacity: {self.scheduling}, p_rbs {punctured_rbs}, rem {remaining_punctures}, old_rb {scheduled_old_rbs}, old_opr {total_slice_opr}, new_opr {sum(slice_opr.values())-total_slice_opr} , sched_mini {scheduled_mini_rbs}, puncturable {puncturable_opr}"
    #punctured_ues = set()

    #print("done")
    return self.scheduling, self.rtx_scheduling, punctured_ues, new_slice_opr

  def scheduleTTI(self, traffic, available_rb, rtx_buffer=None, jump_phase_1=False):
    """Scheduler one transmission time interval

       Arguments:

       slice_traffic - dictionary containing the RB requirements of each slice"""

    # phase prep
    # available_rb = self.bw
    self.scheduling = {}
    self.rtx_scheduling = {}

    slice_traffic = copy.deepcopy(traffic)

    p_sched_raw = []
    total_p_ipt = 0
    total_tokens = 0
    i = 0
    min_over = None
    min_p_over = None
    for s in self.p_slices:
      self._tokenUpdate(s)
      if s in slice_traffic and slice_traffic[s]['rbs'] > 0:
        ipt = min(slice_traffic[s]['rbs'], self.slices[s]['tokens'])
        slice_traffic[s]['ipt'] = ipt
        slice_traffic[s]['opt'] = slice_traffic[s]['rbs'] - ipt
        total_p_ipt += ipt
        total_tokens += self.slices[s]['tokens'] / self.slices[s]['params']['delta']
        self.scheduling[s] = 0
        if min_over is None or self.slices[s]['over'] < min_over:
          min_over = self.slices[s]['over']
          min_p_over = i
        p_sched_raw.append(s)
        i += 1
    p_sched = Util.CircularIterator(p_sched_raw, min_p_over)

    r_sched_raw = []
    total_r = 0
    i = 0
    maximizer = None
    r_begin = None
    for s in self.r_slices:
      full_pool = self.slices[s]['params']['mAR'] * self.slices[s]['params']['TG']
      if self.slices[s]['ewma'] is not None:
        share = max(0, math.ceil(self.slices[s]['params']['mAR'] - (self.slices[s]['ewma'] - self.slices[s]['params']['mAR'])))
        self.slices[s]['ewma'] = (1 - self.slices[s]['alpha']) * self.slices[s]['ewma']
      else:
        if self.frameno % self.slices[s]['params']['TG'] == 0:
          # if self.slices[s]['pool'] > 0 and self.frameno!=0 and s== 0:
          #  print("slice",s,"has pool",self.slices[s]['pool'])
          self.slices[s]['pool'] = full_pool
        time_left = self.slices[s]['params']['TG'] - (self.frameno % self.slices[s]['params']['TG'])
        share = max(0, math.ceil(min(self.pool_multiplier * self.slices[s]['pool'] / time_left, self.slices[s]['pool'])))

      if s in slice_traffic and slice_traffic[s]['rbs'] > 0:
        slice_traffic[s]['ipr'] = share
        total_r += slice_traffic[s]['ipr']
        self.scheduling[s] = 0
        # check what is the slice the needs more resources for phase 3 advantage
        if self.slices[s]['ewma'] is not None:
          if maximizer is None or self.slices[s]['params']['mAR']/self.slices[s]['ewma'] > maximizer:
            maximizer = self.slices[s]['params']['mAR']/self.slices[s]['ewma']
            r_begin = i
        elif maximizer is None or self.slices[s]['pool'] > maximizer:
          maximizer = self.slices[s]['pool']
          r_begin = i
        r_sched_raw.append(s)
        i += 1
      else:
        # if no traffic in the buffer, then consider that the slice is totally fulfilled
        if self.slices[s]['ewma'] is not None:
          self.slices[s]['ewma'] += self.slices[s]['alpha']*self.slices[s]['params']['mAR']
        else:
          self.slices[s]['pool'] -= share
    r_sched = Util.CircularIterator(r_sched_raw, r_begin)

    self.frameno += 1

    # Pre-phase 1 P Slice Rtx
    if rtx_buffer is not None:
      available_rb = self.schedule_rtx(rtx_buffer, available_rb, "P")
    # Pre-phase 2 R Slice Rtx
    if rtx_buffer is not None:
      available_rb = self.schedule_rtx(rtx_buffer, available_rb, "R")

    # phase 1
    # jump phase 1 if puncturing is enabled (phase 1 will be scheduled with puncture() method
    if not jump_phase_1:
      if 0 < total_p_ipt < available_rb:
        for s in p_sched:
          available_rb = self._pSchedule(s, slice_traffic[s]['ipt'], available_rb)
      elif total_p_ipt > 0:
        rbs_to_share = available_rb
        for s in p_sched:
          token_share = int(rbs_to_share *
                            (self.slices[s]['tokens'] / total_tokens) / self.slices[s]['params']['delta'])
          share = min(slice_traffic[s]['ipt'], token_share)
          available_rb = self._pSchedule(s, share, available_rb)
          slice_traffic[s]['ipt'] -= share
          total_p_ipt -= share
        if total_p_ipt > 0:
          rbs_to_share = available_rb
          for s in p_sched:
            share = int(rbs_to_share * slice_traffic[s]['ipt'] / total_p_ipt)
            # TODO isto não faz sentido se available_rb < ipt
            available_rb = self._pSchedule(s, share, available_rb)

    # phase 2
    if 0 < total_r < available_rb and available_rb > 0:
      for s in r_sched:
        amount = min(slice_traffic[s]['ipr'], slice_traffic[s]['rbs'])
        available_rb = self._rSchedule(s, amount, available_rb)
    elif total_r > 0 and available_rb > 0:
      rbs_to_share = available_rb
      for s in r_sched:
        res_share = int(rbs_to_share * slice_traffic[s]['ipr'] / total_r)
        share = min(res_share, slice_traffic[s]['rbs'])
        available_rb = self._rSchedule(s, share, available_rb)
        slice_traffic[s]['rbs'] -= share
      for s in r_sched:
        share = min(slice_traffic[s]['rbs'], available_rb)
        available_rb = self._rSchedule(s, share, available_rb)
        if available_rb == 0:
          break

    # phase 3

    slice_opr = {}
    for s in self.slices:
      slice_opr[s] = 0

    if not self.only_in_profile:
      for s in p_sched:
        share = min(slice_traffic[s]['opt'], available_rb)
        available_rb = self._pScheduleNSU(s, share, available_rb)
        slice_opr[s] += share
        self._set_p_over(s, share)
        if available_rb == 0:
          break
      for s in r_sched:
        share = min(slice_traffic[s]['rbs'] - self.scheduling[s], available_rb)
        slice_opr[s] += share
        available_rb = self._rSchedule(s, share, available_rb)
        if available_rb == 0:
          break

    total = 0
    for s in self.scheduling:
      total += self.scheduling[s]
    assert total <= self.bw
    assert sum(slice_opr.values()) <= sum(self.scheduling.values()), f"{slice_opr, self.scheduling}"

    return self.scheduling, self.rtx_scheduling, slice_opr
