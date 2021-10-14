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
    self.frameno = 0
    self.alpha = 2 / (100 + 1)
    for s in self.slices:
      if 'units' not in self.slices[s]['params']:
        self.slices[s]['params']['units'] = 'rbs'
    #self.only_in_profile = config.get('only-in-profile', False)
    self.only_in_profile = sim.conf['only_in_profile']

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

  def _pSchedule(self, s, share, l):
    self.slices[s]['tokens'] -= share
    self.scheduling[s] += share
    return l - share

  def _rSchedule(self, s, share, l):
    self.slices[s]['pool'] -= share
    self.scheduling[s] += share
    return l - share

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
      self.slices[s]['tokens'] = max(self.slices[s]['tokens'] - share,0)
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

  def scheduleTTI(self, traffic, available_rb, rtx_buffer = None):
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
        total_tokens += self.slices[s]['tokens']
        self.scheduling[s] = 0
        if min_over == None or self.slices[s]['over'] < min_over:
          min_over = self.slices[s]['over']
          min_p_over = i
        p_sched_raw.append(s)
        i += 1
    p_sched = Util.CircularIterator(p_sched_raw, min_p_over)

    r_sched_raw = []
    total_r = 0
    i = 0
    max_pool = None
    max_r_pool = None
    for s in self.r_slices:
      full_pool = self.slices[s]['params']['mAR'] * self.slices[s]['params']['TG']
      if self.frameno % self.slices[s]['params']['TG'] == 0:
        # if self.slices[s]['pool'] > 0 and self.frameno!=0 and s== 0:
        #  print("slice",s,"has pool",self.slices[s]['pool'])
        self.slices[s]['pool'] = full_pool
      time_left = self.slices[s]['params']['TG'] - (self.frameno % self.slices[s]['params']['TG'])
      share = max(0, math.ceil(self.slices[s]['pool'] / time_left))
      if s in slice_traffic and slice_traffic[s]['rbs'] > 0:
        slice_traffic[s]['ipr'] = share
        total_r += slice_traffic[s]['ipr']
        self.scheduling[s] = 0
        if max_pool is None or self.slices[s]['pool'] > max_pool:
          max_pool = self.slices[s]['pool']
          max_r_pool = i
        r_sched_raw.append(s)
        i += 1
      else:
        self.slices[s]['pool'] -= share
    r_sched = Util.CircularIterator(r_sched_raw, max_r_pool)

    self.frameno += 1

    # Pre-phase 1 P Slice Rtx
    if rtx_buffer is not None:
      available_rb = self.schedule_rtx(rtx_buffer, available_rb, "P")
    # Pre-phase 2 R Slice Rtx
    if rtx_buffer is not None:
      available_rb = self.schedule_rtx(rtx_buffer, available_rb, "R")


    # phase 1
    if 0 < total_p_ipt < available_rb:
      for s in p_sched:
        available_rb = self._pSchedule(s, slice_traffic[s]['ipt'], available_rb)
    elif total_p_ipt > 0:
      rbs_to_share = available_rb
      for s in p_sched:
        token_share = int(rbs_to_share * (self.slices[s]['tokens'] / total_tokens) / self.slices[s]['params']['delta'])
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

    if not self.only_in_profile:
      for s in p_sched:
        share = min(slice_traffic[s]['opt'], available_rb)
        available_rb = self._pScheduleNSU(s, share, available_rb)
        self._set_p_over(s, share)
        if available_rb == 0:
          break
      for s in r_sched:
        share = min(slice_traffic[s]['rbs'], available_rb)
        available_rb = self._rSchedule(s, share, available_rb)
        if available_rb == 0:
          break

    total = 0
    for s in self.scheduling:
      total += self.scheduling[s]
    assert total <= self.bw

    return self.scheduling, self.rtx_scheduling
