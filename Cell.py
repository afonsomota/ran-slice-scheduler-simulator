import copy

import Util
import math

AVG_TP_ALPHA = 2 / (100 + 1)
RB_PER_TTI = 1

class CellSlice:

  def __init__(self, sli, cell):
    self.slice = sli
    self.cell = cell

  def schedule(self, ue_info, ue_traffic, available_rb):
    return self.slice.scheduler.schedule(ue_info, ue_traffic, available_rb)


class Cell:

  def __init__(self, inter_scheduler, simulation, tti=0.001, bw=50):
    self.attachedUEs = {}
    self.slices = {}
    self.inter_scheduler = inter_scheduler
    self.tti = tti
    self.bw = bw
    self.buffered_rtxs = []
    self.buffered_rtxs_mini = []
    self.simulation = simulation
    self.ts = -int(self.simulation.conf['warmup'] / self.tti)
    self.debug_csv = open("debug.csv","w")
    if 'cell_metric_definition' not in simulation.conf:
      simulation.conf['cell_metric_definition'] = 1
    cm_def = simulation.conf['cell_metric_definition']
    self.metrics = {
      'x': Util.FineArray(cm_def, aggr_fun=(lambda x: x[0]), error_fun=None),
      'rbs': Util.FineArray(cm_def, aggr_fun=Util.slice_metric_aggregator, error_fun=None),
      'bits': Util.FineArray(cm_def, aggr_fun=Util.slice_metric_aggregator, error_fun=None)
    }
    self.active = True
    self.slice_metrics = {
      'violations': {},  # no SLA violations (td)
      'overbooking': {},  # no of underprovisioned resources (ti version of violation)
      'overprovision': {},  # no of overprovisioned resources (ti)
      'excess': {}  # no of non-allocated resources
    }

  def deactivate(self):
    self.active = False

  def addSlice(self, sli):
    self.slices[sli.id] = CellSlice(sli, self)

  def removeSlice(self, sli):
    del self.slices[sli.id]

  def attachUE(self, ue):
    fixed_mcs = ue.fixed_mcs if ue.fixed_mcs is not None else 10
    self.attachedUEs[ue.id] = {'ue': ue, 'rsrp': 80, 'sinr': 80, 'mcs': fixed_mcs, 'average_tp': 0}

  def detachUE(self, ue):
    del self.attachedUEs[ue.id]

  def _getSlicesTraffic(self, slice_filter=None, mini_slot=None, mini_divide=False):
    st = {}
    for uid in self.attachedUEs:
      u = self.attachedUEs[uid]
      s = u['ue'].slice.id
      if s not in st:
        st[s] = {'rbs': 0, 'bytes': 0, 'ues': {}}
      if slice_filter and u['ue'].slice.type != slice_filter:
        continue
      buf = u['ue'].getBuffer()
      if mini_slot is None and mini_divide:
        # calculate RBs using block groups of mini-slot each
        rbs = 0
        for m in range(7):
          rbs += Util.get_required_rbs(u['mcs'], buf, self.simulation.tti, mini_slot=m)/7.
        rbs = int(math.ceil(rbs/7.0))
      else:
        rbs = Util.get_required_rbs(u['mcs'], buf, self.simulation.tti, mini_slot=mini_slot)
      st[s]['rbs'] += rbs
      st[s]['bytes'] += buf
      st[s]['ues'][uid] = {}
      st[s]['ues'][uid]['rbs'] = rbs
      st[s]['ues'][uid]['bytes'] = buf
      st[s]['ues'][uid]['mcs'] = u['mcs']
    return st

  '''
  slice_traffic = { slice: {'rbs': <rbs>, 'bytes': <bytes>, 'ues :{'mcs': <mcs>, 'rbs': <rbs>, 'bytes': <bytes>}} }
  slice_share = { slice: <rbs> } 
  scheduler_metrics = { slice: {'ipr': <in_profile_resources>, 'opr': <out_profile_resources> }}
  '''

  def register_metrics(self, slice_traffic, scheduled, rtx_scheduled):

    for s in slice_traffic:
      sli = self.slices[s].slice
      if s not in scheduled and s in rtx_scheduled:
        scheduled[s] = {'rbs': 0, 'bytes': 0, 'ues': {}}
      if s in rtx_scheduled:
        scheduled[s]['rbs'] += rtx_scheduled[s]['rbs']
        scheduled[s]['bytes'] += rtx_scheduled[s]['bytes']
        for u in rtx_scheduled[s]['ues']:
          if u not in scheduled[s]['ues']:
            scheduled[s]['ues'][u]['rbs'] = rtx_scheduled[s]['ues'][u]['rbs']
            scheduled[s]['ues'][u]['bytes'] = rtx_scheduled[s]['ues'][u]['bytes']
          else:
            scheduled[s]['ues'][u]['rbs'] += rtx_scheduled[s]['ues'][u]['rbs']
            scheduled[s]['ues'][u]['bytes'] += rtx_scheduled[s]['ues'][u]['bytes']
      sli.updateSchedulingMetrics(self.ts, slice_traffic[s], scheduled[s])

  def update_scheduler(self):
    self.inter_scheduler.updateParams(map((lambda x: x.slice), self.slices.values()))

  def updateUesMcs(self):
    for u in self.attachedUEs.values():
      u['sinr'] = u['ue'].sinr
      if u['ue'].link_adaptation and self.ts % self.simulation.conf['cqi_report_period'] == 0:
        target_ber = u['ue'].slice.params.get('mcs-target-ber', 5 * 10 ** -5)
        u['mcs'] = Util.getShannonMCS(u['ue'].sinr, target_ber)
        if self.ts > 0:
          u['ue'].metrics['mcs'].fineAppend(self.ts * self.simulation.tti * 1000, u['mcs'])

  def schedule(self):
    use_mini_slots = self.simulation.conf.get('mini-slot', False)
    available_rbs = self.bw
    self.updateUesMcs()
    # First schedule RTX
    if not use_mini_slots:
      for uid in self.attachedUEs:
        have_rtx, hps = self.attachedUEs[uid]['ue'].checkRetransmissions()
        if have_rtx:
          for hp in hps:
            hp['uid'] = uid
            hp['usr'] = self.attachedUEs[hp['uid']]
            self.buffered_rtxs.append(hp)

    # temp_buf_rtxs = self.buffered_rtxs.copy()
    # for hp in temp_buf_rtxs:
    #   amount = hp['size']
    #   usr = self.attachedUEs[hp['uid']]
    #   left = available_rbs - amount
    #   if left < 0:
    #     continue
    #   else:
    #     available_rbs -= amount
    #     self.buffered_rtxs.remove(hp)
    #     usr['ue'].dataTxRx(usr['sinr'], hp['mcs'], None, hp['size'], hp)
    #     if self.simulation.conf['rtx_part_of_slice']:
    #       self.inter_scheduler.deduct_from_slice(usr['ue'].slice.id, amount)

    # Then scheduler new
    if use_mini_slots:
      slice_traffic = self._getSlicesTraffic(slice_filter='R', mini_divide=True)
    else:
      slice_traffic = self._getSlicesTraffic()
    if len(slice_traffic) == 0:
      return

    if use_mini_slots:
      # with mini slots, rtxs are only considered mini-slot by mini-slot
      buffered_rtxs = None
    else:
      buffered_rtxs = self.buffered_rtxs
    slice_share, rtx_scheduled, slice_opr = self.inter_scheduler.scheduleTTI(
      slice_traffic, available_rbs, buffered_rtxs)
    scheduled = {}
    user_scheduled = {}
    sent = {}
    data_sent = {}

    for s in self.slices:
      sli = self.slices[s]
      slice_ues = dict(filter(lambda elem: elem[1]['ue'].slice.id == s, self.attachedUEs.items()))
      if s in slice_share:
        user_scheduled[s] = sli.schedule(slice_ues, slice_traffic[s]['ues'], slice_share[s])
      else:
        user_scheduled[s] = sli.schedule(slice_ues, slice_traffic[s]['ues'], 0)

    if use_mini_slots:
      # Mini-slot preemption
      mini_user_scheduled = {}
      tti_slice_share = slice_share
      slice_share = {}
      for s in self.slices:
        scheduled[s] = {'rbs': 0, 'bytes': 0}
        scheduled[s]['ues'] = {}
        slice_share[s] = 0
      for m in range(7):
        # Copy marco scheduling to micro slots
        mini_user_scheduled[m] = {}
        for s in user_scheduled:
          mini_user_scheduled[m][s] = copy.deepcopy(user_scheduled[s])
        # Check for RTX in this mini-slot
        available_rbs_mini = self.bw
        for uid in self.attachedUEs:
          usr = self.attachedUEs[uid]['ue']
          have_rtx, hps = usr.checkRetransmissions()
          if have_rtx:
            for hp in hps:
              hp['uid'] = uid
              hp['usr'] = self.attachedUEs[hp['uid']]
              if hp in self.buffered_rtxs_mini:
                continue
              else:
                assert len(list(filter(lambda x: x['uid'] == uid and x['pid'] == hp['pid'], self.buffered_rtxs_mini))) == 0
              self.buffered_rtxs_mini.append(hp)
          if usr.slice.type == 'R':
            continue

        slice_traffic_mini_p = self._getSlicesTraffic(slice_filter='P', mini_slot=m)
        # slice_traffic_mini = self._getSlicesTraffic(mini_slot=m)
        slice_share_mini, rtx_scheduled_mini, punctured_ues, mini_slice_opr = self.inter_scheduler.puncture(
           slice_traffic_mini_p,
           available_rbs_mini,
           mini_user_scheduled[m],
           tti_slice_share,
           slice_opr,
           m,
           self.buffered_rtxs_mini
        )
        assert sum(slice_share_mini.values()) <= self.bw, f"{slice_share_mini}, {tti_slice_share}, {punctured_ues}"
        for s in self.slices:
          if s in slice_share_mini:
            slice_share[s] += slice_share_mini[s]/7.
          sli = self.slices[s]
          if sli.slice.type == 'P':
            slice_ues = dict(filter(lambda elem: elem[1]['ue'].slice.id == s, self.attachedUEs.items()))
            mini_user_scheduled[m][s] = sli.schedule(slice_ues, slice_traffic_mini_p[s]['ues'], slice_share_mini.get(s, 0))
        #print(self.ts, m, slice_share_mini, mini_slice_opr, slice_traffic_mini_p[1]['rbs'], mini_user_scheduled[m])
        # print(slice_traffic_mini_p, mini_user_scheduled[m])
        # if not last tick 1/7 of normal tick to check for traffic
        sent_t = 0
        for s in mini_user_scheduled[m]:
          for u in mini_user_scheduled[m][s]:
            # print(m, s, u, mini_user_scheduled[m][s][u])
            # data_sent[u] = 0
            usr = self.attachedUEs[u]
            # if u in punctured_ues and mini_user_scheduled[m][s][u] > 0:
            #   sent[u] = Util.get_tbs(usr['mcs'], mini_user_scheduled[m][s][u], self.simulation.tti, mini_slot=m)
            #   usr['ue'].failed_by_puncture(usr['mcs'], sent[u] / 8, mini_user_scheduled[m][s][u], mini_slot=m)
            # elif mini_user_scheduled[m][s][u] > 0:
            if u not in punctured_ues and mini_user_scheduled[m][s][u] > 0:
              mini_user_scheduled[m][s][u] *= RB_PER_TTI
              if u not in sent:
                sent[u] = 0
                data_sent[u] = 0
              assert type(mini_user_scheduled[m][s][u]) is not float, mini_user_scheduled[m][s][u]
              sent_m = Util.get_tbs(usr['mcs'], mini_user_scheduled[m][s][u], self.simulation.tti, mini_slot=m)
              sent_t += mini_user_scheduled[m][s][u]
              sent[u] += sent_m
              _, data_sent_ue = usr['ue'].dataTxRx(usr['sinr'],
                                                   usr['mcs'],
                                                   sent_m / 8,
                                                   mini_user_scheduled[m][s][u],
                                                   mini_slot=m)
              data_sent[u] += data_sent_ue
              #print(round(self.ts + m/7., 3), s, u, sent_m, data_sent_ue, mini_user_scheduled[m][s][u], sep=",", file=self.debug_csv)
              if u not in scheduled[s]['ues']:
                scheduled[s]['ues'][u] = {'rbs': 0, 'bytes': 0, 'mcs': usr['mcs']}
              scheduled[s]['rbs'] += mini_user_scheduled[m][s][u]/7.
              scheduled[s]['ues'][u]['rbs'] += mini_user_scheduled[m][s][u]/7.
              scheduled[s]['bytes'] += data_sent_ue
              scheduled[s]['ues'][u]['bytes'] += data_sent_ue
        # print(self._getSlicesTraffic(slice_filter='R', mini_slot=m))
        assert sent_t <= self.bw*RB_PER_TTI, f"{sent_t}, {mini_user_scheduled[m]}, {punctured_ues}, {slice_share_mini}"
        if m < 6:
          for uid in self.attachedUEs:
            usr = self.attachedUEs[uid]['ue']
            usr.tick(ts=1. / 7)

      if self.active:
        self.register_metrics(slice_traffic, scheduled, rtx_scheduled)
        self.update_scheduler()
    # Else if mini_slot
    else:
      for s in self.slices:
        scheduled[s] = {'rbs': 0, 'bytes': 0}
        scheduled[s]['ues'] = {}
        for u in user_scheduled[s]:
          data_sent[u] = 0
          usr = self.attachedUEs[u]
          if user_scheduled[s][u] > 0:
            user_scheduled[s][u] *= RB_PER_TTI
            sent[u] = Util.get_tbs(usr['mcs'], user_scheduled[s][u], self.simulation.tti)
            _, data_sent[u] = usr['ue'].dataTxRx(usr['sinr'], usr['mcs'], sent[u] / 8, user_scheduled[s][u])
          scheduled[s]['ues'][u] = {}
          scheduled[s]['rbs'] += user_scheduled[s][u]
          scheduled[s]['ues'][u]['rbs'] = user_scheduled[s][u]
          scheduled[s]['bytes'] += data_sent[u]
          scheduled[s]['ues'][u]['bytes'] = data_sent[u]
          scheduled[s]['ues'][u]['mcs'] = usr['mcs']

      if self.active:
        self.register_metrics(slice_traffic, scheduled, rtx_scheduled)
        self.update_scheduler()
      # End if mini_slot

    sent_by_slice = {}
    for s in self.slices:
      sent_by_slice[s] = 0
    for uid in self.attachedUEs:
      if uid in sent:
        given_bits = sent[uid]
        tx_bits = data_sent[uid] * 8
      else:
        given_bits = 0
        tx_bits = 0
      sent_by_slice[self.attachedUEs[uid]['ue'].slice.id] += tx_bits
      self.attachedUEs[uid]['average_tp'] = AVG_TP_ALPHA * (given_bits / self.tti) \
                                            + (1 - AVG_TP_ALPHA) * self.attachedUEs[uid]['average_tp']

    if self.active and self.ts >= 0:
      self.metrics['x'].append(self.ts * self.simulation.tti * 1000)
      for s in self.simulation.slices:
        if s not in sent_by_slice:
          sent_by_slice[s] = 0
        if s not in slice_share:
          slice_share[s] = 0
      self.metrics['bits'].append(sent_by_slice)
      self.metrics['rbs'].append(slice_share)

    self.simulation.logging.info(slice_share)
    self.simulation.logging.info(scheduled)
    self.simulation.logging.info(sent)
    self.simulation.logging.info("")
    self.ts += 1
