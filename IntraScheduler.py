from NRTables import MCS
import copy

SMALL = 0.0000001


class IntraScheduler:

  def schedule(self, ue_info, ue_traffic, available_rb):
    pass


class SchedulerPF(IntraScheduler):

  def __init__(self, allocation_unit=4):
    self.allocation_unit = allocation_unit

  def scheduleOne(self, ue_info, ue_traffic, available_rb):
    max_w = 0
    max_ue = None
    for uid in ue_traffic:
      if ue_traffic[uid]['rbs'] > 0:
        # ue = ue_traffic[uid]
        info = ue_info[uid]
        avg = info['average_tp'] if 'average_tp' in info else 1
        if avg == 0:
          avg = SMALL
        w = MCS[info['mcs']]['se'] / avg
        if w > max_w:
          max_w = w
          max_ue = uid
    if max_ue is not None:
      amount = min(ue_traffic[max_ue]['rbs'], available_rb)
      return max_ue, amount
    else:
      return None, None

  def schedule(self, ue_info, ue_traffic, available_rb):
    rb_to_sched = available_rb
    sched = {}
    for u in ue_info:
      sched[u] = 0
    traffic = copy.deepcopy(ue_traffic)
    uet = copy.deepcopy(traffic)
    while rb_to_sched > 0:
      if self.allocation_unit is not None:
        sched_unit = min(rb_to_sched, self.allocation_unit)
      else:
        sched_unit = rb_to_sched
      uid, amount = self.scheduleOne(ue_info, uet, sched_unit)
      if uid is not None:
        traffic[uid]['rbs'] -= amount
        sched[uid] += amount
        rb_to_sched -= amount
        del uet[uid]
      elif len(uet) == len(traffic):
        # no more traffic to schedule
        break
      else:
        del uet
        uet = copy.deepcopy(traffic)
      assert rb_to_sched >= 0
      if len(uet) == 0:
        del uet
        uet = copy.deepcopy(traffic)
    return sched
