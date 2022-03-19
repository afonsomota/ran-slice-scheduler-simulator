import numpy as np
import statistics
import math
import glob

import Util

FB_SIZE = 100
MV_STEP = 0.01
NO_PROCESSES = 16


class HarqProcesses:

  def __init__(self, ue, max_rv=3, no_processes=NO_PROCESSES, response_time=4, tti=0.001):
    self.processes = {}
    self.max_rv = max_rv
    self.no_processes = no_processes
    self.response_time = response_time
    self.corrupted_packets = []
    self.ue = ue

  def registerSuccess(self, pid):
    del self.processes[pid]

  def registerFailure(self, pid, packets, size, mcs, ts):
    '''
    Registers a transmission failure. May be of a new transmission or a retransmission.
    Returns true if transmission has been made 'max_rv' times and the PDU transmission
      is considered as an error
    '''
    self.ue.metrics['retx']['count'] += 1
    # TODO block if pid occupied!
    if pid not in self.processes:
      self.processes[pid] = {'rv': 0, 'size': size, 'pid': pid, 'mcs': mcs, 'packets': packets}
    proc = self.processes[pid]
    for type in packets:
      assert len(packets[type]) == len(proc['packets'][type]), f"{packets[type]} AND {proc['packets'][type]}"
      for i, pkt in enumerate(packets[type]):
        assert pkt['id'] == proc['packets'][type][i]['id'], "IDs %d %d" % (pkt['id'], proc['packets'][type][i]['id'])
        if type == 'segmented':
          assert pkt['no_segments'] == proc['packets'][type][i]['no_segments'], "Segments %d %d" % (
            pkt['no_segments'], proc['packets'][type][i]['no_segments'])
    assert proc['size'] == size, "%d: %d vs %d and %d vs %d" % (self.ue.id, proc['mcs'], mcs, proc['size'], size)
    proc['rv'] += 1
    if proc['rv'] == self.max_rv:
      for p in self.processes[pid]['packets']['segmented']:
        self.corrupted_packets.append(p)
      del self.processes[pid]
      return True
    else:
      self.processes[pid]['rtx'] = ts + self.response_time
      return False

  def checkRetransmissions(self, ts):
    '''
      Checks if there are retransmissions this timestamp (ts)
      Returns:
        - True/False - if there is a retransmission to do
        - Size - size of retrannsmission
      '''
    rtxs = []
    for pid in self.processes:
      if np.isclose(self.processes[pid]['rtx'], ts) or self.processes[pid]['rtx'] < ts:
        # print(self.ue.id,self.processes[pid]['rtx'],self.processes[pid])
        rtxs.append(self.processes[pid])
    if len(rtxs) > 0:
      return True, rtxs
    else:
      return False, None


class UE:

  def __init__(self, ue_id, sim, sli, traffic_type='full_buffer', params=None, movement=None, start_ts=0,
               generator=None):
    self.id = ue_id
    self.type = traffic_type
    self.params = params
    self.inside_burst = False
    self.active_burst = 0
    self.burst_count = 0
    self.last = 0
    self.next = 0
    self.tti = sim.tti
    self.sim = sim
    self.start_ts = start_ts
    self.ts = start_ts
    self.logging = False
    if generator is not None:
      # self.traffic_generator = np.random.Generator(np.random.PCG64(params['traffic_seed']))
      self.traffic_generator = generator
    else:
      self.traffic_generator = np.random.Generator(np.random.PCG64())
    if 'start' not in params:
      self.calculateNext()
    else:
      self.next = start_ts + params['start'] / self.tti
    self.slice = sli
    self.sinr = params.get('sinr', 80)
    self.metrics = {
      'sinr': Util.FineArray(self.sim.conf['sinr_store_window']),
      'mcs': Util.FineArray(self.sim.conf['sinr_store_window']),
      'packets': {},
      'rejected-packets': {},
      'retx': {'count': 0},
      'slice': sli.id
    }
    self.packet_id = 0
    self.segmented_packets = {}
    self.failed_packets = set()
    if self.type == 'full_buffer':
      self.buffer = [{'size': params['size']}]
    else:
      self.buffer = []
      assert params != None
    if self.sim.conf.get('mini-slot', False) and self.slice.type == 'P':
      self.harqProcesses = HarqProcesses(self, response_time=2. / 7)
    else:
      self.harqProcesses = HarqProcesses(self)
    self.active = True
    self.mv_index = 0
    self.movement = None
    self.fixed_mcs = params.get("mcs")
    if movement is not None:
      self.movement_array = []
      total_time = 0
      mv_files = glob.glob(movement)
      while True:
        # it is not traffic but is also supposed to be the same accross comparative simulations
        r = self.traffic_generator.integers(0, len(mv_files))
        fn = mv_files[r]
        mov_file = open(fn, 'r')
        no_lines = sum(1 for _ in mov_file)
        mov_file.seek(0)
        if no_lines < 10:
          continue
        if total_time == 0:
          self.movement = mov_file
          print("Initial file UE id %d, file %s, lines %d" % (self.id, self.movement.name, no_lines))
          self.sinr = self.getNextSINR()
        else:
          print("Added file UE id %d, file %s, lines %d" % (self.id, self.movement.name, no_lines))
          self.movement_array.append(mov_file)
        total_time += no_lines
        if total_time >= sim.timeout / MV_STEP:
          break
    if movement is not None or 'sinr_fun' in params or ('mcs' not in params and 'sinr' in params):
      self.link_adaptation = True
    else:
      self.link_adaptation = False

  def getNextSINR(self):
    if self.movement:
      if self.ts % (1 / MV_STEP) == 0:
        line = self.movement.readline()
        if line is None or len(line.split(",")) < 4:
          self.movement = self.movement_array.pop(0)
          print("Change file UE id %d, file %s" % (self.id, self.movement.name))
          line = self.movement.readline()
        snr_db = float(line.split(",")[3]) - 30
        if self.logging:
          self.metrics['sinr'].append(snr_db)
        return snr_db
      else:
        return self.sinr
    elif 'sinr_fun' in self.params:
      snr_db = self.params['sinr_fun'](self.sinr, self.ts, self)
      if self.logging:
        self.metrics['sinr'].append(snr_db)
      return snr_db
    else:
      if self.logging:
        self.metrics['sinr'].append(self.sinr)
      return self.sinr

  def triggerBurst(self, global_ts=None):
    if global_ts is None:
      global_ts = self.ts * self.tti
    if 'bursts' in self.params and not self.inside_burst:
      i = 0
      for b in self.params['bursts']:
        if b['start'] < global_ts < b['end']:
          if self.traffic_generator.uniform() < b['probability']:
            self.burst_count = 0
            self.active_burst = i
            self.inside_burst = True
          break
        i += 1

  def triggerActivity(self, global_ts=None):
    if global_ts is not None and global_ts >= 0:
      self.logging = True
    if self.active:
      if (('start' in self.params and global_ts < self.params['start']) or \
          ('end' in self.params and global_ts > self.params['end'])):
        self.active = False
    elif not self.active and \
        ('start' in self.params and global_ts > self.params['start']) and \
        ('end' in self.params and global_ts < self.params['end']):
      self.active = True
      self.calculateNext()

  def checkActivity(self):
    if self.active:
      return True
    else:
      if self.type == 'full_buffer':
        empty_buffer = 'leftover' not in self.buffer[0]
      else:
        empty_buffer = len(self.buffer) == 0
      empty_harq = len(self.harqProcesses.processes) == 0
      return not (empty_buffer and empty_harq)

  def getMetrics(self):
    received = tuple(filter(lambda x: 'recv' in x, self.metrics['packets'].values()))
    no_received = len(received)
    if no_received == 0:
      return {'delivery': (0, 0), 'delay': (None, None), 'tp': (None, None)}
    delivery = no_received / len(self.metrics['packets'])
    if self.type == 'full_buffer':
      delay = (None, None)
    else:
      delays = tuple(map(lambda x: (x['recv'] - x['sent']), received))
      avg = statistics.mean(delays)
      std = statistics.pstdev(delays)
      delay = (avg, 1.96 * std / math.sqrt(no_received))

    tps = tuple(map(lambda x: (x['size'] / (x['recv'] - x['scheduled']) / self.tti), received))
    t_avg = statistics.mean(tps)
    t_std = statistics.pstdev(tps)
    tp = (t_avg, 1.96 * t_std / math.sqrt(no_received))

    if len(self.metrics['sinr']) != 0:
      sinr_avg = statistics.mean(self.metrics['sinr'])
      sinr_ci = 1.96 * statistics.pstdev(self.metrics['sinr'])
    else:
      sinr_avg = 0
      sinr_ci = 0

    return {'delivery': (delivery, 0), 'delay': delay, 'tp': tp, 'sinr': (sinr_avg, sinr_ci)}

  def calculateNext(self):
    last = self.next
    if not self.inside_burst and 'bursts' in self.params:
      self.triggerBurst()
    if self.type == 'full_buffer':
      return
    elif self.type == 'poisson':
      if self.inside_burst:
        avg_int = self.params['bursts'][self.active_burst]['interval']
        self.burst_count += 1
        if self.burst_count > self.params['bursts'][self.active_burst]['total']:
          self.inside_burst = False
      else:
        avg_int = self.params['lambda']
      lbd = avg_int / self.tti
      # exponential gives time between two events following poisson distribution
      interval = 0
      while interval == 0:
        interval = self.traffic_generator.exponential(lbd)
        assert interval >= 0
      self.next = last + interval
    elif self.type == 'cbr':
      if self.inside_burst:
        avg_int = self.params['bursts'][self.active_burst]['interval']
        self.burst_count += 1
        # normalize jitter
        avg_jitter = avg_int * self.params['jitter'] / self.params['interval']
        if self.burst_count > self.params['bursts'][self.active_burst]['total']:
          self.inside_burst = False
      else:
        avg_int = self.params['interval']
        avg_jitter = self.params['jitter']
      times = round((last - self.start_ts) / (avg_int / self.tti))
      jitter = self.traffic_generator.uniform(-avg_jitter, avg_jitter)
      self.next = (times + 1) * (avg_int / self.tti) + jitter / self.tti + self.start_ts
    else:
      assert 1 == 0
    assert self.next > last
    self.last = last

  def tick(self, ts=None):
    last_ts = self.ts
    if ts is None:
      self.ts = int(self.ts + 1)
    else:
      self.ts += ts
    if self.type != 'full_buffer' and self.active and self.ts >= self.next:
      while self.next <= self.ts:
        packet = {'size': self.getSize(), 'sent': self.next / self.tti / 1000, 'id': self.packet_id, 'new': True, 'on_hold': 0}
        self.calculateNext()
        self.buffer.append(packet)
        # self.metrics['packets'][self.packet_id] = packet.copy()
        self.packet_id += 1
      for p in self.buffer:
        p['on_hold'] += (self.ts - last_ts)
    self.sinr = self.getNextSINR()

  def getSize(self):
    if self.type == 'full_buffer':
      return 0
    else:
      return self.params['size']

  def getBuffer(self):
    if self.type == 'full_buffer':
      if self.active:
        buffer = self.buffer * FB_SIZE
      else:
        buffer = self.buffer if 'leftover' in self.buffer[0] else []
    else:
      buffer = self.buffer
    if len(buffer) == 0:
      return 0
    else:
      total = 1  # PDU header
      for p in list(buffer):  # iterate over copy of list to remove elements
        if 'admitted' not in p:
          admitted = True
          if 'admission' in self.slice.target:
            assert 'new' in p and p['new'], f"{p}, {self.id}, {self.slice.id}, {self.slice.target}"
            admitted = self.slice.admission.filter_packet(p['size'])
          p['admitted'] = admitted
          p['new'] = False
        if p['admitted']:
          key = 'size' if 'leftover' not in p else 'leftover'
          total += p[key] + Util.getSDUHeader(p[key])
        else:
          print("PACKET DISCARDED ", self.id, self.ts, p['id'])
          self.registerFailure({'full': [p], 'segmented': []})
          p_copy = p.copy()
          self.metrics['rejected-packets'][p['id']] = p_copy
          buffer.remove(p)
      for p in self.buffer:
        assert p['new'] is False
      return total

  def popBuffer(self, amount):
    packets = {'full': [], 'segmented': []}
    sent = 0
    while sent < amount:
      if self.getBuffer() == 0:
        break
      packet = self.buffer[0]
      if 'leftover' in packet:
        p_size = packet['leftover']
        p_type = 'segmented'
        packet['no_segments'] += 1
      else:
        p_size = packet['size']
        p_type = 'full'
        packet['no_segments'] = 1
        if self.type == 'full_buffer':
          packet['id'] = self.packet_id
          self.packet_id += 1
      if packet['id'] not in self.metrics['packets']:
        p_copy = packet.copy()
        p_copy['scheduled'] = self.ts * self.sim.tti * 1000
        self.metrics['packets'][packet['id']] = p_copy
      else:
        self.metrics['packets'][packet['id']]['no_segments'] = packet['no_segments']
      size = p_size + Util.getSDUHeader(p_size)
      if sent + size <= amount:
        p_copy = packet.copy()
        if p_type == 'segmented':
          p_copy['segment'] = p_size
        packets[p_type].append(p_copy)
        sent += size
        if p_type == 'segmented':
          self.segmented_packets[packet['id']]['segments_sent'] += p_size
        if self.type != 'full_buffer':
          self.buffer.pop(0)
        else:
          if 'leftover' in packet:
            del packet['leftover']
            del packet['no_segments']
      else:
        left = amount - sent
        body_sent = left - (size - p_size)
        # no room for header
        if body_sent <= 0:
          break
        sent = amount
        packet['leftover'] = p_size - body_sent
        c_packet = packet.copy()
        c_packet['segment'] = body_sent
        packets['segmented'].append(c_packet)
        if packet['id'] not in self.segmented_packets:
          sp = {'id': packet['id'], 'segments_sent': 0, 'segments_recv': 0}
          self.segmented_packets[packet['id']] = sp
        self.segmented_packets[packet['id']]['segments_sent'] += body_sent
    return sent, packets

  def attachClosest(self, cells):
    if self not in cells[0].attachedUEs:
      cells[0].attachUE(self)

  def checkRetransmissions(self):
    return self.harqProcesses.checkRetransmissions(self.ts)

  def registerPacketSuccess(self, packet):
    if self.active and self.logging:
      self.slice.updatePacketMetrics(packet)

  def registerSuccess(self, hp_id, packets, mini_slot=None):
    if hp_id is not None:
      self.harqProcesses.registerSuccess(hp_id)
    complete = False
    for p in packets['full']:
      self.metrics['packets'][p['id']][
        'recv'] = self.ts * self.sim.tti * 1000 + self.harqProcesses.response_time * self.sim.tti * 1000
      self.registerPacketSuccess(self.metrics['packets'][p['id']])
    for p in packets['segmented']:
      if 'segments_received' not in self.segmented_packets[p['id']]:
        self.segmented_packets[p['id']]['segments_received'] = 0
        self.segmented_packets[p['id']]['segments_received_lst'] = []
      self.segmented_packets[p['id']]['segments_received'] += p['segment']
      self.segmented_packets[p['id']]['segments_received_lst'].append(p['segment'])
      assert self.segmented_packets[p['id']]['segments_received'] <= p['size']
      if self.segmented_packets[p['id']]['segments_received'] == p['size']:
        self.metrics['packets'][p['id']][
          'recv'] = self.ts * self.sim.tti * 1000 + self.harqProcesses.response_time * self.sim.tti * 1000
        self.registerPacketSuccess(self.metrics['packets'][p['id']])
        del self.segmented_packets[p['id']]

  def registerFailure(self, packets):
    if self.active and self.logging:
      for k in ['full', 'segmented']:
        for p in packets[k]:
          if p['id'] not in self.failed_packets:
            if 'admitted' not in p or p['admitted']:
              self.slice.updatePacketMetrics(self.metrics['packets'][p['id']])
            self.failed_packets.add(p['id'])

  def dataTxRx(self, snr, mcs, size, rbs, hp=None, mini_slot=None):
    if hp is None:
      sent, packets = self.popBuffer(size)
      hp_pid = None
    else:
      packets = hp['packets']
      sent = None
      hp_pid = hp['pid']

    succ_prob = Util.getShannonRxProbability(snr, mcs, rbs, mini_slot)
    # print(succ_prob)
    if np.random.uniform() >= succ_prob:
      if hp is None:
        if mini_slot is None:
          hp_pid = int(self.ts) % NO_PROCESSES
        else:
          hp_pid = round(self.ts*7) % NO_PROCESSES
        is_final = self.harqProcesses.registerFailure(hp_pid, packets, rbs, mcs, self.ts)
      else:
        # TODO block if pid occupied
        is_final = self.harqProcesses.registerFailure(hp_pid, packets, rbs, mcs, self.ts)
      if is_final:
        self.registerFailure(packets)
      return False, sent
    else:
      self.registerSuccess(hp_pid, packets,  mini_slot=None)
      return True, sent

  def failed_by_puncture(self, mcs, size, rbs, mini_slot=None):
    sent, packets = self.popBuffer(size)
    if mini_slot is None:
      hp_pid = int(self.ts) % NO_PROCESSES
    else:
      hp_pid = round(self.ts * 7) % NO_PROCESSES
    self.harqProcesses.registerFailure(hp_pid , packets, rbs, mcs, self.ts)
    return False, sent
