from fractions import Fraction

from NRTables import MCS, TBS, PC_TBS, MCS2CQI, PC_TBS_short
from scipy.stats import norm
import math, statistics
import numpy as np

from functools import reduce

import Medium


# used to generate PC_TBS
def getTBSSize(mcs, rbs, nred=12 * 14, layers=1, dmrs=12):
  npre = nred - dmrs
  nre = min(npre, 156) * rbs
  r = MCS[mcs]['R']
  qm = MCS[mcs]['Qm']
  ninfo = nre * (r / 1024) * qm * layers
  if ninfo <= 3824:
    kb = 3824
    try:
      n = max(3, int(math.log2(ninfo)) - 6)
    except ValueError:
      n = 3
    npinfo = max(24, pow(2, n) * int(ninfo / pow(2, n)))
    _, tbs = treeClosestGE(npinfo, TBS)
    no_cbs = 1
  else:
    kb = 8424
    n = int(math.log2(ninfo - 24)) - 5
    npinfo = max(3840, pow(2, n) * round((ninfo - 24) / pow(2, n)))
    if r / 1024 < 0.25:
      c = math.ceil((npinfo + 24) / 3816)
      tbs = 8 * c * math.ceil((npinfo + 24) / (8 * c)) - 24
      no_cbs = c
    elif npinfo > 8424:
      c = math.ceil((npinfo + 24) / 8424)
      tbs = 8 * c * math.ceil((npinfo + 24) / (8 * c)) - 24
      no_cbs = c
    else:
      tbs = 8 * math.ceil((npinfo + 24) / 8) - 24
      no_cbs = 1

  return tbs, no_cbs


def get_cbs(mcs, rbs):
  A = PC_TBS[mcs][rbs - 1]
  r = MCS[mcs]['R'] / 1024
  if A <= 292 or (A <= 3824 and r <= 0.67) or r <= 0.25:
    bg = 2
    K_cb = 3840
  else:
    bg = 1
    K_cb = 8448

  # plus global CRC
  B = A + 24

  # calculating C
  if B <= K_cb:
    L = 0
    C = 1
    Bp = B
  else:
    L = 24
    C = np.ceil(B / (K_cb - L))
    Bp = B + C * L

  Kp = Bp / C
  if bg == 1:
    K_b = 22
  else:
    if B > 640:
      K_b = 10
    elif B > 560:
      K_b = 9
    elif B > 192:
      K_b = 8
    else:
      K_b = 6

  min_lifting_sizes = np.zeros(8)
  min_lifting_sizes[0] = 2
  min_lifting_sizes[1:] = np.arange(3, 16, 2)
  max_lifting_size = 384

  Z_c = None
  iLS = None

  for i in range(0, len(min_lifting_sizes)):
    zc = min_lifting_sizes[i]
    while zc <= max_lifting_size:
      if K_b * zc >= Kp and (Z_c == None or zc < Z_c):
        Z_c = int(zc)
        iLS = i
      zc *= 2

  if bg == 1:
    K = 22 * Z_c
  else:
    K = 10 * Z_c

  return K, r, bg, iLS, Z_c, C


def getSDUHeader(size):
  assert size <= 65535
  if size <= 255:
    return 2
  else:
    return 3


def getSDUPayload(size):
  assert size != 258
  if size <= 257:
    return size - 2
  else:
    return size - 3


def get_required_rbs(mcs, buf_size, tti):
  amount = buf_size * 8
  pc = get_pc(tti)
  if amount == 0:
    return 0
  if amount > pc[mcs][-1]:
    return len(pc[mcs])
  rb, _ = treeClosestGE(amount, pc[mcs])
  rb = rb + 1  # PC_TBS[0] means 1 RB
  return rb


def get_tbs(mcs, rbs, tti):
  pc = get_pc(tti)
  assert mcs in pc and rbs <= len(pc[mcs])
  return pc[mcs][rbs-1]


def get_pc(tti=0.001):
  pc = None
  if tti == 0.001:
    pc = PC_TBS
  elif tti < 0.0002:
    pc = PC_TBS_short
  assert pc is not None, "TTI not supported"
  return pc


def treeClosestGE(tbs, arr, base=0):
  mid = len(arr) // 2
  if len(arr) == 1:
    return base, arr[0]
  if arr[mid] >= tbs:
    if arr[mid - 1] < tbs:
      return base + mid, arr[mid]
    return treeClosestGE(tbs, arr[0:mid], base)
  else:
    return treeClosestGE(tbs, arr[mid + 1:], base + mid + 1)


def getRXProbability(snr, mcs, tbs, symbol_time=10 ** -3 / 14, bandwidth=15 * 10 ** 3, fec=0):
  # Adapted from Matlab scheduler
  # return min(0.6+MCS2CQI[mcs]/30,0.95)

  # Theoretical AWGN for modulations used
  # l_snr = pow(10,snr/10)
  l_snr = 10 ** (snr / 10)
  # es_n0 = symbol_time * bandwidth * l_snr
  es_n0 = l_snr
  bps = MCS[mcs]['Qm']
  eb_n0 = es_n0 / bps
  # eb_n0 = es_n0
  if bps == 2:
    ber = norm.sf(np.sqrt(2 * eb_n0))
  else:
    ber = (4 / bps) * norm.sf(np.sqrt((3 * eb_n0 * bps) / (2 ** bps - 1)))
    # ber = ((4 * (np.sqrt(2**bps) - 1)) / ((np.sqrt(2**bps) * bps)) * norm.sf(np.sqrt((3 * eb_n0 * bps) / (2 ** bps - 1))))
  return (1 - ber) ** tbs


def getShannonRxProbability(snr, mcs, rbs):
  K, _, _, _, _, C = get_cbs(mcs, rbs)
  snr_lin = 10 ** (snr / 10)
  exponent = -1.5 * snr_lin / (2 ** MCS[mcs]['se'] - 1)
  ber = np.exp(exponent) / 5
  bler = 1 - (1 - ber) ** K
  return (1 - bler) ** C


# cache for MCS calculation
mcs_cache = {}


def getShannonMCS(snr, ber=0.00005, rbs=1, bler=None):
  tpl = (snr, ber, rbs, bler)
  if tpl in mcs_cache:
    return mcs_cache[tpl]
  mcs_ber = np.repeat(ber, len(MCS))
  if bler is not None:
    for m in MCS:
      if m > 27:
        break
      K, _, _, _, _, C = get_cbs(m, rbs)
      mcs_ber[m] = 1 - (1 - bler) ** K

  phy = - np.log(5 * mcs_ber) / 1.5
  shn = np.log2(1 + 10 ** (snr / 10) / phy)
  chosen_mcs = 0
  for m in MCS:
    if m > 27:
      break
    se = MCS[m]['se']
    shn_m = shn[m]
    if se < shn_m:
      chosen_mcs = m
    else:
      break

  mcs_cache[tpl] = chosen_mcs
  return chosen_mcs


def getMeanCIPair(list):
  if len(list) == 0:
    return (0, 0)
  avg = statistics.mean(list)
  std = statistics.pstdev(list)
  lnt = len(list)
  return (avg, 1.96 * std / math.sqrt(lnt))


def orderLegend(handles, labels, order):
  assert len(handles) == len(labels) and len(labels) == len(order)
  reordered_handles = []
  reordered_labels = []
  for i in order:
    reordered_handles.append(handles[i])
    reordered_labels.append(labels[i])
  return reordered_handles, reordered_labels


def filterValues(x, y, x_interval):
  filter = (x >= x_interval[0]) * (x <= x_interval[1])
  return x[filter], y[filter]


def plotCDF(list, plt, style, label, bins=100):
  count, edges = np.histogram(list, bins=bins, density=True)
  cdf = np.cumsum(count)
  plt.plot(edges[1:], cdf / cdf[-1], style, label=label)
  plt.set_ylim(0, 1)


def choose(n, k):
  if k > n // 2: k = n - k
  p = Fraction(1)
  for i in range(1, k + 1):
    p *= Fraction(n - i + 1, i)
  return int(p)


def isSliceFullBuffer(s, ue_conf):
  '''
  Checks if Slice s has UEs with full_buffer
  :param s: slice id
  :param ue_conf: UE configurations
  :return: True if at least one UE configuration belonging to that slice is full_buffer
  '''
  for u in ue_conf:
    if u['traffic'] == 'full_buffer' and u['slice'] == s:
      return True
  return False

def getSINRfromDistance(d2d, los=True, env="macro", config_params=None):
  params = {
    'h_ue': 1.5,  # m
    'h_enb': 20,  # m
  }
  if config_params is not None:
    params.update(config_params)
  enb = Medium.ENB(params['h_enb'])
  ue = Medium.UE(params['h_ue'])
  snr_calculator = Medium.SNRCalculator(None, enb, ue)
  lobes = [0, 120, 240]
  sinr_array = []
  if d2d == 0:
    d2d = 0.00001
  # check all antenna lobes and return the maximum
  for lobe in lobes:
    sinr_array.append(snr_calculator.getSNR((0, 0, params['h_enb']), (0, d2d, params['h_ue']), lobe, pre_los=los))
  return max(sinr_array)


def slice_metric_aggregator(values, fun=np.mean):
  keys = set()
  for v in values:
    for k in v:
      keys.add(k)
  return dict(map((lambda s: (s, fun(list(map((lambda x: x[s] if s in x else 0), values))))), keys))

def check_reliability(unreliable, total, target):
  if unreliable == 0 or (unreliable == 1 and target != 0):
    return True
  elif unreliable/total > target:
    return False



class CircularIterator:

  def __init__(self, lst, root=0):
    self.lst = lst
    self.root = root

  def __iter__(self):
    self.current = self.root
    self.first = True
    return self

  def __next__(self):
    if len(self.lst) == 0:
      raise StopIteration
    curr = self.current
    self.current = (self.current + 1) % len(self.lst)
    if curr == self.root and not self.first:
      raise StopIteration
    else:
      self.first = False
      return self.lst[curr]


class BucketAdmission:
  def __init__(self, rate, capacity, tti):
    self.rate = rate * 1000  # rate in kbytes per second
    self.capacity = capacity
    self.tti = tti
    self.tokens = capacity
    self.ts = 1

  def tick(self):
    tokens_to_add = int(self.rate * self.tti)
    self.tokens = min(self.tokens + tokens_to_add, self.capacity)

  def filter_packet(self, size):
    if self.tokens - size > 0:
      self.tokens -= size
      accepted = True
    else:
      accepted = False
    return accepted


class FineArray:
  def __init__(self, x_interval=100, aggr_fun=np.mean, error_fun=np.std):
    self.fine_values = []
    self.this_x = None
    self.last_x = None
    self.values = []
    self.err = []
    self.x = []
    self.aggr_fun = aggr_fun
    self.error_fun = error_fun
    self.x_interval = x_interval

  def __getstate__(self):
    obj = self.__dict__
    obj['aggr_fun'] = None
    obj['error_fun'] = None
    return obj


  def fineAppend(self, x, y):
    self.this_x = x if self.this_x is None else self.this_x
    aggregated_value = (None, None, None)
    if x - self.this_x >= self.x_interval:
      self.x.append(self.this_x)
      value = self.aggr_fun(self.fine_values)
      self.values.append(value)
      if self.error_fun is not None:
        err = self.error_fun(self.fine_values)
        self.err.append(err)
      else:
        err = None
      self.this_x = x
      self.fine_values = []
      aggregated_value = (self.this_x, value, err)
    self.last_x = x
    self.fine_values.append(y)
    return aggregated_value

  def append(self, y):
    if self.last_x is None:
      x = 0
    else:
      x = self.last_x + 1
    self.fineAppend(x, y)

  def getValues(self):
    return self.values

  def getFineArray(self):
    return self.fine_values

  def getX(self):
    return self.x

  def getError(self):
    return self.err

  def get_x_and_y(self):
    return {
      'x': self.x,
      'y': self.values
    }

  def __getitem__(self, item):
    return self.values[item]

  def __len__(self):
    return len(self.values)

  def getLastValues(self, x_gap):
    last_n = int(x_gap/self.x_interval)
    return self.values[-last_n:]
