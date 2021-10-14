import math

from Simulator import Simulation
from IntraScheduler import SchedulerPF
import Util
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats
from NRTables import PC_TBS
import pickle
from scipy.stats import gaussian_kde
import scipy

los = True

def get_marker(idx):
  markers = []
  i = 0
  for m in Line2D.markers:
    try:
      if len(m) == 1 and m != ' ':
        if i == idx % len(Line2D.markers):
          return m
        else:
          i += 1

    except TypeError:
      pass



def add_distance_plot(x_array,ax1):
  pass
  #secax = ax1.secondary_xaxis('top',functions=(t2d,d2t))
  #secax.set_xlabel("Distance (m)")
  #ax2 = ax1.twiny()
  #x_stamps = []
  #distances = []
  #print("distances")
  #for xt in x_array:
  #  d = int(get_distance_from_time(xt))
  #  if d % 50 == 0 and d not in distances:
  #    print(xt/1000, d)
  #    x_stamps.append(xt)
  #    distances.append(d)
  #ax2.set_xlim(ax1.get_xlim())
  #absolute_distances = abs(np.array(distances)/1000)
  #ax2.set_xticks(x_stamps)
  #ax2.set_xticklabels(absolute_distances)
  #ax2.set_xlabel("Distance (m)")


def plot_time_ue_mcs(ue_metrics,plt):
  plt.set_xlabel("Simulation time (s)")
  plt.set_ylabel("Used MCS")

  x = np.array(ue_metrics[0]['mcs'].x)
  ys = []

  for u in ue_metrics:
    usr = ue_metrics[u]
    if usr['slice'] == 0:
      y = np.array(usr['mcs'],dtype='int')
      ys.append(y)
      #plt.plot(np.array(usr['mcs'].x)/1000, usr['mcs'],label=u,ls='None',marker=get_marker(u),markerfacecolor='none')

  mcs_sum = np.zeros((len(x), 28))
  for y in ys:
    #mcs_sum[:][y[:]] += 1
    for x_i in range(0,len(x)):
      mcs_sum[x_i][y[x_i]] += 1


  no_x = len(x)
  x = x.repeat(len(ys))
  y = np.zeros(no_x*len(ys))
  z = np.zeros(no_x*len(ys))

  for (i,r) in enumerate(ys):
    y[range(0+i,len(y),len(ys))] = r
    for x_i in range(0,len(ue_metrics[0]['mcs'].x)):
      z[0+i + len(ys)*x_i] = mcs_sum[x_i][r[x_i]]



  #ys = np.array(ys).transpose()

  #xy = np.vstack([x,y])
  #z = gaussian_kde(xy)(xy)

  idx = z.argsort()
  x, y, z = x[idx], y[idx], z[idx]

  scatter = plt.scatter(x/1000, y, c=z, s=50, edgecolors='none', label= z)#, alpha=0.6)
  #plt.scatter(x, y, s=10*z)
  plt.legend(*scatter.legend_elements(num=8), ncol=4, columnspacing=1,loc='upper center', bbox_to_anchor=(0.47, -0.2), title="Number of UEs")
  # ax.legend(loc='upper center', bbox_to_anchor=(0.47, -0.2), ncol=3, columnspacing=1)

  add_distance_plot(ue_metrics[0]['mcs'].x,plt)
  plt.set_ylim(0,28)


def plot_time_ipunt_traffic(slice_styles, slice_labels, sizes, plt):
  # input traffic time series
  # plt.set_title("Input traffic time series")
  plt.set_xlabel("Simulation time (s)")
  plt.set_ylabel("Input traffic (Kbytes)")
  binned = {}
  edges = {}
  ymax = 0.000001
  for s in slice_styles:
    if Util.isSliceFullBuffer(s, ue_conf):
      continue
    binned[s], edges[s], _ = stats.binned_statistic(sizes[s]['x'], sizes[s]['y'],
                                                    'sum', bins=100)
    binned[s] = binned[s] / 1000
  for s in binned:
    smax = max(binned[s])
    if smax > ymax: ymax = smax
    plt.plot(edges[s][1:] / 1000, binned[s], slice_styles[s], label=slice_labels[s])
  plt.set_ylim(0, 1.1 * ymax)
  # plt.legend(loc="best")


def plot_time_overuses(slice_styles, slice_labels, results, plt):
  # overuses time series
  plt.set_title("Overused resources per slice")
  plt.set_xlabel("Simulation time (s)")
  plt.set_ylabel("Unused resoureces (RBs)")
  ymax = 0.000001
  # plt.plot(results[1]['overuses']['x'], results[1]['overuses']['y'], results[2]['overuses']['x'],
  #         results[2]['overuses']['y'])
  for s in [2, 0, 1]:
    smax = max(results[s]['overuses']['y'])
    if smax > ymax: ymax = smax
    binned, edges, _ = stats.binned_statistic(results[s]['overuses']['x'], results[s]['overuses']['y'],
                                              'mean', bins=100)
    plt.plot(edges[1:], binned, slice_styles[s], label=slice_labels[s])
  plt.set_ylim(0, 1.1 * ymax)
  plt.legend(loc="best")


def plot_time_excesses(slice_styles, slice_labels, results, plt):
  # excesses time series
  plt.set_title("Unused resources per slice")
  plt.set_xlabel("Simulation time (s)")
  plt.set_ylabel("Unused resoureces (RBs)")
  ymax = 0.000001
  # plt.plot(results[1]['excesses']['x'], results[1]['excesses']['y'], results[2]['excesses']['x'],
  #         results[2]['excesses']['y'], results[0]['excesses']['x'], results[0]['excesses']['y'])
  for s in slice_styles:
    smax = max(results[s]['excesses']['y'])
    if smax > ymax: ymax = smax
    binned, edges, _ = stats.binned_statistic(results[s]['excesses']['x'], results[s]['excesses']['y'],
                                              'mean', bins=100)
    plt.plot(edges[1:], binned, slice_styles[s], label=slice_labels[s])
  plt.set_ylim(0, 1.1 * ymax)
  plt.legend(loc="best")


def plot_time_sched_kbps(slice_styles, slice_labels, results, tti, plt):
  # bps time series
  # plt.set_title("Slice throughput at scheduler time series")
  plt.set_xlabel("Simulation time (s)")
  plt.set_ylabel("Slice throughput (Mbps)")
  binned = {}
  edges = {}
  ymax = 0.000001
  for s in slice_styles:
    #if s != 0: continue
    binned[s], edges[s], _ = stats.binned_statistic(results[s]['cx'], results[s]['bits']['total'] / tti / 1000,
                                                    'mean', bins=100)
  for s in binned:
    smax = max(binned[s])
    if smax > ymax: ymax = smax
    plt.plot(edges[s][1:] / 1000, binned[s] / 1000, slice_styles[s], label=slice_labels[s])
  plt.set_ylim(0, 1.1 * ymax / 1000)
  embb_tm = PC_TBS[10][27 - 1] / 1000
  #plt.axhline(embb_tm, c='r', ls='--', lw=1, label=('%d Mbps' % embb_tm))
  plt.axhline(embb_tm, c='r', ls='--', lw=1, label='eMBB mAR @ MCS:10')
  add_distance_plot(edges[0][1:], plt)
  #plt.set_yscale('log')
  # plt.legend(loc="best")


def plot_time_sched_rbs(slice_styles, slice_labels, results, plt):
  # bps time series
  #  plt.set_title("Slice scheduled resources time series")
  plt.set_xlabel("Simulation time (s)")
  plt.set_ylabel("Resources (RBs)")
  binned = {}
  edges = {}
  ymax = 0.000001
  lines = []
  labels = []
  for s in slice_styles:
    binned[s], edges[s], _ = stats.binned_statistic(results[s]['cx'], results[s]['rbs']['total'],
                                                    'mean', bins=100)
  for s in binned:
    smax = max(binned[s])
    if smax > ymax: ymax = smax
    l = plt.plot(edges[s][1:] / 1000, binned[s], slice_styles[s], label=slice_labels[s])
  plt.axhline(27, c='r', ls='--', lw=1, label='eMBB mAR')
  plt.axhline(slice_conf[1]['params']['MPR'], c='g', ls='--', lw=1, label='URLLC MPR')
  add_distance_plot(edges[0][1:],plt)
  plt.set_ylim(0, 1.1 * ymax)
  # plt.legend(loc="best")

def plot_time_mar(slice_styles, slice_labels, results, plt):
  # mar time series
  #  plt.set_title("Slice scheduled resources time series")
  plt.set_xlabel("Simulation time (s)")
  plt.set_ylabel("mAR (RB/TTI)")
  binned = {}
  edges = {}
  ymax = 0.000001
  for s in [0]:
    binned[s], edges[s], _ = stats.binned_statistic(results[s]['subclass_metrics']['mAR']['x'],
                                                    results[s]['subclass_metrics']['mAR']['y'],
                                                    'mean', bins=100)
  for s in binned:
    smax = max(binned[s])
    if smax > ymax: ymax = smax
    plt.plot(np.array(results[s]['subclass_metrics']['mAR']['x'])/1000, results[s]['subclass_metrics']['mAR']['y'], slice_styles[s], label=slice_labels[s])
  add_distance_plot(results[s]['subclass_metrics']['mAR']['x'],plt)
  plt.set_ylim(0, 1.1 * ymax)
  # plt.legend(loc="best")

def plot_time_reserve(slice_styles, slice_labels, results, plt):
  # mar time series
  #  plt.set_title("Slice scheduled resources time series")
  plt.set_xlabel("Simulation time (s)")
  plt.set_ylabel("RB per TTI")
  binned = {}
  edges = {}
  ymax = 0.000001
  for s in slice_styles:
    if 'reserved_rate' not in results[s]:
      continue
    binned[s] = {}
    edges[s] = {}
    for m in results[s]['reserved_rate']:
      binned[s][m], edges[s][m], _ = stats.binned_statistic(
        results[s]['reserved_rate'][m]['x'],
        results[s]['reserved_rate'][m]['y'],
        'mean',
        bins=100
      )
  for s in binned:
    for m in binned[s]:
      smax = max(binned[s][m])
      if smax > ymax: ymax = smax
      plt.plot(
        np.array(results[s]['reserved_rate'][m]['x'])/1000,
        results[s]['reserved_rate'][m]['y'],
        slice_styles[s],
        label="%s (%s)" % (m, slice_labels[s]),
      )
  plt.set_ylim(0, 1.1 * ymax)
  # plt.legend(loc="best")

def plot_time_delay(slice_styles, slice_labels, time_delays, plt):
  # Delay time series
  #  plt.set_title("Slice delay time series")
  plt.set_xlabel("Simulation time (s)")
  plt.set_ylabel("Slice delay (ms)")
  binned = {}
  edges = {}
  ymax = 0.000001
  for s in slice_styles:
    if len(delays[s]) > 0:
      binned[s], edges[s], _ = stats.binned_statistic(time_delays[s]['x'], time_delays[s]['y'], 'mean', bins=100)
  for s in binned:
    smax = max(binned[s])
    if smax > ymax: ymax = smax
    plt.plot(edges[s][1:] / 1000, binned[s], slice_styles[s], label=slice_labels[s])
  #ymax = 20
  plt.set_ylim(0, ymax * 1.1)
  # plt.legend(loc="best")


def getConfiguration(bc=250, tg=100):
  return {
    1: {'type': 'P', 'params': {'MPR': 9, 'BC': bc, 'delta': 2}, 'scheduler': SchedulerPF()},
    2: {'type': 'R', 'params': {'mAR': 9, 'TG': tg}, 'scheduler': SchedulerPF()},
    0: {'type': 'R', 'params': {'mAR': 27, 'TG': tg}, 'scheduler': SchedulerPF()}
  }

def get_plos(d):
  #return 0
  if not los:
    return 0
  if d <= 18:
    return 1
  else:
    ut = 1.5
    if ut > 13:
      c = ((ut - 13)/10)**1.5
    else:
      c = 0
    return (18/d + math.exp(-d/63)*(1-18/d))*(1+c*(5/4)*((d/100)**3)*math.exp(-d/150))

distance_from_time = {}
init_d = -204 #254
v = 8
start = 2000
end = int(start + 1000 * abs(init_d)*2 / v)
#end = 104000

def d2t(d):
  t = np.zeros(d.shape)
  t[d <= init_d] = start
  t[d >= init_d + (end - start) * v /1000] = end
  mask = (d > init_d) & (d < init_d + (end - start) * v /1000)
  t[mask] = d[mask] * 1000 / v + start - init_d*1000/v
  return t

def t2d(t):
  d = np.zeros(t.shape)
  d[t > end] = init_d + (end - start) * v /1000
  d[t < start] = init_d
  mask = (t >= start) & (t <= end)
  d[mask] = init_d + (t[mask] - start) * v / 1000
  return d

def get_distance_from_time(t):
  if t > end:
    d = init_d + (end - start) * v / 1000
  elif t < start:
    d = init_d
  else:
    d = init_d + v * (t - start) / 1000
  return d

def get_time_from_distance(d):
  assert init_d <= d <= (end - start)*v/1000
  delta = (d - init_d) / (v / 1000)
  return start + delta


#return list(map((lambda x: distance_from_time[x]),t))



def embb_sinr(st, t, u):
  d = get_distance_from_time(t)
  #if t%100 == 0:
  #  print(t,d)
  d2 = abs(d)
  distance_from_time[t] = d2
  is_los = getattr(u, 'is_los', None)
  if is_los is None:
    rnd = np.random.rand()
    plos = get_plos(d2)
    u.is_los = rnd < plos
    if u.is_los: print("LOS,", u.id, d, plos,t)
    else: print("NLOS", u.id, d, plos,t)
    u.los_time = 0
    u.los_pos = d
  else:
    u.los_time += 1
    if abs(d - u.los_pos) > 10:
      rnd = np.random.rand()
      plos = get_plos(d2)
      u.los_pos = d
      u.is_los = rnd < plos
      if is_los != u.is_los:
        if u.is_los:
          print("LOS,", u.id, d, plos, t)
        else:
          print("NLOS", u.id, d, plos, t)
      #if u.is_los: print("NLOS,",u.id)
  return Util.getSINRfromDistance(d, u.is_los)[0] - 30


def makeCDFLine(list, bins, range):
  count, edges = np.histogram(list, bins=bins, range=range, density=True)
  cdf = np.cumsum(count)
  return edges[1:], cdf / cdf[-1]


if __name__ == '__main__':

  setting = 'pa'

  embb_target = {
    'system_throughput': PC_TBS[10][26],
    'throughput_average_time': 100,
    'delay': 100,
    'primary': 'system_throughput',
    'kpi-contract': 'embb-link'
  }

  mmtc_target = {
    'system_throughput': PC_TBS[10][8],
    'throughput_average_time': 100,
    'delay': 100,
    'primary': 'system_throughput'
  }

  urllc_target = {
    'delay': 5,
    'reliability': 10**-5,
    #'admission': {
    #  'rate': 400,  #453,
    #  'capacity': 5000
    #},
    'primary': 'delay',
    'kpi-contract': 'urllc'
  }

  urllc_update = {
    'MPR': {
      'target': 3200,
      'alpha': 1.5
    },
    'BC': {
      'target': 40000,
      'alpha': 1.5
    }
  }

  slice_conf_pa = {
    1: {
      'type': 'P',
      'target': urllc_target,
      'params': {
        'MPR': 9,
        'BC': 82,
        'delta': 2
        #'update': urllc_update,
        #'mcs-target-ber': 10 ** -9
      },
      'scheduler': SchedulerPF()
    },
    2: {
      'type': 'R',
      'target': mmtc_target,
      'params': {
        'mAR': 9,
        'TG': 100,
        #'update': {
        #  'template': 'eMBB',
        #  'mar_step': 0.05,
        #}
        #'update': {
        #  'mAR': {
        #    'target': PC_TBS[10][8]
        #  }
        #}
      },
      'scheduler': SchedulerPF()
    },
    0: {
      'type': 'R',
      'target': embb_target,
      'params': {
        'mAR': 27,
        'TG': 100,
        #'update': {
        #  'template': 'eMBB',
        #  'mar_step': 0.0,
        #  'sensibility': 0.3,
        #  'lower_sensibility': -0.1,
        #  'ue_weight_type': 'custom',
        #  'ue_weight_fun' : (lambda sli, ts, traffic, scheduled, u: u*2 / sum(np.arange(0,len(scheduled['ues']))*2))
        #}
        #'update': {
        #  'mAR': {
        #    'target': PC_TBS[10][26],
        #    'alpha': 1,
        #    'ue_weight_type': 'equal'
        #  }
        #}
      },
      'scheduler': SchedulerPF()
    }
  }
  slice_conf_npa = {
    1: {'type': 'P', 'params': {'MPR': 10, 'BC': 10, 'delta': 1}, 'scheduler': SchedulerPF()},
    2: {'type': 'R', 'params': {'mAR': 9, 'TG': 1}, 'scheduler': SchedulerPF()},
    0: {'type': 'R', 'params': {'mAR': 27, 'TG': 1}, 'scheduler': SchedulerPF()}
  }

  if setting == 'pa':
    slice_conf = slice_conf_pa
    slices = slice_conf_pa.keys()
  else:
    slice_conf = slice_conf_npa
    slices = slice_conf_npa.keys()

  slice_styles = {
    1: 'g-',
    2: 'y-',
    0: 'r-'
  }

  slice_labels = {
    1: 'URLLC',
    2: 'mMTC',
    0: 'eMBB'
  }

  sim_conf = {
    'timeout': 10, #131,
    'tti': 0.001,
    'cqi_report_period': 50,
    'cell_metric_definition': 1,
    'bw': 50,
    'warmup': 0
  }
  print("SIM END", sim_conf['timeout'])
  s1_tps = []
  s2_tps = []
  s0_tps = []
  sla_violations = {}
  counts = np.arange(8, 9, 1).astype(int)
  for s in slices:
    sla_violations[s] = np.zeros(len(counts))

  time_delays = {}
  sizes = {}
  for s in slices:
    sizes[s] = {'x': [], 'y': []}
    time_delays[s] = {}
    for k in ['x', 'y', 'sched']:
      time_delays[s][k] = []

  for idx, uec in enumerate(counts):
    print(uec)

    #traffic_seed = np.random.randint(0,10000,(len(slices),uec))
    traffic_seed = np.array([[745, 4379, 4034, 6676, 2055, 1691],
                             [3638, 6790, 1110, 8794, 1393, 1470],
                             [501, 5996, 7059, 2726, 1905, 1158]])

    ue_conf = [
      {
        'slice': 0,
        'traffic': 'full_buffer',
        'params': {
          'mcs': 10,
          #'sinr': 16,
          #'sinr_fun': (lambda st, t, u: st + 0.003*(5000 - t)/max(0.1, abs(5000 - t))),
          #'sinr_fun': (lambda st, t, u: Util.getSINRfromDistance(-300 + 0.06*t, False)[0] - 30),
          #'sinr_fun': embb_sinr,
          'size': 1500,
          'traffic_seed': traffic_seed[0]
        },
        #    'traffic': 'cbr',
        #    'params' :{
        #      'interval': 0.04,
        #      'jitter': 0.003,
        #      'size': 4000,
        #      'drift': 0.04
        #    },
        #'movement': '../sinr-map/out/trace*',
        'count': uec
      },
      {
        'slice': 1,
        #   'traffic': 'full_buffer',
        #   'params': {'size': 1500},
        'traffic': 'cbr',
        'params': {
          #'sinr_fun': embb_sinr,
          'mcs': 10,
          'interval': 0.01,
          'jitter': 0.002,
          'size': 300,
          'drift': 0.01,
          'traffic_seed': traffic_seed[1],
          'bursts': [
            {'start': 2, 'end': 4, 'probability': 0.1, 'interval': 0.001, 'total': 10},
            {'start': 7, 'end': 9, 'probability': 0.2, 'interval': 0.001, 'total': 10}
           #  {'start': 0, 'end': 131, 'probability': 0.02, 'interval': 0.002, 'total': 5}
          ]
        },
        #    'movement': '../sinr-map/out/trace*',
        'count': uec
      },
      {
        'slice': 2,
        #   'traffic': 'full_buffer',
        #   'params': {'size': 300},
        'traffic': 'cbr',
        'params': {
          #'sinr': 20,
          #'sinr_fun': (lambda st, t, u: st - 0.0015*(5000 - t)/max(0.1, abs(5000 - t))),
          #'sinr_fun': (lambda st, t, u: Util.getSINRfromDistance(-300 + 0.06*t,False)[0]),
          #'sinr_fun': embb_sinr,
          'mcs': 10,
          'interval': 0.01,
          'jitter': 0.002,
          'size': 300,
          'drift': 0.01,
          'traffic_seed': traffic_seed[2]
        },
        #    'movement': '../sinr-map/out/trace*',
        'count': uec
      }
    ]
    #try:
    #  ue_metrics = pickle.load(open("ue_metrics-%d-%s.pkl"%(uec,'los' if los else 'nlos'),'rb'))
    #  results = pickle.load(open("results-%d-%s.pkl"%(uec,'los' if los else 'nlos'),'rb'))
    #  loaded = True
    #except:
    #  ue_metrics = []
    #  results = []
    #  loaded = False

    #loaded = False

    #if not loaded:
    sim = Simulation(in_sim_conf=sim_conf, slice_conf=slice_conf, ue_conf=ue_conf, verbose=False)
    ue_metrics, results = sim.run()
    #pickle.dump(ue_metrics, open("ue_metrics-%d-%s.pkl"%(uec,'los' if los else 'nlos'), 'wb'))
    #pickle.dump(results, open("results-%d-%s.pkl"%(uec,'los' if los else 'nlos'), 'wb'))

    delays = {}
    totals = {}
    for s in slices:
      delays[s] = []
      totals[s] = {}
      for m in ['size', 'packets']:
        totals[s][m] = 0
      totals[s]['ppu'] = {}
      time_delays[s]['rbs'] = results[s]['rbs'][0]
      time_delays[s]['x_rbs'] = range(0, len(results[s]['rbs'][0]))
      for u in ue_metrics:
        if ue_metrics[u]['slice'] != s:
          continue
        totals[s]['packets'] += len(ue_metrics[u]['packets'])
        totals[s]['ppu'][u] = len(ue_metrics[u]['packets'])
        for p in ue_metrics[u]['packets']:
          pkt = ue_metrics[u]['packets'][p]
          if 'recv' not in pkt:
            continue
          if 'sent' in pkt:
            time_delays[s]['x'].append(pkt['sent'])
            delays[s].append(pkt['recv'] - pkt['sent'])
            time_delays[s]['y'].append(pkt['recv'] - pkt['sent'])
            int_drift = pkt['scheduled'] % 1
            time_delays[s]['sched'].append(pkt['scheduled'] - int_drift - np.ceil(pkt['sent'] - int_drift))
            if pkt['scheduled'] > sim_conf['timeout'] * 1000:
              continue
            sizes[s]['x'].append(pkt['sent'])
          else:
            if pkt['scheduled'] > sim_conf['timeout'] * 1000:
              continue
            time_delays[s]['x'].append(pkt['scheduled'])
            sizes[s]['x'].append(pkt['scheduled'])
          sizes[s]['y'].append(pkt['size'])
          totals[s]['size'] += pkt['size']

    ue_cnt = len(ue_metrics)
    ue_ids = np.zeros(ue_cnt)
    sinr = np.zeros(ue_cnt)
    sinr_ci = np.zeros(ue_cnt)
    ue_labels = {}
    for i, u in enumerate(ue_metrics):
      metrics = ue_metrics[u]
      ue_ids[i] = u
      sinr[i] = ue_metrics[u]['final']['sinr'][0]
      sinr_ci[i] = ue_metrics[u]['final']['sinr'][1]
      ue_labels[u] = "%ds%d" % (u, ue_metrics[u]['slice'])
      s = ue_metrics[u]['slice']

    plt_h = 3
    plt_w = 4

    leg_x = 0.47
    leg_y = -0.3

    col_space = 0.9


    target = {}
    for s in slices:
      target_list = results[s]['target_metric'][slice_conf[s]['target']['primary']]['y']
      x_target, target[s] = makeCDFLine(target_list,100,[0,2])

    fname = "target-%d-%s"%(uec,"alt-los" if los else "nlos")
    fig, ax = plt.subplots()
    fig.set_figheight(plt_h)
    fig.set_figwidth(plt_w)
    ax.set_xlabel("Target metric value")
    ax.axvline(1, c='b', ls='--', lw=1)
    for s in slices:
      ax.plot(x_target, target[s], slice_styles[s], label=slice_labels[s])
    plt.subplots_adjust(bottom=0.25)
    ax.legend(loc='upper center', bbox_to_anchor=(leg_x, leg_y), ncol=2, columnspacing=col_space)
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)

    fname = "ue-sinr-" + str(uec)
    fig, ax = plt.subplots()
    fig.set_figheight(plt_h)
    fig.set_figwidth(plt_w)
    ax.set_title("Average SINR per UE")
    ax.set_xlabel("UE")
    ax.set_ylabel("SINR (dB)")
    ax.bar(ue_ids,sinr,yerr=sinr_ci)
    ax.set_xticks(ue_ids)
    ax.set_xticklabels([ue_labels[u] for u in ue_ids])
    #plt.show()
    plt.close(fig)
    for s in slices:
      print("RETX, slice",s,", ",results[s]['retx'])

    # Delay CDF
    fname = "delay-cdf-" + str(uec) + "-" + setting
    fig, ax = plt.subplots()
    fig.set_figheight(plt_h)
    fig.set_figwidth(plt_w)
    for s in slices:
      if len(delays[s]) > 0:
        Util.plotCDF(delays[s], ax, slice_styles[s], slice_labels[s])
    # plt.show()
    ax.legend(loc='upper center', bbox_to_anchor=(leg_x, leg_y), ncol=3, columnspacing=col_space)
    # fig.savefig(fname)
    plt.close(fig)

    fname = "delay-time-" + str(uec) + "-" + setting
    fig, ax = plt.subplots()
    fig.set_figheight(plt_h)
    fig.set_figwidth(plt_w)
    plot_time_delay(slice_styles, slice_labels, time_delays, ax)
    plt.subplots_adjust(bottom=0.25)
    ax.legend(loc='upper center', bbox_to_anchor=(leg_x, leg_y), ncol=2, columnspacing=col_space)
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)

    fname = "kbps-time-" + str(uec) + "-" + setting
    fig, ax = plt.subplots()
    fig.set_figheight(plt_h)
    fig.set_figwidth(plt_w)
    plot_time_sched_kbps(slice_styles, slice_labels, results, sim_conf['tti'], ax)
    plt.subplots_adjust(bottom=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(leg_x, leg_y), ncol=2, columnspacing=col_space)
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)

    fname = 'excesses-time-' + str(uec) + "-" + setting
    fig, ax = plt.subplots()
    fig.set_figheight(plt_h)
    fig.set_figwidth(plt_w)
    plot_time_excesses(slice_styles, slice_labels, results, ax)
    ax.legend(loc='upper center', bbox_to_anchor=(leg_x, leg_y), ncol=3, columnspacing=col_space)
    # fig.savefig(fname)
    plt.close(fig)

    fname = 'overuses-time-' + str(uec) + "-" + setting
    fig, ax = plt.subplots()
    fig.set_figheight(plt_h)
    fig.set_figwidth(plt_w)
    plot_time_overuses(slice_styles, slice_labels, results, ax)
    ax.legend(loc='upper center', bbox_to_anchor=(leg_x, leg_y), ncol=3, columnspacing=col_space)
    # fig.savefig(fname)
    plt.close(fig)

    fname = 'input-time-' + str(uec) + "-" + setting
    fig, ax = plt.subplots()
    fig.set_figheight(plt_h)
    fig.set_figwidth(plt_w)
    plot_time_ipunt_traffic(slice_styles, slice_labels, sizes, ax)
    plt.subplots_adjust(bottom=0.25)
    ax.legend(loc='upper center', bbox_to_anchor=(leg_x, leg_y), ncol=3, columnspacing=col_space)
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)

    fname = 'sched-rbs-' + str(uec) + "-" + setting
    fig, ax = plt.subplots()
    fig.set_figheight(plt_h)
    fig.set_figwidth(plt_w)
    plot_time_sched_rbs(slice_styles, slice_labels, results, ax)
    plt.subplots_adjust(bottom=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(leg_x, leg_y), ncol=3, columnspacing=col_space)
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)

    #fname = 'reserves-' + str(uec) + "-" + setting
    #fig, ax = plt.subplots()
    #fig.set_figheight(plt_h)
    #fig.set_figwidth(plt_w)
    ##plot_time_mar(slice_styles, slice_labels, results, ax)
    #plot_time_reserve(slice_styles,slice_labels,results,ax)
    #plt.subplots_adjust(bottom=0.3)
    #ax.legend(loc='upper center', bbox_to_anchor=(0.47, -0.2), ncol=3, columnspacing=1)
    #fig.savefig(fname, bbox_inches='tight')
    #plt.close(fig)


    #fname = 'mcs-' + str(uec) + "-" +setting
    #fig, ax = plt.subplots(constrained_layout=True)
    #fig.set_figheight(plt_h)
    #fig.set_figwidth(plt_w)
    #plot_time_ue_mcs(ue_metrics,ax)
    #plt.subplots_adjust(bottom=0.3)
    ##ax.legend(loc='upper center', bbox_to_anchor=(0.47, -0.2), ncol=3, columnspacing=1)
    #fig.savefig(fname, bbox_inches='tight')
    #plt.close(fig)

    #no_plots = 4
    #fig, axs = plt.subplots(1, 3, sharex=True)
    #fig.set_figheight(3)
    #fig.set_figwidth(14)

    # plt.subplot(no_plots,1,1)
    #plot_time_delay(slice_styles, slice_labels, time_delays, axs[2])

    # plt.subplot(no_plots,1,2)
    # plot_time_excesses(slice_styles,slice_labels,results,axs[0,1])
    # plot_time_overuses(slice_styles,slice_labels,results,axs[1,0])

    # plt.subplot(no_plots,1,3)
    #plot_time_sched_rbs(slice_styles, slice_labels, results, axs[1])

    # plt.subplot(no_plots,1,4)
    #plot_time_ipunt_traffic(slice_styles, slice_labels, sizes, axs[0])

    #for ax in axs:
    #  # for ax in axh:
    #  ax.legend(loc='best')  # , bbox_to_anchor=(0.9, 0.5),fancybox=True,shadow=True)

    fname = "bundle-" + str(uec) + "-" + setting

    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)

    if results[0]['tp'][0] is not None:
      s0_tps.append(results[0]['tp'][0] / 1000)
    else:
      s0_tps.append(0)
    if results[1]['tp'][0] is not None:
      s1_tps.append(results[1]['tp'][0] / 1000)
    else:
      s1_tps.append(0)
    if results[2]['tp'][0] is not None:
      s2_tps.append(results[2]['tp'][0] / 1000)
    else:
      s2_tps.append(0)

    for s in [0, 1, 2]:
      sla_violations[s][idx] = results[s]['violations']

  # plt.subplot(3,1,2)
  plt.plot(counts, s1_tps, counts, s2_tps, counts, s0_tps)
  # plt.show()
  # plt.yscale('log')
  #
  # plt.subplot(3,1,3)
  for s in slices:
    plt.plot(counts, sla_violations[s], slice_styles[s], label=slice_labels[s])
  plt.legend()
  # plt.show()
