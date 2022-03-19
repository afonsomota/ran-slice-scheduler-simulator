from Simulator import Simulation
from IntraScheduler import SchedulerPF
import Util
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pickle
from NRTables import PC_TBS
import os
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def plot_time_ipunt_traffic(slice_styles, slice_labels, sizes,  plt, bins= 100):
  # input traffic time series
  plt.set_title("Input traffic time series")
  plt.set_xlabel("Simulation time (s)")
  plt.set_ylabel("Input traffic (Kbytes)")
  binned = {}
  edges = {}
  ymax = 0.000001
  ret = np.zeros((2,len(slice_labels),bins))
  for s in slice_styles:
    if Util.isSliceFullBuffer(s,ue_conf):
      continue
    binned[s], edges[s], _ = stats.binned_statistic(sizes[s]['x'], sizes[s]['y'],
                                                    'sum', bins=bins)
    binned[s] = binned[s]/1000
  for s in binned:
    smax = max(binned[s])
    if smax > ymax: ymax = smax
    plt.plot(edges[s][1:], binned[s], slice_styles[s], label=slice_labels[s])
    ret [0,s] = edges[s][1:]
    ret [1,s] = binned[s]
  plt.set_ylim(0, 1.1 * ymax)
  plt.legend(loc="best")
  return ret



def plot_time_overuses(slice_styles,slice_labels,results,plt,bins= 100):
  # overuses time series
  plt.set_title("Overused resources per slice")
  plt.set_xlabel("Simulation time (s)")
  plt.set_ylabel("Unused resoureces (RBs)")
  ymax = 0.000001
  ret = np.zeros((2,len(slice_labels),bins))
  #plt.plot(results[1]['overuses']['x'], results[1]['overuses']['y'], results[2]['overuses']['x'],
  #         results[2]['overuses']['y'])
  for s in [2,0,1]:
    smax = max(results[s]['overuses']['y'])
    if smax > ymax: ymax = smax
    binned, edges, _ = stats.binned_statistic(results[s]['overuses']['x'], results[s]['overuses']['y'],
                                                    'mean', bins=bins)
    plt.plot(edges[1:],binned,slice_styles[s],label=slice_labels[s])
    ret[0, s] = edges[1:]
    ret[1, s] = binned
  plt.set_ylim(0,1.1*ymax)
  plt.legend(loc="best")
  return ret

def plot_time_excesses(slice_styles,slice_labels,results,plt,bins=100):
  # excesses time series
  plt.set_title("Unused resources per slice")
  plt.set_xlabel("Simulation time (s)")
  plt.set_ylabel("Unused resoureces (RBs)")
  ymax = 0.000001
  #plt.plot(results[1]['excesses']['x'], results[1]['excesses']['y'], results[2]['excesses']['x'],
  #         results[2]['excesses']['y'], results[0]['excesses']['x'], results[0]['excesses']['y'])
  ret = np.zeros((2,len(slice_labels),bins))
  for s in slice_styles:
    smax = max(results[s]['excesses']['y'])
    if smax > ymax: ymax = smax
    binned, edges, _ = stats.binned_statistic(results[s]['excesses']['x'], results[s]['excesses']['y'],
                                              'mean', bins=bins)
    plt.plot(edges[1:],binned,slice_styles[s],label=slice_labels[s])
    ret[0, s] = edges[1:]
    ret[1, s] = binned
  plt.set_ylim(0,1.1*ymax)
  plt.legend(loc="best")
  return ret


def plot_time_sched_kbps(slice_styles,slice_labels,results,plt,bins=100,x_interval=None,slice_tgs=None):
  # bps time series
  plt.set_title("Slice throughput at scheduler time series")
  plt.set_xlabel("Simulation time (s)")
  plt.set_ylabel("Slice throughput (kbps)")
  binned = {}
  edges = {}
  ymax = 0.000001
  bins_per_slice = {}
  if slice_tgs is not None:
    for s in slice_tgs:
      bins_per_slice[s] = min(10000/slice_tgs[s],bins*10000/(x_interval[1]-x_interval[0]))
  else:
    for s in slice_labels:
      bins_per_slice[s] = bins
  for s in slice_styles:
    x = np.array(results[s]['cx'])
    y = np.array(results[s]['bits']['total'])
    binned[s], edges[s], _ = stats.binned_statistic(x,y,
                                                    'mean', bins=bins_per_slice[s])
  ret = np.zeros((2,len(slice_labels),bins))
  for s in binned:
    smax = max(binned[s])
    if smax > ymax: ymax = smax
    x = edges[s][1:]
    y = binned[s]
    if x_interval is not None:
      x,y = Util.filterValues(x,y,x_interval)
    plt.plot(x, y, slice_styles[s], label=slice_labels[s])
    ret[0, s][0:len(x)] = x
    ret[1, s][0:len(x)] = y
  plt.set_ylim(0, 1.1 * ymax)
  plt.legend(loc="best")
  return ret

def plot_time_sched_rbs(slice_styles,slice_labels,results,plt,bins=100,x_interval=None):
  # bps time series
  plt.set_title("Slice scheduled resources time series")
  plt.set_xlabel("Simulation time (s)")
  plt.set_ylabel("Resources (RBs)")
  binned = {}
  edges = {}
  ymax = 0.000001
  ret = np.zeros((2,len(slice_labels),bins))
  for s in slice_styles:
    x = np.array(results[s]['cx'])
    y = np.array(results[s]['rbs']['total'])
    if x_interval is not None:
      x,y = Util.filterValues(x,y,x_interval)
      bins = min(bins, len(x))
      if s == 0:
        ret = np.zeros((2, len(slice_labels), bins))
    binned[s], edges[s], _ = stats.binned_statistic(x, y,
                                                    'mean', bins=bins)
  for s in binned:
    smax = max(binned[s])
    if smax > ymax: ymax = smax
    plt.plot(edges[s][1:], binned[s], slice_styles[s], label=slice_labels[s])
    ret[0, s] = edges[s][1:]
    ret[1, s] = binned[s]
  plt.axhline(slice_conf[0]['params']['mAR'],c='r',ls='--',lw=1,label='eMBB mAR')
  plt.axhline(slice_conf[1]['params']['MPR'],c='g',ls='--',lw=1,label='URLLC MPR')
  plt.set_ylim(0, 1.1 * ymax)
  plt.legend(loc="best")
  return ret

def plot_time_delay(slice_styles,slice_labels,time_delays,plt,bins=100,x_interval=None,aggregate=True):
  # Delay time series
  plt.set_title("Slice delay time series")
  plt.set_xlabel("Simulation time (s)")
  plt.set_ylabel("Slice delay (ms)")
  binned = {}
  edges = {}
  ymax = 0.000001
  ret = np.zeros((2,len(slice_labels),bins))
  for s in slice_styles:
    if aggregate:
      if len(delays[s]) > 0:
        x = np.array(time_delays[s]['x'])
        y = np.array(time_delays[s]['y'])
        if x_interval is not None:
          x,y = Util.filterValues(x,y,x_interval)
          bins = min(bins, len(x))
          if s == 0:
            ret = np.zeros((2, len(slice_labels), bins))
        binned[s], edges[s], _ = stats.binned_statistic(x, y, 'mean', bins=bins)
    else:
      edges[s] = np.array([0]+time_delays[s]['x'])
      binned[s] = np.array(time_delays[s]['y'])

  for s in binned:
    smax = max(binned[s])
    if smax > ymax: ymax = smax
    plt.plot(edges[s][1:], binned[s], slice_styles[s], label=slice_labels[s])
    ret[0, s] = edges[s][1:]
    ret[1, s] = binned[s]
  ymax = 20
  plt.set_ylim(0, 1.1 * ymax)
  plt.legend(loc="best")
  return ret


def getSliceConfiguration(bc=82, tg=100):
  if bc == -1:
    mpr = 50
    bcm = 10000000
  else:
    mpr = 9
    bcm = max(mpr, bc)
  return {
    1: {'type': 'P', 'params': {'MPR': mpr, 'BC': bcm, 'delta': 2}, 'scheduler': SchedulerPF()},
    2: {'type': 'R', 'params': {'mAR': 9, 'TG': tg}, 'scheduler': SchedulerPF()},
    0: {'type': 'R', 'params': {'mAR': 27, 'TG': tg}, 'scheduler': SchedulerPF()}
  }

def getUEConfiguration(traffic_seed,uec):
  return [
      {
        'slice': 0,
        'traffic': 'full_buffer',
        'params': {
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
    #    'movement': '../sinr-map/out/trace*',
        'count': uec
      },
      {
        'slice': 1,
     #   'traffic': 'full_buffer',
     #   'params': {'size': 1500},
        'traffic': 'cbr',
        'params': {
          'interval': 0.01,
          'jitter': 0.002,
          'size': 300,
          'drift': 0.01,
          'traffic_seed': traffic_seed[1],
          'bursts': [
            {'start': 2, 'end': 4, 'probability': 0.1, 'interval': 0.001, 'total': 10},
            {'start': 7, 'end': 9, 'probability': 0.2, 'interval': 0.001, 'total': 10}
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


def fill_dls_tps(dls, tps, ue_metrics):
  max_dl = max_tp = 0
  for u in ue_metrics:
    s = ue_metrics[u]['slice']
    if s not in dls:
      if dls is not None: dls[s] = []
      if tps is not None: tps[s] = []
    for p in ue_metrics[u]['packets']:
      pkt = ue_metrics[u]['packets'][p]
      if 'recv' in pkt and 'sent' in pkt:
        delay = pkt['recv'] - pkt['sent']
        if delay > max_dl:
          max_dl = delay
        if dls is not None: dls[s].append(delay)
      if 'recv' in pkt and 'scheduled' in pkt:
        tp = pkt['size'] * 8 / (pkt['recv'] - pkt['scheduled']) / sim_conf['tti'] / 1000
        if tp > max_tp:
          max_tp = tp
        if tps is not None: tps[s].append(tp)
  return max_dl, max_tp


if __name__ == '__main__':

  simulation_dir = "group_2"

  if simulation_dir is not None:
    if not os.path.exists(simulation_dir):
      os.makedirs(simulation_dir)
    os.chdir(simulation_dir)

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
    'timeout': 10,
    'tti': 0.001,
    'warmup': 0
  }
  #counts = np.arange(5,13,1).astype(int)
  burst_capacities = [0,25,41,82,-1]
  time_gaps = [1,25,50,100,500]
  #time_gaps = [100]

  measures = {
    'delay': 0,
    'throughput': 1,
    'excesses': 2,
    'overuses': 3,
    'input': 4,
    'scheduled': 5,
    'delay-zoom': 6,
    'throughput-zoom':7,
    'delay-xzoom':8,
    'in-profile':9
  }

  time_delays = {}
  sizes = {}
  slices = slice_labels.keys()
  for s in slices:
    sizes[s] = {'x':[], 'y':[]}
    time_delays[s] = {}
    for k in ['x','y','sched']:
      time_delays[s][k] = []
  bins = 100

  try:
    lines = pickle.load(open("lines.pkl", 'rb'))
    loaded = True
  except:
    lines = np.zeros((len(measures), len(burst_capacities), len(time_gaps), 2, len(slices), bins))
    loaded = False
  uec = 8
  #traffic_seed = np.random.randint(0,10000,(len(slices),uec))
  if not loaded:
    for ibc,bc in enumerate(burst_capacities):
      for itg,tg in enumerate(time_gaps):
        print(bc,tg)

        slice_conf = getSliceConfiguration(bc=bc,tg=tg)
        slices = slice_conf.keys()

        traffic_seed = np.array([[ 745, 4379, 4034, 6676, 2055, 1691],
           [3638, 6790, 1110, 8794, 1393, 1470],
           [ 501, 5996, 7059, 2726, 1905, 1158]])

        ue_conf = getUEConfiguration(traffic_seed,uec)

        sim = Simulation(slice_conf=slice_conf, ue_conf=ue_conf,verbose=False)
        ue_metrics,results = sim.run()

        delays = {}
        totals = {}
        for s in slices:
          delays[s] = []
          totals[s] = {}
          for m in ['size','packets']:
            totals[s][m] = 0
          totals[s]['ppu'] = {}
          time_delays[s]['rbs'] = results[s]['rbs'][0]
          time_delays[s]['x_rbs'] = range(0,len(results[s]['rbs'][0]))
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
                delays[s].append(pkt['recv']-pkt['sent'])
                time_delays[s]['y'].append(pkt['recv']-pkt['sent'])
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
              totals[s]['size']+=pkt['size']

        dls = {}

        fill_dls_tps(dls,None,ue_metrics)

        ue_cnt = len(ue_metrics)
        ue_ids = np.zeros(ue_cnt)
        sinr = np.zeros(ue_cnt)
        sinr_ci = np.zeros(ue_cnt)
        ue_labels = {}
        for i,u in enumerate(ue_metrics):
          metrics = ue_metrics[u]
          ue_ids[i] = u
          sinr[i] = ue_metrics[u]['final']['sinr'][0]
          sinr_ci[i] = ue_metrics[u]['final']['sinr'][1]
          ue_labels[u] = "%ds%d" % (u,ue_metrics[u]['slice'])

    #    fname = "ue-sinr-" + str(uec)
    #    fig, ax = plt.subplots()
    #    ax.set_title("Average SINR per UE")
    #    ax.set_xlabel("UE")
    #    ax.set_ylabel("SINR (dB)")
    #    ax.bar(ue_ids,sinr,yerr=sinr_ci)
    #    ax.set_xticks(ue_ids)
    #    ax.set_xticklabels([ue_labels[u] for u in ue_ids])
    #    plt.show()
    #    plt.close(fig)

        plt.subplots_adjust(bottom=0.35)
        #Delay CDF
        fname = "delay-cdf-bc_%d-tg_%d"%(bc,tg)
        fig, ax = plt.subplots()
        for s in slices:
          if len(delays[s]) > 0:
            Util.plotCDF(delays[s],ax,slice_styles[s],slice_labels[s])
        #plt.show()
        ax.legend(loc='upper center', bbox_to_anchor=(0.47, -0.2), ncol=3, columnspacing=1)
        fig.savefig(fname+".pdf", format="pdf", bbox_inches='tight')
        plt.close(fig)

        fname = "delay-time-bc_%d-tg_%d"%(bc,tg)
        fig, ax = plt.subplots()
        lines[measures['delay'],ibc,itg] = plot_time_delay(slice_styles,slice_labels,time_delays,ax,bins)
        ax.legend(loc='upper center', bbox_to_anchor=(0.47, -0.2), ncol=3, columnspacing=1)
        fig.savefig(fname+".pdf", format="pdf", bbox_inches='tight')
        plt.close(fig)

        total = 0
        fast = 0
        for dl in time_delays[1]['y']:
          total += 1
          if dl <= 4:
            fast += 1


        print("TG:",tg,"BC:",bc,"In contract:",fast/total)
        lines[measures['in-profile'],ibc,itg,1,1,0] = fast/total

        # fname = "delay-time-zoom-bc_%d-tg_%d" % (bc, tg)
        # fig, ax = plt.subplots()
        # lines[measures['delay'], ibc, itg] = plot_time_delay(slice_styles, slice_labels, time_delays, ax, bins)
        # ax.legend(loc='upper center', bbox_to_anchor=(0.47, -0.2), ncol=3, columnspacing=1)
        # fig.savefig(fname)
        # plt.close(fig)

        fname = "delay-time-zoom-bc_%d-tg_%d" % (bc, tg)
        fig, ax = plt.subplots()
        lines[measures['delay-zoom'], ibc, itg] = plot_time_delay(slice_styles, slice_labels, time_delays, ax, bins,x_interval=[7000,9000])
        ax.legend(loc='upper center', bbox_to_anchor=(0.47, -0.2), ncol=3, columnspacing=1)
        fig.savefig(fname+".pdf", format="pdf", bbox_inches='tight')
        plt.close(fig)


      #  fname = "delay-time-xzoom-bc_%d-tg_%d" % (bc, tg)
      #  fig, ax = plt.subplots()
      #  lines[measures['delay-xzoom'], ibc, itg] = plot_time_delay(slice_styles, slice_labels, time_delays, ax, bins,x_interval=[7500,7750],aggregate=False)
      #  ax.legend(loc='upper center', bbox_to_anchor=(0.47, -0.2), ncol=3, columnspacing=1)
      #  fig.savefig(fname+".pdf", format="pdf", bbox_inches='tight')
      #  plt.close(fig)

        fname = "kbps-time-bc_%d-tg_%d"%(bc,tg)
        fig, ax = plt.subplots()
        lines[measures['throughput'],ibc,itg] = plot_time_sched_kbps(slice_styles,slice_labels,results,ax,bins)
        ax.legend(loc='upper center', bbox_to_anchor=(0.47, -0.2), ncol=3, columnspacing=1)
        #fig.savefig(fname+".pdf", format="pdf", bbox_inches='tight')
        plt.close(fig)

        fname = "kbps-time-zoom-bc_%d-tg_%d" % (bc, tg)
        fig, ax = plt.subplots()
        lines[measures['throughput-zoom'], ibc, itg] = plot_time_sched_kbps(slice_styles, slice_labels, results, ax, bins,x_interval=[7000,9000],slice_tgs={0:tg, 1:tg,2:tg})
        ax.legend(loc='upper center', bbox_to_anchor=(0.47, -0.2), ncol=3, columnspacing=1)
        #fig.savefig(fname+".pdf", format="pdf", bbox_inches='tight')
        plt.close(fig)

        fname = "excesses-time-bc_%d-tg_%d"%(bc,tg)
        fig, ax = plt.subplots()
        lines[measures['excesses'],ibc,itg] = plot_time_excesses(slice_styles,slice_labels,results,ax,bins)
        ax.legend(loc='upper center', bbox_to_anchor=(0.47, -0.2), ncol=3, columnspacing=1)
        #fig.savefig(fname+".pdf", format="pdf", bbox_inches='tight')
        plt.close(fig)

        fname = "overuses-time-bc_%d-tg_%d"%(bc,tg)
        fig, ax = plt.subplots()
        lines[measures['overuses'],ibc,itg] = plot_time_overuses(slice_styles,slice_labels,results,ax,bins)
        ax.legend(loc='upper center', bbox_to_anchor=(0.47, -0.2), ncol=3, columnspacing=1)
        #fig.savefig(fname+".pdf", format="pdf", bbox_inches='tight')
        plt.close(fig)


        no_plots=4
        fig,axs = plt.subplots(1,3,sharex=True)
        fig.set_figheight(3)
        fig.set_figwidth(14)

        #plt.subplot(no_plots,1,1)
        plot_time_delay(slice_styles,slice_labels,time_delays,axs[2],bins)

        #plt.subplot(no_plots,1,2)
        #plot_time_excesses(slice_styles,slice_labels,results,axs[0,1])
        #plot_time_overuses(slice_styles,slice_labels,results,axs[1,0])

        #plt.subplot(no_plots,1,3)
        lines[measures['scheduled'],ibc,itg] = plot_time_sched_rbs(slice_styles,slice_labels,results,axs[1],bins)

        #plt.subplot(no_plots,1,4)
        lines[measures['input'],ibc,itg] = plot_time_ipunt_traffic(slice_styles,slice_labels,sizes,axs[0],bins)

        for ax in axs:
          #for ax in axh:
          ax.legend(loc='upper center', bbox_to_anchor=(0.47, -0.2), ncol=3, columnspacing=1)

        fname = "bundle-time-bc_%d-tg_%d"%(bc,tg)

        fig.savefig(fname+".pdf", format="pdf", bbox_inches='tight')
        plt.close(fig)

  if not loaded:
    pickle.dump(lines, open('lines.pkl', 'wb'))
    loaded = True
  slice_conf = getSliceConfiguration()

  # fname = "delays-tg_100"
  # fig, ax = plt.subplots()
  # itg = time_gaps.index(100)
  # l = lines[measures['delay']]
  # for ibc, bc in enumerate(burst_capacities):
  #   ax.plot(l[ibc,itg,0,1],l[ibc,itg,1,1],label=str(bc))
  # ax.legend(loc='upper center', bbox_to_anchor=(0.47, -0.2), ncol=3, columnspacing=1)
  # #fig.savefig(fname, bbox_inches='tight')
  # plt.close(fig)
  #
  # fname = "delays-bc_50"
  # fig, ax = plt.subplots()
  # ibc = burst_capacities.index(50)
  # l = lines[measures['delay']]
  # for itg, tg in enumerate(time_gaps):
  #   ax.plot(l[ibc,itg,0,1],l[ibc,itg,1,1],label=str(tg))
  # ax.legend(loc='upper center', bbox_to_anchor=(0.47, -0.2), ncol=3, columnspacing=1)
  # #fig.savefig(fname, bbox_inches='tight')
  # plt.close(fig)

  # fname = "delays-bc_250"
  # fig, ax = plt.subplots()
  # ibc = burst_capacities.index(250)
  # l = lines[measures['delay']]
  # for itg, tg in enumerate(time_gaps):
  #   ax.plot(l[ibc,itg,0,1],l[ibc,itg,1,1],label=str(tg))
  # ax.legend(loc='upper center', bbox_to_anchor=(0.47, -0.2), ncol=3, columnspacing=1)
  # fig.savefig(fname, bbox_inches='tight')
  # plt.close(fig)

  stgs = [1, 50, 100]
  sbcs = [0, 41, 82, -1]

  # plt_h = 4.5
  # plt_w = 5.5
  plt_h = 3
  plt_w = 4

  leg_x = 0.47
  leg_y = -0.3

  leg_size = 9

  bold_lw = 1.2

  fname = "delays-mix"
  fig, ax = plt.subplots()
  fig.set_figheight(plt_h)
  fig.set_figwidth(plt_w)
  l = lines[measures['delay']]
  for bc in sbcs:
    for tg in stgs:
      if bc == 0 and tg != 1:
        continue
      if tg == 1 and bc != 0:
        continue
      if bc == -1 and tg != 100:
        continue
      itg = time_gaps.index(tg)
      ibc = burst_capacities.index(bc)
      label = "BC:%d,TG:%d"%(bc,tg)
      lw = 1
      if bc == -1:
        label = "BC: \u221E"
        lw = bold_lw
      elif tg == 1:
        lw = bold_lw
      ax.plot(l[ibc,itg,0,1]/1000,l[ibc,itg,1,1],label=label, lw=lw)
  ax.set_xlabel("Simulation time (s)")
  ax.set_ylabel("Delay (ms)")
  plt.subplots_adjust(bottom=0.35)
  ax.legend(loc='upper center', bbox_to_anchor=(leg_x, leg_y), ncol=3, columnspacing=1, prop={'size': leg_size})
  fig.savefig(fname+".pdf", format="pdf", bbox_inches='tight')
  ax.set_ylim(0,22)
  #plt.show()
  plt.close(fig)

  fname = "kbps-mix"
  fig, ax = plt.subplots()
  fig.set_figheight(plt_h)
  fig.set_figwidth(plt_w)
  l = lines[measures['throughput']]
  for bc in sbcs:
    for tg in stgs:
      if bc == 0 and tg != 1:
        continue
      if tg ==1 and bc != 0:
        continue
      if bc == -1 and tg != 100:
        continue
      itg = time_gaps.index(tg)
      ibc = burst_capacities.index(bc)
      label = "BC:%d,TG:%d"%(bc,tg)
      lw = 1
      if bc == -1:
        label = "BC: \u221E"
        lw = bold_lw
      elif tg == 1:
        lw = bold_lw
      ax.plot(l[ibc,itg,0,0]/1000,l[ibc,itg,1,0]/1000,label= label, lw=lw)
  ax.set_xlabel("Simulation time (s)")
  ax.set_ylabel("Slice throughput (Mbps)")
  plt.subplots_adjust(bottom=0.35)
  handles, labels = ax.get_legend_handles_labels()
  ax.axhline(PC_TBS[10][slice_conf[0]['params']['mAR']-1]/1000,c='r',ls='--',lw=1,label='eMBB mAR (MCS:10)')
  ax.legend(loc='upper center', bbox_to_anchor=(leg_x, leg_y), ncol=3, columnspacing=0.5, prop={'size': leg_size})
  fig.savefig(fname+".pdf", format="pdf", bbox_inches='tight')
  #plt.show()
  plt.close(fig)


  fname = "delays-mix-zoom"
  fig, ax = plt.subplots()
  fig.set_figheight(plt_h)
  fig.set_figwidth(plt_w)
  l = lines[measures['delay-zoom']]
  for bc in sbcs:
    for tg in stgs:
      if bc == 0 and tg != 1:
        continue
      if tg ==1 and bc != 0:
        continue
      if bc == -1 and tg != 100:
        continue
      itg = time_gaps.index(tg)
      ibc = burst_capacities.index(bc)
      label = "BC:%d,TG:%d"%(bc,tg)
      lw = 1
      if bc == -1:
        label = "BC: \u221E"
        lw = bold_lw
      elif tg == 1:
        lw = bold_lw
      ax.plot(l[ibc,itg,0,1]/1000,l[ibc,itg,1,1],label=label, lw=lw)
  ax.set_xlabel("Simulation time (s)")
  ax.set_ylabel("Delay (ms)")
  plt.subplots_adjust(bottom=0.35)
  ax.legend(loc='upper center', bbox_to_anchor=(leg_x, leg_y), ncol=3, columnspacing=1, prop={'size': leg_size})
  fig.savefig(fname+".pdf", format="pdf", bbox_inches='tight')
  ax.set_ylim(0,22)
  #plt.show()
  plt.close(fig)

  fname = "kbps-mix-zoom"
  fig, ax = plt.subplots()
  fig.set_figheight(plt_h)
  fig.set_figwidth(plt_w)
  l = lines[measures['throughput-zoom']]
  for bc in sbcs:
    for tg in stgs:
      if bc == 0 and tg != 1:
        continue
      if tg ==1 and bc != 0:
        continue
      if bc == -1 and tg != 100:
        continue
      itg = time_gaps.index(tg)
      ibc = burst_capacities.index(bc)
      label = "BC:%d,TG:%d"%(bc,tg)
      lw = 1
      if bc == -1:
        label = "BC: \u221E"
        lw = bold_lw
      elif tg == 1:
        lw = bold_lw
      x = l[ibc,itg,0,0]/1000
      y = l[ibc,itg,1,0]/1000
      filter = (x != 0)*(y != 0)
      ax.plot(x[filter],y[filter],label= label, lw=lw)
  ax.set_xlabel("Simulation time (s)")
  ax.set_ylabel("Slice throughput (Mbps)")
  plt.subplots_adjust(bottom=0.35)
  handles, labels = ax.get_legend_handles_labels()
  ax.axhline(PC_TBS[10][slice_conf[0]['params']['mAR']-1]/1000,c='r',ls='--',lw=1,label='eMBB mAR (MCS:10)')
  ax.legend(loc='upper center', bbox_to_anchor=(leg_x, leg_y), ncol=3, columnspacing=0.5, prop={'size': leg_size})
  fig.savefig(fname+".pdf", format="pdf", bbox_inches='tight')
  #plt.show()
  plt.close(fig)


  #
  # fname = "delays-mix-xzoom"
  # fig, ax = plt.subplots()
  # fig.set_figheight(plt_h)
  # fig.set_figwidth(plt_w)
  # l = lines[measures['delay-xzoom']]
  # for bc in sbcs:
  #   for tg in stgs:
  #     if tg ==1 and bc != 0:
  #       continue
  #     if bc == -1 and tg != 1:
  #       continue
  #     itg = time_gaps.index(tg)
  #     ibc = burst_capacities.index(bc)
  #     label = "BC:%d,TG:%d"%(bc,tg)
  #     if bc == -1:
  #       label = "BC: \u221E"
  #     ax.plot(l[ibc,itg,0,1]/1000,l[ibc,itg,1,1],label=label, lw=1)
  # ax.set_xlabel("Simulation time (s)")
  # ax.set_ylabel("Delay (ms)")
  # plt.subplots_adjust(bottom=0.35)
  # ax.legend(loc='upper center', bbox_to_anchor=(leg_x, leg_y), ncol=3, columnspacing=1, prop={'size': leg_size})
  # fig.savefig(fname, bbox_inches='tight')
  # ax.set_ylim(0,22)
  # #plt.show()
  # plt.close(fig)

  for ibc,bc in enumerate(sbcs):
    for itg,tg in enumerate(stgs):
      p = (ibc,itg)
      if (bc,tg) in [(0,1),(250,100),(-1,100)] or True:
        print((bc,tg),lines[measures['in-profile'], p[0], p[1], 0,0,0])

