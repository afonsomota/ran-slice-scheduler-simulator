from Simulator import Simulation
from IntraScheduler import SchedulerPF
import Util
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scipy

def plot_time_ipunt_traffic(slice_styles, slice_labels, sizes,  plt):
  # input traffic time series
  plt.set_title("Input traffic time series")
  plt.set_xlabel("Simulation time (s)")
  plt.set_ylabel("Input traffic (Kbytes)")
  binned = {}
  edges = {}
  ymax = 0.000001
  for s in slice_styles:
    if Util.isSliceFullBuffer(s,ue_conf):
      continue
    binned[s], edges[s], _ = stats.binned_statistic(sizes[s]['x'], sizes[s]['y'],
                                                    'sum', bins=100)
    binned[s] = binned[s]/1000
  for s in binned:
    smax = max(binned[s])
    if smax > ymax: ymax = smax
    plt.plot(edges[s][1:], binned[s], slice_styles[s], label=slice_labels[s])
  plt.set_ylim(0, 1.1 * ymax)
  plt.legend(loc="best")



def plot_time_overuses(slice_styles,slice_labels,results,plt):
  # overuses time series
  plt.set_title("Overused resources per slice")
  plt.set_xlabel("Simulation time (s)")
  plt.set_ylabel("Unused resoureces (RBs)")
  ymax = 0.000001
  #plt.plot(results[1]['overuses']['x'], results[1]['overuses']['y'], results[2]['overuses']['x'],
  #         results[2]['overuses']['y'])
  for s in [2,0,1]:
    smax = max(results[s]['overuses']['y'])
    if smax > ymax: ymax = smax
    binned, edges, _ = stats.binned_statistic(results[s]['overuses']['x'], results[s]['overuses']['y'],
                                                    'mean', bins=100)
    plt.plot(edges[1:],binned,slice_styles[s],label=slice_labels[s])
  plt.set_ylim(0,1.1*ymax)
  plt.legend(loc="best")

def plot_time_excesses(slice_styles,slice_labels,results,plt):
  # excesses time series
  plt.set_title("Unused resources per slice")
  plt.set_xlabel("Simulation time (s)")
  plt.set_ylabel("Unused resoureces (RBs)")
  ymax = 0.000001
  plt.plot(results[1]['excesses']['x'], results[1]['excesses']['y'], results[2]['excesses']['x'],
           results[2]['excesses']['y'], results[0]['excesses']['x'], results[0]['excesses']['y'])
  for s in slice_styles:
    smax = max(results[s]['excesses']['y'])
    if smax > ymax: ymax = smax
    plt.plot(results[s]['excesses']['x'],results[s]['excesses']['y'],slice_styles[s],label=slice_labels[s])
  plt.set_ylim(0,1.1*ymax)
  plt.legend(loc="best")


def plot_time_sched_kbps(slice_styles,slice_labels,results,plt):
  # bps time series
  plt.set_title("Slice throughput at scheduler time series")
  plt.set_xlabel("Simulation time (s)")
  plt.set_ylabel("Slice throughput (kbps)")
  binned = {}
  edges = {}
  ymax = 0.000001
  for s in slice_styles:
    binned[s], edges[s], _ = stats.binned_statistic(results[s]['cx'], results[s]['bits']['total'],
                                                    'mean', bins=100)
  for s in binned:
    smax = max(binned[s])
    if smax > ymax: ymax = smax
    plt.plot(edges[s][1:], binned[s], slice_styles[s], label=slice_labels[s])
  plt.set_ylim(0, 1.1 * ymax)
  plt.legend(loc="best")

def plot_time_sched_rbs(slice_styles,slice_labels,results,plt):
  # bps time series
  plt.set_title("Slice scheduled resources time series")
  plt.set_xlabel("Simulation time (s)")
  plt.set_ylabel("Resources (RBs)")
  binned = {}
  edges = {}
  ymax = 0.000001
  for s in slice_styles:
    binned[s], edges[s], _ = stats.binned_statistic(results[s]['cx'], results[s]['rbs']['total'],
                                                    'mean', bins=100)
  for s in binned:
    smax = max(binned[s])
    if smax > ymax: ymax = smax
    plt.plot(edges[s][1:], binned[s], slice_styles[s], label=slice_labels[s])
  plt.axhline(slice_conf[0]['params']['mAR'],c='r',ls='--',lw=1,label='eMBB mAR')
  plt.axhline(slice_conf[1]['params']['MPR'],c='g',ls='--',lw=1,label='URLLC MPR')
  plt.set_ylim(0, 1.1 * ymax)
  plt.legend(loc="best")

def plot_time_delay(slice_styles,slice_labels,time_delays,plt):
  # Delay time series
  plt.set_title("Slice delay time series")
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
    plt.plot(edges[s][1:], binned[s], slice_styles[s], label=slice_labels[s])
  ymax = 10
  plt.set_ylim(0, 1.1 * ymax)
  plt.legend(loc="best")


if __name__ == '__main__':

  setting = 'npa'

  slice_conf_pa = {
    1: {'type': 'P', 'params': {'MPR': 9, 'BC': 250}, 'scheduler': SchedulerPF()},
    2: {'type': 'R', 'params': {'mAR': 9, 'TG': 100}, 'scheduler': SchedulerPF()},
    0: {'type': 'R', 'params': {'mAR': 27, 'TG': 100}, 'scheduler': SchedulerPF()}
  }
  slice_conf_npa = {
    1: {'type': 'P', 'params': {'MPR': 9, 'BC': 9}, 'scheduler': SchedulerPF()},
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
    'timeout': 10,
    'tti': 0.001
  }
  counts = np.arange(5,13,1).astype(int)
  s1_tps = []
  s2_tps = []
  s0_tps = []
  sla_violations = {}
  for s in slices:
    sla_violations[s] = np.zeros(len(counts))

  time_delays = {}
  sizes = {}
  for s in slices:
    sizes[s] = {'x':[], 'y':[]}
    time_delays[s] = {}
    for k in ['x','y','sched']:
      time_delays[s][k] = []

  for idx,uec in enumerate(counts):
    print(uec)

    #traffic_seed = np.random.randint(0,10000,(len(slices),uec))
    traffic_seed = np.array([[ 745, 4379, 4034, 6676, 2055, 1691],
       [3638, 6790, 1110, 8794, 1393, 1470],
       [ 501, 5996, 7059, 2726, 1905, 1158]])

    ue_conf = [
      {
        'slice': 0,
        'traffic': 'full_buffer',
        'params': {
          'size': 3000,
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
            if pkt['scheduled'] > sim_conf['timeout'] / sim_conf['tti']:
              continue
            sizes[s]['x'].append(pkt['sent'])
          else:
            if pkt['scheduled'] > sim_conf['timeout'] / sim_conf['tti']:
              continue
            time_delays[s]['x'].append(pkt['scheduled'])
            sizes[s]['x'].append(pkt['scheduled'])
          sizes[s]['y'].append(pkt['size'])
          totals[s]['size']+=pkt['size']

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

    #Delay CDF
    fname = "delay-cdf-" + str(uec) + "-" + setting
    fig, ax = plt.subplots()
    for s in slices:
      if len(delays[s]) > 0:
        Util.plotCDF(delays[s],ax,slice_styles[s],slice_labels[s])
    #plt.show()
    ax.legend(loc="best")
    fig.savefig(fname)
    plt.close(fig)

    fname = "delay-time-" + str(uec) + "-" + setting
    fig, ax = plt.subplots()
    plot_time_delay(slice_styles,slice_labels,time_delays,ax)
    fig.savefig(fname)
    plt.close(fig)

    fname = "kbps-time-" + str(uec) + "-" + setting
    fig, ax = plt.subplots()
    plot_time_sched_kbps(slice_styles,slice_labels,results,ax)
    fig.savefig(fname)
    plt.close(fig)

    fname = 'excesses-time-' + str(uec) + "-" + setting
    fig, ax = plt.subplots()
    plot_time_excesses(slice_styles,slice_labels,results,ax)
    fig.savefig(fname)
    plt.close(fig)

    fname = 'overuses-time-' + str(uec) + "-" + setting
    fig, ax = plt.subplots()
    plot_time_overuses(slice_styles,slice_labels,results,ax)
    fig.savefig(fname)
    plt.close(fig)


    no_plots=4
    fig,axs = plt.subplots(1,3,sharex=True)
    fig.set_figheight(3)
    fig.set_figwidth(14)

    #plt.subplot(no_plots,1,1)
    plot_time_delay(slice_styles,slice_labels,time_delays,axs[2])

    #plt.subplot(no_plots,1,2)
    #plot_time_excesses(slice_styles,slice_labels,results,axs[0,1])
    #plot_time_overuses(slice_styles,slice_labels,results,axs[1,0])

    #plt.subplot(no_plots,1,3)
    plot_time_sched_rbs(slice_styles,slice_labels,results,axs[1])

    #plt.subplot(no_plots,1,4)
    plot_time_ipunt_traffic(slice_styles,slice_labels,sizes,axs[0])

    for ax in axs:
      #for ax in axh:
        ax.legend(loc='best')#, bbox_to_anchor=(0.9, 0.5),fancybox=True,shadow=True)

    fname = "bundle-" + str(uec) + "-" + setting

    fig.savefig(fname,bbox_inches='tight')
    plt.close(fig)

    if results[0]['tp'][0] != None:
      s0_tps.append(results[0]['tp'][0]/1000)
    else:
      s0_tps.append(0)
    if results[1]['tp'][0] != None:
      s1_tps.append(results[1]['tp'][0]/1000)
    else:
      s1_tps.append(0)
    if results[2]['tp'][0] != None:
      s2_tps.append(results[2]['tp'][0]/1000)
    else:
      s2_tps.append(0)

    for s in [0,1,2]:
      sla_violations[s][idx] = results[s]['violations']



  #plt.subplot(3,1,2)
  plt.plot(counts,s1_tps,counts,s2_tps,counts,s0_tps)
  plt.show()
  ##plt.yscale('log')
  #
  #plt.subplot(3,1,3)
  for s in slices:
    plt.plot(counts,sla_violations[s],slice_styles[s],label=slice_labels[s])
  plt.legend()
  plt.show()

