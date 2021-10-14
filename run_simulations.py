from Simulator import Simulation
from IntraScheduler import SchedulerPF
import Util
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import stats
import scipy
import pickle
from functools import reduce
from NRTables import PC_TBS


def fill_dls_tps(dls, tps, ue_metrics):
  max_dl = max_tp = 0
  for u in ue_metrics:
    s = ue_metrics[u]['slice']
    if s not in dls:
      dls[s] = []
      tps[s] = []
    for p in ue_metrics[u]['packets']:
      pkt = ue_metrics[u]['packets'][p]
      if 'recv' in pkt and 'sent' in pkt:
        delay = pkt['recv'] - pkt['sent']
        if delay > max_dl:
          max_dl = delay
        dls[s].append(delay)
      if 'recv' in pkt and 'scheduled' in pkt:
        tp = pkt['size'] * 8 / (pkt['recv'] - pkt['scheduled']) / sim_conf['tti'] / 1000
        if tp > max_tp:
          max_tp = tp
        tps[s].append(tp)
  return max_dl, max_tp


def makeCDFLine(list, bins, range):
  count, edges = np.histogram(list, bins=bins, range=range, density=True)
  cdf = np.cumsum(count)
  return edges[1:], cdf / cdf[-1]


if __name__ == '__main__':

  mpl.rcParams['legend.fontsize'] = 9

  urllc_target = {'delay': 5, 'reliability': 10 ** -5, 'primary': 'delay'}
  embb_target = {'system_throughput': PC_TBS[10][26], 'throughput_average_time': 100, 'delay': 100,
                 'primary': 'system_throughput'}
  mmtc_target = {'system_throughput': PC_TBS[10][8], 'throughput_average_time': 100, 'delay': 1000,
                 'primary': 'system_throughput'}

  slice_conf_pa = {
    1: {'type': 'P', 'target': urllc_target, 'params': {'MPR': 9, 'BC': 82, 'delta': 2}, 'scheduler': SchedulerPF()},
    2: {'type': 'R', 'target': mmtc_target, 'params': {'mAR': 9, 'TG': 100}, 'scheduler': SchedulerPF()},
    0: {'type': 'R', 'target': embb_target, 'params': {'mAR': 27, 'TG': 100}, 'scheduler': SchedulerPF()}
  }
  slice_conf_npa = {
    1: {'type': 'P', 'target': urllc_target, 'params': {'MPR': 10, 'BC': 10, 'delta': 2}, 'scheduler': SchedulerPF()},
    2: {'type': 'R', 'target': mmtc_target, 'params': {'mAR': 9, 'TG': 1}, 'scheduler': SchedulerPF()},
    0: {'type': 'R', 'target': embb_target, 'params': {'mAR': 27, 'TG': 1}, 'scheduler': SchedulerPF()}
  }
  slice_conf_up = {
    1: {'type': 'P', 'target': urllc_target, 'params': {'MPR': 50, 'BC': 100000000, 'delta': 2}, 'scheduler': SchedulerPF()},
    2: {'type': 'R', 'target': mmtc_target, 'params': {'mAR': 9, 'TG': 100}, 'scheduler': SchedulerPF()},
    0: {'type': 'R', 'target': embb_target, 'params': {'mAR': 27, 'TG': 100}, 'scheduler': SchedulerPF()}
  }

  target = {
    'delay': 0,  # ms
    'system_throughput': 0,  # kbits
    'user_throughput': 0,  # kbits
    'throughput_average_time': 0,  # ms
    'reliability': 10 ** -0,
    'packet_loss': 10 ** -0  # reliability = packet_loss +
  }

  slices = slice_conf_pa.keys()

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
    'bw': 50,
    'tti': 0.001,
    'cqi_report_period': 1,
    'warmup': 0
  }
  counts = [8, 4, 6, 10]
  s1_tps = []
  s2_tps = []
  s0_tps = []
  mcss = [(None, 10.5), (None, 14.5), (None, 19.5), (10, 80)]

  targets_per_ue = {}
  for iuec, uec in enumerate(counts):
    targets_per_mcs = {}
    for idx, (mcs, sinr) in enumerate(mcss):

      traffic_seed = np.random.randint(0, 12000, (3, 6))

      ue_conf = [
        {
          'slice': 0,
          'traffic': 'full_buffer',
          'params': {
            'size': 1500,
            'traffic_seed': traffic_seed[0]
          },
          'count': uec
        },
        {
          'slice': 1,
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
          'count': uec
        },
        {
          'slice': 2,
          'traffic': 'cbr',
          'params': {
            'interval': 0.01,
            'jitter': 0.002,
            'size': 300,
            'drift': 0.01,
            'traffic_seed': traffic_seed[2]
          },
          'count': uec
        }
      ]

      for u in ue_conf:
        if mcs is not None:
          u['params']['mcs'] = mcs
        if sinr is not None:
          u['params']['sinr'] = sinr

      if mcs is None:
        mcs = -1

      try:
        ue_metrics_pa = pickle.load(open("ue_metrics_pa-%d-%d-%d.pkl" % (uec, mcs, sinr), 'rb'))
        results_pa = pickle.load(open("results_pa-%d-%d-%d.pkl" % (uec, mcs, sinr), 'rb'))
        ue_metrics_npa = pickle.load(open("ue_metrics_npa-%d-%d-%d.pkl" % (uec, mcs, sinr), 'rb'))
        results_npa = pickle.load(open("results_npa-%d-%d-%d.pkl" % (uec, mcs, sinr), 'rb'))
        ue_metrics_up = pickle.load(open("ue_metrics_up-%d-%d-%d.pkl" % (uec, mcs, sinr), 'rb'))
        results_up = pickle.load(open("results_up-%d-%d-%d.pkl" % (uec, mcs, sinr), 'rb'))
        loaded = True
      except:
        loaded = False

      bins = 100
      no_its = 3


      if not loaded:
        ue_metrics_pa = []
        results_pa = []
        ue_metrics_up = []
        results_up = []
        ue_metrics_npa = []
        results_npa = []
        for i in range(no_its):
          sim_pa = Simulation(in_sim_conf=sim_conf, slice_conf=slice_conf_pa, ue_conf=ue_conf, verbose=False)
          uem, slim = sim_pa.run()
          ue_metrics_pa.append(uem)
          results_pa.append(slim)

          sim_npa = Simulation(in_sim_conf=sim_conf, slice_conf=slice_conf_npa, ue_conf=ue_conf, verbose=False)
          uem, slim = sim_npa.run()
          ue_metrics_npa.append(uem)
          results_npa.append(slim)

          sim_up = Simulation(in_sim_conf=sim_conf, slice_conf=slice_conf_up, ue_conf=ue_conf, verbose=False)
          uem, slim = sim_up.run()
          ue_metrics_up.append(uem)
          results_up.append(slim)

        pickle.dump(ue_metrics_pa, open('ue_metrics_pa-%d-%d-%d.pkl' % (uec, mcs, sinr), 'wb'))
        pickle.dump(ue_metrics_npa, open('ue_metrics_npa-%d-%d-%d.pkl' % (uec, mcs, sinr), 'wb'))
        pickle.dump(results_pa, open('results_pa-%d-%d-%d.pkl' % (uec, mcs, sinr), 'wb'))
        pickle.dump(results_npa, open('results_npa-%d-%d-%d.pkl' % (uec, mcs, sinr), 'wb'))
        pickle.dump(ue_metrics_up, open('ue_metrics_up-%d-%d-%d.pkl' % (uec, mcs, sinr), 'wb'))
        pickle.dump(results_up, open('results_up-%d-%d-%d.pkl' % (uec, mcs, sinr), 'wb'))
      print(uec, "done!", flush=True)

      delays_pa = []
      delays_npa = []
      delays_up = []
      throughputs_pa = []
      throughputs_npa = []
      throughputs_up = []
      max_dl = max_tp = 0
      for i in range(no_its):
        dls_pa = {}
        tps_pa = {}
        dls_npa = {}
        tps_npa = {}
        dls_up = {}
        tps_up = {}
        mdlspa, mtpspa = fill_dls_tps(dls_pa, tps_pa, ue_metrics_pa[i])
        mdlsnpa, mtpsnpa = fill_dls_tps(dls_npa, tps_npa, ue_metrics_npa[i])
        mdlsup, mtpsup = fill_dls_tps(dls_up, tps_up, ue_metrics_up[i])
        delays_pa.append(dls_pa)
        if mdlspa > max_dl:
          max_dl = mdlspa
        delays_npa.append(dls_npa)
        if mdlsnpa > max_dl:
          max_dl = mdlsnpa
        throughputs_pa.append(tps_pa)
        if mtpspa > max_tp:
          max_tp = mtpspa
        throughputs_npa.append(tps_npa)
        if mtpsnpa > max_tp:
          max_tp = mtpsnpa
        delays_up.append(dls_up)
        if mdlsup > max_dl:
          max_dl = mdlsup
        throughputs_up.append(tps_up)
        if mtpsup > max_tp:
          max_tp = mtpsup

      max_stp = reduce(max,
                       map(lambda x: reduce(max,
                                            map(lambda y: reduce(max,
                                                                 map(lambda z: max(z['bits']['total']),
                                                                     y.values())
                                                                 ),
                                                x)
                                            ),
                           [results_pa, results_npa, results_up]
                           )
                       )
      max_target = reduce(max,
                          map(lambda x: reduce(max,
                                               map(lambda y: reduce(max,
                                                                    map(lambda z: max(z[1]['target_metric'][
                                                                                        slice_conf_pa[z[0]]['target'][
                                                                                          'primary']]['y']),
                                                                        y.items())
                                                                    ),
                                                   x)
                                               ),
                              [results_pa, results_npa, results_up]
                              )
                          )

      dls_cum_pa = np.zeros((len(slices), no_its, bins))
      dls_cum_npa = np.zeros((len(slices), no_its, bins))
      dls_cum_up = np.zeros((len(slices), no_its, bins))
      tps_cum_pa = np.zeros((len(slices), no_its, bins))
      tps_cum_npa = np.zeros((len(slices), no_its, bins))
      dtps_cum_pa = np.zeros((len(slices), no_its, bins))
      dtps_cum_npa = np.zeros((len(slices), no_its, bins))
      stps_cum_pa = np.zeros((len(slices), no_its, bins))
      stps_cum_npa = np.zeros((len(slices), no_its, bins))
      stps_cum_up = np.zeros((len(slices), no_its, bins))
      sftps_cum_pa = np.zeros((len(slices), no_its, bins))
      sftps_cum_npa = np.zeros((len(slices), no_its, bins))
      target_cum = np.zeros((3, len(slices), no_its, bins))
      x_dls, x_tps, x_dtps, x_stps, x_target = None, None, None, None, None

      for i in range(no_its):
        for s in slices:
          max_dl = 20
          _, dls_cum_pa[s][i] = makeCDFLine(delays_pa[i][s], bins, (4, max_dl))
          _, dls_cum_up[s][i] = makeCDFLine(delays_up[i][s], bins, (4, max_dl))
          x_dls, dls_cum_npa[s][i] = makeCDFLine(delays_npa[i][s], bins, (4, max_dl))
          _, tps_cum_pa[s][i] = makeCDFLine(throughputs_pa[i][s], bins, (0, max_tp))
          x_tps, tps_cum_npa[s][i] = makeCDFLine(throughputs_npa[i][s], bins, (0, max_tp))
          _, sftps_cum_pa[s][i] = makeCDFLine(results_pa[i][s]['bits']['total'], bins, (0, max_stp))
          tocdf, _, _ = stats.binned_statistic(results_pa[i][s]['cx'], results_pa[i][s]['bits']['total'], 'mean',
                                               bins=100)
          _, stps_cum_pa[s][i] = makeCDFLine(tocdf, bins, (0, max_stp))
          tocdf, _, _ = stats.binned_statistic(results_up[i][s]['cx'], results_up[i][s]['bits']['total'], 'mean',
                                               bins=100)
          _, stps_cum_up[s][i] = makeCDFLine(tocdf, bins, (0, max_stp))
          _, sftps_cum_npa[s][i] = makeCDFLine(results_npa[i][s]['bits']['total'], bins, (0, max_stp))
          tocdf, _, _ = stats.binned_statistic(results_npa[i][s]['cx'], results_npa[i][s]['bits']['total'], 'mean',
                                               bins=100)
          x_stps, stps_cum_npa[s][i] = makeCDFLine(tocdf, bins, (0, max_stp))
          max_target = 2.0
          _, target_cum[0][s][i] = makeCDFLine(
            results_pa[i][s]['target_metric'][slice_conf_pa[s]['target']['primary']]['y'], bins, (0, max_target))
          _, target_cum[1][s][i] = makeCDFLine(
            results_npa[i][s]['target_metric'][slice_conf_pa[s]['target']['primary']]['y'], bins, (0, max_target))
          x_target, target_cum[2][s][i] = makeCDFLine(
            results_up[i][s]['target_metric'][slice_conf_up[s]['target']['primary']]['y'], bins, (0, max_target))
          s_size = None
          max_size = 0
          for u in ue_conf:
            size = u['params']['size']
            if size > max_size:
              max_size = size
            if s == u['slice']:
              s_size = size

          _, dtps_cum_pa[s][i] = makeCDFLine(uec * s_size * 8 / np.array(delays_pa[i][s]), bins,
                                             (0, 4 * max_size * 8 / 3))
          x_dtps, dtps_cum_npa[s][i] = makeCDFLine(uec * s_size * 8 / np.array(delays_npa[i][s]), bins,
                                                   (0, 4 * max_size * 8 / 3))


      dls_cdf_avg_pa = np.mean(dls_cum_pa, axis=1)
      dls_cdf_avg_npa = np.mean(dls_cum_npa, axis=1)
      dls_cdf_avg_up = np.mean(dls_cum_up, axis=1)
      tps_cdf_avg_pa = np.mean(tps_cum_pa, axis=1)
      tps_cdf_avg_npa = np.mean(tps_cum_npa, axis=1)
      dtps_cdf_avg_pa = np.mean(dtps_cum_pa, axis=1)
      dtps_cdf_avg_npa = np.mean(dtps_cum_npa, axis=1)
      stps_cdf_avg_pa = np.mean(stps_cum_pa, axis=1)
      stps_cdf_avg_npa = np.mean(stps_cum_npa, axis=1)
      stps_cdf_avg_up = np.mean(stps_cum_up, axis=1)

      dls_cdf_std_pa = np.std(dls_cum_pa, axis=1)
      dls_cdf_std_npa = np.std(dls_cum_npa, axis=1)
      dls_cdf_std_up = np.std(dls_cum_up, axis=1)
      tps_cdf_std_pa = np.std(tps_cum_pa, axis=1)
      tps_cdf_std_npa = np.std(tps_cum_npa, axis=1)
      dtps_cdf_std_pa = np.std(dtps_cum_pa, axis=1)
      dtps_cdf_std_npa = np.std(dtps_cum_npa, axis=1)
      stps_cdf_std_pa = np.std(stps_cum_pa, axis=1)
      stps_cdf_std_npa = np.std(stps_cum_npa, axis=1)
      stps_cdf_std_up = np.std(stps_cum_up, axis=1)

      ci = (lambda std: 1.96 * std / np.sqrt(len(std)))

      # plt_h = 4.5
      # plt_w = 5.5
      plt_h = 3
      plt_w = 4

      leg_x = 0.47
      leg_y = -0.3

      # Target metric CDF
      fname = "target-cdf-ue_%d-mcs_%d-%d" % (uec, mcs, sinr)
      fig, ax = plt.subplots()
      fig.set_figheight(plt_h)
      fig.set_figwidth(plt_w)
      ax.set_xlabel("Target metric value")
      for s in [2, 0, 1]:
        y = np.mean(target_cum[0][s], axis=0)
        err = ci(np.std(target_cum[0][s], axis=0))
        ax.errorbar(x_target, y, fmt=slice_styles[s], yerr=err, lw=2, label="%s PA" % slice_labels[s])
        y = np.mean(target_cum[1][s], axis=0)
        err = ci(np.std(target_cum[1][s], axis=0))
        ax.errorbar(x_target, y, fmt=slice_styles[s], yerr=err, lw=1, label="%s non-PA" % slice_labels[s])
        y = np.mean(target_cum[2][s], axis=0)
        err = ci(np.std(target_cum[2][s], axis=0))
        ax.errorbar(x_target, y, fmt=slice_styles[s], yerr=err, lw=1, ls="--", label="%s UP" % slice_labels[s])
      ax.axvline(1, c='b', ls='--', lw=1)

      plt.subplots_adjust(bottom=0.35)
      ax.legend(loc='upper center', bbox_to_anchor=(leg_x, leg_y), ncol=3, columnspacing=1)
      fig.savefig(fname, bbox_inches='tight')
      plt.close(fig)

      if uec == 8 and mcs == 10:
        fname = "target-example"
        fig, ax = plt.subplots()
        fig.set_figheight(plt_h)
        fig.set_figwidth(plt_w)
        ax.set_xlabel("Target metric value")
        optimal_x = (x_target - 1)*0.4 + 1.015
        optimal_x[0] = 0
        mask = optimal_x <= 2
        optimal = np.mean(target_cum[0][2], axis=0)[mask]
        optimal_x = optimal_x[mask]
        mask = x_target <= 2
        sub_optimal = np.mean(target_cum[0][1], axis=0)[mask]
        sub_optimal_x = x_target[mask]
        bad = np.mean(target_cum[0][1], axis=0)
        bad_x = x_target * 1.3
        last_x = x_target[-1]
        mask = bad_x <= 2
        bad = bad[mask]
        bad_x = bad_x[mask]

        ax.axvline(1,  ls='--', lw=1)
        ax.errorbar(optimal_x, optimal, c='orange', lw=2, yerr=0, label="Slice 1")
        ax.errorbar(sub_optimal_x, sub_optimal, c='blue', lw=2, yerr=0, label="Slice 2")
        ax.errorbar(bad_x, bad, lw=2, c = 'brown', yerr=0, label="Slice 3")
        plt.subplots_adjust(bottom=0.35)
        ax.legend(loc='upper center', bbox_to_anchor=(leg_x, leg_y), ncol=3, columnspacing=1)
        fig.savefig(fname, bbox_inches='tight')
        plt.close(fig)

      targets_per_mcs[mcs] = {'pa': {}, 'npa': {}, 'up': {}}
      for s in slices:
        for k in targets_per_mcs[mcs]:
          targets_per_mcs[mcs][k][s] = {}
        targets_per_mcs[mcs]['pa'][s]['y'] = np.mean(target_cum[0][s], axis=0)
        targets_per_mcs[mcs]['pa'][s]['err'] = ci(np.std(target_cum[0][s], axis=0))
        targets_per_mcs[mcs]['npa'][s]['y'] = np.mean(target_cum[1][s], axis=0)
        targets_per_mcs[mcs]['npa'][s]['err'] = ci(np.std(target_cum[1][s], axis=0))
        targets_per_mcs[mcs]['up'][s]['y'] = np.mean(target_cum[2][s], axis=0)
        targets_per_mcs[mcs]['up'][s]['err'] = ci(np.std(target_cum[2][s], axis=0))
      targets_per_mcs[mcs]['pa']['x'] = x_target
      targets_per_mcs[mcs]['npa']['x'] = x_target
      targets_per_mcs[mcs]['up']['x'] = x_target

      # Delay CDF
      fname = "delay-cdf-ue_%d-mcs_%d-%d" % (uec, mcs, sinr)
      fig, ax = plt.subplots()
      fig.set_figheight(plt_h)
      fig.set_figwidth(plt_w)

      idx = np.abs(x_dls - 5).argmin()
      print(idx, x_dls[idx])

      # ax.set_title("CDF for delay (ms)")
      ax.set_xlabel("Delay (ms)")
      ax.axvline(5, c='b', ls='--', lw=1, label="Scheduled on 1st TTI")
      for s in slices:
        if len(delays_pa[0][s]) == 0:
          continue
        y = dls_cdf_avg_pa[s]
        err = ci(dls_cdf_std_pa[s])
        if s == 1:
          sla_pa = y[idx]
        ax.errorbar(x_dls, y, fmt=slice_styles[s], yerr=err, lw=2, label="%s PA" % slice_labels[s])
        y = dls_cdf_avg_npa[s]
        err = ci(dls_cdf_std_npa[s])
        if s == 1:
          sla_npa = y[idx]
        ax.errorbar(x_dls, y, fmt=slice_styles[s], yerr=err, lw=1, label="%s non-PA" % slice_labels[s])
        y = dls_cdf_avg_up[s]
        err = ci(dls_cdf_std_up[s])
        if s == 1:
          sla_up = y[idx]
        ax.errorbar(x_dls, y, fmt=slice_styles[s], yerr=err, lw=1, ls="--", label="%s UP" % slice_labels[s])

      sla = (lambda xs: sum(np.array(xs) < 5) / len(xs))
      print(uec, mcs, "PA:", sla_pa, sla(dls_pa[1]), "NPA:", sla_npa, sla(dls_npa[1]), "UP:", sla_up, sla(dls_up[1]))
      handles, labels = ax.get_legend_handles_labels()
      handles, labels = Util.orderLegend(handles, labels, [1, 2, 3, 4, 5, 6, 0])
      plt.subplots_adjust(bottom=0.25)
      ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(leg_x, leg_y), ncol=2, columnspacing=1)
      # plt.show()
      ax.set_xlim(3, 20)
      ax.set_ylim(0, 1.05)
      fig.savefig(fname, bbox_inches='tight')
      plt.close(fig)

      # Throughput CDF
      fname = "ptp-cdf-" + str(uec)
      fig, ax = plt.subplots()
      ax.set_title("CDF for scheduled throughput (kbps)")
      for s in slices:
        y = tps_cdf_avg_pa[s]
        err = ci(tps_cdf_std_pa[s])
        ax.errorbar(x_tps, y, fmt=slice_styles[s], yerr=err, lw=2, label="%s PA" % slice_labels[s])
        y = tps_cdf_avg_npa[s]
        err = ci(tps_cdf_std_npa[s])
        ax.errorbar(x_tps, y, fmt=slice_styles[s], yerr=err, lw=1, label="%s non-PA" % slice_labels[s])

      ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
      # fig.savefig(fname,bbox_inches='tight')
      # plt.show()

      plt.close(fig)

      # Throughput (delay-based) CDF
      fname = "dtp-cdf-" + str(uec)
      fig, ax = plt.subplots()
      ax.set_title("CDF for throughput (kbps)")
      for s in slices:
        y = dtps_cdf_avg_pa[s]
        err = ci(dtps_cdf_std_pa[s])
        ax.errorbar(x_dtps, y, fmt=slice_styles[s], yerr=err, lw=2, label="%s PA" % slice_labels[s])
        y = dtps_cdf_avg_npa[s]
        err = ci(dtps_cdf_std_npa[s])
        ax.errorbar(x_dtps, y, fmt=slice_styles[s], yerr=err, lw=1, label="%s non-PA" % slice_labels[s])

      ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
      # fig.savefig(fname,bbox_inches='tight')
      # plt.show()

      plt.close(fig)

      # Throughput at scheduler CDF
      fname = "tp-cdf-ue_%d-mcs_%d-%d" % (uec, mcs, sinr)
      fig, ax = plt.subplots()
      fig.set_figheight(plt_h)
      fig.set_figwidth(plt_w)
      # ax.set_title("CDF for throughput at the scheduler (kbps)")
      ax.set_xlabel("Throughput (Mbps)")
      for s in slices:
        y = stps_cdf_avg_pa[s]
        err = ci(stps_cdf_std_pa[s])
        ax.errorbar(x_stps / 1000, y, fmt=slice_styles[s], yerr=err, lw=2, label="%s PA" % slice_labels[s])
        y = stps_cdf_avg_npa[s]
        err = ci(stps_cdf_std_npa[s])
        ax.errorbar(x_stps / 1000, y, fmt=slice_styles[s], yerr=err, lw=1, label="%s non-PA" % slice_labels[s])
        y = stps_cdf_avg_up[s]
        err = ci(stps_cdf_std_up[s])
        ax.errorbar(x_stps / 1000, y, fmt=slice_styles[s], yerr=err, lw=1, ls='--', label="%s UP" % slice_labels[s])
      ax.axvline(x=PC_TBS[10][26] / 1000, c='r', ls='--', lw=0.5, label="eMBB mAR (MCS: 10)")
      ax.axvline(x=PC_TBS[10][8] / 1000, c='y', ls='--', lw=0.5, label="mMTC mAR (MCS:10)")
      # ax.axvline(x=29,c='r',ls='--',lw=1)
      # ax.axvline(x=8,c='y',ls='--',lw=1)

      handles, labels = ax.get_legend_handles_labels()
      handles, labels = Util.orderLegend(handles, labels, [2, 3, 4, 5, 6, 7,  8, 9, 10, 0, 1 ])
      plt.subplots_adjust(bottom=0.35)
      ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(leg_x, leg_y), ncol=2, columnspacing=0.5)
      fig.savefig(fname, bbox_inches='tight')
      # plt.show()

      plt.close(fig)

      # Fine throughput at scheduler CDF
      fname = "sftp-cdf-" + str(uec)
      fig, ax = plt.subplots()
      ax.set_title("CDF for fine throughput at the scheduler (kbps)")
      for s in slices:
        y = np.mean(sftps_cum_pa[s], axis=0)
        err = ci(np.std(sftps_cum_pa[s], axis=0))
        ax.errorbar(x_stps, y, fmt=slice_styles[s], yerr=err, lw=2, label="%s PA" % slice_labels[s])
        y = np.mean(sftps_cum_npa[s], axis=0)
        err = ci(np.std(sftps_cum_npa[s], axis=0))
        ax.errorbar(x_stps, y, fmt=slice_styles[s], yerr=err, lw=1, label="%s non-PA" % slice_labels[s])
      ax.axvline(x=PC_TBS[10][26], c='r', ls='--', lw=1)
      ax.axvline(x=PC_TBS[10][8], c='y', ls='--', lw=1)
      # ax.axvline(x=29,c='r',ls='--',lw=1)
      # ax.axvline(x=8,c='y',ls='--',lw=1)

      ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
      # fig.savefig(fname,bbox_inches='tight')
      # plt.show()

      plt.close(fig)

    if len(mcss) > 1:
      fname = "mcs-target-%d" % uec
      fig, axs = plt.subplots(1, len(targets_per_mcs))
      fig.set_figheight(4)
      fig.set_figwidth(len(targets_per_mcs) * 5)
      for i, m in enumerate(targets_per_mcs.keys()):
        ax = axs[i]
        ax.set_title("MCS: %d" % m)
        ax.set_xlabel("Target metric value")
        for s in slices:
          y = targets_per_mcs[m]['pa'][s]['y']
          err = targets_per_mcs[m]['pa'][s]['err']
          x = targets_per_mcs[m]['pa']['x']
          ax.errorbar(x, y, fmt=slice_styles[s], yerr=err, lw=2, label="%s PA" % slice_labels[s])
          y = targets_per_mcs[m]['npa'][s]['y']
          err = targets_per_mcs[m]['npa'][s]['err']
          x = targets_per_mcs[m]['npa']['x']
          ax.errorbar(x, y, fmt=slice_styles[s], yerr=err, lw=1, label="%s non-PA" % slice_labels[s])
          y = targets_per_mcs[m]['up'][s]['y']
          err = targets_per_mcs[m]['up'][s]['err']
          x = targets_per_mcs[m]['up']['x']
          ax.errorbar(x, y, fmt=slice_styles[s], yerr=err, lw=1, ls='--', label="%s UP" % slice_labels[s])
        ax.axvline(1, c='b', ls='--', lw=1)
        handles, labels = ax.get_legend_handles_labels()
      fig.legend(handles, labels, loc='lower center',  # bbox_to_anchor=(0, 0),
                 shadow=True, ncol=9)
      plt.subplots_adjust(bottom=0.35)
      fig.savefig(fname, bbox_inches='tight')
      # plt.show()

    targets_per_ue[uec] = targets_per_mcs[10]

  if len(targets_per_ue) > 1:
    fname = "ue-target"
    fig, axs = plt.subplots(1, len(targets_per_ue))
    fig.set_figheight(4)
    fig.set_figwidth(len(targets_per_ue) * 5)
    for i, m in enumerate(targets_per_ue.keys()):
      ax = axs[i]
      ax.set_title("%d UEs" % m)
      ax.set_xlabel("Target metric value")
      for s in slices:
        y = targets_per_ue[m]['pa'][s]['y']
        err = targets_per_ue[m]['pa'][s]['err']
        x = targets_per_ue[m]['pa']['x']
        ax.errorbar(x, y, fmt=slice_styles[s], yerr=err, lw=2, label="%s PA" % slice_labels[s])
        y = targets_per_ue[m]['npa'][s]['y']
        err = targets_per_ue[m]['npa'][s]['err']
        x = targets_per_ue[m]['npa']['x']
        ax.errorbar(x, y, fmt=slice_styles[s], yerr=err, lw=1, label="%s non-PA" % slice_labels[s])
        y = targets_per_ue[m]['up'][s]['y']
        err = targets_per_ue[m]['up'][s]['err']
        x = targets_per_ue[m]['up']['x']
        ax.errorbar(x, y, fmt=slice_styles[s], yerr=err, lw=1, ls='--', label="%s UP" % slice_labels[s])
      ax.axvline(1, c='b', ls='--', lw=1)
      handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',  # bbox_to_anchor=(0, 0),
               shadow=True, ncol=9)
    plt.subplots_adjust(bottom=0.25)
    fig.savefig(fname, bbox_inches='tight')
    # plt.show()
