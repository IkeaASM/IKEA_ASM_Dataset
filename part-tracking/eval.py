# MOT Challenge evaluation code from: https://github.com/cheind/py-motmetrics

from __future__ import print_function
import motmetrics as mm
from utils.evaluation import Evaluator_mot
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', default="0002_white_floor_05_02_2019_08_19_17_47", required=True)
parser.add_argument('-f', help='name of furniture from [Kallax_Shelf_Drawer, Lack_Side_Table, Lack_Coffee_Table, Lack_TV_Bench]', required=True)
parser.add_argument('-root', help='path to GT up to name of directory before furniture name', required=True)

args = parser.parse_args()

sample = args.f + '/' + args.s
seq_names = [sample]

accs = []
for seq in seq_names:
        evaluator = Evaluator_mot(args.root, seq + '/dev3', 'mot')
        accs.append(evaluator.eval_file('tracking_1_percent_' + args.s + '.txt'))


metrics = mm.metrics.motchallenge_metrics
mh = mm.metrics.create()
summary = Evaluator_mot.get_summary(accs, seq_names, metrics)
strsummary = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
print(strsummary)
