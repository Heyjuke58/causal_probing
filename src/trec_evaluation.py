import argparse
import subprocess
from pathlib import Path
import os
from typing import Dict, Any

import pandas as pd
from ranking_utils import write_trec_eval_file
from scipy import stats
import scipy.stats
import pytrec_eval


def get_map_score(ifname, METRIC_TYPE, qrel_file, trec_eval_file):
    metric = 0.0
    ARG = trec_eval_file + " -m map -m recip_rank -m P.10,20 -m ndcg_cut.10,20 " + qrel_file + " " + ifname
    ARG.split()

    p = subprocess.Popen(ARG, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout.readlines():
        line = line.split()
        T = str(line[0], encoding='utf-8')
        if T == METRIC_TYPE:
            metric = float(line[2])
            break
    return metric


# ToDo: get statistical significance score with other metrics
def trec_evaluation(out_file, model_name, qrel_file, trec_eval_file, result_file, score):
    fo = open(out_file, 'a')
    # fo.write('alpha' + '\t' + 'MAP' + '\t' + 'nDCG20' + '\t' + 'P20' + '\t' + 'RR' + '\t' + model_name + '\n')
    MAP = get_map_score(result_file, 'map', qrel_file, trec_eval_file)
    NDCG_10 = get_map_score(result_file, 'ndcg_cut_10', qrel_file, trec_eval_file)
    NDCG = get_map_score(result_file, 'ndcg_cut_20', qrel_file, trec_eval_file)
    P10 = get_map_score(result_file, 'P_10', qrel_file, trec_eval_file)
    P20 = get_map_score(result_file, 'P_20', qrel_file, trec_eval_file)
    RR = get_map_score(result_file, 'recip_rank', qrel_file, trec_eval_file)

    s = str(model_name) + '\t' + str(MAP) + '\t' + str(NDCG_10) + '\t' + str(NDCG) + '\t' + str(P10) + '\t' + str(
        P20) + '\t' + str(RR) + '\t' + str(score) + '\t' + str('')
    print(s)
    fo.write(s + '\n')
    fo.close()


def statistical_significance(qrel_file, file_1, file_2, metric):
    with open(qrel_file, 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)

    with open(file_1, 'r') as f_run:
        first_run = pytrec_eval.parse_run(f_run)

    with open(file_2, 'r') as f_run:
        second_run = pytrec_eval.parse_run(f_run)

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel, {metric})

    first_results = evaluator.evaluate(first_run)
    second_results = evaluator.evaluate(second_run)

    query_ids = list(
        set(first_results.keys()) & set(second_results.keys()))

    first_scores = [
        first_results[query_id][metric] for query_id in query_ids]
    second_scores = [
        second_results[query_id][metric] for query_id in query_ids]

    score = scipy.stats.ttest_rel(first_scores, second_scores)
    print('Metric:{} , statistical score:{}'.format(metric, score))
    return score


class TrecEvaluation:
    def __init__(self, hparams: Dict[str, Any]):
        self.test_name = hparams['test_name']
        self.qrel_file_20 = hparams['qrel_file_20']
        self.qrel_file = hparams['qrel_file']
        self.trec_eval_file = hparams['trec_eval_file']
        self.save_dir = hparams['save_dir']
        self.stat_sig_base_path = hparams['stat_sig_base_path']
        self.stat_sig_base_path_20 = hparams['stat_sig_base_path_20']

    def log_and_results(self, log_path, out_file_name: str, test_trecdl_20: bool = False):
        store_model_path = [log_path]

        prediction_files = []
        for file in os.listdir(log_path):
            if file.endswith(".pkl"):
                prediction_files.append(os.path.join(log_path, file))

        print('Reading and Writing predictions for TREC evaluation')
        predictions = read_predictions(map(Path, prediction_files))
        if test_trecdl_20:
            model_name = self.test_name + '_20'
            result_file = log_path + 'trec_20_' + str(self.test_name) + '.tsv'
            qrel_file = self.qrel_file_20
            print('Qrel file:{}'.format(qrel_file))

        else:
            model_name = self.test_name
            result_file = log_path + 'trec_' + str(self.test_name) + '.tsv'
            qrel_file = self.qrel_file
            print('Qrel file:{}'.format(qrel_file))
        print("Result_file: {}".format(result_file))
        write_trec_eval_file(Path(result_file), predictions, self.test_name)

        # statistical Significance Test
        if test_trecdl_20:
            print('Significance computed with(baseline):{}'.format(self.stat_sig_base_path))
            score = statistical_significance(self.qrel_file_20, self.stat_sig_base_path_20, result_file, 'map')
        else:
            print('Significance computed with(baseline):{}'.format(self.stat_sig_base_path))
            score = statistical_significance(self.qrel_file, self.stat_sig_base_path, result_file, 'map')

        # For trec evaluation
        out_file = self.save_dir + '/' + out_file_name
        trec_eval_file = self.trec_eval_file
        print('TREC EVAL file path: {}'.format(trec_eval_file))
        trec_evaluation(out_file, model_name, qrel_file, trec_eval_file, result_file, score)
        print('Result file after trec evaluation:{}'.format(result_file))
        print('Final Evaluation path: {}'.format(out_file))


def stattistical_ttest_rel(base_file, final_file):
    colnames = ['logit', 'rev_prob', 'prob']
    base = pd.read_csv(base_file, delimiter="\t", names=colnames, header=None)
    final = pd.read_csv(final_file, delimiter="\t", names=colnames, header=None)
    print(stats.ttest_rel(base['prob'], final['prob']))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--save_dir', default='out', help='Directory for logs, checkpoints and predictions')
    ap.add_argument('--test_name', type=str, default="", help='Name of the test')
    ap.add_argument('--test_trecdl_20', action='store_true', help='Set True for TRECDL-20 test')
    ap.add_argument('--qrel_file', type=str, default="/home/hinrichs/causal_probing/assets/msmarco/2019-qrels-pass.txt", help='Path to qrel file')
    ap.add_argument('--qrel_file_20', type=str, default="/home/aanand/qrels/2020-qrels-doc.txt",
                    help='Path to 2020 qrel file')
    ap.add_argument('--trec_eval_file', type=str, default="/home/hinrichs/causal_probing/trec_eval",
                    help='Path to trec evalation script')
    ap.add_argument('--version_num', type=str, default="version_0", help='Version to calculate results')
    ap.add_argument('--stat_sig_base_path', type=str,
                    default="/home/aanand/data/model/scl/test_vic_1k_glove/lightning_logs/version_7/trec_vic_test_1k_glove.tsv",
                    help='base file for stat significance')
    ap.add_argument('--trec_eval', action='store_true', help='Set True for TRECDL test')

    args = ap.parse_args()
    if args.trec_eval:
        eval = TrecEvaluation(vars(args))
        out_dir = Path(args.save_dir)
        log_path = str(out_dir / 'lightning_logs' / str(args.version_num)) + str('/')
        eval.log_and_results(log_path, 0.4, 1, 'results.txt')

        if args.test_trecdl_20:
            print('Prediction for TRECDL-20')
            eval.log_and_results(log_path, 'results_20.txt', True)
    else:
        base_file = '/home/aanand/data/result/vic/vic_test_cont/0.1/0.0/fold1_robust_9.txt'
        final_file = '/home/aanand/data/result/vic/vic_test_cont/0.1/1.0/fold1_robust_0.txt'
        # statistical_significance(args.qrel_file, base_file, final_file)
        stattistical_ttest_rel(base_file, final_file)
