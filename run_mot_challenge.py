

import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

ROOT=os.path.dirname(__file__)
# Run code
from trackeval.eval import Evaluator
from trackeval.datasets.mot_challenge_2d_box import MotChallenge2DBox
from trackeval.metrics import *

if __name__ == '__main__':

    config={}

    config['TRACKERS_FOLDER'] = ROOT+'/data/predect_mot'
    config['GT_FOLDER'] = ROOT+'/data/mot17_gt'  # 给出gt路径
    config['OUTPUT_FOLDER'] = ROOT+'/data/out_dir'

    # 确定文件内gt.txt的路径，gt_folder=config['GT_FOLDER']，seq为os.listdir(gt_folder)列表
    config['GT_LOC_FORMAT'] = '{gt_folder}/{seq}/gt/gt.txt'
    config['CLASSES_TO_EVAL'] = ['pedestrian']  # 确定预测指标的类别





    dataset = MotChallenge2DBox(config)  # dataset_list是存放数据信息列表

    evaluator = Evaluator()

    metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
    metrics_list = []
    for metric in [HOTA, CLEAR, Identity, VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    evaluator.evaluate(dataset, metrics_list)
