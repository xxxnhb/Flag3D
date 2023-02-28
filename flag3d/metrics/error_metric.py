from collections import OrderedDict
from scipy import stats
import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS
from mmengine.logging import MMLogger
from mmengine.dist import get_world_size


@METRICS.register_module()
class ErrorMetric(BaseMetric):

    def __init__(self, prefix, collect_device='cpu'):
        super(ErrorMetric, self).__init__(prefix=prefix,
                                          collect_device=collect_device)

    def process(self, data_batch, data_samples):
        score, label = data_samples
        self.results.append({
            'score': score,
            'true_score': label
        })

    def compute_metrics(self, results):
        logger = MMLogger.get_current_instance()
        logger.info(f"{len(results)} results were collected from "
                    f"{get_world_size()} ranks.")
        pred_scores = []
        true_scores = []
        for res in results:
            for i in res['score']:
                pred_scores.append(i)
            for j in res['true_score']:
                true_scores.append(j)
        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        rho, p = stats.spearmanr(pred_scores, true_scores)
        # L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (max(true_scores) - min(true_scores)), 2).sum() / \
              true_scores.shape[0]
        metrics = {'correlation': -rho,
                   'RL2': RL2
                   }
        return metrics
