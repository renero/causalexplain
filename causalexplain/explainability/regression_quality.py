import logging
from typing import List, Set, Union

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator

log = logging.getLogger(__name__)


class RegQuality(BaseEstimator):

    def __init__(self):
        super().__init__()

    @staticmethod
    def predict(
            scores: List[float],
            gamma_shape: float = 1,
            gamma_scale: float = 1,
            threshold: float = 0.9,
            verbose: bool = False) -> Set[int]:
        scores = np.array(scores)
        gamma_indices = RegQuality._gamma_criteria(
            scores, gamma_shape, gamma_scale, threshold, verbose=verbose)
        outliers_indices = RegQuality._mad_criteria(scores, verbose=verbose)
        return set(gamma_indices).intersection(outliers_indices)

    @staticmethod
    def _mad_criteria(scores, verbose=False) -> Set[int]:
        median = np.median(scores)
        ad = [np.abs(score-median) for score in scores]
        mad = np.median(ad)
        if mad == 0:
            return {idx for idx, score in enumerate(scores) if score != median}
        M = [np.abs(.6745 * (score-median) / mad) for score in scores]
        mad_indices = [idx for idx, m in enumerate(M) if m > 3.5]
        log.debug("MAD median=%.4f  mad=%.4f", median, mad)
        for score, m in zip(scores, M):
            log.debug("  Score: %.4f, M: %.4f", score, m)
        return set(mad_indices)

    @staticmethod
    def _gamma_criteria(
            scores,
            gamma_shape=1,
            gamma_scale=1,
            threshold=0.9,
            verbose=False) -> Union[None, Set[int]]:
        gamma_indices = []
        for idx, score in enumerate(scores):
            pdf = stats.gamma.pdf(score, a=gamma_shape, scale=gamma_scale)
            if pdf <= threshold:
                gamma_indices.append(idx)
            log.debug("  Score: %.4f, PDF: %.4f, criteria: %s", score, pdf, pdf < threshold)
        return set(gamma_indices)
