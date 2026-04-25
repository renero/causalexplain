from typing import List, Set, Union
import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator


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
        """
        Returns the indices of features that are both gamma and outliers. Both criteria
        are applied to the given scores to determine if the MSE error obtained from
        the regression is bad compared with the rest of regressions for the other features
        in the dataset, and thus the feature should be considered a parent node.

        Parameters
        ----------
        scores: List[float]
            List of scores

        Returns
        -------
        Set[int]
            List of indices of features that are both gamma and outliers
        """
        scores = np.array(scores)
        gamma_indices = RegQuality._gamma_criteria(
            scores, gamma_shape, gamma_scale, threshold, verbose=verbose)
        outliers_indices = RegQuality._mad_criteria(scores, verbose=verbose)

        return set(gamma_indices).intersection(outliers_indices)

    @staticmethod
    def _mad_criteria(scores, verbose=False) -> Set[int]:
        """
        Returns indices of outliers in the given scores, using the MAD method.
        Taken from https://stats.stackexchange.com/a/78617
        
        Parameters
        ----------
        scores: List[float]
            List of scores
        verbose: bool
            Whether to print the score and M for each score
            
        Returns
        -------
        Set[int]
            List of indices of outliers in the given scores
        """
        median = np.median(scores)
        ad = [np.abs(score-median) for score in scores]
        mad = np.median(ad)
        if mad == 0:
            return {idx for idx, score in enumerate(scores) if score != median}
        M = [np.abs(.6745 * (score-median) / mad) for score in scores]
        mad_indices = [idx for idx, m in enumerate(M) if m > 3.5]

        if verbose:
            print(f"Median: {median:.4f}")
            print(f"Median of absolute differences: {mad:.4f}")
            for score, m in zip(scores, M):
                print(f"Score: {score:.4f}, M: {m:.4f}")

        return set(mad_indices)

    @staticmethod
    def _gamma_criteria(
            scores,
            gamma_shape=1,
            gamma_scale=1,
            threshold=0.9,
            verbose=False) -> Union[None, Set[int]]:
        """
        Returns a list of booleans indicating whether the score is below the threshold

        Parameters
        ----------
        scores: List[float]
            List of scores
        gamma_shape: float
            Shape parameter for gamma distribution
        gamma_scale: float
            Scale parameter for gamma distribution
        threshold: float
            Threshold for gamma criteria
        verbose: bool
            Whether to print the score and gamma criteria for each score

        Returns
        -------
        Set[int]
            List of booleans indicating whether the score is below the threshold
        """
        gamma_indices = []
        for idx, score in enumerate(scores):
            pdf = stats.gamma.pdf(score, a=gamma_shape, scale=gamma_scale)
            if pdf <= threshold:
                gamma_indices.append(idx)
            if verbose:
                print(
                    f"Score: {score:.4f}, PDF: {pdf:.4f}, criteria: {pdf < threshold}")

        return set(gamma_indices)
