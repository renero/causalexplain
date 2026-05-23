"""
Module for the edge orientation algorithm.
"""
import logging

from hyppo.independence import Hsic

from .regressors import fit_and_get_residuals

log = logging.getLogger(__name__)


def get_edge_orientation(data, x, y, iters=20, method='gpr', verbose=False):
    """
    ANM test of independence for pairs with high correlation. Repeated
    runs stabilise the inferred causal direction.

    Args:
        data: Dataset with samples for the features.
        x: Source feature name.
        y: Target feature name.
        iters: Nr of repetitions of the test. Defaults to 20.
        method: 'gpr' or 'gam'. Defaults to 'gpr'.
        verbose: Unused; kept for API compatibility.

    Returns:
        +1 if direction is x->y, -1 if x<-y, 0 if undetermined.
    """
    res_y = fit_and_get_residuals(data[x].values, data[y].values, method=method)
    res_x = fit_and_get_residuals(data[y].values, data[x].values, method=method)

    r1 = Hsic().test(res_y, data[x].values, reps=iters).pvalue
    r2 = Hsic().test(res_x, data[y].values, reps=iters).pvalue
    mark = "**" if r1 < 0.05 and r2 < 0.05 else ""
    if r1 > r2:
        log.debug("    > %3s-->%-3s  [p(%s->%s): %8.6f; p(%s->%s): %8.6f] %s",
                  x, y, x, y, r1, y, x, r2, mark)
        return +1
    elif r1 < r2:
        log.debug("    > %3s<--%-3s  [p(%s->%s): %8.6f; p(%s->%s): %8.6f] %s",
                  x, y, x, y, r1, y, x, r2, mark)
        return -1
    else:
        log.debug("    > %3s·-·%-3s  [p(%s->%s): %8.6f; p(%s->%s): %8.6f]",
                  x, y, x, y, r1, y, x, r2)
        return 0
