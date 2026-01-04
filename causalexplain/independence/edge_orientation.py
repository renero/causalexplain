"""
Module for the edge orientation algorithm.
"""
from hyppo.independence import Hsic

from .regressors import fit_and_get_residuals


def get_edge_orientation(data, x, y, iters=20, method='gpr', verbose=False):
    """
    This is an ANM test of independence for the pairs between which
    a lot of correlation has been seen. If the test is repeated a sufficient
    number of times (100) the correct causal direction almost always comes
    out -- and in cases where it is not, it is enough to repeat the test
    an odd number of times (5) to see that the result Yeah that's right.

    Args:
        data (pandas.DataFrame): The dataset with the samples for the features
        x (str): the name of the source feature
        y (str): the name of the target feature
        iters (int, optional): Nr of repetitions of the test. Defaults to 100.
        method (str, optional): Can be 'gpr' or 'gam'. Defaults to 'gpr'.
        verbose (bool, optional): Verbosity. Defaults to False.

    Returns:
        int: Returns +1 if direction is x->y, or -1 if direction is x<-y
            Returns 0 if no direction can be set.
    """
    res_y = fit_and_get_residuals(
        data[x].values, data[y].values, method=method)
    res_x = fit_and_get_residuals(
        data[y].values, data[x].values, method=method)

    r1 = Hsic().test(res_y, data[x].values, reps=iters).pvalue
    r2 = Hsic().test(res_x, data[y].values, reps=iters).pvalue
    mark = "**" if r1 < 0.05 and r2 < 0.05 else ""
    if r1 > r2:
        if verbose:
            print(f"    > {x:>3s}-->{y:<3s}  "
                  f"[p({x:>3s}->{y:<3s}): {r1:8.6f};"
                  f" p({y:>3s}->{x:<3s}): {r2:8.6f}] {mark}")
        return +1
    elif r1 < r2:
        if verbose:
            print(f"    > {x:>3s}<--{y:<3s}  "
                  f"[p({x:>3s}->{y:<3s}): {r1:8.6f};"
                  f" p({y:>3s}->{x:<3s}): {r2:8.6f}] {mark}")
        return -1
    else:
        if verbose:
            print(f"    > {x:>3s}·-·{y:<3s}", end="")
            print(
                f"  [p({x:>3s}->{y:<3s}): {r1:8.6f}; p({y:>3s}->{x:<3s}): {r2:8.6f}]")
        return 0
