import numpy as np
import pandas as pd

from scipy import stats


def power_divergence(X, Y, Z, data, boolean=True, lambda_="cressie-read", **kwargs):
    """Compute the power divergence test for conditional independence."""

    # Step 1: Check if the arguments are valid and type conversions.
    if hasattr(Z, "__iter__"):
        Z = list(Z)
    else:
        raise (f"Z must be an iterable. Got object type: {type(Z)}")

    if (X in Z) or (Y in Z):
        raise ValueError(
            f"The variables X or Y can't be in Z. Found {X if X in Z else Y} in Z."
        )

    # Step 2: Do a simple contingency test if there are no conditional variables.
    if len(Z) == 0:
        chi, p_value, dof, expected = stats.chi2_contingency(
            data.groupby([X, Y]).size().unstack(Y, fill_value=0), lambda_=lambda_
        )

    # Step 3: If there are conditionals variables, iterate over unique states and do
    #         the contingency test.
    else:
        chi = 0
        dof = 0
        for z_state, df in data.groupby(Z):
            try:
                c, _, d, _ = stats.chi2_contingency(
                    df.groupby([X, Y]).size().unstack(Y, fill_value=0), lambda_=lambda_
                )
                chi += c
                dof += d
            except ValueError:
                # If one of the values is 0 in the 2x2 table.
                if not isinstance(z_state, str):
                    z_str = ", ".join(
                        [f"{var}={state}" for var, state in zip(Z, z_state)]
                    )
        p_value = 1 - stats.chi2.cdf(chi, df=dof)

    # Step 4: Return the values
    if boolean:
        return p_value >= kwargs["significance_level"]
    else:
        return chi, p_value, dof


def chi_square(X, Y, Z, data, boolean=True, **kwargs):
    """Compute the chi-square test for conditional independence."""
    return power_divergence(
        X=X, Y=Y, Z=Z, data=data, boolean=boolean, lambda_="pearson", **kwargs
    )


def pearsonr(X, Y, Z, data, boolean=True, **kwargs):
    r"""
    Computes Pearson correlation coefficient and p-value for testing non-correlation.
    Should be used only on continuous data. In case when :math:`Z != \null` uses
    linear regression and computes pearson coefficient on residuals.

    Parameters
    ----------
    X: str
        The first variable for testing the independence condition X \u27C2 Y | Z

    Y: str
        The second variable for testing the independence condition X \u27C2 Y | Z

    Z: list/array-like
        A list of conditional variable for testing the condition X \u27C2 Y | Z

    data: pandas.DataFrame
        The dataset in which to test the indepenedence condition.

    boolean: bool
        If boolean=True, an additional argument `significance_level` must
            be specified. If p_value of the test is greater than equal to
            `significance_level`, returns True. Otherwise returns False.

        If boolean=False, returns the pearson correlation coefficient and p_value
            of the test.

    Returns
    -------
    Pearson's correlation coefficient: float
    p-value: float

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    [2] https://en.wikipedia.org/wiki/Partial_correlation#Using_linear_regression
    """
    # Step 1: Test if the inputs are correct
    if not hasattr(Z, "__iter__"):
        raise ValueError(
            f"Variable Z. Expected type: iterable. Got type: {type(Z)}")
    else:
        Z = list(Z)

    if not isinstance(data, pd.DataFrame):
        raise ValueError(
            f"Variable data. Expected type: pandas.DataFrame. Got type: {type(data)}"
        )

    # Step 2: If Z is empty compute a non-conditional test.
    if len(Z) == 0:
        coef, p_value = stats.pearsonr(data.loc[:, X], data.loc[:, Y])

    # Step 3: If Z is non-empty, use linear regression to compute residuals and test independence on it.
    else:
        X_coef = np.linalg.lstsq(data.loc[:, Z], data.loc[:, X], rcond=None)[0]
        Y_coef = np.linalg.lstsq(data.loc[:, Z], data.loc[:, Y], rcond=None)[0]

        residual_X = data.loc[:, X] - data.loc[:, Z].dot(X_coef)
        residual_Y = data.loc[:, Y] - data.loc[:, Z].dot(Y_coef)
        coef, p_value = stats.pearsonr(residual_X, residual_Y)

    if boolean:
        if p_value >= kwargs["significance_level"]:
            return True
        else:
            return False
    else:
        return coef, p_value
