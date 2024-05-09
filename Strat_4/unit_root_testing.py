from statsmodels.tsa.stattools import adfuller

def test_unit_root(price_series, significance_level=0.05):
    """
    Test if a price series contains a unit root using Augmented Dickey-Fuller test.
    
    Parameters:
    - price_series: A pandas Series containing the price series.
    - significance_level: The significance level for the test (default is 0.05).
    
    Returns:
    - A tuple (test_statistic, p_value, is_unit_root), where:
        - test_statistic: The test statistic from the Augmented Dickey-Fuller test.
        - p_value: The p-value from the test.
        - is_unit_root: True if the null hypothesis (presence of a unit root) is not rejected, False otherwise.
    """
    adf_result = adfuller(price_series, autolag='AIC')
    test_statistic, p_value = adf_result[0], adf_result[1]
    
    is_unit_root = p_value > significance_level
    
    return test_statistic, p_value, is_unit_root