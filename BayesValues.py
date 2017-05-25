def Bayes(mean, var):
    """ calculates the bayes values on a list of std and mean values
    ADJUST THIS TO VARIANCE!!
    """
    import numpy as np
    std_new = np.sqrt( (var[0] * var[1]) / (var[0] + var[1]))

    weight, mean_new = [], []
    weight.append((1 / var[0]) / ((1 / var[1]) + (1 / var[0])))
    weight.append((1 / var[1]) / ((1 / var[1]) + (1 / var[0])))

    mean_new.append(weight[0] * mean[0])
    mean_new.append(weight[1] * mean[1])

    mean_out = sum(mean_new)
    return mean_out, std_new, weight


def BayesPanda(means, var):
    """ this functiion calculates Bayesian mean, std and weights
    with pandas dataframes
    """
    import pdb
    import numpy as np

    # get the labels
    keys = var.keys()
    keys2 = means.keys()

    # replace 0 to 0.001 to prevent nan outcomes
    var = var.replace(to_replace=0, value=0.001)
    var = var.replace(to_replace=np.nan, value=0.001)

    # calculate the Bayes optimal variance (sigma^2) following Fetsch et al. 2012
    var_new = (var[keys[0]] * var[keys[1]]) / (var[keys[0]] + var[keys[1]])

    # calculate the weights for each single target case
    weight_one = ((1 / var[keys[0]]) / ((1 / var[keys[1]]) + (1 / var[keys[0]])))
    weight_two = ((1 / var[keys[1]]) / ((1 / var[keys[1]]) + (1 / var[keys[0]])))

    # multiple weights * means of single targets
    mean_one = weight_one * means[keys2[0]]
    mean_two = weight_two * means[keys2[1]]

    # sum both to the Bayesian predicted warmth for the Combined Target
    mean_out = mean_one + mean_two

    return mean_out, var_new
