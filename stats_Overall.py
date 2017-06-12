def test_means(AllData_warmth, vecUniTargets):
    """ this function calculates the means, tests whether they are different or not
    and saves the statistical values 
    """
    import numpy as np
    import scipy.stats as stats
    import pdb
    import os
    import pandas as pd

    dict_means = {}
    count = 0
    vecTarget= []
    for h in range(len(AllData_warmth)):
        df = AllData_warmth[h]
        for targ in df.keys():
            if targ[-1] == 'r':
                continue
            vecTarget.append(targ)

            try:
                dict_means[targ]
            except:
                dict_means[targ] = {}
            try:
                dict_means[targ][count]
            except:
                dict_means[targ][count] = []

            if len(dict_means[targ][0]) != 0:
                count = len(dict_means[targ])
            else:
                count = len(dict_means[targ])-1

            dict_means[targ][count] = df[targ].dropna().values
            count = 0

    vecTarget = list(set(vecTarget))
    test_val,test_type, test_val_var, mean_dif, var_dif, repeats = {}, {}, {}, {}, {}, {}
    for Targ in vecTarget:
        if len(dict_means[Targ]) == 2:
            test_val_var[Targ] = stats.levene(dict_means[Targ][0], dict_means[Targ][1])
            if test_val_var[Targ][1] < 0.05:
                equal_variance = False
            else:
                equal_variance = True
            test_val[Targ] = stats.ttest_ind(dict_means[Targ][0], dict_means[Targ][1], equal_var=equal_variance)

            mean_dif[Targ] = abs(np.mean(dict_means[Targ][0]) - np.mean(dict_means[Targ][1]))
            var_dif[Targ] = abs(np.var(dict_means[Targ][0]) - np.var(dict_means[Targ][1]))

            test_type[Targ] = 'T_test'
        elif len(dict_means[Targ]) == 3:
            test_val_var[Targ] = stats.levene(dict_means[Targ][0], dict_means[Targ][1], dict_means[Targ][2])

            test_val[Targ] = stats.f_oneway(dict_means[Targ][0], dict_means[Targ][1], dict_means[Targ][2])

            mean_dif[Targ] = []
            mean_dif[Targ].append(abs(np.mean(dict_means[Targ][0]) - np.mean(dict_means[Targ][1])))
            mean_dif[Targ].append(abs(np.mean(dict_means[Targ][1]) - np.mean(dict_means[Targ][2])))
            mean_dif[Targ].append(abs(np.mean(dict_means[Targ][0]) - np.mean(dict_means[Targ][2])))
            mean_dif[Targ] = np.mean(mean_dif[Targ])

            var_dif[Targ] = []
            var_dif[Targ].append(abs(np.var(dict_means[Targ][0]) - np.var(dict_means[Targ][1])))
            var_dif[Targ].append(abs(np.var(dict_means[Targ][1]) - np.var(dict_means[Targ][2])))
            var_dif[Targ].append(abs(np.var(dict_means[Targ][0]) - np.var(dict_means[Targ][2])))
            var_dif[Targ] = np.mean(var_dif[Targ])

            test_type[Targ] = 'ANOVA'
        elif len(dict_means[Targ]) == 4:
            test_val_var[Targ] = stats.levene(dict_means[Targ][0], dict_means[Targ][1], dict_means[Targ][2], dict_means[Targ][3])
            test_val[Targ] = stats.f_oneway(dict_means[Targ][0], dict_means[Targ][1], dict_means[Targ][2], dict_means[Targ][3])

            mean_dif[Targ] = []
            mean_dif[Targ].append(abs(np.mean(dict_means[Targ][0]) - np.mean(dict_means[Targ][1])))
            mean_dif[Targ].append(abs(np.mean(dict_means[Targ][1]) - np.mean(dict_means[Targ][2])))
            mean_dif[Targ].append(abs(np.mean(dict_means[Targ][2]) - np.mean(dict_means[Targ][3])))
            mean_dif[Targ].append(abs(np.mean(dict_means[Targ][0]) - np.mean(dict_means[Targ][2])))
            mean_dif[Targ].append(abs(np.mean(dict_means[Targ][0]) - np.mean(dict_means[Targ][3])))
            mean_dif[Targ] = np.mean(mean_dif[Targ])

            var_dif[Targ] = []
            var_dif[Targ].append(abs(np.var(dict_means[Targ][0]) - np.var(dict_means[Targ][1])))
            var_dif[Targ].append(abs(np.var(dict_means[Targ][1]) - np.var(dict_means[Targ][2])))
            var_dif[Targ].append(abs(np.var(dict_means[Targ][2]) - np.var(dict_means[Targ][3])))
            var_dif[Targ].append(abs(np.var(dict_means[Targ][0]) - np.var(dict_means[Targ][2])))
            var_dif[Targ].append(abs(np.var(dict_means[Targ][0]) - np.var(dict_means[Targ][3])))
            var_dif[Targ] = np.mean(var_dif[Targ])

            test_type[Targ] = 'ANOVA'

        if len(dict_means[Targ]) != 1:
            repeats[Targ] = len(dict_means[Targ])

    # save the model coeffs
    print(test_type)
    df_test_mean = pd.DataFrame(test_val , index=['Mean_statistic', 'Mean_p_value'])
    df_test_var = pd.DataFrame(test_val_var, index=['Var_statistic', 'Var_p_value'])
    df_var_diff = pd.DataFrame(var_dif, index=['average variance diff'])
    df_mean_diff = pd.DataFrame(mean_dif, index=['average mean diff'])
    df_test_type = pd.DataFrame(test_type, index=['Type of test'])
    df_reps = pd.DataFrame(repeats, index=['# repeats'])
    df_out = pd.concat([df_test_mean, df_test_var, df_mean_diff, df_var_diff, df_test_type, df_reps])
    filename = 'Test_values.csv'
    os.chdir("C:/Users/Thomas Dolman/Documents/UvA/MasterStage2_MingHsu/Onderzoek/Data/Overall")
    df_out.to_csv(filename)
