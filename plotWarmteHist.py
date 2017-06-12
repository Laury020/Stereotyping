def plotWarmteHist(df, Key, Name):
    import StringSplit
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    import pdb

    label, label_title = StringSplit.StringSplit(Key)
    plt.figure(figsize=[14, 7])
    color = ['purple', 'blue', 'red']
    for n in range(1, 4):
        if n <= 3:
            plt.subplot(3, 1, n)
            # plt.subplot(2, 3, n)
            plt.hist(df[label[n]].dropna(), bins=70, color=color[n - 1])
            plt.vlines(np.nanmean(df[label[n]]), 0, 8, colors='black')
            plt.title(label_title[n] + " n = " + str(len(df[label[n]].dropna())) + ' mean= ' + str(
                np.mean(df[label[n]].dropna()))[0:5])
            plt.xlim([0, 100])
            plt.xticks(np.arange(0, 100, 5))
            plt.xlabel("Mean Warmth")
            plt.ylabel("#")
            # else:
            # plt.subplot(2, 3, n)
            # label_new = label[n-3][0:len(label[n-3])-5]
            # label_new = label_new + "_var"
            # plt.hist(df[label_new].dropna(), bins=70, color= color[n-4])
            # plt.vlines(np.nanmean(df[label_new]), 0, 8, colors='black')
            # plt.title(label_title[n-3])
            # plt.xlim([0, 40])
            # plt.xlabel("Std over Pers Traits")
            # plt.ylabel("#")

    plt.tight_layout()
    fileloc = "C:/Users/Thomas Dolman/Documents/UvA/MasterStage2_MingHsu/Onderzoek/Data/Overall/MeanHist/"
    os.chdir(fileloc)
    plt.savefig(label[1] + str(Name) + '.eps', format='eps')
    plt.savefig(label[1] + str(Name) + '.png', format='png')
    plt.close('all')


def meanWarmthProba(df, Key, Name):
    import StringSplit
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.mlab as mlab
    import os

    label, label_title = StringSplit.StringSplit(Key)
    plt.figure(figsize=[15, 8])
    color = ['blue', 'red', 'purple']

    vecX = np.linspace(0, 100, 100)
    for n in range(1, 6):
        plt.subplot(5, 1, n)
        plt.xlabel('Warmth score')
        plt.ylabel('Probability')
        if n < 4:
            plt.plot(mlab.normpdf(vecX, np.nanmean(df[label[n]]), np.nanstd(df[label[n]])), color=color[n - 1],
                     label=label_title[n])
            plt.vlines(np.nanmean(df[label[n]]), 0, 0.02, colors='black')
            plt.legend(loc='best')
            ax = plt.gca()
            ax.set_xlim([0, 100])
        elif n == 4:
            mean = (np.nanmean(df[label[2]]) + np.nanmean(df[label[3]])) / 2
            std = (np.nanstd(df[label[2]]) + np.nanstd(df[label[3]])) / 2
            plt.plot(mlab.normpdf(vecX, mean, std), color=color[2], label='Average Integrated Pred')
            plt.vlines(mean, 0, 0.02, colors='black')
            plt.legend(loc='best')
            ax = plt.gca()
            ax.set_xlim([0, 100])
        elif n == 5:
            mean = [np.nanmean(df[label[2]]), np.nanmean(df[label[3]])]
            var = [np.nanvar(df[label[2]]), np.nanvar(df[label[3]])]
            import BayesValues as BV
            mean_Bay, std_Bay, weight = BV.Bayes(mean, var)
            # print('weight of factors = ')
            # print(weight)
            plt.plot(mlab.normpdf(vecX, mean_Bay, std_Bay), color=color[2], label='Bayes Integrated Pred')
            plt.vlines(mean_Bay, 0, 0.02, colors='black')
            plt.legend(loc='best')
            ax = plt.gca()
            ax.set_xlim([0, 100])

    plt.tight_layout()
    fileloc = "C:/Users/Thomas Dolman/Documents/UvA/MasterStage2_MingHsu/Onderzoek/Data/Overall/ProbPlot/"
    os.chdir(fileloc)
    plt.savefig(label[1] + str(Name) + '.eps', format='eps')
    plt.savefig(label[1] + str(Name) + '.png', format='png')
    plt.close('all')


def scatWarmthInt(df, vecUniTargets, Name):
    import matplotlib.pyplot as plt
    import StringSplit
    import numpy as np
    import BayesValues

    # create a plot that examines the change between mean and predicted mean
    plt.figure('Bayes_VS_Integrated', figsize=(15, 8))

    for Key in df.keys():
        if Key in vecUniTargets:
            continue
        elif Key[-1] == "r":
            continue
        else:
            import pdb

            label, label_title = StringSplit.StringSplit(Key)
            plt.figure(label_title[1], figsize=(15, 8))

            plt.xlabel('Ethnicity')
            plt.ylabel('Occupation')
            plt.scatter(np.nanmean(df[label[2]]), np.nanmean(df[label[3]]), len(df[label[1]].dropna()))
            label_use, x, y = 'Combined', np.nanmean(df[label[2]]), np.nanmean(df[label[3]])
            plt.annotate(label_use,
                         xy=(x, y), xytext=(-20, 20),
                         textcoords='offset points', ha='right', va='bottom',
                         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

            plt.scatter(np.nanmean(df[label[1]]), np.nanmean(df[label[3]]), len(df[label[3]].dropna()))
            label_use, x, y = label_title[1], np.nanmean(df[label[3]]), np.nanmean(df[label[3]])
            plt.annotate(label_use,
                         xy=(x, y), xytext=(-20, 20),
                         textcoords='offset points', ha='right', va='bottom',
                         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

            plt.tight_layout()
            import os
            os.chdir('C:\\Users\\Thomas Dolman\\Documents\\UvA\\MasterStage2_MingHsu\\Onderzoek\\Data\\Overall')
            plt.savefig("Predictions/" + "Bayes_VS_Integrated_" + label_title[1] + str(Name) +'.eps', format='eps')
            plt.savefig("Predictions/" + "Bayes_VS_Integrated_" + label_title[1] + str(Name) + '.png', format='png')



def plotTargetVar(df, vecUniTargets, Name):
    import matplotlib.pyplot as plt
    import numpy as np
    import pdb
    import os

    Keys_new = []
    for n in df.keys():
        if n in vecUniTargets:
            continue
        elif n[-1] == 'r':
            continue
        else:
            Keys_new.append(n)

    vecSTD = np.nanstd(df[Keys_new].values, axis=0)
    # pdb.set_trace()

    plt.figure(figsize=[14, 7])
    xline = np.linspace(0, len(Keys_new), len(Keys_new))
    plt.bar(xline, vecSTD)
    plt.xticks(xline, Keys_new, rotation=45)

    plt.ylim([np.min(vecSTD) - 1, np.max(vecSTD) + 1])
    plt.tight_layout()

    fileloc = "C:/Users/Thomas Dolman/Documents/UvA/MasterStage2_MingHsu/Onderzoek/Data/Overall/VarBars/"
    os.chdir(fileloc)
    plt.savefig(Keys_new[0] + str(Name) + '_Variance.eps', format='eps')
    plt.savefig(Keys_new[0] + str(Name) + '_Variance.png', format='png')
    plt.close('all')


def ChangeWarmthHist(df, Key, Name):
    import matplotlib.pyplot as plt
    import numpy as np
    import pdb
    import StringSplit
    import os

    label, label_title = StringSplit.StringSplit(Key)
    plt.figure(figsize=[14, 7])
    color = ['blue', 'red', 'purple']
    vecTitle = ["Eth - Occ", "Comb - Eth", 'Comb - Occ']
    for n in range(1, 4):
        if n <= 3:
            plt.subplot(3, 1, n)
            if n == 3 or n == 2:
                diff = df[label[1]].dropna() - df[label[n]].dropna()
                plt.title(label_title[3] + " - " + label_title[n])
            elif n == 1:
                diff = df[label[2]].dropna() - df[label[3]].dropna()
                plt.title(label_title[2] + " - " + label_title[1])

            plt.hist(diff.dropna(), bins=70, color=color[n - 1])
            plt.vlines(np.nanmean(diff), 0, 8, colors='black')
            plt.xlim([-60, 60])
            plt.xlabel("Differences")
            plt.ylabel("#")
            plt.title(vecTitle[n - 1])

    plt.tight_layout()

    fileloc = "C:/Users/Thomas Dolman/Documents/UvA/MasterStage2_MingHsu/Onderzoek/Data/Overall/PopulationChange/"
    os.chdir(fileloc)
    plt.savefig(label_title[1] + str(Name) + "_Changes.eps", format='eps')
    plt.savefig(label_title[1] + str(Name) + "_Changes.png", format='png')
    plt.close('all')


def ScatterWarmth(df, Key, Name):
    """This function makes a scatter plot between the uniTargets """
    import StringSplit
    import matplotlib.pyplot as plt
    import os
    import scipy.stats as sct
    import numpy as np

    label, label_title = StringSplit.StringSplit(Key)
    vecX = np.linspace(0, 100, 100)
    CorrValues = []
    Pvals = []

    plt.figure(figsize=[14, 7])
    plt.subplot(3, 1, 1)
    plt.scatter(df[label[1]].dropna(), df[label[2]].dropna())
    PlotValues = sct.linregress(df[label[1]].dropna(), df[label[2]].dropna())
    plt.plot(vecX, PlotValues[0] * vecX + PlotValues[1], color='red')
    if PlotValues[3] > 0.05:
        Pval = str(PlotValues[3])[0:4]
    else:
        Pval = 'below 0.05'
    N = len(df[label[1]].dropna())
    plt.title(" n = " + str(N) + ' R = ' + str(PlotValues[2])[0:4] + '  ' + 'P = ' + Pval)
    plt.xlim([0, 100])
    plt.xlabel(label_title[1])
    plt.ylabel(label_title[2])
    CorrValues.append(PlotValues[2])
    Pvals.append(Pval)

    plt.subplot(3, 1, 2)
    plt.scatter(df[label[1]].dropna(), df[label[3]].dropna())
    PlotValues = sct.linregress(df[label[1]].dropna(), df[label[3]].dropna())
    plt.plot(vecX, PlotValues[0] * vecX + PlotValues[1], color='red')
    if PlotValues[3] > 0.05:
        Pval = str(PlotValues[3])[0:4]
    else:
        Pval = 'below 0.05'
    plt.title(" n = " + str(N) + ' R = ' + str(PlotValues[2])[0:4] + '  ' + 'P = ' + Pval)
    plt.xlim([0, 100])
    plt.xlabel(label_title[1])
    plt.ylabel(label_title[3])
    CorrValues.append(PlotValues[2])
    Pvals.append(Pval)

    plt.subplot(3, 1, 3)
    plt.scatter(df[label[2]].dropna(), df[label[3]].dropna())
    PlotValues = sct.linregress(df[label[2]].dropna(), df[label[3]].dropna())
    plt.plot(vecX, PlotValues[0] * vecX + PlotValues[1], color='red')
    if PlotValues[3] > 0.05:
        Pval = str(PlotValues[3])[0:4]
    else:
        Pval = 'below 0.05'
    plt.title(" n = " + str(N) + ' R = ' + str(PlotValues[2])[0:4] + '  ' + 'P = ' + Pval)
    plt.xlim([0, 100])
    plt.xlabel(label_title[2])
    plt.ylabel(label_title[3])
    CorrValues.append(PlotValues[2])
    Pvals.append(Pval)

    plt.tight_layout()
    fileloc = "C:/Users/Thomas Dolman/Documents/UvA/MasterStage2_MingHsu/Onderzoek/Data/Overall/MeanHist/"
    os.chdir(fileloc)
    plt.savefig(label[1] + str(Name) + '_scatter.eps', format='eps')
    plt.savefig(label[1] + str(Name) + '_scatter.png', format='png')
    plt.close('all')

    return CorrValues, Pvals


def plotBayesProb(df, key, preset, train_size, randomstate, types, allDeviation, allDataWarmth, AllUniTarg):
    """ this function plots the histogram over all subjects and creates model statistics for the models
    its output is a histogram with the model predictions and model performance
    Depending on the random state it reports different sized output
    """
    import StringSplit
    import matplotlib.pyplot as plt
    import os
    import BayesValues
    import pdb
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_squared_error
    import modelfit
    from sklearn.model_selection import train_test_split
    import math
    import scipy.stats as stat

    rmse, bestmodel, difference, score, outof_score = preset[0], preset[1], preset[2], preset[3], preset[4]

    # separate key into labels and label_title (label without _mean at the end)
    label, label_title = StringSplit.StringSplit(key)

    # create the relevant labels which are needed in selecting the right data
    keysmean = [label[2], label[3]]
    keysvar = [label_title[2] + "_var", label_title[3] + "_var"]

    # split data into train and test sets
    X_train_mean, X_test_mean, y_train, y_test2 = train_test_split(df[keysmean], df[label[1]], train_size=train_size,
                                                                 random_state=randomstate)
    X_train_var, X_test_var, y_train, y_test2 = train_test_split(df[keysvar], df[label[1]], train_size=train_size,
                                                             random_state=randomstate)
    # creates X and y sets for the modelfit.
    X = np.array([df[label[2]].dropna(), df[label[3]].dropna()])
    y = df[label[1]].dropna().values
    X = X.T
    # calculated the weighted mean of the combined targed by estimating weights on a subset and extrapolating that
    # meanWeightPred, unimp, Aver_weights = modelfit.WeightAver(X, y, train_size, randomstate)
    X_train, X_test, y_train, y_test2 = train_test_split(X, y, train_size=train_size, random_state=randomstate)

    import modelfit
    Regres_scores, Regres_rmse, regress_model = modelfit.RegressionFit(X, y, train_size, randomstate)
    Regress_pred = regress_model.predict(X_test)

    # bayes based upon within subject variance (over Pers traits
    meanBayPred, varbayPred, BayWeight = BayesValues.BayesPanda(X_test_mean, X_train_var)
    # print(train_size)
    # print("Bayes weights: " + str(BayWeight))
    # bayes based upon population variance (over all warmth values)
    meanBayPred_pop, varbayPred_pop, BayWeight_pop = BayesValues.BayesPopulation(X_test, X_train)
    # print(train_size)
    # print("Bayes weights pop: " + str(BayWeight_pop))

    meanbayPred_totalpop, varbayPred_totalpop, BayWeight_totalpop = BayesValues.BayesPopulation(X, X)

    # Use the average of both single targets which predicts the combined warmth target
    meanAverPred_mean = np.nanmean(X_train, axis=1)
    meanAverPred = np.nanmean(X_test, axis=1)

    # put weights of each model in output var.
    modelNames = ['Bayes', 'Regress', 'BayPop', 'BaytotalPop']
    weights_out = [np.mean(BayWeight, axis=1), regress_model.coef_, BayWeight_pop, BayWeight_totalpop]
    # create data frame with all the weights of each model
    df_weights= pd.DataFrame(weights_out, index= modelNames, columns= keysmean)

    # calculate regress R^2 (score)
    Regress_score = regress_model.score(X_test, y_test2)

    # Regress sum of squares = ((y_true - y_pred) **2).sum()
    u = ((y_test2 - meanAverPred) ** 2).sum()
    # residual sum of squares = ((y_true - Y_true.mean()) **2).sum()
    v = ((y_test2 - y_test2.mean()) ** 2).sum()
    # coefficient R^2 = (1- RegSS / ResSS)
    meanAver_score = round((1 - (u / v) ),4)

    u = ((y_test2 - meanBayPred) ** 2).sum()
    v = ((y_test2 - y_test2.mean()) ** 2).sum()
    Bayes_score = round((1 - (u / v) ),4)

    # u = ((y_test2 - meanWeightPred) ** 2).sum()
    # v = ((y_test2 - y_test2.mean()) ** 2).sum()
    # meanWeight_score = round((1 - (u / v) ),4)

    u = ((y_test2 - meanBayPred_pop) ** 2).sum()
    v = ((y_test2 - y_test2.mean()) ** 2).sum()
    Baypop_score = round((1 - (u / v)), 4)

    u = ((y - meanbayPred_totalpop) ** 2).sum()
    v = ((y - y.mean()) ** 2).sum()
    Baytotalpop_score = round((1 - (u / v)), 4)

    # create subset of combined targets which is the same as the subset made in WeightedAver.WeightAver(X, y) due to
    # random state set
    X_train1, X_combshort, y_train2, y_test2 = train_test_split(df[label[1]].dropna().values,
                                                              df[label[1]].dropna().values,
                                                              train_size=train_size, random_state=randomstate)

    # calculate the residual mean squared error for each prediction
    rmsebay = mean_squared_error(X_combshort, meanBayPred)
    rmseAver = mean_squared_error(X_train1, meanAverPred_mean)
    # rmseWeight = mean_squared_error(X_combshort, meanWeightPred)
    rmseRegress = mean_squared_error(X_combshort, Regress_pred)
    rmseBayPop = mean_squared_error(X_combshort, meanBayPred_pop)
    rmseBaytotalPop = mean_squared_error(y, meanbayPred_totalpop)

    # calculate differences between predicted and actual combined target warmth reports
    BayesDiff = np.mean(X_combshort) - np.mean(meanBayPred)
    AverDiff = np.mean(X_combshort) - np.mean(meanAverPred_mean)
    # WeightDiff = np.mean(X_combshort) - np.mean(meanWeightPred)
    RegressDiff = np.mean(X_combshort) - np.mean(Regress_pred)
    BayPopDiff = np.mean(X_combshort) - np.mean(meanBayPred_pop)
    BaytotalPopDiff = np.mean(y) - np.mean(meanbayPred_totalpop)

    # examine variance and standdev in relation to weights/coeffs
    variance = np.var(df[keysmean], axis=0)
    variance[label[1]] = np.var(df[label[1]], axis=0)
    variance['mean'] = np.mean(variance[keysmean])
    standdev = np.std(df[keysmean], axis=0)
    standdev[label[1]] = np.std(df[label[1]], axis=0)
    standdev['mean'] = np.mean(standdev[keysmean])

    variation = {'variance': variance, 'standdev': standdev}
    # loop over variance and stddev
    vecDict, prob1 = [], []
    tel = 0
    for step in ['variance', 'standdev']:
        # pdb.set_trace()
        allDeviation[step][key] = variation[step][keysmean]
            # try:
            #     allDeviation[step] = pd.concat([allDeviation[step], variation[step][keysmean]], axis=1)
            # except:
            #     allDeviation[step] = pd.DataFrame(variation[step][keysmean])
            # create a normalized variance and standdev.
            # import StringSplit
            # label, label_title = StringSplit.StringSplit(key)
            # prob1.append(allDeviation[step][key][label[2]])

        prob1.append([allDeviation[step][key][label[2]] / (allDeviation[step][key][label[2]] + allDeviation[step][key][label[3]]),
                          allDeviation[step][key][label[3]] / (allDeviation[step][key][label[2]] + allDeviation[step][key][label[3]])])
        vecDict.append('Ratios_' + step)
        prob1.append([allDeviation[step][key][label[2]], allDeviation[step][key][label[3]]])
        vecDict.append(step)
    # pdb.set_trace()
    # prob1_d = dict(prob1, vecDict)
    proba = pd.DataFrame(prob1, index=vecDict, columns= keysmean)

    # # save metrics into dictionary
    if types >= 2:
        difference[key] = {"Bayes_diff": abs(BayesDiff), "Aver_diff": abs(AverDiff),
                           "Regres_diff": abs(RegressDiff), "BayPop_diff": abs(BayPopDiff), "BayTotalPop_diff": abs(BaytotalPopDiff)}
        rmse[key] = {"Bayes_rmse": rmsebay, "Aver_rmse": rmseAver,
                     "Regress_rmse": rmseRegress, "BayPop_rmse": rmseBayPop, "BayTotalPop_rmse": rmseBaytotalPop}
        score[key] = {"Bayes_score": Bayes_score, "Aver_score": meanAver_score,
                 "Regress_score": Regress_score, "BayPop_score": Baypop_score, "BayTotalPop_score": Baytotalpop_score}

    # add score to the bestmodel to quanpngy which model does better predicitons
    min_rmse, min_diff, max_score = list(rmse[key].values()), list(difference[key].values()), list(score[key].values())
    rmseKeys, diffKeys, scoreKeys = list(rmse[key].keys()), list(difference[key].keys()), list(score[key].keys())
    LowRMSE_model = rmseKeys[min_rmse.index(min(min_rmse))]
    LowDiff_model = diffKeys[min_diff.index(min(min_diff))]
    HighScore_model = scoreKeys[max_score.index(max(max_score))]

    # TODO verzin een beter distinctie voor wanneer een model beter is dan de rest, hier is een vorm van stat test nodig.
    # import pdb
    # pdb.set_trace()
    rmse_low = min_rmse[min_rmse.index(min(min_rmse))]
    min_rmse.pop(min_rmse.index(min(min_rmse)))
    stat_vals = stat.ttest_1samp(min_rmse, rmse_low)
    diff_low = min_diff[min_diff.index(min(min_diff))]
    min_diff.pop(min_diff.index(min(min_diff)))
    stat_vals2 = stat.ttest_1samp(min_diff, diff_low)
    score_high = max_score[max_score.index(max(max_score))]
    max_score.pop(max_score.index(max(max_score)))
    stat_vals3 = stat.ttest_1samp(max_score, score_high)
    # update the beat model
    if train_size != 0.3:
        if types == 0 or types == 3:
            if stat_vals[1] < 0.05:
                bestmodel[LowRMSE_model] += 1

            if stat_vals2[1] < 0.05:
                bestmodel[LowDiff_model] += 1
            if stat_vals3[1] < 0.05:
                bestmodel[HighScore_model] += 1
        else:
            bestmodel[LowRMSE_model] += 1
            bestmodel[LowDiff_model] += 1
            bestmodel[HighScore_model] += 1

    # outof_score = {}
    if train_size == 0.3:
        # plot the histograms for uni, uni, measured, predicted bayes, predicted average
        plt.figure(figsize=(24, 13))
        vecHists = [df[label[1]].dropna(), meanBayPred.dropna(), meanAverPred, Regress_pred, meanBayPred_pop]
        titles = ["Integrated", "Bayes predicted", "Average predicted",
                  "Regress predicted", "Bayes population predicted"]
        color = ["purple", "cyan", "green", "orange", 'yellow']
        labels = list(difference[key].keys())
        for n in range(len(vecHists)):
            ax = plt.subplot(math.ceil(len(vecHists) / 2), 2, n + 1)
            plt.hist(vecHists[n], bins=50, color=color[n])
            plt.vlines(np.mean(vecHists[n]), 0, 8, colors='black', linewidth=3.0)
            plt.vlines(np.mean(df[keysmean[0]].dropna()), 0, 8, colors='red', linewidth=3.0)
            plt.vlines(np.mean(df[keysmean[1]].dropna()), 0, 8, colors='blue', linewidth=3.0)
            plt.vlines(np.mean(df[label[1]].dropna()), 0, 8, colors='purple', linewidth=3.0)
            if n ==0:
                plt.title(titles[n] + " diff = 0")
            else:
                plt.title(titles[n] + " diff = " + str(difference[key][labels[n-1]])[0:5])
            plt.xlabel("warmth, and the train_size = " + str(train_size))
            plt.ylabel("count")
            plt.xticks(np.arange(0, 100, 5))
            ax.set_xlim(0, 100)

        plt.tight_layout()

        # save the figure
        fileloc = "C:/Users/Thomas Dolman/Documents/UvA/MasterStage2_MingHsu/Onderzoek/Data/Overall/PredHists/"
        os.chdir(fileloc)
        plt.savefig(label[1] + " " + str(train_size) + '.eps', format='eps')
        plt.savefig(label[1] + " " + str(train_size) + '.png', format='png')
        plt.close('all')

        # build a function for this, it tests the best model outside of the fitted dataset
        for n in range(len(allDataWarmth)):
            # loop over the other questionaires and check if the model works there as well
            if n == 2:
                continue
            # vecUniTargets = AllUniTarg[n]
            df = allDataWarmth[n]
            for Key_test in df.keys():
                if Key_test == key:
                    label_test, label_title_test = StringSplit.StringSplit(Key_test)
                    try:
                        predict
                    except:
                        predict = {}
                    try:
                        outof_score[key + str(n)]
                    except:
                        outof_score[key + str(n)] = {}
                    # make a prediction
                    for model in df_weights.index:
                        predict[model] = df[label_test[2]] * df_weights[label_test[2]][model] + df[label_test[3]] * df_weights[label_test[3]][model]
                        # Regress sum of squares = ((y_true - y_pred) **2).sum()
                        u = ((df[label_test[1]] - predict[model]) ** 2).sum()
                        # residual sum of squares = ((y_true - Y_true.mean()) **2).sum()
                        v = ((df[label_test[1]] - df[label_test[1]].mean()) ** 2).sum()
                        # coefficient R^2 = (1- RegSS / ResSS)
                        outof_score[key + str(n)][model] = round((1 - (u / v)), 4)

                    meanAverPred_out = np.nanmean([df[label_test[2]], df[label_test[3]]], axis=0)
                    # Regress sum of squares = ((y_true - y_pred) **2).sum()
                    u = ((df[label_test[1]] - meanAverPred_out) ** 2).sum()
                    # residual sum of squares = ((y_true - Y_true.mean()) **2).sum()
                    v = ((df[label_test[1]] - df[label_test[1]].mean()) ** 2).sum()
                    # coefficient R^2 = (1- RegSS / ResSS)
                    meanAver_score_out = round((1 - (u / v)), 4)

                    outof_score[key+str(n)]['Aver_score'] = meanAver_score_out

    return rmse, bestmodel, difference, allDeviation, df_weights, score, proba, outof_score
