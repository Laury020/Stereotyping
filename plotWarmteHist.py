def plotWarmteHist(df, Key):
    import StringSplit
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    import pdb

    label, label_title = StringSplit.StringSplit(Key)
    plt.figure(figsize=[14, 7])
    color = ['blue', 'red', 'purple']
    for n in range(1, 4):
        if n <= 3:
            plt.subplot(3,1, n)
            #plt.subplot(2, 3, n)
            plt.hist(df[label[n]].dropna(), bins=70, color= color[n-1])
            plt.vlines(np.nanmean(df[label[n]]), 0, 8, colors='black')
            plt.title(label_title[n] + " n = " + str(len(df[label[n]].dropna())))
            plt.xlim([0, 100])
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
    plt.savefig(label[1] + '.png')
    plt.close('all')

def meanWarmthProba(df, Key):
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
            plt.plot(mlab.normpdf(vecX, np.nanmean(df[label[n]]), np.nanstd(df[label[n]])), color= color[n-1],
                     label=label_title[n])
            plt.vlines(np.nanmean(df[label[n]]), 0, 0.02, colors='black')
            plt.legend(loc='best')
            ax = plt.gca()
            ax.set_xlim([0, 100])
        elif n == 4:
            mean = ( np.nanmean(df[label[2]]) + np.nanmean(df[label[3]]) ) / 2
            std = ( np.nanstd(df[label[2]]) + np.nanstd(df[label[3]]) ) / 2
            plt.plot(mlab.normpdf(vecX, mean, std), color=color[2], label='Average Integrated Pred')
            plt.vlines(mean, 0, 0.02, colors='black')
            plt.legend(loc='best')
            ax = plt.gca()
            ax.set_xlim([0, 100])
        elif n == 5:
            mean = [np.nanmean(df[label[2]]),  np.nanmean(df[label[3]])]
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
    plt.savefig(label[1] + '.png')
    plt.close('all')

def scatWarmthInt(df, vecUniTargets):
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
            label_use, x, y  = 'Combined', np.nanmean(df[label[2]]), np.nanmean(df[label[3]])
            plt.annotate(label_use,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

            plt.scatter(np.nanmean(df[label[1]]), np.nanmean(df[label[3]]), len(df[label[3]].dropna()))
            label_use, x, y  = label_title[1], np.nanmean(df[label[3]]), np.nanmean(df[label[3]])
            plt.annotate(label_use,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

            plt.tight_layout()
            import os
            os.chdir('C:\\Users\\Thomas Dolman\\Documents\\UvA\\MasterStage2_MingHsu\\Onderzoek\\Data\\Overall')
            plt.savefig("Predictions/" + "Bayes_VS_Integrated_" + label_title[1])

def plotTargetVar(df, vecUniTargets):
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

    vecSTD = np.nanstd(df[Keys_new].values, axis= 0)
    # pdb.set_trace()

    plt.figure(figsize=[14, 7])
    xline = np.linspace(0, len(Keys_new), len(Keys_new))
    plt.bar(xline, vecSTD)
    plt.xticks(xline, Keys_new, rotation= 45)

    plt.ylim([np.min(vecSTD)-1, np.max(vecSTD)+1])
    plt.tight_layout()

    fileloc = "C:/Users/Thomas Dolman/Documents/UvA/MasterStage2_MingHsu/Onderzoek/Data/Overall/VarBars/"
    os.chdir(fileloc)
    plt.savefig(Keys_new[0] + '_Variance.png')
    plt.close('all')

def ChangeWarmthHist(df, Key):
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
            plt.title(vecTitle[n-1])

    plt.tight_layout()

    fileloc = "C:/Users/Thomas Dolman/Documents/UvA/MasterStage2_MingHsu/Onderzoek/Data/Overall/PopulationChange/"
    os.chdir(fileloc)
    plt.savefig(label_title[1] + "_Changes.png")
    plt.close('all')

def ScatterWarmth(df, Key):
    """This function makes a scatter plot between the uniTargets """
    import StringSplit
    import matplotlib.pyplot as plt
    import os

    label, label_title = StringSplit.StringSplit(Key)
    plt.figure(figsize=[14, 7])
    plt.subplot(3,1,1)
    plt.scatter(df[label[1]].dropna(), df[label[2]].dropna())
    plt.title(label_title[1] + " n = " + str(len(df[label[1]].dropna())))
    plt.xlim([0, 100])
    plt.xlabel(label_title[1])
    plt.ylabel(label_title[2])

    plt.subplot(3, 1, 2)
    plt.scatter(df[label[1]].dropna(), df[label[3]].dropna())
    plt.title(label_title[1] + " n = " + str(len(df[label[1]].dropna())))
    plt.xlim([0, 100])
    plt.xlabel(label_title[1])
    plt.ylabel(label_title[3])

    plt.subplot(3, 1, 3)
    plt.scatter(df[label[2]].dropna(), df[label[3]].dropna())
    plt.title(label_title[1] + " n = " + str(len(df[label[1]].dropna())))
    plt.xlim([0, 100])
    plt.xlabel(label_title[2])
    plt.ylabel(label_title[3])

    plt.tight_layout()
    fileloc = "C:/Users/Thomas Dolman/Documents/UvA/MasterStage2_MingHsu/Onderzoek/Data/Overall/MeanHist/"
    os.chdir(fileloc)
    plt.savefig(label[1] + '_scatter.png')
    plt.close('all')

def plotBayesProb(df, key, rmse, bestmodel, difference):
    """ this function plots the histogram over all subjects and uses
    Individual variance between personality traits to predict the bayesian integration
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

    # separate key into labels and label_title (label without _mean at the end)
    label, label_title = StringSplit.StringSplit(key)
    # create the relevant labels which are needed in selecting the right data
    keysmean = [label[2], label[3]]
    keysvar = [label_title[2] + "_var", label_title[3] + "_var"]
    # drop nan values from the DataFrame

    # retrieve the predictions
    # retrieve the Bayes predicted mean values from BayesValues
    meanBayPred, varbayPred = BayesValues.BayesPanda(df[keysmean], df[keysvar])
    # Use the average of both single targets which predicts the combined warmth target
    meanAverPred = np.nanmean(df[keysmean].dropna(), axis=1)
    # calculated the weighted mean of the combined targed by estimating weights on a subset and extrapolating that
    X = np.array([df[label[3]].dropna(), df[label[2]].dropna()])
    y = df[label[1]].dropna().values
    X = X.T
    meanWeightPred, y_test1 = modelfit.WeightAver(X, y)

    # create subset of combined targets which is the same as the subset made in WeightedAver.WeightAver(X, y) due to
    # random state set
    X_train, X_test, y_train, y_test2 = train_test_split(df[label[1]].dropna().values, df[label[1]].dropna().values,
                                                         train_size=0.3, random_state=42)
    # calculate the residual mean squared error for each prediction
    rmsebay = mean_squared_error(df[label[1]].dropna().values, meanBayPred.dropna())
    rmseAver = mean_squared_error(df[label[1]].dropna().values, meanAverPred)
    rmseWeight = mean_squared_error(X_test, meanWeightPred)

    # print("rmse of Bayesian predicted is : {}".format(rmsebay))
    # print("rmse of Average predicted is : {}".format(rmseAver))

    # calculate differences between predicted and actual combined target warmth reports
    BayesDiff = np.mean(df[label[1]].dropna()) - np.mean(meanBayPred.dropna())
    AverDiff = np.mean(df[label[1]].dropna()) - np.mean(meanAverPred)
    WeightDiff = np.mean(X_test) - np.mean(meanWeightPred)

    # save metrics into dictionary
    difference[key] = {"Bayes_diff": abs(BayesDiff), "Aver_diff": abs(AverDiff), "Weight_diff": abs(WeightDiff)}
    rmse[key] = {"Bayes_rmse": rmsebay, "Aver_rmse": rmseAver, "Weight_rmse": rmseWeight}

    # add score to the bestmodel to quantify which model does better predicitons
    min_rmse, min_diff = list(rmse[key].values()), list(difference[key].values())
    rmseKeys, diffKeys = list(rmse[key].keys()), list(difference[key].keys())
    LowRMSE_model = rmseKeys[min_rmse.index(min(min_rmse))]
    LowDiff_model = diffKeys[min_diff.index(min(min_diff))]

    # update the beat model
    bestmodel[LowRMSE_model] += 1
    bestmodel[LowDiff_model] += 1

    # plot the histograms for uni, uni, measured, predicted bayes, predicted average
    plt.figure(figsize=(24, 13))
    vecHists = [df[keysmean[0]].dropna(), df[keysmean[1]].dropna(), df[label[1]].dropna(), meanBayPred.dropna(), meanAverPred, meanWeightPred]
    titles = [label_title[2], label_title[3], label_title[1], "Bayes predicted", "Average predicted", "Weighted predicted"]
    color = ["red", "blue", "purple", "cyan", "green", "orange"]
    for n in range(len(vecHists)):
        ax = plt.subplot(len(vecHists), 1, n + 1)
        plt.hist(vecHists[n], bins=50, color=color[n])
        plt.vlines(np.mean(vecHists[n]), 0, 8, colors='black')
        plt.title(titles[n])
        plt.xlabel("warmth")
        plt.ylabel("count")
        ax.set_xlim(0, 100)

    plt.tight_layout()

    # save the figure
    fileloc = "C:/Users/Thomas Dolman/Documents/UvA/MasterStage2_MingHsu/Onderzoek/Data/Overall/BayesProb/"
    os.chdir(fileloc)
    plt.savefig(label[1] + '.png')

    return rmse, bestmodel, difference

