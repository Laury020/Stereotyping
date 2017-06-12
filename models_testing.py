def all(AllData_warmth, AllUniTarg, vecTrainSize, randomstate, types):
    """ this function either test how a different test size affects model fitting or just runs
    the models with a set test size"""
    import numpy as np
    import modelfit
    import plotWarmteHist as plot
    import pandas as pd
    import StringSplit
    import os
    import matplotlib.pyplot as plt
    import pdb

    # initialize some values which will store model performance. These will be filled in plotBayesProb
    preset = [{}, {}, {}, {}, {}]
    allModels, allDeviation = {}, {}
    allDeviation['variance'], allDeviation['standdev'] = {},{}
    labels = ['Bayes_rmse', 'Aver_rmse', 'Regress_rmse', 'BayPop_rmse', "BayTotalPop_rmse",
              'Bayes_diff', 'Aver_diff', 'Regres_diff', 'BayPop_diff', "BayTotalPop_diff",
              'Bayes_score', 'Aver_score', 'Regress_score', 'BayPop_score', "BayTotalPop_score"]

    # initialize model performane variables, rmse, mean_difference and R^2 (score)
    if len(vecTrainSize) != 1:
        data_rmse, data_diff, data_score = {}, {}, {}
        data_rmse_out, data_diff_out, data_score_out, data_bestmodel = {}, {}, {}, {}

    # loop over all Test size options
    for train_size in vecTrainSize:

        # create dicts were needed
        if len(vecTrainSize) != 1:
            data_rmse[train_size] = {}
            data_diff[train_size] = {}
            data_score[train_size] = {}
            allModels[train_size] = {}
            # data_bestmodel[train_size] = []

        # create presets to be filled in plotBayes
        for l in labels:
            preset[1][l] = 0

        outof_score_all = []

        # loop over the relevant targets
        for Key in AllData_warmth[2].keys():
            # this section ensures not all keys are used in the functions.
            # this increases efficiency of the code
            vecUniTargets = AllUniTarg[2]
            if Key in vecUniTargets:
                continue
            elif Key[-1] == 'r':
                continue

                # if len(vecTrainSize) == 1:
                # print(Key)
                # print("\n")


            rmse, bestmodel, difference, allDeviation, df_weights, score, proba, outof_score = plot.plotBayesProb(AllData_warmth[2], Key, preset,
                                                                      train_size, randomstate, types, allDeviation, [], [])


            # update the allModels DF with the weights of each model and then the ratios of Normalized variance and standdev
            # pdb.set_trace()
            # df_dev = pd.DataFrame(allDeviation[Key])
            df_append = pd.concat([df_weights, proba])
            try:
                allModels[train_size] = pd.concat([allModels[train_size], df_append], axis=1)
            except:
                allModels[train_size] = df_append


            if len(vecTrainSize) != 1:

                for rsme_lab in labels[0:5]:
                    try:
                        data_rmse[train_size][rsme_lab]
                    except:
                        data_rmse[train_size][rsme_lab] = []

                    data_rmse[train_size][rsme_lab].append(rmse[Key][rsme_lab])

                # bestmodel_all[train_size] = bestmodel
                for dif_lab in labels[5:10]:
                    try:
                        data_diff[train_size][dif_lab]
                    except:
                        data_diff[train_size][dif_lab] = []

                    data_diff[train_size][dif_lab].append(difference[Key][dif_lab])

                for model_sc in score[Key].keys():
                    try:
                        data_score[train_size][model_sc]
                    except:
                        data_score[train_size][model_sc] = []

                    data_score[train_size][model_sc].append(score[Key][model_sc])

                    # data_score[model_sc][Key].append(score[Key][model_sc])

                # performance_all[train_size] = performance
                # data_struct[2][train_size][Key].append(R_sqr_best)
                # pdb.set_trace()
                data_bestmodel[train_size] = bestmodel

        # create mean of difference and RMSE overall the Targets
        if len(vecTrainSize) != 1:
            for yup in labels[0:5]:
                try:
                    data_rmse_out[yup]
                except:
                    data_rmse_out[yup] = []

                data_rmse_out[yup].append(np.mean(data_rmse[train_size][yup]))
            for puy in labels[5:10]:
                try:
                    data_diff_out[puy]
                except:
                    data_diff_out[puy] = []

                data_diff_out[puy].append(np.mean(data_diff[train_size][puy]))
            for ypu in labels[10:15]:
                try:
                    data_score_out[ypu]
                except:
                    data_score_out[ypu] = []

                data_score_out[ypu].append(np.mean(data_score[train_size][ypu]))

    # write scores to pandas dataframe

    if len(vecTrainSize) != 1:
        # TODO huh de afname is er niet, wat gebeurt hier :O
        # pdb.set_trace()
        indexSampSize = len(AllData_warmth[2]) * vecTrainSize
        indexSampSize = indexSampSize.round(2)
        df_RMSE = pd.DataFrame(data_rmse_out, index=indexSampSize)
        df_Diff = pd.DataFrame(data_diff_out, index=indexSampSize)
        df_score = pd.DataFrame(data_score_out, index=indexSampSize)
        # df_weights = pd.DataFrame(allModels, index=indexSampSize)
        scores_out = [df_RMSE, df_Diff, df_score]

        # df_score.plot()
        # plt.xlabel('train size')
        # plt.ylabel('R^2 predictive power')
        # os.chdir("C:/Users/Thomas Dolman/Documents/UvA/MasterStage2_MingHsu/Onderzoek/Data/Overall/ModelScores")
        # plt.savefig('R_squared' + str(randomstate) + '.png')
        #
        #
        # df_Diff.plot()
        # plt.xlabel('train size')
        # plt.ylabel('Mean Difference')
        # plt.savefig('Difference' + str(randomstate) + '.png')
        #
        # df_RMSE.plot()
        # plt.xlabel('train size')
        # plt.ylabel('RMSE average')
        # plt.savefig('RMSE' + str(randomstate) + '.png')
        # ind = round(len(AllData_warmth[2]) * 0.30060606, 2)
        # plt.hlines(df_RSquare_Regress.mean(axis=1)[ind], 0, ind)
        # plt.vlines(ind, 0, df_RSquare_Regress.mean(axis=1)[ind])

        # plt.close('all')

        # import csv
        # save dataframes
        # # TODO: examine this  code to write data frames to one .csv file
        # # https://stackoverflow.com/questions/20982165/pandas-dataframe-to-csv
        os.chdir("C:/Users/Thomas Dolman/Documents/UvA/MasterStage2_MingHsu/Onderzoek/Data/Overall")

        filename = "Varying_train_size_Rsquare.csv"
        df_score.to_csv(filename)

        filename = 'Varying_train_size_RMSE.csv'
        df_RMSE.to_csv(filename)

        filename = 'Varying_train_size_diff.csv'
        df_Diff.to_csv(filename)


    return scores_out, allModels, allDeviation, bestmodel, outof_score_all
