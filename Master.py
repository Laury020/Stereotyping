import pandas as pd
import os
import numpy as np
import RavenCorr as Rave
import CombineTraits
import getTargets_Traits
import RemoveSame
import plotWarmteHist as plot
import Individual as Ind
import modelfit
import StringSplit
import pdb
import matplotlib.pyplot as plt

Saving = False

# load the data in
os.chdir("C:/Users/Thomas Dolman/Documents/UvA/MasterStage2_MingHsu/Onderzoek/Data/Integration_March_1Block")
df_march1block = pd.read_csv("Integration_March17_1Block_Pilot_PythonReady.csv")
# load the data in
os.chdir("C:/Users/Thomas Dolman/Documents/UvA/MasterStage2_MingHsu/Onderzoek/Data/Names")
df_names = pd.read_csv("data//Names_preprocessed.csv")
# load the data in
os.chdir("C:/Users/Thomas Dolman/Documents/UvA/MasterStage2_MingHsu/Onderzoek/Data/Integration_March_MultiBlock")
df_marchmultblock = pd.read_csv("MultipleBlocks_Integration_March_Preprossed(1).csv")
# load the data in
os.chdir("C:/Users/Thomas Dolman/Documents/UvA/MasterStage2_MingHsu/Onderzoek/Data/Integration_Feb")
df_feb = pd.read_csv("SingleBlock_Integration_Feb_Preprossed.csv")
os.chdir("C:/Users/Thomas Dolman/Documents/UvA/MasterStage2_MingHsu/Onderzoek/Data/Overall")

AllData = [df_feb, df_march1block, df_marchmultblock, df_names]
Names = ["Febuary", "March_1block", "March_4block", 'May_Names']
# pilot, second, first and third study

# preprocess the data, first get Ravencorrect. Use this to throw participants away
# remove the Raven questions and combine the different occurences of the same Target
AllData_new, AllData_warmth, AllUniTarg, AllData_CorrVals, AllData_Pvals = [], [], [], [], []
CorrVals, Pvals, = {}, {}
for n in range(len(AllData)):
    df = AllData[n]
    allTargets, allTraits, vecUniTargets = getTargets_Traits.getTargets_Traits(n)
    AllUniTarg.append(vecUniTargets)

    if n < 3:
        df_ready = CombineTraits.combineTraits(df, allTargets, allTraits, n)
    else:
        df_ready = df

    if n < 2:
        Samevalids = RemoveSame.RemoveSame(df_ready, 'all', allTargets, allTraits, n)
        df_new = RemoveSame.selectUsers(df_ready, Samevalids, Names[n], 'all')

        Samevalids = RemoveSame.RemoveSame(df_ready, 'Targ', allTargets, allTraits, n)
        df_new = RemoveSame.selectUsers(df_ready, Samevalids, Names[n], 'Targ')
    elif n == 2:
        Ravenvalids = Rave.RavenCorr(df)
        df_new = Rave.selectUsers(df_ready, Ravenvalids, Names[n])
        # add remove same based upon std scores
        Samevalids = RemoveSame.RemoveSame(df_new, 'all', allTargets, allTraits, n)
        df_new = RemoveSame.selectUsers(df_new, Samevalids, Names[n], 'all')
        Samevalids = RemoveSame.RemoveSame(df_new, 'Targ', allTargets, allTraits, n)
        df_new = RemoveSame.selectUsers(df_new, Samevalids, Names[n], 'Targ')
    else:
        Ravenvalids = Rave.RavenCorr(df)
        df_new = Rave.selectUsers(df_ready, Ravenvalids, Names[n])

    print('People excluded: ' + str(len(df) - len(df_new)))
    print('Original subjects: ' + str(len(df)))
    AllData_new.append(df_new)

    # calculate the std values over the personality traits and save this in Target_std
    df_warm_var = Ind.SuvjVarAnalys(df_new, allTargets, allTraits, n)

    for Key in df_warm_var.keys():
        df_val = df_warm_var[Key].value_counts(dropna=False)
        if df_val[np.nan] > 5:
            df_warm_var[Key] = df_warm_var[Key].dropna()
        else:
            df_warm_var[Key] = df_warm_var[Key].replace(np.nan, np.mean(df_warm_var[Key]))

    percIncl = (len(df_warm_var) / len(df)) * 100
    print(Names[n])
    print("users included based upon all criteria: " + str(percIncl) + "%")

    AllData_warmth.append(df_warm_var)

    if Saving:
        for Key in df_warm_var.keys():
            if Key in vecUniTargets:
                continue
            elif Key[-1] == 'r':
                continue
            else:
                plot.meanWarmthProba(df_warm_var, Key, Names[n])
                plot.plotWarmteHist(df_warm_var, Key, Names[n])
                plot.ChangeWarmthHist(df_warm_var, Key, Names[n])
                CorrValues, PValuess = plot.ScatterWarmth(df_warm_var, Key, Names[n])
                try:
                    CorrVals[Key]
                    try:
                        CorrVals[Key + '.1']
                        pdb.set_trace()
                    except:
                        CorrVals[Key + '.1'] = CorrValues
                        Pvals[Key + '.1'] = PValuess
                except:
                    CorrVals[Key] = CorrValues
                    Pvals[Key] = PValuess

        plot.scatWarmthInt(df_warm_var, vecUniTargets, Names[n])
        plot.plotTargetVar(df_warm_var, vecUniTargets, Names[n])

import models_testing

vecTrainSize = np.linspace(0.03, 0.5, 100)

bestmodel_file, RMSE_file, diff_file, Rsq_file = {}, {}, {}, {}
# RMSE_file_plt, diff_file_plt, Rsq_file_plt = {},{},{}
allstates = range(1,501)
# allstates = range(1,101)
# allstates = range(1, 11)
# allstates = range(1, 2)

# allstates = [39]
# vecTrainSize = [0.3]
for types in [2]:
    for randomState in allstates:
        scores, allModels, allDeviation, bestmodel, outof_score = models_testing.all(AllData_warmth, AllUniTarg,
                                                                                     vecTrainSize, randomState, types)
        print("random state = " + str(randomState))
        bestmodel_file[randomState] = bestmodel
        Trainsize = 70.64
        RMSE_file[randomState] = scores[0].loc[Trainsize, :]
        diff_file[randomState] = scores[1].loc[Trainsize, :]
        Rsq_file[randomState] = scores[2].loc[Trainsize, :]
        # RMSE_file_plt[randomState] = scores[0]
        # diff_file_plt[randomState] = scores[1]
        # Rsq_file_plt[randomState] = scores[2]
        # pdb.set_trace()
        try:
            RMSE_file_plt1 = RMSE_file_plt1 + scores[0]
        except NameError:
            RMSE_file_plt1 = scores[0]
        try:
            diff_file_plt1 = diff_file_plt1 + scores[1]
        except NameError:
            diff_file_plt1 = scores[1]
        try:
            Rsq_file_plt1 = Rsq_file_plt1 + scores[2]
        except NameError:
            Rsq_file_plt1 = scores[2]

        train_size = 0.3
        preset = [{}, {}, {}, {}, {}]
        for Key in AllData_warmth[2].keys():
            # pdb.set_trace()
            if Key[-1] == 'r':
                continue
            if Key in AllUniTarg[2]:
                continue
            rmse, bestmodel, difference, allDeviation, df_weights, score, proba, outof_score = plot.plotBayesProb(
                AllData_warmth[2], Key, preset, train_size, randomState, types, allDeviation, AllData_warmth,
                AllUniTarg)

        df_score = pd.DataFrame(score)
        df_score_outof = pd.DataFrame(outof_score)
        try:
            score_file = score_file + df_score
        except NameError:
            score_file = df_score
        try:
            score_out_file = score_out_file + df_score_outof
        except NameError:
            score_out_file = df_score_outof

            # pdb.set_trace()
            # print(Rsq_file_plt1.iloc[0, :])
            # print(score_file.mean(axis=1))

    df_bestModel = pd.DataFrame(bestmodel_file)
    df_bestModel_out = df_bestModel.sum(axis=1)
    df_bestModel_out = pd.DataFrame(df_bestModel_out)
    df_bestModel_out['ratio'] = df_bestModel_out / df_bestModel.sum(axis=1).sum(axis=0)
    df_bestModel_out.columns = ['Summed scores', 'ratio']
    df_RMSE = pd.DataFrame(RMSE_file)
    df_RMSE_out = df_RMSE.mean(axis=1)
    df_RMSE_out = pd.DataFrame(df_RMSE_out)
    df_RMSE_out['ratio'] = df_RMSE_out / df_RMSE.sum(axis=1).sum(axis=0)
    df_RMSE_out.columns = ['average RMSE', 'ratio']
    df_Diff = pd.DataFrame(diff_file)
    df_Diff_out = df_Diff.mean(axis=1)
    df_Diff_out = pd.DataFrame(df_Diff_out)
    df_Diff_out['ratio'] = df_Diff_out / df_Diff.sum(axis=1).sum(axis=0)
    df_Diff_out.columns = ['average difference', 'ratio']
    df_Rsq = pd.DataFrame(Rsq_file)
    df_Rsq_out = df_Rsq.mean(axis=1)
    df_Rsq_out = pd.DataFrame(df_Rsq_out)
    df_Rsq_out['ratio'] = df_Rsq_out / df_Rsq.sum(axis=1).sum(axis=0)
    df_Rsq_out.columns = ['average Rsquared', 'ratio']

    Rsq_file_plt1 = Rsq_file_plt1 / len(allstates)
    df_Rsq_file_plt = pd.DataFrame(Rsq_file_plt1)
    # df_Rsq_file_plt_out = df_Rsq_file_plt.mean(axis=1)
    # df_Rsq_file_plt_out = pd.DataFrame(df_Rsq_file_plt_out)

    diff_file_plt1 = diff_file_plt1 / len(allstates)
    df_diff_file_plt = pd.DataFrame(diff_file_plt1)
    # df_diff_file_plt_out = df_diff_file_plt.mean(axis=1)
    # df_diff_file_plt_out = pd.DataFrame(df_diff_file_plt_out)

    RMSE_file_plt1 = RMSE_file_plt1 / len(allstates)
    df_RMSE_file_plt = pd.DataFrame(RMSE_file_plt1)
    # df_RMSE_file_plt_out = df_RMSE_file_plt.mean(axis=1)
    # df_RMSE_file_plt_out = pd.DataFrame(df_RMSE_file_plt_out)

    os.chdir("C:/Users/Thomas Dolman/Documents/UvA/MasterStage2_MingHsu/Onderzoek/Data/Overall/ModelScores")
    # pdb.set_trace()
    df_Rsq_file_plt = df_Rsq_file_plt.rename(columns={"Aver_score": 'Average', "BayPop_score": 'Bayes test population',
                                    "BayTotalPop_score": 'Bayes total population', "Bayes_score":'Bayes subject wise', "Regress_score":'Regression'})
    df_Rsq_file_plt.plot()
    plt.xlabel('train size')
    plt.ylabel('R^2 predictive power')
    plt.savefig('Average_R_squared_' + str(len(allstates)) + '_.eps', format='eps')
    plt.savefig('Average_R_squared_' + str(len(allstates)) + '_.png', format='png')
    df_diff_file_plt = df_diff_file_plt.rename(columns={"Aver_diff": 'Average', "BayPop_diff": 'Bayes test population',
                                    "BayTotalPop_diff": 'Bayes total population', "Bayes_diff": 'Bayes subject wise',
                                    "Regres_diff": 'Regression'})
    df_diff_file_plt.plot()
    plt.xlabel('train size')
    plt.ylabel('Difference to mean')
    plt.savefig('Average_Difference_' + str(len(allstates)) + '_.eps', format='eps')
    plt.savefig('Average_Difference_' + str(len(allstates)) + '_.png', format='png')
    df_RMSE_file_plt = df_RMSE_file_plt.rename(columns={"Aver_rmse": 'Average', "BayPop_rmse": 'Bayes test population',
                                    "BayTotalPop_rmse": 'Bayes total population', "Bayes_rmse": 'Bayes subject wise',
                                    "Regress_rmse": 'Regression'})
    df_RMSE_file_plt.plot()
    plt.xlabel('train size')
    plt.ylabel('RMSE')
    plt.savefig('Average_RMSE_' + str(len(allstates)) + '_.eps', format='eps')
    plt.savefig('Average_RMSE_' + str(len(allstates)) + '_.png', format='png')

    # save the dataframes
    os.chdir("C:/Users/Thomas Dolman/Documents/UvA/MasterStage2_MingHsu/Onderzoek/Data/Overall")
    vecType = ['_allmod_ttest', '_allmod_NOttest', '_submod_NOttest', '_submod_ttest']
    filename = 'bestmodel_' + 'n' + str(len(allstates)) + vecType[types] + '.csv'
    df_bestModel_out.to_csv(filename)
    filename = 'RMSE_' + 'n' + str(len(allstates)) + vecType[types] + '.csv'
    df_RMSE_out.to_csv(filename)
    filename = 'Difference_' + 'n' + str(len(allstates)) + vecType[types] + '.csv'
    df_Diff_out.to_csv(filename)
    filename = 'Rsquared_' + 'n' + str(len(allstates)) + vecType[types] + '.csv'
    df_Rsq_out.to_csv(filename)

indexSampSize = len(AllData_warmth[2]) * vecTrainSize
indexSampSize = indexSampSize.round(2)
vecTrainSize[indexSampSize == 70.64]

# statistical analyses.....
# analysis of the different surveys
vecTrainSize = [0.3]
if len(vecTrainSize) == 1:
    import stats_Overall

    stats_Overall.test_means(AllData_warmth, vecUniTargets)

# Save the values if one test size is used, not a vector

if len(vecTrainSize) == 1:
    # write scores to csv file
    # pdb.set_trace()
    df_out_score = score_out_file / len(allstates)
    # df_out_score = pd.DataFrame(score_out_file1)
    # pdb.set_trace()
    filename = 'Out_of_sample_Predictions' + str(len(allstates)) + '.csv'
    os.chdir("C:/Users/Thomas Dolman/Documents/UvA/MasterStage2_MingHsu/Onderzoek/Data/Overall")
    df_out_score.to_csv(filename)
    print(filename + '  saved')

    df_in_score = score_file / len(allstates)
    # df_in_score = pd.DataFrame(score_file1)
    filename = 'In_sample_Predictions' + str(len(allstates)) + '.csv'
    os.chdir("C:/Users/Thomas Dolman/Documents/UvA/MasterStage2_MingHsu/Onderzoek/Data/Overall")
    df_in_score.to_csv(filename)

    plt.close('all')

    # save the model coeffs and the relation plot
    import seaborn as sns
    # pdb.set_trace()
    modelselect = 0.30060606060606054
    df_Models = pd.DataFrame(allModels[modelselect])
    df_Models = df_Models.T
    sns.lmplot(x='Bayes', y='Regress', data=df_Models)
    filename = 'Models' + str(round(modelselect, 2)) + 'B-R.png'
    plt.savefig(filename)
    sns.lmplot(x='Ratios_variance', y='Regress', data=df_Models)
    filename = 'Models' + str(round(modelselect, 2)) + 'ratioVar-R.png'
    plt.savefig(filename)
    sns.lmplot(x='Ratios_variance', y='Bayes', data=df_Models)
    filename = 'Models' + str(round(modelselect, 2)) + 'ratioVar-B.png'
    plt.savefig(filename)
    sns.lmplot(x='variance', y='Regress', data=df_Models)
    filename = 'Models' + str(round(modelselect, 2)) + 'Var-R.png'
    plt.savefig(filename)
    sns.lmplot(x='standdev', y='Regress', data=df_Models)
    filename = 'Models' + str(round(modelselect, 2)) + 'std-R.png'
    plt.savefig(filename)
    plt.savefig(filename)
    sns.lmplot(x='variance', y='Bayes', data=df_Models)
    filename = 'Models' + str(round(modelselect, 2)) + 'Var-B.png'
    plt.savefig(filename)
    # pdb.set_trace()
    filename = 'Models' + str(round(modelselect,2)) + '.csv'
    df_Models.to_csv(filename)
    #
    #
    # modelselect =0.1012121212121212
    # df_Models = pd.DataFrame(allModels[modelselect])
    # df_Models = df_Models.T
    # sns.lmplot(x='Bayes', y='Regress', data=df_Models)
    # filename = 'Models' + str(round(modelselect, 2)) + 'B-R.png'
    # plt.savefig(filename)
    # sns.lmplot(x='Ratios_variance', y='Regress', data=df_Models)
    # filename = 'Models' + str(round(modelselect, 2)) + 'Var-R.png'
    # plt.savefig(filename)
    # sns.lmplot(x='Ratios_variance', y='Bayes', data=df_Models)
    # filename = 'Models' + str(round(modelselect, 2)) + 'Var-B.png'
    # plt.savefig(filename)
    # filename = 'Models' + str(round(modelselect,2)) + '.csv'
    # df_Models.to_csv(filename)
    #
    # modelselect = 0.5
    # df_Models = pd.DataFrame(allModels[modelselect])
    # df_Models = df_Models.T
    # sns.lmplot(x='Bayes', y='Regress', data=df_Models)
    # filename = 'Models' + str(round(modelselect, 2)) + 'B-R.png'
    # plt.savefig(filename)
    # sns.lmplot(x='Ratios_variance', y='Regress', data=df_Models)
    # filename = 'Models' + str(round(modelselect, 2)) + 'Var-R.png'
    # plt.savefig(filename)
    # sns.lmplot(x='Ratios_variance', y='Bayes', data=df_Models)
    # filename = 'Models' + str(round(modelselect, 2)) + 'Var-B.png'
    # plt.savefig(filename)
    # filename = 'Models' + str(round(modelselect,2)) + '.csv'
    # df_Models.to_csv(filename)


    # save the model coeffs
    df_STD = pd.DataFrame(allDeviation['standdev'])
    filename = 'Deviation_STD.csv'
    df_STD.to_csv(filename)

    # save the model coeffs
    df_var = pd.DataFrame(allDeviation['variance'])
    filename = 'Deviation_var.csv'
    df_var.to_csv(filename)

    print('finished')

if Saving:
    # save the correlation values, p and n
    df_corr_out = pd.DataFrame(CorrVals, index=['Comb - Eth', 'Comb - Occu', 'Eth - Occu'])
    filename = 'Correlation_Scores.csv'
    os.chdir("C:/Users/Thomas Dolman/Documents/UvA/MasterStage2_MingHsu/Onderzoek/Data/Overall")
    df_corr_out.to_csv(filename)

    # save the correlation values, p and n
    df_corr_P_out = pd.DataFrame(Pvals, index=['Comb - Eth', 'Comb - Occu', 'Eth - Occu'])
    filename = 'Correlation_Ps.csv'
    os.chdir("C:/Users/Thomas Dolman/Documents/UvA/MasterStage2_MingHsu/Onderzoek/Data/Overall")
    df_corr_P_out.to_csv(filename)
