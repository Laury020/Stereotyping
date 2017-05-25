
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

# preprocess the data, first get Ravencorrect. Use this to throw participants away
# remove the Raven questions and combine the different occurences of the same Target
AllData_ready, AllData_warmth, AllUniTarg = [], [], []
for n in range(len(AllData)):
    df = AllData[n]
    allTargets, allTraits, vecUniTargets = getTargets_Traits.getTargets_Traits(n)
    AllUniTarg.append(vecUniTargets)
    if n < 2:
        Samevalids = RemoveSame.RemoveSame(df, allTargets, allTraits)
        df_new = RemoveSame.selectUsers(df, Samevalids, Names[n])
    else:
        Ravenvalids = Rave.RavenCorr(df)
        df = Rave.selectUsers(df, Ravenvalids, Names[n])
        df_new = Rave.removeRaven(df)

    if n < 3:
        df_ready = CombineTraits.combineTraits(df_new, allTargets, allTraits, n)
    else:
        df_ready = df_new

    AllData_ready.append(df_ready)
    # pdb.set_trace()
    # calculate the std values over the personality traits and save this in Target_std
    df_warm_var = Ind.SuvjVarAnalys(df_ready, allTargets, allTraits, n)

    for Key in df_warm_var.keys():
        df_warm_var[Key] = df_warm_var[Key].replace(np.nan, np.mean(df_warm_var[Key]))

    AllData_warmth.append(df_warm_var)

    for Key in df_warm_var.keys():
        if Key in vecUniTargets:
            continue
        elif Key[-1] == 'r':
            continue
        else:
            plot.meanWarmthProba(df_warm_var, Key)
            plot.plotWarmteHist(df_warm_var, Key)
            plot.ChangeWarmthHist(df_warm_var, Key)
            plot.ScatterWarmth(df_warm_var, Key)

    plot.scatWarmthInt(df_warm_var, vecUniTargets)
    plot.plotTargetVar(df_warm_var, vecUniTargets)

# for n in range(len(AllData_warmth)):
#     df = AllData_warmth[n]
#     Key = df.keys()[0]
#     label, label_title = StringSplit.StringSplit(Key)
#     print("Key: " + Key)
#     print("Labels: {}".format(label))
# //

# calculate warmth over personality traits and save this in Target_mean
# df_warmth = getWarmthValue.getWarmthValue(df_ready, allTargets, allTraits)

# initialize some values which will store model performance. These will be filled in plotBayesProb
bestmodel, difference, rmse = {}, {}, {}
labels = ['Bayes_rmse', 'Aver_rmse', 'Weight_rmse', 'Aver_diff', 'Bayes_diff', 'Weight_diff']
for l in labels:
    bestmodel[l] = 0
count = np.array(np.zeros((3, 1)))
performance, scores = {},{}
for n in range(4):
    performance[n] = {}

for Key in AllData_warmth[2].keys():
    # this section ensures not all keys are used in the functions.
    # this increases efficiency of the code
    vecUniTargets = AllUniTarg[2]
    if Key in vecUniTargets:
        continue
    elif Key[-1] == 'r':
        continue

    print(Key)
    # plot.plotWarmteHist(df_warmth, Key)
    # plot.ChangeWarmthHist(df_warmth, Key)
    # plot.meanWarmthProba(df_warmth, Key)
    print("\n")
    count, modelNames, performance, models = modelfit.SelectModel(AllData_warmth[2], Key, count, performance)
    # plot.plotWarmteHist(df_warmth, Key)
    rmse, bestmodel, difference = plot.plotBayesProb(AllData_warmth[2], Key, rmse, bestmodel, difference)

    for n in range(len(AllData_warmth)):
        # loop over the other questionaires and check if the model works there as well
        if n == 2:
            continue
        vecUniTargets = AllUniTarg[n]
        df = AllData_warmth[n]
        # pdb.set_trace()
        for Key_test in df.keys():

            if Key_test in vecUniTargets:
                continue
            elif Key_test[-1] == 'r':
                continue
            try:
                scores[Key]
            except KeyError:
                scores[Key] = {}

            try:
                label, label_title = StringSplit.StringSplit(Key_test)
                label2, label_title2 = StringSplit.StringSplit(Key)
            except ValueError:
                pdb.set_trace()

            # if label[3] in label2[3] or label[2] in label2[2]:

            X = np.array([df[label[3]].dropna(), df[label[2]].dropna()])
            y = df[label[1]].dropna().values
            X = X.T
            lasso_pred = models[2].predict(X)
            scores[Key][Key_test] = models[2].score(X, y)
            # print("the Key on which the model is fitted: {}".format(Key))
            # print("the label of the tested data: {}".format(Key_test))
            # print(scores[Key][Key_test])

print("\n")
print(rmse)
print(bestmodel)
print(difference)
print("\n")
count = list(count)
print("best model overall is : {}".format(modelNames[count.index(max(count))]))
print(count)
print("targets with poor R^2 : {}".format(performance[0]))
print("total {}".format(len(performance[0])))
print("targets with medium R^2 : {}".format(performance[1]))
print("total {}".format(len(performance[1])))
print("targets with good R^2 : {}".format(performance[2]))
print("total {}".format(len(performance[2])))
print("targets with very good R^2 : {}".format(performance[3]))
print("total {}".format(len(performance[3])))

# write scores to csv file
df_out = pd.DataFrame(scores)
filename = 'Out_of_sample_Predictions.csv'
os.chdir("C:/Users/Thomas Dolman/Documents/UvA/MasterStage2_MingHsu/Onderzoek/Data/Overall")
df_out.to_csv(filename)
