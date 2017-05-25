def RemoveSame(df,allTargets, allTraits):
    import numpy as np

    vecValid, exclude = [], []

    for u in range(len(df)):
        if np.nanstd(df.iloc[u]) < 2:
            exclude.append(u)
        else:
            vecValid.append(u)

    return vecValid

def selectUsers(df, Users, Name):

    #portion included:
    percIncl = (len(Users) / len(df)) * 100
    print(Name)
    print("users included: " + str(percIncl) + "%")

    df_out = df.iloc[Users]
    return df_out