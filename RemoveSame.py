def RemoveSame(df, kind, allTargets, vecTraits, n):
    import numpy as np

    vecValid, exclude = [], []
    import pdb
    # pdb.set_trace()
    if kind == 'all':
        for u in range(len(df)):
            if np.nanstd(df.iloc[u]) < 1:
                exclude.append(u)
            else:
                vecValid.append(u)

    else:

        endpos = int(len(vecTraits) / 2)
        TraitsPos = vecTraits[0:endpos]
        TraitsNeg = vecTraits[endpos: len(vecTraits)]
        for Target in allTargets:
            veckeyPos, veckeyNeg = [], []
            for Trait in vecTraits:
                if Trait in TraitsPos:
                    if n == 3:
                        veckeyPos.append(Target + " " + Trait)
                    else:
                        veckeyPos.append(Target + "_" + Trait)
                elif Trait in TraitsNeg:
                    if n == 3:
                        veckeyNeg.append(Target + " " + Trait)
                    else:
                        veckeyNeg.append(Target + "_" + Trait)

            for u in range(len(df)):
                if np.nanstd(df[veckeyNeg+veckeyPos].iloc[u]) < 1:
                    exclude.append(int(u))

        exclude_uni = set(exclude)
        for k in range(len(df)):
            if k in exclude_uni:
                continue
            else:
                vecValid.append(k)

    return vecValid

def selectUsers(df, Users, Name, kind):

    #portion included:
    percIncl = (len(Users) / len(df)) * 100
    print(Name)
    if kind == 'all':
        print("users included based upon overall std < 2: " + str(percIncl) + "%")
    elif kind == 'Targ':
        print("users included based upon std < 2 on a targ: " + str(percIncl) + "%")

    df_out = df.iloc[Users,:]
    # import pdb
    # pdb.set_trace()
    return df_out