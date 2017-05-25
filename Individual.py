def SuvjVarAnalys(df, allTargets, vecTraits, n):
    """ this function examines the within subject variance per Target over Traits
    """

    import pdb
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # initialize new DataFrame with the right Columns
    Columns = []
    for Target in allTargets:
        Columns.append(Target + '_mean')
    Index = range(1, df.shape[0] + 1)
    df_new = pd.DataFrame(index=Index, columns=Columns)

    # Distinction between pos and negative Traits
    endpos = int(len(vecTraits) / 2)
    TraitsPos = vecTraits[0:endpos]
    TraitsNeg = vecTraits[endpos: len(vecTraits)]

    # loop over Targets and Traits to create a vector with positive and negative ratings
    for Target in allTargets:
        veckeyPos,veckeyNeg = [], []
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


        # transform negative to the inverse
        df[veckeyNeg] = 100 - df[veckeyNeg]

        # calculate variance
        # import pdb
        # pdb.set_trace()
        var = pd.DataFrame(np.nanvar(df[veckeyNeg+veckeyPos], axis= 1))
        mean = pd.DataFrame(np.nanmean(df[veckeyNeg + veckeyPos], axis=1))
        # save variance
        df_new[Target + "_var"] = var
        df_new[Target + "_mean"] = mean

    return df_new
