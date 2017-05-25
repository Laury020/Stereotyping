def combineTraits(df, allTargets, allTraits, n):
    import numpy as np
    import pandas as pd

    if n == 0:
        repeats = 3
    elif n == 1:
        repeats = 2
    elif n == 2:
        repeats = 4

    # Target in vecHeaders
    # determine size of dataframe_new
    Columns = []
    for Target in allTargets:
        for Trait in allTraits:
            Columns.append(Target + '_' + Trait)

    Index = range(1, df.shape[0] + 1)
    df_new = pd.DataFrame(index=Index, columns=Columns)

    # combine the multiple same Trait presentations to one Trait
    for Target in allTargets:
        for Trait in allTraits:
            matrixTrait = {}
            for m in range(0, repeats):
                if m == 0:
                    if n == 0:
                        label = Target + " " + Trait
                    else:
                        label = Target + '  ' + Trait
                    matrixTrait[m] = df[label].values
                else:
                    if n == 0:
                        label = Target + " " + Trait + '.' + str(m)
                    else:
                        label = Target + '  ' + Trait + '.' + str(m)

                    try:
                        matrixTrait[m] = df[label].values
                    except KeyError:
                        continue
            if len(matrixTrait) == 4:
                npMatrixTrait = [matrixTrait[0], matrixTrait[1], matrixTrait[2], matrixTrait[3]]
            elif len(matrixTrait) == 3:
                npMatrixTrait = [matrixTrait[0], matrixTrait[1], matrixTrait[2]]
            elif len(matrixTrait) == 2:
                npMatrixTrait = [matrixTrait[0], matrixTrait[1]]
            elif len(matrixTrait) == 1:
                npMatrixTrait = [matrixTrait[0]]

            meanTrait = np.nanmean(npMatrixTrait, axis=0)
            df_new[Target + '_' + Trait] = meanTrait

    return df_new
