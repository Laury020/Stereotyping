def RavenCorr(df):
    import numpy as np

    if len(df) == 122:
        vecRaven = ['Raven_1', 'Raven_2']
        correct = [3, 3]
        matCorr = {}
        for j in range(0, len(vecRaven)):
            RavenVal = df[vecRaven[j]]
            vecCorr = np.repeat(correct[j], len(RavenVal))
            matCorr[j] = RavenVal == vecCorr

        vecValid = []
        for l in range(0, len(matCorr[0])):
            if matCorr[0][l] == True or matCorr[1][l] == True:
                vecValid.append(l)

    elif len(df) == 302:
        vecRaven = ['Raven_1 ', 'Raven_2 ', 'Raven_3 ', 'Raven_4 ']
        correct = [2, 5, 8, 7]
        matCorr = {}

        for j in range(0, 4):
            RavenVal = df[vecRaven[j]]
            vecCorr = np.repeat(correct[j], len(RavenVal))
            matCorr[j] = RavenVal == vecCorr

        vecValid = []
        for l in range(0,len(matCorr[0])):
            if matCorr[0][l] == True or matCorr[1][l] == True or matCorr[2][l] == True or matCorr[3][l] == True:
                vecValid.append(l)

    return vecValid

def selectUsers(df, Users, Name):

    #portion included:
    percIncl = (len(Users) / len(df)) * 100
    print(Name)
    print("users included: " + str(percIncl) + "%")

    df_out = df.iloc[Users]
    return df_out


def removeRaven(df):

    if len(df) == 302:
        vecRaven = ['Raven_1 ', 'Raven_2 ', 'Raven_3 ', 'Raven_4 ']
        for m in range(0,4):
            if m != 0:
                for n in range(0,len(vecRaven)):
                    df = df.drop(vecRaven[n] + '.' + str(m), axis= 1)
            else:
                for n in range(0,len(vecRaven)):
                    # import pdb
                    # pdb.set_trace()
                    df = df.drop(vecRaven[n], axis= 1)

    elif len(df) == 122:
        vecRaven = ['Raven_1', 'Raven_2']
        for m in range(0,len(vecRaven)):
            #import pdb; pdb.set_trace()
            df = df.drop(vecRaven[m], axis= 1)

    return df

