def RegressionFit(X, y, train_size, randomstate):
    """ This function will fit a regression model to a training set and test it on the testing
    set. The model is fitted per Key which is a label """

    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score

    # Create training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_size, random_state=randomstate)

    # Create the regressor: reg_all
    reg_all = LinearRegression(normalize= True)

    # Print the 5-fold cross-validation scores
    #print(cv_scores)

    #print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

    # Fit the regressor to the training data
    reg_all.fit(X_train, y_train)

    # params = reg_all.get_params()
    # print(params)
    # import pdb
    # pdb.set_trace()


    # Predict on the test data: y_pred
    y_pred = reg_all.predict(X_test)

    # Compute and print R^2 and RMSE
    #print("R^2: {}".format(reg_all.score(X_test, y_test)))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    #print("Root Mean Squared Error: {}".format(rmse))
    if train_size == 0.3:
        # Compute 5-fold cross-validation scores: cv_scores
        cv_scores = cross_val_score(reg_all, X_train, y_train, cv=5)
        score_out = cv_scores
    else:
        # score_out = reg_all.score(X_test, y_test)
        score_out = reg_all.score(X_test, y_test)

    return score_out, rmse, reg_all

def RidgeRegres(X, y, train_size, randomstate):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import numpy as np
    from sklearn.model_selection import cross_val_score

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_size, random_state = randomstate)
    ridge = Ridge(alpha=0.1, normalize=True)

    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)

    #get RMSE
    rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))

    if train_size == 0.3:
        # Compute 5-fold cross-validation scores: cv_scores
        cv_scores = cross_val_score(ridge, X_train, y_train, cv=5)
        score_out = cv_scores
    else:
        score_out = ridge.score(X_test, y_test)
    return score_out, rmse, ridge

def LassoRegres(X, y, train_size, randomstate):
    """ Deze functie bekijkt hoe goed een lasso regressie functie past op de data
    """

    from sklearn.linear_model import Lasso
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import numpy as np
    from sklearn.model_selection import cross_val_score

    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = train_size, random_state = randomstate)
    lasso = Lasso(alpha=0.1, normalize=True)

    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)
    #scores = lasso.score(X_test, y_test)

    # get RMSE
    rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))

    if train_size == 0.3:
        # Compute 5-fold cross-validation scores: cv_scores
        cv_scores = cross_val_score(lasso, X_train, y_train, cv=5)
        score_out = cv_scores
    else:
        score_out = lasso.score(X_test, y_test)

    return score_out, rmse, lasso

def WeightAver(X, y, train_size, randomstate):
    from sklearn.model_selection import train_test_split
    import numpy as np
    from sklearn.metrics import mean_squared_error

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=randomstate)

    #initialize possible weights
    weights_one = np.linspace(0, 1, 100)
    weights_two = 1 - weights_one

    X_train_means = np.mean(X_train, axis=0)
    y_train_mean = np.mean(y_train, axis=0)

    # calculate all possible new means, with all possible weights
    new_mean = (X_train_means[0] * weights_one + X_train_means[1] * weights_two)
    # get the difference between the new means and the combined target warmth mean
    difference = list(abs(y_train_mean - new_mean))
    # retrieve the actual weights used
    weights = [weights_one[difference.index(min(difference))], weights_two[difference.index(min(difference))]]

    WeightAver_pred = np.sum(weights * X_test, axis=1)
    # print(WeightAver_pred)
    rmse = mean_squared_error(y_test, WeightAver_pred)

    mean_new = new_mean[difference.index(min(difference))]
    return WeightAver_pred, rmse, weights

def SelectModel(df, Key, count, performance, train_size, randomstate):
    """ This selects the best model from the options used below and
    prints the % correctly classified and the used model
    """

    import StringSplit
    import numpy as np

    #preprocess data
    label, label_title = StringSplit.StringSplit(Key)
    # impute nan with mean
    for n in range(1,4):
        mean = np.mean(df[label[n]], axis=0)
        df[label[n]] = df[label[n]].replace(to_replace= np.nan, value= mean)

    coef_names =[label[2], label[3]]
    X = np.array([df[label[2]].dropna(), df[label[3]].dropna()])
    y = df[label[1]].dropna().values
    X = X.T
    Regres_scores, Regres_rmse, regress = RegressionFit(X, y, train_size, randomstate)
    Ridge_scores, Ridge_rmse, ridge = RidgeRegres(X, y, train_size, randomstate)
    Lasso_scores, Lasso_rmse, lasso = LassoRegres(X, y, train_size, randomstate)
    # WeightAver_scores, WeightAver_rmse, weights = WeightAver(X, y)

    # print("model results")
    # print("\n")
    # print("5-Fold CV Regress Score: {}".format(Regres_scores))
    # print("Average of CV: {}".format(np.mean(Regres_scores)))
    # print("RMSE : {}".format(Regres_rmse))
    # print("\n")
    # print("5-Fold CV Ridge Scores: {}".format(Ridge_scores))
    # print("Average of CV: {}".format(np.mean(Ridge_scores)))
    # print("RMSE : {}".format(Ridge_rmse))
    # print("\n")
    # print("5-Fold CV Lasso Scores: {}".format(Lasso_scores))
    # print("Average of CV: {}".format(np.mean(Lasso_scores)))
    # print("RMSE : {}".format(Lasso_rmse))

    # import pdb
    # print(Key)
    # print(Regres_scores)
    # print(Ridge_scores)
    # print(Lasso_scores)
    # import pdb
    # pdb.set_trace()

    vecScores = [np.mean(Regres_scores), np.mean(Ridge_scores), np.mean(Lasso_scores)]
    # print(vecScores)
    vecRMSE = [Regres_rmse, Ridge_rmse, Lasso_rmse]
    # print(vecRMSE)
    modelsNames = ['Regress', 'Ridge', 'Lasso']
    modelsNames1 = ['Regress']
    import pandas as pd
    models = pd.DataFrame([regress], index=modelsNames1)
    count[:,0][vecScores.index(max(vecScores))] += 1
    count[:,1][vecRMSE.index(min(vecRMSE))] += 1

    R_sqr_best = abs(np.mean(vecScores[vecScores.index(max(vecScores))]))
    if R_sqr_best < 0.3:
        performance[0][Key] = R_sqr_best
    elif R_sqr_best < 0.6 and R_sqr_best > 0.3:
        performance[1][Key] = R_sqr_best
    elif R_sqr_best > 0.6 and R_sqr_best < 0.7:
        performance[2][Key] = R_sqr_best
    elif R_sqr_best > 0.7:
        performance[3][Key] = R_sqr_best

    return count, modelsNames, performance, models, vecScores,coef_names