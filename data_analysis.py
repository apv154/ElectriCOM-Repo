def function_creator(input_dir, output_dir):
      from sklearn import preprocessing
      import pandas as pd
      import matplotlib.pyplot as plt
      from sklearn.model_selection import train_test_split
      from sklearn.ensemble import RandomForestRegressor
      from sklearn.ensemble import AdaBoostRegressor
      from sklearn.metrics import mean_squared_error, make_scorer, r2_score
      from sklearn.model_selection import GridSearchCV

      elect = pd.read_csv(input_dir)
      elect = elect.drop(columns=['time',
                                  'Global_active_power',
                                  'Global_reactive_power',
                                  'Voltage,Global_intensity',])
      elect = elect.dropna()
      X = elect.loc[:, elect.columns != 'Sub_metering_3']
      y = elect.loc[:, elect.columns == 'Sub_metering_1'] + 
          elect.loc[:, elect.columns == 'Sub_metering_2'] +
          elect.loc[:, elect.columns == 'Sub_metering_3']
      X = X.drop(columns=['Sub_metering_1', 'Sub_metering_2'])
      standardized_X = preprocessing.scale(X)
      xtrain, xtest, ytrain, ytest=train_test_split(standardize_X, y, test_size=0.25)
      # Choose the type of classifier.
      abreg = AdaBoostRegressor()
      # Choose some parameter combinations to try
      params = {
       'n_estimators': [50, 100],
       'learning_rate' : [0.01, 0.05, 0.1, 0.5],
       'loss' : ['linear', 'square', 'exponential']
       }
      score = make_scorer(mean_squared_error)

      gridsearch=GridSearchCV(abreg, params, scoring=score, cv=5, return_train_score=True)
      gridsearch.fit(standardize_X, y)

      # Choose the type of classifier.
      abreg = AdaBoostRegressor()
      # Choose some parameter combinations to try
      params = {
       'n_estimators': [50, 100],
       'learning_rate' : [0.01, 0.05, 0.1, 0.5],
       'loss' : ['linear', 'square', 'exponential']
       }
      score = make_scorer(mean_squared_error)

      gridsearch=GridSearchCV(abreg, params, scoring=score, cv=5, return_train_score=True)
      gridsearch.fit(standardize_X, y)

      best_estim=gridsearch.best_estimator_

      best_estim.fit(xtrain,ytrain)
      ytr_pred=best_estim.predict(xtrain)
      mse = mean_squared_error(ytr_pred,ytrain)
      r2 = r2_score(ytr_pred,ytrain)

      print('The MSE error for the train is: ', mse)
      print('The correlation for the train is: ', r2)

      ypred=best_estim.predict(xtest)
      mse = mean_squared_error(ytest, ypred)
      r2 = r2_score(ytest, ypred)

      print('The MSE error for the test is: ', mse)
      print('The correlation for the test is: ', r2)

      plt.scatter(ytest, ypred)
      plt.show()
