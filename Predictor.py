import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import joblib


class models:
    def __init__(self):

        self.df=pd.read_csv("seattle-weather.csv")

        self.df['weather']=LabelEncoder().fit_transform(self.df['weather'])

        self.features=["precipitation", "temp_max", "wind"]
        X=self.df[self.features]
        y=self.df.weather
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(X, y,random_state = 42)

    def model1(self):
        model = DecisionTreeRegressor(random_state=1)
        param_grid = {
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8, 16],
        'max_features': [None, 'auto', 'sqrt', 'log2']
        }
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
        grid_search.fit(self.train_X, self.train_y)
        best_model = grid_search.best_estimator_
        joblib.dump(best_model, 'decesion_tree_model.pkl')
        best_model.fit(self.train_X, self.train_y)
        pred1=best_model.predict(self.test_X)
        print("Mean Absolute Error(DecisionTree): " , mean_absolute_error(self.test_y, pred1))
        print("Score (DecesionTree): ", r2_score(self.test_y, pred1))
        plt.scatter(self.test_X['precipitation'], self.test_y, color = 'red', s = 30)
        plt.scatter(self.test_X['precipitation'], pred1, color = 'blue', s = 10)
        plt.show()
        print()

    
    def model2(self):
        model = RandomForestRegressor(random_state=1)
        param_grid = {
        'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']

        }
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
        grid_search.fit(self.train_X, self.train_y)
        best_model = grid_search.best_estimator_
        joblib.dump(best_model, 'random_forest_model.pkl')
        best_model.fit(self.train_X, self.train_y)
        pred2=best_model.predict(self.test_X)
        print("Mean Absolute Error(RandomForest): ",  mean_absolute_error(self.test_y, pred2))
        print("Score (RandomForest): ", r2_score(self.test_y, pred2))
        plt.scatter(self.test_X['precipitation'], self.test_y, color = 'red', s = 30)
        plt.scatter(self.test_X['precipitation'], pred2, color = 'blue',  s = 10)
        plt.show()
        print()
        

    def model3(self):  
        model = XGBRegressor()
        param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
        }
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
        grid_search.fit(self.train_X, self.train_y)
        best_model = grid_search.best_estimator_
        joblib.dump(best_model, 'xgb_model.pkl')
        best_model.fit(self.train_X, self.train_y)
        pred3=best_model.predict(self.test_X)
        print("Mean Absolute Error(XGB): " , mean_absolute_error(self.test_y, pred3))
        print("Score (XGB): ", r2_score(self.test_y, pred3))
        plt.scatter(self.test_X['precipitation'], self.test_y, color = 'red', s = 30)
        plt.scatter(self.test_X['precipitation'], pred3, color = 'blue', s = 10)
        plt.show()
        


reg = models()
reg.model1()
reg.model2()
reg.model3()
