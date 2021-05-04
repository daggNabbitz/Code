import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
import sklearn_pandas as spd
from sklearn.preprocessing import LabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from insretl.Storebrand import data_util
from sklearn.model_selection import cross_val_score

class HousingData():

    def create_pipelines(self,df):
        ord_cols,num_cols,obj_cols = self.create_columns(df)
        num_pipeline = Pipeline([("num", SimpleImputer(strategy="median")), ("std", StandardScaler())])
        obj_pipe_ord = Pipeline([("obj", SimpleImputer(strategy="most_frequent")), ("encoder", OrdinalEncoder())])
        obj_pipe_hot = Pipeline([("obj", SimpleImputer(strategy="most_frequent")), ("hot", OneHotEncoder())])
        combined_pipe = ColumnTransformer(
            [("num", num_pipeline, num_cols), ("ordinal", obj_pipe_ord, ord_cols), ("hot", obj_pipe_hot, obj_cols)])
        return combined_pipe

    def create_columns(self,df):
        ord_cols = ['ExterQual', 'ExterCond',
                    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                    'HeatingQC', 'kitchenqual', 'FireplaceQu', 'GarageQual',
                    'GarageCond']
        ord_cols = pd.Series(ord_cols).apply(lambda x: x.lower())
        num_cols = df.columns[df.dtypes != object]
        obj_cols = df.columns[(df.dtypes == object) & (~df.columns.isin(ord_cols))]
        return ord_cols,num_cols,obj_cols

    def initialize_algorithms(self):
        svr = SVR()
        rnd_forest = RandomForestRegressor()
        xgb_regressor = GradientBoostingRegressor()
        return svr,rnd_forest,xgb_regressor

    def mse(self,true_y,pred_y):
        mse = ((true_y - pred_y)**2).sum()/len(true_y)
        smse = np.sqrt(mse)
        return smse

    def pick_best_algorithm(self,alg1, alg2, alg3, X_train, y_train):
        scores = []
        lowest = np.inf
        algorithms = [alg1, alg2, alg3]
        for alg in algorithms:
            score = abs(cross_val_score(estimator=alg, cv=3, X=X_train, y=y_train, scoring="neg_root_mean_squared_error"))
            scores.append((alg,score.mean()))
            print(f"average error for {str(alg)} is {score.mean()}")
        for s in scores:
            if s[1] < lowest:
                lowest = s[1]
                best_alg = s[0]
        print(f"The best algorithm is {str(best_alg)}")
        return best_alg







