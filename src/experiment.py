import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from data_reader import load_communities, load_insurance, load_lsac, load_german, load_heart, load_synthetic
from density_balance import DensityBalance
from metrics import Metrics


class Experiment():

    def __init__(self, data="Community", regressor="Linear", balance="None", density_model="Neighbor"):
        datasets = {"Community": load_communities, "Insurance": load_insurance,
                    "LSAC": load_lsac, "German": load_german,
                    "Heart": load_heart, "Synthetic": load_synthetic}
        regressors = {"SVR": SVR(kernel="linear"), "Linear": LinearRegression(positive=True),
                      "Logistic": LogisticRegression(), "DT": DecisionTreeRegressor(max_depth=8),
                      "RF": RandomForestClassifier()}
        self.X, self.y, self.protected = datasets[data]()
        self.regressor = regressors[regressor]
        self.balance = balance
        self.density_model = density_model
        self.preprocessor = None

    def run(self):
        self.train_test_split()
        self.preprocess(self.X_train)
        X_train = self.preprocessor.transform(self.X_train)
        X_test = self.preprocessor.transform(self.X_test)
        y_train = self.y_train

        y_test = self.y_test
        if self.balance == "None":
            sample_weight = None
        else:
            sample_weight = DensityBalance(model=self.density_model).weight(self.X_train[self.protected].to_numpy(),
                                                                            np.transpose([self.y_train]),
                                                                            treatment=self.balance)
        self.regressor.fit(X_train, y_train, sample_weight)
        result = self.test(X_test, y_test)
        return result

    def preprocess(self, X):
        numerical_columns_selector = selector(dtype_exclude=object)
        categorical_columns_selector = selector(dtype_include=object)

        numerical_columns = numerical_columns_selector(X)
        categorical_columns = categorical_columns_selector(X)

        categorical_preprocessor = OneHotEncoder(handle_unknown='ignore')
        numerical_preprocessor = StandardScaler()
        self.preprocessor = ColumnTransformer([
            ('OneHotEncoder', categorical_preprocessor, categorical_columns),
            ('StandardScaler', numerical_preprocessor, numerical_columns)])
        self.preprocessor.fit(X)

    def test(self, X, y):
        y_pred = self.regressor.predict(X)
        m = Metrics(y, y_pred)
        result = {
            "MSE": m.mse(),
            "R2": m.r2()
            # "Accuracy": m.accuracy(),
            # "F1": m.f1(),
        }
        for key in self.protected:
            result["AOD_" + str(key)] = m.AOD(np.array(self.X_test[key]))
            result["AODc_" + str(key)] = m.AODc(np.array(self.X_test[key]))
        return result

    def train_test_split(self, train_ratio=0.5):
        n = len(self.y)
        train_ind = list(np.random.choice(range(n), int(n * train_ratio), replace=False))
        test_ind = list(set(range(n)) - set(train_ind))
        self.X_train = self.X.iloc[train_ind]
        self.X_test = self.X.iloc[test_ind]
        self.y_train = self.y[train_ind]
        self.y_test = self.y[test_ind]
