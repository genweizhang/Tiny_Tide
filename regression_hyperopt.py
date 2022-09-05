import numpy as np
from sklearn import linear_model, neural_network, gaussian_process, svm, ensemble, metrics, neighbors
from sklearn.preprocessing import StandardScaler

from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, Matern, ConstantKernel, RBF, RationalQuadratic, ExpSineSquared

from scipy import stats
import xgboost

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

class Regression:
    def __init__(self, model_type = 'Linear'):
        self.__model_type = model_type
        
        self._model_types = {
            'Linear': self._linreg,
            'Ridge': self._ridge,
            'Lasso': self._lasso,
#             'TheilSen': self._theilsen,
            'SGD': self._sgd,
            'MLP': self._mlp,
            'GP-Matern': self._gp_matern,
            'GP-RBF': self._gp_rbf,
            'SVR': self._svr,
            'RandomForest': self._randomforest,
            'XGBoost': self._xgboost,
            'GradientBoosting': self._gradboost,
            'KNN': self._knn,
        }
    
        self._model_params = {}
    
    def _knn(self, X, y):
        self.model = BayesSearchCV(
            neighbors.KNeighborsRegressor(),
            {
                'weights': Categorical(['uniform', 'distance']),
                'leaf_size': Integer(10, 100, prior='log-uniform'),
                'n_neighbors': Integer(2, 20, prior='log-uniform'),
                'algorithm': Categorical(['auto', 'ball_tree', 'kd_tree', 'brute']),
                'p': Integer(1, 5),
            },
            n_iter=10,
            random_state=0,
            cv=3,
            n_points=5,
            n_jobs=-1,

        )
        self.model.fit(X, y)
    
    def _gradboost(self, X, y):
        self.model = BayesSearchCV(
            ensemble.GradientBoostingRegressor(),
            {
                'loss': Categorical(['ls', 'lad', 'huber', 'quantile']),
                'learning_rate': Real(1e-2, 1, prior='log-uniform'),
                'n_estimators': Integer(10, 1000, prior='log-uniform'),
                'criterion': Categorical(['friedman_mse', 'mse', 'mae',]),
                'max_depth': Integer(1, 10),
            },
            n_iter=10,
            random_state=0,
            cv=3,
            n_points=5,
            n_jobs=-1,

        )
        self.model.fit(X, y)
        
    def _linreg(self, X, y):
        self.model = BayesSearchCV(
            linear_model.LinearRegression(),
            {
                'fit_intercept': Categorical([True, False]),
                'normalize': Categorical([True, False]),
            },
            n_iter=10,
            random_state=0,
            cv=3,
            n_points=5,
            n_jobs=-1,

        )
        self.model.fit(X, y)
                
    def _ridge(self, X, y):
        self.model = BayesSearchCV(
            linear_model.Ridge(),
            {
                'fit_intercept': Categorical([True, False]),
                'normalize': Categorical([True, False]),
                'alpha': Real(1e-3, 1e+3, 'log-uniform'),
                'solver': Categorical(['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']),
            },
            n_iter=10,
            random_state=0,
            cv=3,
            n_points=5,
            n_jobs=-1,

        )
        self.model.fit(X, y)
        
    def _lasso(self, X, y):
        self.model = BayesSearchCV(
            linear_model.Lasso(),
            {
                'fit_intercept': Categorical([True, False]),
                'normalize': Categorical([True, False]),
                'alpha': Real(1e-3, 1e+3, 'log-uniform'),
                'precompute': Categorical([True, False]),
                'selection': Categorical(['cyclic', 'random']),
            },
            n_iter=10,
            random_state=0,
            cv=3,
            n_points=5,
            n_jobs=-1,

        )
        self.model.fit(X, y)
            
    def _theilsen(self, X, y):
        self.model = BayesSearchCV(
            linear_model.TheilSenRegressor(),
            {
                'fit_intercept': Categorical([True, False]),
                'max_subpopulation': Integer(1e+1, 1e+4, 'log-uniform'),
            },
            n_iter=10,
            random_state=0,
            cv=3,
            n_points=5,
            n_jobs=-1,

        )
        self.model = linear_model.TheilSenRegressor()
        self.model.fit(X, y)
        
    def _sgd(self, X, y):
        self.model = BayesSearchCV(
            linear_model.SGDRegressor(),
            {
                'loss': Categorical(['squared_loss', 'epsilon_insensitive', 'huber', 'squared_epsilon_insensitive']),
                'penalty': Categorical(['l1', 'l2', 'elasticnet']),
                'alpha': Real(1e-3, 1e+3, 'log-uniform'),
                'l1_ratio': Real(1e-3, 1e-1, prior='log-uniform'),
                'fit_intercept': Categorical([True, False]),
                'learning_rate': Categorical(['invscaling', 'constant', 'optimal', 'adaptive']),
                'epsilon': Real(1e-3, 1e+3, 'log-uniform'),
                'eta0': Real(1e-2, 1e+1, prior='log-uniform'),
                'power_t': Real(1e-2, 1e+1, prior='log-uniform'),
                'average': Categorical([True, False]),
            },
            n_iter=10,
            random_state=0,
            cv=3,
            n_points=5,
            n_jobs=-1,

        )
        self.model.fit(X, y)
        
    def _mlp(self, X, y):
        self.model = BayesSearchCV(
            neural_network.MLPRegressor(),
            {
                'activation': Categorical(['identity', 'logistic', 'tanh', 'relu']),
                'solver': Categorical(['lbfgs', 'sgd', 'adam']),
                'beta_1': Real(1e-2, 1, 'log-uniform'),
                'beta_2': Real(1e-2, 1, 'log-uniform'),
                'alpha': Real(1e-3, 1, 'log-uniform'),
                'epsilon': Real(1e-9, 1e-6, 'log-uniform'),
                'power_t': Real(1e-2, 1e+1, prior='log-uniform'),
                'learning_rate_init': Real(1e-3, 1, prior='log-uniform'),
                'learning_rate': Categorical(['invscaling', 'constant', 'adaptive']),
            },
            n_iter=10,
            random_state=0,
            cv=3,
            n_points=5,
            n_jobs=-1,
        )
        self.model.fit(X, y)
        
    def _gp_rbf(self, X, y):
        kernel = gaussian_process.kernels.RBF()
        self.model = BayesSearchCV(
            gaussian_process.GaussianProcessRegressor(kernel=kernel),
            {
                'alpha': Real(1e-11, 1e-6, 'log-uniform'),
                'n_restarts_optimizer': Integer(0, 10),
            },
            n_iter=10,
            random_state=0,
            cv=3,
            n_points=5,
            n_jobs=-1,
        )
        self.model.fit(X, y)
        
    def _gp_matern(self, X, y):
        kernel = gaussian_process.kernels.Matern()
        self.model = BayesSearchCV(
            gaussian_process.GaussianProcessRegressor(kernel=kernel),
            {
                'alpha': Real(1e-11, 1e-6, 'log-uniform'),
                'n_restarts_optimizer': Integer(0, 10),
            },
            n_iter=10,
            random_state=0,
            cv=3,
            n_points=5,
            n_jobs=-1,
        )
        self.model.fit(X, y)
    
    def _svr(self, X, y):
        self.model = BayesSearchCV(
            svm.SVR(cache_size=5000),
            {
                'kernel': Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
                'degree': Integer(1, 6),
                'gamma': Real(1e-6, 1e+1, 'log-uniform'),
                'C': Real(1e-2, 1e+1, 'log-uniform'),
                'epsilon': Real(1e-3, 1e+1, 'log-uniform'),
                'shrinking': Categorical([True, False]),
            },
            n_iter=10,
            random_state=0,
            cv=3,
            n_points=5,
            n_jobs=-1,

        )
        self.model.fit(X, y)
        
    def _randomforest(self, X, y):
        self.model = BayesSearchCV(
            ensemble.RandomForestRegressor(),
            {
                'criterion': Categorical(['mse', 'mae']),
                'n_estimators': Integer(10, 1000, prior='log-uniform'),
                'max_depth': Integer(1, 10),
            },
            n_iter=10,
            random_state=0,
            cv=3,
            n_points=5,
            n_jobs=-1,
        )
        self.model.fit(X, y)
        
    def _xgboost(self, X, y):
        self.model = BayesSearchCV(
            xgboost.XGBRegressor(),
            {
                'gamma': Real(1e-6, 1e+1, 'log-uniform'),
                'eta': Real(1e-3, 1, prior='log-uniform'),
                'max_depth': Integer(1, 10),
                'tree_method': Categorical(['auto', 'exact', 'approx', 'hist']),
                'alpha': Real(1e-3, 1e+3, 'log-uniform'),
                'lambda': Real(1e-3, 1e+3, 'log-uniform'),
                'sketch_eps': Real(1e-3, 1, 'log-uniform'),
            },
            n_iter=10,
            random_state=0,
            cv=3,
            n_points=5,
            n_jobs=-1,
        )
        self.model.fit(X, y)
                
    def train(self, X, y, model_type = None): # Implement hparams optimization
        if model_type is None:
            model_type = self.__model_type
        
        self.model_function = self._model_types[model_type]
        self.model_function(X, y)
        
    def predict(self, X): # Implement scaling function
        return self.model.predict(X)
    
    def model_sweep(self, X, y, X_test, y_test, scaling_function = None):
        self._model_dict = {}
        self._r2 = []
        self._pearson = []
        self._mse = []
        
        for key in self._model_types.keys():
            print(key)
            try:
                self.train(X, y, model_type=key)
                predicted_scores = self.predict(X_test)

                if scaling_function is not None:
                    y_true = scaling_function.inverse_transform(y_test)
                    y_pred = scaling_function.inverse_transform(predicted_scores)
                else:
                    y_true = y_test
                    y_pred = predicted_scores

                self._model_dict[key] = self.model

                self._r2 += [metrics.r2_score(y_true, y_pred)]
                self._mse += [metrics.mean_squared_error(y_true, y_pred)]
                self._pearson += [stats.pearsonr(y_true.reshape(-1), y_pred.reshape(-1))[0]]
            except:
                pass