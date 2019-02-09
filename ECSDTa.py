__author__ = 'amir'

from sklearn.base import BaseEstimator
import sklearn.ensemble

from costcla.metrics import costs
from costcla.models import cost_tree
import costcla.models.regression as regression

from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement
import numpy as np

inducers = {"Bagging": 0, "Pasting": 1, "RF": 2, "RP": 3}
combinators = {"MV": 0, "CSWV": 1, "CSS": 2}


class ECSDT(object):
    def __init__(self, combinator, inducer, num_estimators, samples, max_features, pruned):
        self.num_estimators = num_estimators
        self.max_samples = samples
        self.max_features = max_features
        self.pruned = pruned
        if inducer in inducers.keys():
            self.inducer = inducers[inducer]
        # The default inducer is Bagging
        else:
            self.inducer = inducers["Bagging"]

        if combinator in combinators.keys():
            self.combinator = combinators[combinator]
        # The default combinator is majority voting
        else:
            self.combinator = combinators["MV"]

        self.classes_names = None
        self.f_staking = None
        self.num_classes = 0
        self.models = []
        self.features_drawn = []
        self.left_samples = []
        self.savings = []

    def fit(self, X, y, cost_mat):
        self.classes_names = np.unique(y, return_inverse=True)[0]
        self.num_classes = len(self.classes_names)
        bootstrap = True
        bootstrap_features = False

        # Bagging
        if self.inducer == 0:
            bootstrap = True
            bootstrap_features = False
        # Pasting
        if self.inducer == 1:
            bootstrap = False
            bootstrap_features = False
        # RF
        if self.inducer == 2:
            bootstrap = True
            bootstrap_features = False
        # RP
        if self.inducer == 3:
            bootstrap = True
            bootstrap_features = True

        # Step 1: Create the set of base classifiers
        for i in range(self.num_estimators):
            S, features, target, costs, s_oob, target_oob, costs_oob = self.sample(X, y, cost_mat, bootstrap,
                                                                                   bootstrap_features)
            classifier = cost_tree.CostSensitiveDecisionTreeClassifier(max_features=self.max_features,
                                                                       pruned=self.pruned)
            classifier.fit(S, target, costs)

            self.models.append(classifier)
            self.features_drawn.append(features)

            self.left_samples.append(s_oob)
            classifier_predict = classifier.predict(s_oob)
            self.savings.append(self.saving(classifier_predict, target_oob, costs_oob))

        if self.combinator == 2:  # in case of stacking combinator, build logistic reggresion model to estimate B
            self.f_staking = regression.CostSensitiveLogisticRegression(fit_intercept=False,max_iter=70)
            self.f_staking.fit(self.create_stacking_matrix(X), y, cost_mat)

        return self

    def create_stacking_matrix(self, X):
        n_samples = X.shape[0]
        valid_estimators = np.nonzero(self.savings)[0]
        n_valid_estimators = valid_estimators.shape[0]
        X_stacking = np.zeros((n_samples, n_valid_estimators))
        for estimator in range(valid_estimators):
            X_stacking[:, estimator] = self.models[valid_estimators[estimator]].predict(
                X[:, self.features_drawn[valid_estimators[estimator]]])
        return X_stacking

    # Step 2: Combine the different base classifiers
    def predict(self, X, cost_mat):
        predictions = np.zeros((X.shape[0], self.num_classes))
        # MV
        if self.combinator == 0:
            for model, features in zip(self.models, self.features_drawn):
                model_predictions = model.predict(X[:, features])
                for i in range(X.shape[0]):
                    predictions[i, int(model_predictions[i])] += 1
            return self.classes_names.take(np.argmax(predictions, axis=1), axis=0)

        # CSWV
        if self.combinator == 1:
            for model, features, weight in zip(self.models, self.features_drawn, self.savings):
                model_predictions = model.predict(X[:, features])
                for i in range(X.shape[0]):
                    predictions[i, int(model_predictions[i])] += 1 * weight
            return self.classes_names.take(np.argmax(predictions, axis=1), axis=0)
        # CSS
        if self.combinator == 2:
            return self.f_staking.predict(self.create_stacking_matrix(X))

    def sample(self, X, y, cost_mat, bootstrap, bootstrap_features):
        random = check_random_state(seed=None)
        n_samples, n_features = X.shape
        if bootstrap_features:
            selected_features = random.randint(0, n_features, self.max_features)
        else:
            selected_features = sample_without_replacement(n_features, self.max_features, random_state=random)

        if bootstrap:
            indices = random.randint(0, n_samples, self.max_samples)
        else:
            indices = sample_without_replacement(n_samples, self.max_samples, random_state=random)
        oob_indices = [x for x in range(n_samples) if x not in indices]

        S = (X[indices])[:, selected_features]
        return S, selected_features, y[indices], cost_mat[indices, :], (X[oob_indices])[:, selected_features], y[
            oob_indices], cost_mat[oob_indices, :]

    def saving(self, predictions, y, costs_mat):
        return max(0, costs.savings_score(predictions, y, costs_mat))


def saving(predictions, y, costs_mat):
    return max(0, costs.savings_score(predictions, y, costs_mat))

    # def MV(self,X,models):
#
#
# def fit_bagging(self,X,y,cost_mat):
#     bootstrap=True
#     bootstrap_features=False
#
#
#
#
#
#
# def fit_Pasting(self,X,y,cost_mat):
#     bootstrap=False
#     bootstrap_features=False
#
# def fit_RF(self,X,y,cost_mat):
#     bootstrap=True
#     bootstrap_features=False
#
# def fit_RP(self,X,y,cost_mat):
#     bootstrap=False
#     bootstrap_features=False


# def	fit(self):
# def	predict(self):
# def	get_params(self):
# def	set_params(self):
