from costcla.metrics import costs
from costcla.models import cost_tree,bagging
import costcla.models.regression as regression
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement
import numpy as np


class ECSDT(object):

    def _inducer_info(self,i):
        switcher = {
            # Bagging
            0: (True, False),
            # Pasting
            1: (False, False),
            # RF
            2: (True, False),
            # RP
            3: (True, True)
        }
        return switcher.get(i, (True, False))

    def _data_sampling(self, X, y, cost_mat):
        random = check_random_state(seed=None)
        n_samples, n_features = X.shape
        if self.bootstrap_features:
            selected_features = random.randint(0, n_features, self.num_of_features)
        else:
            selected_features = sample_without_replacement(n_features, self.num_of_features, random_state=random)

        if self.bootstrap:
            indexes = random.randint(0, n_samples, self.num_of_samples)
        else:
            indexes = sample_without_replacement(n_samples, self.num_of_samples, random_state=random)

        oob_indexes = [x for x in range(n_samples) if x not in indexes]
        S = (X[indexes])[:, selected_features]
        return S, selected_features, y[indexes], cost_mat[indexes, :], (X[oob_indexes])[:, selected_features], y[
            oob_indexes], cost_mat[oob_indexes, :]

    def _create_predict_matrix(self, X):
        n_samples = X.shape[0]
        valid_estimators = np.nonzero(self.alphas)[0]
        n_valid_estimators = valid_estimators.shape[0]
        X_stacking = np.zeros((n_samples, n_valid_estimators))
        for estimator in range(n_valid_estimators):
            X_stacking[:, estimator] = self.models[valid_estimators[estimator]].predict(
                X[:, self.features_drawn[valid_estimators[estimator]]])
        return X_stacking

    def _create_fitting_matrix(self, X,y,cost_mat):
        random = check_random_state(seed=None)
        indexes = sample_without_replacement(X.shape[0], self.num_of_samples, random_state=random)
        dummy_x = X[indexes, :]
        dummy_y = y[indexes]
        dummy_cost_mat = cost_mat[indexes, :]
        n_samples = dummy_x.shape[0]
        valid_estimators = np.nonzero(self.alphas)[0]
        n_valid_estimators = valid_estimators.shape[0]
        X_stacking = np.zeros((n_samples, n_valid_estimators))
        for estimator in range(n_valid_estimators):
            X_stacking[:, estimator] = self.models[valid_estimators[estimator]].predict(
                dummy_x[:, self.features_drawn[valid_estimators[estimator]]])
        return X_stacking,dummy_y,dummy_cost_mat


    def __init__(self, T, Ne, Nf, combiner='Bagging', inducer='MV'):
        self.inducers = {"Bagging": 0, "Pasting": 1, "RandomForest": 2, "RandomPatches": 3}
        self.combiners = {"MV": 0, "CSWV": 1, "CSS": 2}
        self.num_of_iterations = T
        self.num_of_samples = Ne
        self.num_of_features = Nf
        self.models = []
        self.alphas = []
        self.s_oobs = []
        self.features_drawn = []
        self.classes_names = None
        self.num_classes = 0

        if inducer in self.inducers.keys():
            self.inducer = self.inducers[inducer]
            # The default inducer is Bagging
        else:
            self.inducer = self.inducers["Bagging"]

        if combiner in self.combiners.keys():
            self.combiner = self.combiners[combiner]
        # The default combinator is majority voting
        else:
            self.combinator = self.combiners["MV"]

        self.bootstrap, self.bootstrap_features = self._inducer_info(self.inducer)

    def fit(self, X, y, cost_mat):
        self.classes_names = np.unique(y)
        self.num_classes = len(self.classes_names)

        #step 1: create the set of base classifires
        for i in range(self.num_of_iterations):
            S, features, target, mat_costs, s_oob, target_oob, mat_costs_oob = self._data_sampling(X, y, cost_mat)
            csdt_clf = cost_tree.CostSensitiveDecisionTreeClassifier(max_depth=6,max_features=self.num_of_features,pruned=True)
            csdt_clf.fit(S, target, mat_costs)

            self.models.append(csdt_clf)
            self.features_drawn.append(features)
            self.s_oobs.append(s_oob)
            self.alphas.append(costs.savings_score(target_oob,csdt_clf.predict(s_oob),mat_costs_oob))

        #lies for alphas!
        pos = any(n > 0 for n in self.alphas)
        if pos:
            self.alphas = [0.0 if n < 0 else n for n in self.alphas]
        else:
            real_max_alpha = max(self.alphas)
            max_alpha = real_max_alpha
            if real_max_alpha==0:
                max_alpha=0.001
            self.alphas = [0.0 if n != real_max_alpha else abs(max_alpha) for n in self.alphas]
        temp = np.array(self.alphas)
        alpha_sum = sum(temp)
        if alpha_sum>0:
            temp = temp/sum(temp)
        self.alphas = temp.tolist()

        if self.combiner == 2:
            self.staking_m = regression.CostSensitiveLogisticRegression()
            x_stacking, y_stacking, cost_mat_stacking = self._create_fitting_matrix(X,y,cost_mat)
            if x_stacking.shape[1] > 1:
                self.staking_m.fit(x_stacking, y_stacking, cost_mat_stacking)
            else:
                self.combiner=0
        return self

    #step2: combine the different base classifiers
    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.num_classes))
        # MV
        if self.combiner == 0:
            for model, features in zip(self.models, self.features_drawn):
                model_predictions = model.predict(X[:, features])
                for i in range(X.shape[0]):
                    predictions[i, int(model_predictions[i])] += 1
            return self.classes_names.take(np.argmax(predictions, axis=1), axis=0)
        # CSWV
        if self.combiner == 1:
            for model, features, weight in zip(self.models, self.features_drawn, self.alphas):
                model_predictions = model.predict(X[:, features])
                for i in range(X.shape[0]):
                    predictions[i, int(model_predictions[i])] += 1 * weight
            return self.classes_names.take(np.argmax(predictions, axis=1), axis=0)
        # CSS
        if self.combiner == 2:
            return self.staking_m.predict(self._create_predict_matrix(X))





