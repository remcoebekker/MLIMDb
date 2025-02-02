import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, export_text
from BaselineEstimator import BaselineEstimator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import shap
import matplotlib.pyplot as plt


class ExperimentExecutor:
    RANDOM_STATE = 43
    LOOPS_FOR_EXECUTION = 20
    LOOPS_FOR_HYPERPARAMETER_OPTIMIZATION = 1
    TEST_SET_SIZE_FOR_EXECUTION = 0.5
    MODEL_NAMES = ["Baseline", "Polynomial regression", "Decision tree regression", "SVR"]
    MODEL_RMSE_NAMES = ["Baseline RMSE", "Polynomial regression RMSE", "Decision tree regression RMSE", "SVR RMSE"]

    def __init__(self, dataset: pd.DataFrame, test_set_size: float):
        self.X_data = dataset.drop(columns="averageRating")
        self.y_data = dataset["averageRating"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_data,
                                                                                self.y_data,
                                                                                test_size=test_set_size,
                                                                                random_state=self.RANDOM_STATE)
        self.models = []
        self.models.append(BaselineEstimator())
        self.models.append(Pipeline([("poly", PolynomialFeatures()), ("linear", LinearRegression())]))
        self.models.append(DecisionTreeRegressor())
        self.models.append(SVR())
        self.parameter_ranges = []
        self.parameter_ranges.append({})
        self.parameter_ranges.append({"poly__degree": [1, 2, 3], "poly__interaction_only": [False, True]})
        self.parameter_ranges.append({"max_leaf_nodes": [10, 50, 100, 250, 500]})
        self.parameter_ranges.append({"C": [0.1, 1.0, 10], "epsilon": [0.1, 0.5, 1.0]})
        self.best_models = [{} for _ in self.models]

    def determine_optimal_hyperparameters(self):
        # We look for the optimal hyperparameters in a number of runs. In each run, we determine the optimal
        # hyperparameters for each model. For each model we count how often a parameter_combination
        # is the best combination.
        parameter_combination_count = [{} for _ in self.models]
        for i in range(0, self.LOOPS_FOR_HYPERPARAMETER_OPTIMIZATION):
            print(i)
            # We set up the folds in accordance with Diettrich (5x2), and vary the split using
            # a random state
            folds = KFold(n_splits=5, shuffle=True, random_state=i + self.RANDOM_STATE)
            # We loop through the models
            for j in range(len(self.models)):
                print("j", j)
                model = self.models[j]
                # The baseline estimator does not need to be optimized, so we skip it
                if type(model) != type(BaselineEstimator):
                    # Search the optimal parameters, scoring is RMSE
                    grid = GridSearchCV(model, self.parameter_ranges[j], cv=folds,
                                        scoring="neg_root_mean_squared_error")
                    # We normalize the X_data first before fitting
                    scaler = StandardScaler()
                    grid.fit(scaler.fit_transform(self.X_train), self.y_train)
                    print(grid.best_params_)
                    best_model = grid.best_estimator_
                    best_params = grid.best_params_
                else:
                    best_model = model
                    best_params = {}
                # We count how often this parameter combination is the best combination
                param_dict = parameter_combination_count[j]
                current_dict = param_dict.setdefault(str(best_params), [0, best_model])
                current_dict[0] = current_dict[0] + 1
                param_dict[str(best_params)] = current_dict

        # We loop through the counters to see what the best model was...
        print(parameter_combination_count)
        for c in range(len(parameter_combination_count)):
            current_dict = parameter_combination_count[c]
            max_number = 0
            for item in current_dict.values():
                if item[0] > max_number:
                    self.best_models[c] = item[1]
        # And we return the best models...
        return self.best_models

    def execute_experiments(self):
        # We execute the experiment a number of times so that we get a number of test results.
        # The results per run are stored in this data frame
        rmse_results = pd.DataFrame(columns=["Run"] + self.MODEL_RMSE_NAMES)

        for i in range(0, self.LOOPS_FOR_EXECUTION):
            print(i)
            # On each run, we split up the data set aside for testing in training and test differently
            # In this way, we get a new test score on each run
            X_train, X_test, y_train, y_test = train_test_split(self.X_test, self.y_test,
                                                                test_size=self.TEST_SET_SIZE_FOR_EXECUTION,
                                                                random_state=i + self.RANDOM_STATE)
            # We store the RMSE's for this run in a list
            rmse = []
            # We loop through the models
            for j in range(len(self.best_models)):
                print("j", j)
                # The best parameters have already been set
                model = self.best_models[j]
                # We train the model with normalized X_data
                scaler = StandardScaler()
                model.fit(scaler.fit_transform(X_train), y_train)
                # And then, we let the model predict the average rating for the test data
                y_pred = model.predict(scaler.fit_transform(X_test))
                # We determine the RMSE and add it to the list
                rmse.append(root_mean_squared_error(y_test, y_pred))
            # We store the results in the data frame
            print(rmse)
            rmse_results.loc[len(rmse_results)] = [i + 1, *rmse]

        return rmse_results

    def print_best_models(self):
        tree_rules = export_text(self.best_models[1], feature_names=self.X_test.columns)
        print(self.best_models[3].get_params())
        print(tree_rules)

    def create_shapley_plot(self, best_model, with_budget_only):
        # Creating a Shapley plot is very resource intensive, so a subset of the data is used to
        # create it
        dataset_for_shapley = pd.concat([self.X_test, self.y_test], axis=1)
        if with_budget_only:
            dataset_for_shapley = dataset_for_shapley.sample(frac=0.5)
        else:
            dataset_for_shapley = dataset_for_shapley.sample(frac=0.1)
        # And split it up in X_data and y_data
        X_data = dataset_for_shapley.drop(columns="averageRating")
        y_data = dataset_for_shapley["averageRating"]
        # And then split it up in training and test data
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,
                                                            test_size=self.TEST_SET_SIZE_FOR_EXECUTION,
                                                            random_state=self.RANDOM_STATE)
        # In order to prevent some issues with a boolean, we turn the isAdult feature into an integer
        X_train["isAdult"] = X_train["isAdult"].astype(int)
        X_test["isAdult"] = X_test["isAdult"].astype(int)
        print(X_test["isAdult"], X_test["isAdult"].describe())
        # And fit the model
        best_model.fit(X_train, y_train)
        # And setup the Shap explainer
        explainer = shap.Explainer(best_model.predict, X_train)
        shap_values = explainer(X_test)
        # And generate the Shap summary plot and save the figure
        fig = plt.figure(figsize=(16, 8))
        shap.summary_plot(shap_values, X_test)

        if with_budget_only:
            file_name = "with budget only Shapley summary plot.png"
        else:
            file_name = "Shapley summary plot.png"

        plt.savefig(file_name, bbox_inches="tight", dpi=300)

