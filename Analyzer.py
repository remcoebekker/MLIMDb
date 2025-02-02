from scipy.stats import f_oneway, ttest_rel
import numpy as np


class Analyzer:

    def run_anova_on_results(self, rmse_results, model_names, with_budget_only):
        # We turn the data frame into a list so we can feed it to the ANOVA test
        rmse_results_list = []
        for i in range(len(model_names)):
            rmse_results_list.append(rmse_results[model_names[i]])
        f_statistic, p_value = f_oneway(*rmse_results_list)

        return f_statistic, p_value

    def run_ttests_on_results(self, rmse_results, model_names, model_rmse_names):
        # We run a ttest on  all model rmse_results against the baseline
        # We assume that the first column is the baseline
        ttest_results = []
        for i in range(1, len(model_names)):
            t_statistic, p_value = ttest_rel(rmse_results[model_rmse_names[i]],
                                             rmse_results[model_rmse_names[0]])
            ttest_results.append(["Baseline vs. " + model_names[i], np.round(t_statistic, 5), np.round(p_value, 5)])
        # We print the results
        print(ttest_results)
        return ttest_results
