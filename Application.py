import numpy as np
import pandas as pd

from DatasetCollector import DatasetCollector
from DataExplorer import DataExplorer
from ExperimentExecutor import ExperimentExecutor
from Analyzer import Analyzer

TEST_SIZE = 0.7


def run(create_dataset: bool, explore_data: bool, with_budget_only: bool):
    # First we collect the dataset...depending on the boolean a new pre-processed data set is created from the base
    # files or a previously preprocessed data set is read in.
    dataset = collect_data(create_dataset)
    # The following statistics are relevant for the report
    print("After pre-processing the data set contains", len(dataset),
          "rows that are completely filled apart from budget.")
    print("After pre-processing the mean average rating is ", np.round(np.mean(dataset["averageRating"]), 2), ".",
          sep="")
    print("After pre-processing ", np.round((len(dataset[dataset["budget"] > 0]) / len(dataset)) * 100, 2),
          "% of the rows have a budget > 0.", sep="")
    print("After pre-processing ", np.round((len(dataset[dataset["isAdult"]]) / len(dataset)) * 100, 2),
          "% of the rows have isAdult == True.", sep="")
    # We only use the numerical and boolean columns for our regression experiments
    numerical_dataset = dataset.select_dtypes(include=["number", "boolean"])
    print(numerical_dataset.describe(include="all").T)
    if with_budget_only:
        numerical_dataset = numerical_dataset[numerical_dataset["budget"] > 0]
        # And then apply the experiments on the partial data set including budget
    else:
        # Since only for a small portion of the films budget is present, we drop the column
        numerical_dataset = numerical_dataset.drop(columns=["budget"])
    if explore_data:
        # Next up is the data exploration...
        execute_data_exploration(numerical_dataset, with_budget_only)
    # We are ready to start the experiments
    rmse_results, executor = execute_experiments(numerical_dataset, TEST_SIZE, with_budget_only)
    print(rmse_results)
    # Now that we have the results, we can apply statistics to determine whether there are significant differences
    # between the different models and with regards to the baseline
    analyze_results(rmse_results, executor, with_budget_only)


def collect_data(create_dataset: bool):
    # We setup the data collector
    collector = DatasetCollector()
    # And retrieve our preprocessed data set
    dataset = collector.read_preprocessed_dataset(create_dataset)
    return dataset


def execute_data_exploration(dataset, with_budget_only):
    data_explorer = DataExplorer()
    # First we run the basic data exploration steps...
    data_explorer.execute_basic_data_analysis(dataset, with_budget_only)
    # Next we generate some graphs that will be part of the report
    data_explorer.generate_boxplots(dataset, with_budget_only)
    data_explorer.generate_histplots(dataset, with_budget_only)
    data_explorer.generate_correlation_heatmap(dataset, with_budget_only)
    data_explorer.generate_average_rating_throughout_years_plot(dataset, with_budget_only)
    data_explorer.generate_correlation_with_genres_heatmap(dataset, with_budget_only)
    data_explorer.generate_pair_plot(dataset, with_budget_only)


def execute_experiments(dataset, test_set_size, with_budget_only):
    # We create en experiment executor that will do the experiments for us
    executor = ExperimentExecutor(dataset, test_set_size)
    # First we determine the optimal hyperparameters
    best_models = executor.determine_optimal_hyperparameters()
    print(best_models)
    # Next up is executing the experiments
    results = executor.execute_experiments()
    # We save the RMSE results as csv file, so we can generate a png later on. In colab
    # we cannot generate the png directly with dfi-image.
    if with_budget_only:
        file_name = "with budget only RMSE results.csv"
    else:
        file_name = "RMSE results.csv"
    results.to_csv(file_name, index=False)

    return results, executor


def analyze_results(rmse_results, executor: ExperimentExecutor, with_budget_only):
    # We create the analyzer that will analyze the RMSE's for us
    analzyer = Analyzer()
    # We first run the one_way ANOVA
    f_statistic, p_value = analzyer.run_anova_on_results(rmse_results,
                                                         ExperimentExecutor.MODEL_RMSE_NAMES,
                                                         with_budget_only)
    if with_budget_only:
        print(f"Results partial data set with budget: F-statistic={np.round(f_statistic, 5)}, "
              f"P-value={np.round(p_value, 5)}")
    else:
        print(f"Results full data set without budget: F-statistic={np.round(f_statistic, 5)}, "
              f"P-value={np.round(p_value, 5)}")

    # Then, the ttests to look for individual differences
    ttest_results = analzyer.run_ttests_on_results(rmse_results,
                                                   ExperimentExecutor.MODEL_NAMES,
                                                   ExperimentExecutor.MODEL_RMSE_NAMES)
    # We save the ttest results as csv file, so we can generate a png later on. In colab
    # we cannot generate the png directly with dfi-image.
    if with_budget_only:
        file_name = "with budget only ttest results.csv"
    else:
        file_name = "ttest results.csv"
    ttest_results_df = pd.DataFrame(data=ttest_results, columns=["Comparison", "t-statistic", "p-value"])
    ttest_results_df.to_csv(file_name, index=False)
    # We determine which model outperformed the others in terms of ttest statistic
    min_ttest_statistic = 0
    min_ttest_index = None
    for i in range(len(ttest_results)):
        if ttest_results[i][1] < min_ttest_statistic:
            min_ttest_statistic = ttest_results[i][1]
            min_ttest_index = i
    if min_ttest_index is not None:
        print("The best performing model is: ",
              ttest_results[min_ttest_index],
              # We need to offset with 1 because the ttest tesults compare against the baseline
              executor.best_models[min_ttest_index + 1])
        # For this model, we use Shapley to determine the most important features
        executor.create_shapley_plot(executor.best_models[min_ttest_index + 1], with_budget_only)


# If this module is run, it will call the run function
if __name__ == "__main__":
    # The application is kicked off with 3 boolean parameters:
    # - Whether the preprocessed data set needs to be created or that the already preprocessed data set is used
    # - Whether data exploration needs to be performed
    # - Whether the experiments are only run on films that have a budget associated with them
    run(False, False, True)
