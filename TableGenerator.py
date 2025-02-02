import pandas as pd
import dataframe_image as dfi
import matplotlib_inline


def run():
    ttest_results = pd.read_csv("ttest results.csv")
    generate_ttest_results_table(ttest_results, False)
    ttest_results = pd.read_csv("with budget only ttest results.csv")
    generate_ttest_results_table(ttest_results, True)

    rmse_results = pd.read_csv("rmse results.csv")
    generate_rmse_results_table(rmse_results, False)
    rmse_results = pd.read_csv("with budget only rmse results.csv")
    generate_rmse_results_table(rmse_results, True)


def determine_background_color(cell_value):
    # Default background color is white
    if isinstance(cell_value, float):
        color = "white"
    else:
        # The cell is textual (row and column headers), so background is off-white
        color = "#FAF9F6"
    return f"background-color: {color}"


def generate_ttest_results_table(ttest_results_df, with_budget_only):
    # We put the t-test results in a dataframe
    ttest_results_df["p-value"] = ttest_results_df["p-value"].apply(lambda x: "<0.0001" if x < 0.0001 else x)
    # And create a styler object to layout the dataframe as a table we can save as an image
    styler = ttest_results_df.style.format(lambda s: ("%.4f" % s).replace("-0", "-").lstrip("0") if isinstance(s, float) else s)
    styler.map(determine_background_color).hide()
    styler.set_table_styles(
        [{
            'selector': 'th',
            'props': [
                ('background-color', '#40826D'),
                ('color', 'white')],
            "table_conversion": "chrome"
        }])
    # Export the formatted table as an image
    if with_budget_only:
        file_name = "with budget only ttest results.png"
    else:
        file_name = "ttest results.png"

    dfi.export(styler, file_name)


def generate_rmse_results_table(rmse_results, with_budget_only):
    # And create a styler object to layout the dataframe as a table we can save as an image
    rmse_results.loc["Average"] = rmse_results.mean()
    rmse_results["Run"] = rmse_results["Run"].astype(int).astype(str)
    rmse_results.loc["Average", "Run"] = "Average"
    styler = rmse_results.style.format(
        lambda s: ("%.4f" % s).replace("-0", "-").lstrip("0") if isinstance(s, float) else s)
    styler.map(determine_background_color).hide()
    styler.set_table_styles(
        [{
            'selector': 'th',
            'props': [
                ('background-color', '#40826D'),
                ('color', 'white')],
            "table_conversion": "chrome"}])
    # Export the formatted table as aimage
    if with_budget_only:
        file_name = "with budget only RMSE results.png"
    else:
        file_name = "RMSE results.png"

    dfi.export(styler, file_name)


# If this module is run, it will call the run function
if __name__ == "__main__":
    run()
