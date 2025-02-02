import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter


class DataExplorer:

    FEATURES_PAIR_PLOT = ["averageRating", "startYear", "runtimeMinutes", "isAdult",
                             "director_experience", "writer_experience", "actor_experience",
                             "actress_experience", "producer_experience"]
    FEATURES_PAIR_PLOT_WITH_BUDGET = FEATURES_PAIR_PLOT + ["budget"]

    def execute_basic_data_analysis(self, df: pd.DataFrame, with_budget_only: bool):
        # We execute basic data analysis
        # We extract the categorical columns and the numeric columns
        category_columns = df.dtypes[df.dtypes == "object"]
        non_category_columns = df.dtypes[df.dtypes != "object"]
        # What are the dimensions of the data set?
        self.separate("Dimensions")
        self.dimensions(df)
        # We inspect the data
        self.separate("Data inspection")
        self.inspect_data(df)
        # Are there duplicates in the data?
        self.separate("Duplicates")
        self.duplicates(df)
        # How well balanced is the data?
        self.separate("Balance")
        self.balance(df, category_columns)
        # Are there values missing?
        self.separate("Missing values")
        self.missing_values(df)
        # How skewed is the data?
        self.separate("Skewness and kurtosis")
        self.skewness(df)

    def separate(self, header: str):
        print("-" * 100)
        print(header)

    def dimensions(self, df):
        print("Glance at the contents:")
        print(df.head())
        print("Dimensions:", df.shape)
        print(df.info())

    def inspect_data(self, df):
        df2 = pd.DataFrame({"Data Type": df.dtypes,
                            "No of Levels": df.apply(lambda x: x.nunique(), axis=0),
                            "Levels": df.apply(lambda x: str(x.unique()), axis=0)})
        print("Types and levels:")
        print(df2)
        print("Description")
        print(df.describe(include="all"))

    def duplicates(self, df):
        print("Number of duplicates:")
        print(df.duplicated().sum())
        print("Duplicates:")
        print(df[df.duplicated()])

    def balance(self, df, category_columns):
        for i in range(0, len(category_columns)):
            print(df[category_columns.keys()[i]].value_counts())

    def missing_values(self, df):
        print(df.isnull().sum(axis=0))

    def skewness(self, df):
        print(df.skew(numeric_only=True))
        print(df.kurt(numeric_only=True))

    def count_plot(self, df, category_columns):
        for i in range(0, len(category_columns)):
            print(category_columns.keys()[i])
            plt.title("Count plot:" + category_columns.keys()[i])
            sns.countplot(data=df, x=category_columns.keys()[i], hue=category_columns.keys()[i])
            plt.show()

    def correlation_heatmap(self, df):
        sns.heatmap(df.corr(numeric_only=True), cmap="Blues", annot=True)
        df.head()
        plt.show()

    def distribution(df, category_columns, non_category_columns):
        if len(category_columns) > 0:
            for col in non_category_columns.keys():
                sns.FacetGrid(df, hue=category_columns.index[0], height=5).map(sns.histplot, col, kde=True).add_legend()
            plt.show()

    def univariate_analysis(self, df, category_columns, non_category_columns):
        if len(category_columns) > 0:
            for col in non_category_columns.keys():
                sns.boxplot(df, y=col, x=category_columns.index[0], hue=category_columns.index[0])
                plt.show()

    def find_outliers_IQR(self, df):
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        IQR = q3 - q1
        outliers = df[((df < (q1 - 1.5 * IQR)) | (df > (q3 + 1.5 * IQR)))]
        return outliers

    def generate_pair_plot(self, dataset, with_budget_only):
        fig = plt.figure(figsize=(16, 8))

        if with_budget_only:
            features = self.FEATURES_PAIR_PLOT_WITH_BUDGET
        else:
            features = self.FEATURES_PAIR_PLOT

        sns.pairplot(data=dataset.loc[:, features], height=3)
        plt.tight_layout()
        if with_budget_only:
            file_name = "with budget only pair plot.png"
        else:
            file_name = "pair plot.png"
        plt.savefig(file_name, bbox_inches="tight", dpi=300)

    def generate_correlation_heatmap(self, dataset, with_budget_only):
        fig = plt.figure(figsize=(16, 8))
        if with_budget_only:
            dataset = dataset[dataset["budget"] > 0]
            features = self.FEATURES_PAIR_PLOT_WITH_BUDGET
        else:
            features = self.FEATURES_PAIR_PLOT
        sns.heatmap(np.round(dataset.loc[:, features].corr(), 2), cmap="Blues", annot=True)
        plt.tight_layout()
        if with_budget_only:
            file_name = "with budget only correlation heatmap.png"
        else:
            file_name = "correlation heatmap.png"
        plt.savefig(file_name, bbox_inches="tight", dpi=300)

    def generate_correlation_with_genres_heatmap(self, dataset, with_budget_only):
        l = []
        for i in dataset.columns:
            if i.startswith("genre_"):
                l.append(i)
        l.append("averageRating")
        l = sorted(l)

        fig = plt.figure(figsize=(16, 8))
        sns.heatmap(np.round(dataset[l].corr(), 2), cmap="Blues", annot=True)

        if with_budget_only:
            file_name = "with budget only correlation with genres.png"
        else:
            file_name = "correlation with genres.png"

        plt.savefig(file_name, bbox_inches="tight", dpi=300)

    def generate_histplots(self, dataset, with_budget_only):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        sns.histplot(data=dataset, x="startYear", kde=True, ax=axes[0])
        axes[0].set_xlabel("Year of film release")
        sns.histplot(data=dataset, x="averageRating", kde=True, ax=axes[1])
        axes[1].set_xlabel("Average weighted vote average")
        plt.tight_layout()

        if with_budget_only:
            file_name = "with budget only histplots.png"
        else:
            file_name = "histplots.png"

        plt.savefig(file_name, bbox_inches="tight", dpi=300)

    def generate_boxplots(self, dataset, with_budget_only):
        if with_budget_only:
            fig, axes = plt.subplots(4, 2, figsize=(12, 10))
        else:
            fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        sns.boxplot(data=dataset, x="runtimeMinutes", ax=axes[0, 0])
        axes[0, 0].set_xlabel("Runtime of the film in minutes")
        sns.boxplot(data=dataset, x="director_experience", ax=axes[0, 1])
        axes[0, 1].set_xlabel("Director experience (number of films as lead director)")
        sns.boxplot(data=dataset, x="writer_experience", ax=axes[1, 0])
        axes[1, 0].set_xlabel("Writer experience (number of films as lead writer)")
        sns.boxplot(data=dataset, x="actor_experience", ax=axes[1, 1])
        axes[1, 1].set_xlabel("Actor experience (number of films as lead actor)")
        sns.boxplot(data=dataset, x="actress_experience", ax=axes[2, 0])
        axes[2, 0].set_xlabel("Actress experience (number of films as lead actress)")
        sns.boxplot(data=dataset, x="producer_experience", ax=axes[2, 1])
        axes[2, 1].set_xlabel("Producer experience (number of films as lead producer)")
        if with_budget_only:
            file_name = "with budget only boxplots.png"
            sns.boxplot(x=dataset.loc[dataset["budget"] > 0, "budget"] / 1000000, ax=axes[3, 0])
            axes[3, 0].set_xlabel("Budget (where available, in million $)")
            axes[3, 0].xaxis.set_major_formatter(ScalarFormatter())
            axes[3, 0].ticklabel_format(style='plain', axis='x')
        else:
            file_name = "boxplots.png"

        plt.tight_layout()
        plt.savefig(file_name, bbox_inches="tight", dpi=300)

    def generate_average_rating_throughout_years_plot(self, dataset, with_budget_only):
        fig = plt.figure(figsize=(16, 8))
        dataset_avg_rating = dataset.groupby("startYear").mean(numeric_only=True)
        sns.lineplot(x=dataset_avg_rating.index, y=dataset_avg_rating["averageRating"])
        plt.xlabel("Year of film release")
        plt.ylabel("Average of the yearly average rating")

        if with_budget_only:
            file_name = "with budget only average_rating_throughout_years.png"
        else:
            file_name = "average_rating_throughout_years.png"

        plt.savefig(file_name, bbox_inches="tight", dpi=300)
