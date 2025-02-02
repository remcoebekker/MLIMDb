import pandas as pd


class DatasetCollector:
    PRINCIPAL_CATEGORIES = ["director", "writer", "actor", "actress", "producer"]

    def read_preprocessed_dataset(self, update: bool):
        if update:
            # We create a new dataset...

            # First we read in the data available for download on the IMDb website
            print("We are reading in the basic imdb files...")
            dataset = self.read_in_imdb_files()
            # We now join in the principals (such as directors, writers, actors, and actresses)
            print("We are reading in the principals data set...", len(dataset), dataset.info())
            dataset = self.read_in_imdb_principals(dataset)
            # The genres feature contains up tot 3 genres that are associated with a film, we turn the genres
            # into boolean variables
            # First, we add a column to the data frame for each individual genre
            print("We are adding the genre columns to the data set...", len(dataset), dataset.info())
            dataset = self.add_genre_to_separate_columns(dataset)
            # Next up, we add a derived experience levels for the principals
            print("We are adding the experience levels to the data set...", len(dataset), dataset.info())
            dataset = self.add_experience_levels(dataset)
            # Then we load the budget feature available from a TMDb data set
            print("We are reading in the tmdb data set to get the budget feature...", len(dataset), dataset.info())
            dataset = self.add_budget_feature_from_tmdb(dataset)
            # We store the preprocessed data set for easy retrieval
            print("We are saving the data set...", len(dataset), dataset.info())
            dataset.to_csv("dataset.csv", index=False)

        # We read in an already stored dataset...
        dataset = pd.read_csv("dataset.csv")

        return dataset

    def add_budget_feature_from_tmdb(self, dataset):
        # We read in the 5000 movies file
        tmdb = pd.read_csv("tmdb_5000_movies.csv", usecols=["budget", "original_title", "release_date"])
        # Drop all rows that have missing values since we need all three features
        tmdb = tmdb.dropna(axis=1, how="all")
        # And then turn the budget into a float
        tmdb["budget"] = tmdb["budget"].astype(float)
        # We extract the release year out of the release_date
        tmdb["release_year"] = pd.to_datetime(tmdb["release_date"], format="%Y-%m-%d").dt.year
        # We drop the release date as we don't need it anymore
        tmdb = tmdb.drop(columns="release_date")
        # And strip any leading spaces from the title field
        tmdb["original_title"] = tmdb["original_title"].str.strip()
        # We left outer join the tmdb dataset with the imdb data set on the combination of title and year
        # This will help us in determining which combination is not unique and could lead to wrong mappings
        tmdb_joined = pd.merge(tmdb, dataset,
                               left_on=["original_title", "release_year"],
                               right_on=["primaryTitle", "startYear"], how="left")
        # We are interested in finding those combinations that occur more than once
        counts = tmdb_joined[["original_title", "release_year"]].value_counts()
        counts = counts.reset_index()
        duplicates = counts[counts["count"] > 1]
        # We filter out the combinations of title and year that cause mapping issues
        tmdb = pd.merge(tmdb, duplicates, on=["original_title", "release_year"], how="left")
        tmdb = tmdb[tmdb["count"].isna()]
        tmdb = tmdb.drop(columns="count")

        dataset = pd.merge(dataset, tmdb, left_on=["primaryTitle", "startYear"],
                           right_on=["original_title", "release_year"], how="left")

        dataset = dataset.drop(columns=["original_title", "release_year"])
        return dataset

    def read_in_imdb_files(self):
        # Read in the title basics file
        imdb = pd.read_csv("title basics.txt", sep="\t", na_values="\\N")
        # Filter for movies and drop unnecessary columns
        imdb = imdb[imdb["titleType"] == "movie"].drop(columns=["endYear", "titleType"])
        # Convert columns to appropriate data types
        imdb = imdb.astype({
            "startYear": "Int64",
            "runtimeMinutes": "Float64",
            "isAdult": "boolean"
        })
        # Read in the title ratings file and merge with the imdb dataframe
        imdb_rating = pd.read_csv("title ratings.txt", sep="\t", na_values="\\N")
        imdb = imdb.merge(imdb_rating, on="tconst", how="inner")
        # Convert averageRating to Float64 and drop numVotes column
        imdb["averageRating"] = imdb["averageRating"].astype("Float64")
        imdb = imdb.drop(columns=["numVotes"])
        # Filter out rows with missing values in specific columns
        imdb = imdb.dropna(subset=["startYear", "runtimeMinutes", "isAdult", "genres"])

        return imdb

    def read_in_imdb_principals(self, imdb: pd.DataFrame):
        # First we read in the principals file
        principals = pd.read_csv("title principals.txt", sep="\t", na_values="\\N",
                                 usecols=["tconst" , "nconst", "category"])
        # We merge it with the basic data file
        imdb_principals = pd.merge(imdb, principals, on="tconst", how="left")
        # We create a pivot table so that the categories are turned into columns with the first principal
        pivot = imdb_principals.pivot_table(index="tconst", columns="category", values="nconst", aggfunc="first")
        # We only keep these 5 categories
        pivot = pivot.loc[:,self.PRINCIPAL_CATEGORIES]
        # And joint it with the data set
        imdb = imdb.set_index("tconst").join(pivot).reset_index()

        # We link the names to the princpal ID's
        imdb_names = pd.read_csv("title name basics.txt", sep="\t", na_values="\\N")
        imdb_names["nconst"] = imdb_names["nconst"].astype(str)
        imdb = self.link_name_to_principal("director", imdb, imdb_names)
        imdb = self.link_name_to_principal("writer", imdb, imdb_names)
        imdb = self.link_name_to_principal("actor", imdb, imdb_names)
        imdb = self.link_name_to_principal("actress", imdb, imdb_names)
        imdb = self.link_name_to_principal("producer", imdb, imdb_names)

        return imdb

    def link_name_to_principal(self, category, imdb, imdb_names):
        # We merge the imdb file and the names data set on the category
        imdb_plus_names = pd.merge(imdb, imdb_names, left_on=category, right_on="nconst", how="inner")
        # We append the name of the person in parentheses behind the id
        imdb_plus_names[category] = imdb_plus_names.apply(lambda row:
                                                          (row["primaryName"] if pd.notnull(
                                                              row["primaryName"]) else "") +
                                                          " (" +
                                                          (row[category] if pd.notnull(row[category]) else ""),
                                                          axis=1) + ")"
        # We drop the columns we are not interested in
        imdb_plus_names = imdb_plus_names.drop(
            columns=["primaryName", "nconst", "birthYear", "deathYear", "primaryProfession",
                     "knownForTitles"])
        print(imdb_plus_names.head())
        return imdb_plus_names

    def process_row(self, row, set_of_genres):
        genres = row["genres"]
        parts = genres.split(",")
        parts = [part.strip() for part in parts]
        [set_of_genres.add(part) for part in parts]

    def get_list_of_genres(self, imdb: pd.DataFrame):
        set_of_genres = set()
        # We loop through the data set and create a set of unique genres
        imdb.apply(lambda row: self.process_row(row, set_of_genres), axis=1)
        return set_of_genres

    def add_genre_to_separate_columns(selfs, imdb: pd.DataFrame):
        # We split up the genres column into separate columns
        genres_split = imdb["genres"].str.get_dummies(sep=",")
        # We add a prefix
        genres_split = genres_split.add_prefix("genre_")
        # And link it to the imdb data set
        imdb = pd.concat([imdb, genres_split], axis=1)
        # We don't need the old column anymore
        imdb = imdb.drop(columns="genres")
        return imdb

    def add_experience_levels(self, dataset: pd.DataFrame):
        # For all principal categories we count the number of films the person has performed in this role
        for category in self.PRINCIPAL_CATEGORIES:
            grouped_dataset = dataset.groupby(category).size().reset_index(name=category + "_experience")
            dataset = pd.merge(dataset, grouped_dataset, on=category)
        return dataset