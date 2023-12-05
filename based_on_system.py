import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from warnings import simplefilter
from sklearn.impute import SimpleImputer
from itertools import cycle

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


def calculate_global_score(row):
    # calculate the global score as the average of user_score and critic_score
    user_score = row['User_Score']
    critic_score = row['Critic_Score']
    if not pd.isnull(user_score) and not pd.isnull(critic_score):
        return (user_score + critic_score) / 2
    else:
        # if either score is missing, return NaN
        return np.nan


def cluster():
    # load the game dataset from the CSV file
    file_path = 'Video_Games.csv'
    game_dataset = pd.read_csv(file_path)

    # select relevant columns for clustering
    selected_columns = ['Name', 'Genre', 'User_Score', 'Critic_Score', 'Global_Sales', 'Platform']
    game_subset = game_dataset[selected_columns].copy()

    # convert 'tbd' values to NaN and convert columns to numeric
    game_subset['User_Score'] = pd.to_numeric(game_subset['User_Score'], errors='coerce')
    game_subset['Critic_Score'] = pd.to_numeric(game_subset['Critic_Score'], errors='coerce')

    # calculate the average score
    game_subset['Global_Score'] = (game_subset['User_Score'] + game_subset['Critic_Score']) / 2

    # drop rows with missing values in the selected columns
    game_subset = game_subset.dropna(subset=['User_Score', 'Critic_Score'])

    # standardize the scores for clustering
    scaler = StandardScaler()
    game_subset[['User_Score', 'Critic_Score', 'Global_Score', 'Global_Sales']] = scaler.fit_transform(
        game_subset[['User_Score', 'Critic_Score', 'Global_Score', 'Global_Sales']])

    # get unique genres
    unique_genres = game_subset['Genre'].unique()

    # create a color cycle for plotting
    colors = cycle('bgrcmyk')

    # initialize a figure for plotting
    fig, ax = plt.subplots(figsize=(15, 10))

    # initialize a dictionary to track assigned clusters for each genre
    assigned_clusters = {}

    # load the user input dataset
    user_input_file = 'user_input.csv'
    user_input_data = pd.read_csv(user_input_file)

    # check if the necessary columns are present in the user input data
    required_columns = ['Name', 'Genre', 'Global_Sales', 'User_Score', 'Platform']
    if not all(column in user_input_data.columns for column in required_columns):
        raise ValueError(f"Columns {required_columns} not found in the user input data.")

    # calculate missing features for user input data
    if 'Global_Score' not in user_input_data.columns:
        user_input_data['Global_Score'] = user_input_data.apply(calculate_global_score, axis=1)

    # impute missing values in the user input data
    imputer = SimpleImputer(strategy='mean')  # you can also use 'median' or 'constant'
    user_input_data[['User_Score', 'Critic_Score']] = imputer.fit_transform(
        user_input_data[['User_Score', 'Critic_Score']])

    # perform one-hot encoding for the 'Platform' column
    preprocessor = ColumnTransformer(
        transformers=[('platform', OneHotEncoder(), ['Platform'])],
        remainder='passthrough'
    )

    # use a pipeline to apply one-hot encoding and clustering
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('kmeans', KMeans(n_clusters=1, random_state=42))])

    # fit and predict clusters for the user input data
    user_input_data['cluster'] = pipeline.fit_predict(user_input_data[['User_Score', 'Global_Sales', 'Platform']])

    # plot user input data alone
    for genre in unique_genres:
        genre_subset = user_input_data[user_input_data['Genre'] == genre].copy()
        ax.scatter(
            genre_subset['User_Score'],
            genre_subset['Global_Sales'],
            label=f'{genre} (User Input)',
            marker='X',
            s=100,
            color='black'  # you can customize the color for user input data
        )

    # set plot labels and title
    ax.set_title('Game Clusters Based on Global Scores by Genre (User Input Only)')
    ax.set_xlabel('User Score')
    ax.set_ylabel('Global Sales')

    # create a legend for user input data
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    plt.show()

    # plot user input data along with the cluster of all data
    fig, ax = plt.subplots(figsize=(15, 10))

    # plot user input data
    for genre in unique_genres:
        genre_subset = user_input_data[user_input_data['Genre'] == genre].copy()
        ax.scatter(
            genre_subset['User_Score'],
            genre_subset['Global_Sales'],
            label=f'{genre} (User Input)',
            marker='X',
            s=100,
            color='black'  # you can customize the color for user input data
        )

    # iterate over each genre and perform kmeans clustering
    for genre, color in zip(unique_genres, colors):
        # filter games of the current genre
        genre_subset = game_subset[game_subset['Genre'] == genre].copy()

        # check if there are enough games for clustering
        if len(genre_subset) > 1:
            # check if the genre has already been assigned a cluster
            if genre in assigned_clusters:
                cluster_index = assigned_clusters[genre]
            else:
                # perform kmeans clustering for the current genre
                kmeans = KMeans(n_clusters=1, random_state=42)
                genre_subset['cluster'] = kmeans.fit_predict(genre_subset[['Global_Score', 'Global_Sales']])

                # assign a unique color to the cluster
                cluster_color = next(colors)

                # scatter plot for each cluster
                cluster_index = 0  # since there's only one cluster per genre
            ax.scatter(
                genre_subset['Global_Score'],
                genre_subset['Global_Sales'],
                label=f'{genre}',
                color=cluster_color,
            )

            # update the assigned cluster for the genre
            assigned_clusters[genre] = cluster_index

    # set plot labels and title
    ax.set_title('Game Clusters Based on Global Scores by Genre (With User Input)')
    ax.set_xlabel('Global Score')
    ax.set_ylabel('Global Sales')
    ax.legend()

    # use nearest neighbors to find the three closest points to the user input
    nn = NearestNeighbors(n_neighbors=3)
    nn.fit(game_subset[['User_Score', 'Global_Sales']])
    _, indices = nn.kneighbors(user_input_data[['User_Score', 'Global_Sales']])

    # calculate the average score of the three closest points
    average_score = game_subset.iloc[indices.flatten()]['Global_Score'].mean()

    # recommend a game based on the average score
    recommended_game = game_subset.loc[
        game_subset['Global_Score'].sub(average_score).abs().idxmin(), 'Name']

    print(f"\nBased on your input, we recommend the game: {recommended_game}")

    plt.show()


# put the starting logic here
cluster()
