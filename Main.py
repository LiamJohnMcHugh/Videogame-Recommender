import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from itertools import cycle
import numpy as np
import warnings
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


def cluster():

    # load the game dataset from csv file
    file_path = 'Video_Games.csv'


    game_dataset = pd.read_csv(file_path)

    # select relevant columns for clustering
    selected_columns = ['Name', 'Genre', 'User_Score', 'Critic_Score', 'Global_Sales']
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
    ax.set_title('Game Clusters Based on Global Scores by Genre')
    ax.set_xlabel('Global Score')
    ax.set_ylabel('Global Sales')
    ax.legend()
    plt.show()

#Put the starting logic here
cluster()