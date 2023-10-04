import multiprocessing
import sys
from multiprocessing import pool
import random
import h5py
from itertools import repeat
from itertools import count
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import struct
import polars as pl
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from time import perf_counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import dask.dataframe as dd
from sklearn.metrics.pairwise import pairwise_kernels
from numba import jit, cuda
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import seaborn as sn
from sklearn.metrics import accuracy_score
import sklearn.preprocessing as pp
import scipy.sparse as sp
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import os
os.environ['PYTHONMALLOC'] = 'debug'


import warnings
warnings.filterwarnings('ignore')


class Song:

    def __init__(self, custom_distance, artist, song, preview, type, id):
        self.custom_distance = custom_distance
        self.artist = artist
        self.song_name = song
        self.preview = preview
        self.type = type
        self.id = id



class SongReccomender:

    def __init__(self, filename):
        self.df = pd.read_excel(filename)
        self.remove_categorical_Data()
        self.pca()
        self.train_and_add_categorical_columns()

    def get_song_dataframe(self):
        return self.df


    def remove_categorical_Data(self):
        self.cols = ['artist', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                     'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature', 'label']
        self.non_categorical = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness',
                                'liveness', 'valence', 'tempo', 'duration_ms']
        self.categorical = ['artist', 'key', 'mode', 'time_signature', 'label']

        # %ms = MinMaxScaler()
        self.df[self.non_categorical] = self.df[self.non_categorical].apply(
            lambda x: (x - x.min()) / (x.max() - x.min()))

    def pca(self):
        pca = PCA(6)
        self.data = self.df.drop_duplicates()
        self.data = pca.fit_transform(self.data[self.non_categorical]) #de inteles
        #cluster_plot(pd.DataFrame(self.data))
        self.data = pd.DataFrame(self.data)
        #self.cluster_plot(self.data)
        return

    def train(self, df_train):
        n = 1
        for _ in range(n):  #When you are not interested in some values returned by a function we use underscore in place of variable name . Basically it means you are not interested in how many times the loop is run till now just that it should run some specific number of times overall.
            self.km = KMeans(
                n_clusters=7, init='k-means++', #initial n_clusters = 4, iar init='random', iar apoi n_clusters = 6
                n_init=10, max_iter=1000,
                tol=1e-04, random_state=0
            )
            y_km = self.km.fit(df_train)
        #return self.km
        return y_km

    #@jit(target_backend='cuda')
    def k_mean_distance(self, center_coordinates, data_coordiantes):
        summ = 0
        #mag = 0
        for i in range(len(data_coordiantes)):
            summ += (center_coordinates[i] - data_coordiantes[i]) ** 2
            #mag += (data_coordiantes[i]) ** 2
        return (summ) ** 0.5

    @jit(target_backend='cuda')
    def computing_all_k_mean_distances(self, dummy_df, song):
        arr = []
        for i in range(len(dummy_df)):
            dist = self.k_mean_distance(dummy_df[i], song.values[0][0:6])
            #print(dist)
            arr.append(dist)

        return arr

    def custom_distance(self, current_distance, array_dist_min, array_dist_max, min_popularity, max_popularity, current_cosine_similarity, dummy_df):
        current_distance = 1 - (current_distance - array_dist_min) * 2 / (array_dist_max - array_dist_min)
        normalized_popularity = - 1 + (dummy_df[10] - min_popularity) * 2 /(max_popularity - min_popularity)
        custom_dist = (normalized_popularity + current_cosine_similarity + current_distance) / 3
        current_song = Song(custom_dist, dummy_df[7],
                    dummy_df[8], dummy_df[9],
                    dummy_df[11], dummy_df[13])
        return current_song

    @jit(target_backend='cuda')
    def computing_custom_distances(self, array_dist, dummy_df, cosine_similarities):
        array_dist_min = min(array_dist)
        array_dist_max = max(array_dist)
        min_popularity = min(dummy_df['popularity'])
        max_popularity = max(dummy_df['popularity'])
        dummy_df = dummy_df.to_numpy()
        songs = []
        for i in range(len(dummy_df)):
            current_distance = 1 - (array_dist[i] - array_dist_min) * 2 / (array_dist_max - array_dist_min)
            normalized_popularity = - 1 + (dummy_df[i][10] - min_popularity) * 2 / (max_popularity - min_popularity)
            custom_distance = (current_distance + normalized_popularity + cosine_similarities[i][len(cosine_similarities) - 1]) / 3
            current_song = Song(custom_distance, dummy_df[i][7],
                    dummy_df[i][8], dummy_df[i][9],
                    dummy_df[i][11], dummy_df[i][13])

            songs.append(current_song)
        return songs

    #@jit(target_backend='cude')
    def compute_cosine_similarity(self, lyrics_matrix):
        #return pd.DataFrame(cosine_similarity(lyrics_matrix.astype('float32')))
        normed_lyrics_matrix = pp.normalize(lyrics_matrix.tocsc(), axis = 0)
        return normed_lyrics_matrix.T * normed_lyrics_matrix

    def train_and_add_categorical_columns(self):
        self.train(self.data)
        self.data['label'] = self.km.labels_
        self.data['artist'] = self.df.artist
        self.data['name'] = self.df.name
        self.data['preview'] = self.df.preview
        self.data['popularity'] = self.df.popularity
        self.data['type'] = self.df.label
        self.data['lyrics'] = self.df.lyrics
        self.data['id'] = self.df.id

        #de refactorizat partea asta
        self.X = self.data
        self.Y = self.data['label']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.25, shuffle=False)
        knn = KNeighborsClassifier(n_neighbors=25)
        knn.fit(self.X_train.drop(['label', 'artist', 'name', 'preview', 'type', 'popularity', 'lyrics', 'id'], axis=1),
                self.y_train)
        self.y_pred = knn.predict(
            self.X_test.drop(['label', 'artist', 'name', 'preview', 'type', 'popularity', 'lyrics', 'id'], axis=1))
        self.tfidf = TfidfVectorizer(analyzer='word', stop_words='english', lowercase=True)



    @jit(target_backend='cuda')
    def parallel_fit_transform(self, lyrics):
        return self.tfidf.fit_transform(lyrics.values.astype('U'))

    #@jit(target_backend='cuda')
    def recommend_songs(self, song): #needs refactoring
        start_time = perf_counter()
        try:
            arr_df = pd.read_csv('../Data/Evaluations/custom_distances ' + song['id'][song.index.values[0]] + '.csv')
            arr_df.drop('Unnamed: 0', inplace=True, axis=1)
        except:
            print("Current time passed at this point : " + str(perf_counter() - start_time) + " seconds.")
            # moved to train_and_add_categorical_columns
            #X = self.data  # .drop(['label', 'artist', 'name', 'preview', 'type', 'popularity'],axis=1)#,'lyrics'], axis=1)
            #Y = self.data['label']
            #X_train, X_test, y_train, y_test = train_test_split(
                #X, Y, test_size=0.25, shuffle=False)  # random_state=42)
            #knn = KNeighborsClassifier(n_neighbors=25)
            #knn.fit(X_train.drop(['label', 'artist', 'name', 'preview', 'type', 'popularity', 'lyrics','id'], axis=1),
                    #y_train)
            #y_pred = knn.predict(X_test.drop(['label', 'artist', 'name', 'preview', 'type', 'popularity', 'lyrics','id'], axis=1))
            #tfidf = TfidfVectorizer(analyzer='word', stop_words='english', lowercase=True)
            #print("Current time passed at this point after k-nn and tf-idf preparation : " + str(
                #perf_counter() - start_time) + " seconds.")
            print("Song is in cluster " + str(song.label.values[0]))
            print("Song index : " + str(song.index.values[0]))
            if song.index.values[0] in self.X_train.index:
                dummy_df = self.X_train.loc[self.y_train == song.label.values[0] & (self.X_train.index != song.index.values[0])]
                print("Song is in Training")
            elif song.index.values[0] in self.X_test.index:
                dummy_df = self.X_train.loc[self.y_train == self.y_pred[song.index.values[0] - len(self.y_train)]] #& (X_test.index != song.index.values[0])]#X_test.loc[y_pred == song.label.values[0] & (X_test.index != song.index.values[0])]
                print("Song is in Testing")
            print(len(dummy_df))
            lyrics = dummy_df['lyrics']
            lyrics = lyrics.append(song['lyrics'])
            self.song = song
            print("Current time passed at this point : " + str(perf_counter() - start_time) + " seconds.")
            #pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            #lyrics_matrix = pool.starmap(self.tfidf.fit_transform, list(zip(lyrics.values.astype('U')[0:len(lyrics)])))
            lyrics_matrix = self.tfidf.fit_transform(lyrics.values.astype('U'))
            print("Current time passed at this point : " + str(perf_counter() - start_time) + " seconds.")
            cosine_similarities = pairwise_kernels(lyrics_matrix.astype('float32'), metric='cosine', n_jobs=-1)#cosine_similarity(lyrics_matrix.astype('float32'))
            print("Current time passed at this point : " + str(perf_counter() - start_time) + " seconds.")
            #pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            #array_dist = pool.starmap(self.k_mean_distance, list(zip(dummy_df.values[0:len(dummy_df.values)], repeat(song.values[0][0:6]))))
            array_dist = self.computing_all_k_mean_distances(dummy_df.to_numpy(),song)#.to_numpy(), song)

            print("Current time passed at this point after computing Euclidean distances : " + str(perf_counter() - start_time) + " seconds")
            arr = self.computing_custom_distances(array_dist, dummy_df, cosine_similarities) #min_popularity, max_popularity)#.to_numpy(), cosine_similarities, min_popularity, max_popularity)
            print("Current time passed at this point after computing custom distances: " + str(perf_counter() - start_time) + " seconds.")
            arr.sort(key = lambda x: x.custom_distance, reverse=True)
            print("Current time passed at this point after sorting: " + str(perf_counter() - start_time) + " seconds.")
            arr_df = pd.DataFrame([t.__dict__ for t in arr])
            print(song['id'][song.index.values[0]])
            arr_df.to_csv('../Data/Evaluations/custom_distances ' + song['id'][song.index.values[0]] + '.csv')
            print("Current time after csv export : " + str(perf_counter() - start_time) + " seconds.")
        return arr_df



    def song_print(self, song): #ma gandesc poate sa redenumesc numele metodei
        print('=' * 200)
        print('Artist:  ', song.artist.values[0])
        print('Song Name:   ', song.name.values[0])
        print('Type:   ', song['type'].values[0])
        print('Preview link:   ', song.preview.values[0])
        print('=' * 200)

    def print_song_reccomendation(self, song_number, number_of_recommendations):
        try:
            song = self.data.loc[[song_number]]
        except:
            raise ValueError(str(song_number) + " is out of range.")
        #ans = self.song_recommendation(song, self.data,number_of_recommendations)
        ans = self.recommend_songs(song)
        self.song_print(song)
        j = 1
        for song in ans.itertuples():#[::-1]:
            if (j > number_of_recommendations):
                break
            print('Number:  ', j)
            print('Custom Distance Value:  ', song.custom_distance)
            print('Artist:  ', song.artist)
            print('Song Name:   ', song.song_name)
            print('Type:   ', song.type)
            print('Preview link:   ', song.preview)
            print('-' * 100)
            j += 1
        return

    def get_recommendations(self, song_number_or_id):
        #self.print_song_reccomendation(song_number, number_of_recommendations)
        if isinstance(song_number_or_id, int):
            try:
                song = self.data.loc[[song_number_or_id]]
            except:
                raise ValueError(str(song_number_or_id) + " is out of range.")
        else:
            try:
                song = self.data[self.data['id'] == song_number_or_id]
            except:
                raise ValueError(str(song_number_or_id) + " doesn't exist as an id.")
            # ans = self.song_recommendation(song, self.data,number_of_recommendations)
        ans = self.recommend_songs(song)
        self.song_print(song)
        return ans

    def prepare_user_data_evaluation(self, filename):

        self.interactions = pd.read_csv(filename)
        self.interactions = self.interactions['song_id'].unique()

    def get_user_data_evaluation(self):
        return self.interactions


def compute_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

def one_song_average_f1_evaluation(song_id, user_song_data_interactions_df, recommendations, number_of_recommendations=None):
    if number_of_recommendations is None:
        number_of_recommendations = len(recommendations)

    # Filter relevant users who have listened to the given song_id
    song_listened_by_users = user_song_data_interactions_df[user_song_data_interactions_df['song_id'] == song_id]

    # Filter out the rows where the 'song_id' is the same as the target song_id
    filtered_song_data = user_song_data_interactions_df[user_song_data_interactions_df['song_id'] != song_id]

    # Calculate the total number of listened songs by each user
    total_number_of_listened_songs_by_user = filtered_song_data.groupby('user_id').size().reindex(
        filtered_song_data['user_id'].unique(),
        fill_value=0
    )

    # Convert the recommendations DataFrame to a set for faster look-up
    recommended_song_ids = set(recommendations['id'][:number_of_recommendations].values)

    # Calculate hits using set intersection and vectorized operations
    songs_listened_by_current_user = filtered_song_data[
        filtered_song_data['user_id'].isin(song_listened_by_users['user_id'])
    ]
    song_hit_mask = songs_listened_by_current_user['song_id'].isin(recommended_song_ids)
    hits_by_user = songs_listened_by_current_user[song_hit_mask].groupby('user_id').size().reindex(
        song_listened_by_users['user_id'].unique(),
        fill_value=0
    )

    # Calculate precision and recall using vectorized operations
    precision = hits_by_user / number_of_recommendations
    recall = hits_by_user / total_number_of_listened_songs_by_user.reindex(
        song_listened_by_users['user_id'].unique(),
        fill_value=0
    )

    # Handle the case when both precision and recall are 0
    zero_mask = np.logical_and(precision == 0, recall == 0)
    f1_score = compute_f1_score(precision, recall)

    # Set F1 score to 0 where both precision and recall are 0
    f1_score[zero_mask] = 0

    average = np.nanmean(f1_score)  # Compute the average F1 score, ignoring NaN values



    return average



def process_song(args):
    current_song_id,user_song_data_interactions_df, number_of_recommendations = args

    new_start_time = perf_counter()
    #recommendations = pd.read_csv("../Data/Evaluations/custom_distances " + str(current_song_id) + ".csv")

    recommendations = pd.read_csv("../Data/Evaluations for random generated songs/random_recommendations " + str(current_song_id) + ".csv")
    #print("Time required for reading generated recommendations : " + str(perf_counter()-new_start_time) + " seconds.")
    # Filter relevant users who have listened to the given song_id
    song_listened_by_users = user_song_data_interactions_df[user_song_data_interactions_df['song_id'] == current_song_id]

    #song_listened_by_users.to_csv("../Data/Evaluations/song listened by users/song " + str(current_song_id) + " listened by users.csv")


    # Filter out the rows where the 'song_id' is the same as the target song_id
    filtered_song_data = user_song_data_interactions_df[user_song_data_interactions_df['song_id'] != current_song_id]

    # Calculate the total number of listened songs by each user
    total_number_of_listened_songs_by_user = filtered_song_data.groupby('user_id').size().reindex(
        filtered_song_data['user_id'].unique(),
        fill_value=0
    )

    # Calculate hits using set intersection and vectorized operations
    songs_listened_by_current_user = filtered_song_data[
        filtered_song_data['user_id'].isin(song_listened_by_users['user_id'])
    ]
        #songs_listened_by_current_user.to_csv(
            #"../Data/Evaluations/songs listened by current users/songs listened by current users " + str(current_song_id) + ".csv")

    # Convert the recommendations DataFrame to a set for faster look-up
    recommended_song_ids = set(recommendations['id'][:number_of_recommendations].values)

    song_hit_mask = songs_listened_by_current_user['song_id'].isin(recommended_song_ids)
    hits_by_user = songs_listened_by_current_user[song_hit_mask].groupby('user_id').size().reindex(
        song_listened_by_users['user_id'].unique(),
        fill_value=0
    )

    # Calculate precision and recall using vectorized operations
    precision = hits_by_user / number_of_recommendations
    recall = hits_by_user / total_number_of_listened_songs_by_user.reindex(
        song_listened_by_users['user_id'].unique(),
        fill_value=0
    )

    # Handle the case when both precision and recall are 0
    zero_mask = (precision == 0) & (recall == 0)

    # Calculate average recall and average precision separately
    average_recall = np.nanmean(recall)
    average_precision = np.nanmean(precision)

    print("Len np mean recall : " + str(len(np.nanmean(recall))))
    print("Len np mean precision : " + str(len(np.nanmean(recall))))

    f1_score = compute_f1_score(precision, recall)

    # Set F1 score to 0 where both precision and recall are 0
    f1_score = np.where(zero_mask, 0, f1_score)

    average_f1_score = np.nanmean(f1_score)  # Compute the average F1 score, ignoring NaN values
    print("Len np mean f1 score : " + str(len(np.nanmean(f1_score))))

    #print("Current time elapsed : " + str(perf_counter() - new_start_time) + " seconds for song id" + str(current_song_id))
    return average_f1_score, average_precision, average_recall


#de modificat numele si eventual si metoda
def average_f1_score_evaluation_parallel(user_song_data_interactions_df,evaluation_songs, number_of_recommendations=None):
    if number_of_recommendations is None:
        number_of_recommendations = 28510

    print("Number of processes : " + str(multiprocessing.cpu_count()))
    # Create a pool of processes with the maximum number of available CPU cores (-1)
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Prepare the arguments for each song ID to be processed
        song_args = [(current_song_id,user_song_data_interactions_df, number_of_recommendations) for current_song_id in evaluation_songs]

        # Use the pool to map the tasks to the processes and get the results
        #one_song_average_f1_scores, one_song_average_precision_scores, one_song_average_recall_scores = pool.map(process_song, song_args)
        results = pool.map(process_song, song_args)
        one_song_average_f1_scores, one_song_average_precision_scores, one_song_average_recall_scores = zip(*results)


    average_f1_score = sum(one_song_average_f1_scores) / len(one_song_average_f1_scores)
    average_precision_score = sum(one_song_average_precision_scores) / len(one_song_average_precision_scores)
    average_recall_score = sum(one_song_average_recall_scores) / len(one_song_average_recall_scores)

    return average_f1_score, average_precision_score, average_recall_score




def find_optimal_number_of_recommendations_for_recommendations(lower_number,upper_number):

    begin = lower_number
    end = upper_number
    user_song_data_interactions_df = pd.read_csv("../Data/new_song-user_interactions.csv")
    # Create an empty list to store the best_average_f1_mid values
    best_average_f1_list = []
    #evaluation_songs is a gloabal variable for now.....
    current_start_time = perf_counter()
    max_best_average_f1 = average_f1_score_evaluation_parallel(user_song_data_interactions_df, evaluation_songs, end)
    result_dict = { "max_best_average_f1": max_best_average_f1,
    "number of recommendations": end}
    best_average_f1_list.append(result_dict)
    #print("Current time elapsed : " + str(perf_counter() - current_start_time) + " seconds for value " + str(end))
    while end - begin > 1:
        current_start_time = perf_counter()
        mid = (begin + end)//2
        best_average_f1_mid = average_f1_score_evaluation_parallel(user_song_data_interactions_df, evaluation_songs, mid)
        result_dict = {"max_best_average_f1": best_average_f1_mid,
                       "number of recommendations": mid}
        best_average_f1_list.append(result_dict)
        if best_average_f1_mid > max_best_average_f1:
            max_best_average_f1 = best_average_f1_mid
            end = mid
            current_optimal_no_recommendations = end
        else:
            #max_best_average_f1 = one_song_average_f1_evaluation(song_id, user_song_data_interactions_df,recommendations, end)
            begin = mid + 1
            current_optimal_no_recommendations = begin
        print("Current time elapsed : " + str(perf_counter() - current_start_time) + " seconds for value " + str(mid))
    print("Best average f1 score : " + str(max_best_average_f1))

    best_average_f1_df = pd.DataFrame( best_average_f1_list)
    output_file = "output_14.csv"
    best_average_f1_df.to_csv(output_file, index=False)
    return current_optimal_no_recommendations


def generate_random_recommendations(song_id, song_dataframe):
    # Filter the song_dataframe to exclude the song with the same ID as song_id
    filtered_dataframe = song_dataframe[song_dataframe['id'] != song_id]

    # Check if there are enough rows in the filtered DataFrame to generate recommendations
    if len(filtered_dataframe) < len(pd.read_csv("../Data/Evaluations/custom_distances " + str(song_id) + ".csv")):
        raise ValueError("Not enough songs in the DataFrame to generate recommendations.")

    # Randomly sample num_recommendations rows from the filtered DataFrame
    random_recommendations = filtered_dataframe.sample(n=len(pd.read_csv("../Data/Evaluations/custom_distances " + str(song_id) + ".csv")))

    return random_recommendations

def remove_some_columns():
    # Define the directory containing the CSV files
    directory = "../Data/Evaluations for random generated songs"

    # Define the columns you want to keep
    columns_to_keep = ['id', 'artist', 'name', 'label', 'preview']

    # Iterate through all files in the directory
    i = 1
    for filename in os.listdir(directory):
        start_time = perf_counter()

        if filename.endswith(".csv"):
            # Construct the full path to the CSV file
            file_path = os.path.join(directory, filename)

            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Select only the desired columns
            df = df[columns_to_keep]

            # Save the modified DataFrame back to the same file
            df.to_csv(file_path, index=False)
        print("Time required for current song : " + str(perf_counter()-start_time) + " seconds.")


if __name__ == '__main__':

    #remove_some_columns()
    start_time = perf_counter()
    sr = SongReccomender('../Data/final_data_with_lyrics_101.xlsx')
    print("Time needed for constructor : " + str(perf_counter() - start_time) + " seconds.")
    #recommendations = sr.get_recommendations(song_number_or_id="2PUKnv6IOIkq72pZ7Agqsf")
    '''
    number_of_recommendations = [100,250,500,1000,1500,3000]
    times = []
    for current_no_of_recommendations in number_of_recommendations:
        start_time = perf_counter()
        sr.print_song_reccomendation(song_number=51172, number_of_recommendations= current_no_of_recommendations)#.get_recommendations(song_number=51172, number_of_recommendations= current_no_of_recommendations) #initial song_number = 2980
        time = perf_counter() - start_time
        print("Time needed for recommendations : " + str(perf_counter() - start_time) + " seconds.")
        times.append(time)

    print(times)
    plt.plot(number_of_recommendations, times)
    plt.title("Time performance in relation to number of recommendations")
    plt.xlabel("Number of recommended songs")
    plt.ylabel("Time required in seconds")
    plt.show()
    #print("Time needed for recommendations : " + str(perf_counter() - start_time) + " seconds.")
    '''
    sr.prepare_user_data_evaluation("../Data/new_song-user_interactions_with_deleted_labels.csv")
    evaluation_songs = sr.get_user_data_evaluation()
    for current_id in evaluation_songs:
        current_time = perf_counter()
        if (os.path.exists("../Data/Evaluations for random generated songs/random_recommendations " + str(current_id) + ".csv")):
            continue
            #print("../Data/Evaluations/custom_distances " + str(current_id) + ".csv exists.")
        else :
            recommendations = generate_random_recommendations(current_id, sr.get_song_dataframe())
            recommendations.to_csv("../Data/Evaluations for random generated songs/random_recommendations " + str(current_id) + ".csv")
        print("Time elapsed : " + str(perf_counter() - current_time) + " seconds.")

    #random generated recommendations

    #pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    #recommendations = pool.starmap(sr.get_recommendations, list(zip(evaluation_songs[0:len(evaluation_songs)])))

    #current_no = 0
    #for current_id in evaluation_songs:
        #current_no+=1
        #print(current_no)
        #if current_no >= 8273 :
            #current_song_recommendations = sr.get_recommendations(song_number_or_id= current_id)
            #recommendations.append(current_song_recommendations)

    #print(recommendations)

    #for current_id in evaluation_songs:
        #start_time = perf_counter()
        #current_recommendations = pd.read_csv("../Data/Evaluations/custom_distances " + str(current_id)+ ".csv")
        #current_no += 1
        #current_optimal_no_recommendations = find_best_one_song_average_f1_evaluation(current_id, "../Data/new_song-user_interactions_with_deleted_labels.csv", current_recommendations)
        #print("Current optimal number of recommendations : " + str(current_optimal_no_recommendations))
        #print("Time required for one song evaluation :" + str(perf_counter() - start_time) + " seconds.")
    #print("Time required for evaluation songs: " + str(perf_counter() - start_time) + " seconds.")



    #testing for a random number
    user_song_data_interactions_df = pd.read_csv("../Data/new_song-user_interactions.csv")
    #new_start_time = perf_counter()
    #average_f1_score_for_a_value = average_f1_score_evaluation_parallel(user_song_data_interactions_df, evaluation_songs, 28510)
    #print("Value average f1 score for top 100 songs : " + str(average_f1_score_for_a_value))
    #print("Time required for average f1 score for top 100 songs : " + str(perf_counter() - new_start_time))
    upper = 28510
    lower = 21382
    new_start_time = perf_counter()
    #optimal_number_of_recommendations = find_optimal_number_of_recommendations_for_recommendations(lower,upper)
    #print("Optimal number of recommendations in range [" + str(lower) + "," + str(upper) + "] is " + str(optimal_number_of_recommendations))
    numbers = [14265,14263,14255,7128,3564,1782]
    for current_number in reversed(numbers) :
        print("Time required : " + str(perf_counter() - new_start_time) + " seconds.")
        current_average_f1, current_precision, current_recall = average_f1_score_evaluation_parallel(user_song_data_interactions_df,evaluation_songs, number_of_recommendations=current_number)
        print("Current value : " + str(current_number))
        print("Current average f1 : " + str(current_average_f1))
        print("Current average precision : " + str(current_precision))
        print("Current average recall : " + str(current_recall))
    #pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    #multiple_optimal_recommendation_numbers = pool.starmap(find_best_one_song_average_f1_evaluation, list(zip(evaluation_songs[0:len(evaluation_songs)], repeat("../Data/new_song-user_interactions_with_deleted_labels.csv"), recommendations[0:len(recommendations)])))
    #print("Time elapsed after picking all multiple optimal recommendation numbers: " + str(perf_counter() - start_time) + " seconds.")
       # print("Time elapsed after operation was done: " + str(perf_counter() - start_time) + " seconds.")