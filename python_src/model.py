import multiprocessing
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from time import perf_counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_kernels
from numba import jit,cuda
import os

class Song:

    def __init__(self, custom_distance, artist, song, preview, type, id):
        self.custom_distance = custom_distance
        self.artist = artist
        self.song_name = song
        self.preview = preview
        self.type = type
        self.id = id

class SongReccomender:

    """ This is the class of the recommender system

        Parameters:
            track_dataframe: A pandas dataframe which contains the tracks used for the development of the recommender system
            number_of_principal_components: The number of principal components for converting non-categorical audio features into a number of principal components
            number_of_clusters: The number of clusters for K-Means clustering

        Attributes:
            filename: The file path of the track dataset used for generating recommendations


    """

    def __init__(self, filename, number_of_principal_components=6, number_of_clusters=6):
        self.track_dataframe = pd.read_excel(filename)
        self.number_of_principal_components = number_of_principal_components
        self.number_of_clusters=number_of_clusters
        self.__remove_categorical_data()
        self.__pca(self.number_of_principal_components)
        self.__cluster_dataframe_in_numbered_labels_with_k_means()
        self.__readd_categorical_columns()
        self.__classify_track_labels_by_knn()
        self.__compute_tf_idf_matrices_by_label_for_all_tracks()


    def __remove_categorical_data(self):
        self.non_categorical_audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness',
                                'liveness', 'valence', 'tempo', 'duration_ms']

        self.track_dataframe[self.non_categorical_audio_features] = self.track_dataframe[self.non_categorical_audio_features].apply(
            lambda x: (x - x.min()) / (x.max() - x.min()))

    def __pca(self, number_of_components=6):
        pca = PCA(number_of_components)
        self.new_track_dataframe = pca.fit_transform(self.track_dataframe[self.non_categorical_audio_features])
        self.new_track_dataframe = pd.DataFrame(self.new_track_dataframe)
        #return

    def __cluster_dataframe_in_numbered_labels_with_k_means(self):
        self.km = KMeans(n_clusters=self.number_of_clusters, init='k-means++',n_init=10, max_iter=1000,tol=1e-04, random_state=0)
        self.km.fit(self.new_track_dataframe)
        self.track_dataframe['label'] = self.km.labels_

    def __compute_current_pca_based_distance(self, center_coordinates, data_coordiantes):
        sum = 0
        for i in range(len(data_coordiantes)):
            sum += (center_coordinates[i] - data_coordiantes[i]) ** 2
        return (sum) ** 0.5

    @jit(target_backend='cuda')
    def __compute_all_pca_based_distances(self, dummy_df, song):
        list_of_distances = []
        for i in range(len(dummy_df)):
            current_distance = self.__compute_current_pca_based_distance(dummy_df[i], song.values[0][0:self.number_of_principal_components])
            list_of_distances.append(current_distance)
        return list_of_distances


    @jit(target_backend='cuda')
    def __computing_custom_distances(self, array_dist, dummy_df, cosine_similarities):
        array_dist_min = min(array_dist)
        array_dist_max = max(array_dist)
        min_popularity = min(dummy_df['popularity'])
        max_popularity = max(dummy_df['popularity'])
        songs = []
        for i in range(len(dummy_df)):
            current_distance = 1 - (array_dist[i] - array_dist_min) / (array_dist_max - array_dist_min) # re-scaling to [0,1] such that for array_dist[i]=array_dist_min, the best re-scaled euclidean distance is 1 and 0 the worst
            normalized_popularity = (dummy_df['popularity'][i] - min_popularity) / (max_popularity - min_popularity)
            custom_distance = (current_distance + normalized_popularity + cosine_similarities[i][0]) / 3
            current_song = Song(custom_distance, dummy_df['artist'][i],
                    dummy_df['name'][i], dummy_df['preview'][i],
                    dummy_df['type'][i], dummy_df['id'][i])

            songs.append(current_song)
        return songs

    def __readd_categorical_columns(self):
        self.remaining_track_features = ['label', 'artist', 'name', 'preview', 'popularity', 'type', 'lyrics', 'id']

        for current_feature in self.remaining_track_features:
            self.new_track_dataframe[current_feature] = self.track_dataframe[current_feature]

    def __classify_track_labels_by_knn(self):

        self.X = self.new_track_dataframe
        self.Y = self.new_track_dataframe['label']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.20,
                                                                                shuffle=False)
        knn = KNeighborsClassifier(n_neighbors=25)
        knn.fit(self.X_train.drop(self.remaining_track_features, axis=1), self.y_train)
        self.y_pred = knn.predict(self.X_test.drop(self.remaining_track_features, axis=1))
        self.y_pred = pd.Series(self.y_pred)

    def __compute_tf_idf_matrices_by_label_for_all_tracks(self):

        self.tfidf = TfidfVectorizer(analyzer='word', stop_words='english', lowercase=True)
        self.Y = pd.concat([self.y_train, self.y_pred], ignore_index=True)
        labels = self.Y.unique()
        self.X_labeled = [self.X.loc[self.Y == label] for label in range(0, len(labels))]
        self.lyrics_tf_idf_matrices = [self.tfidf.fit_transform(self.X_labeled[label]['lyrics'].values.astype('U')) for label in range(0, len(labels))]


    def recommend_songs(self, track):
        start_time = perf_counter()
        try:
            arr_df = pd.read_csv('../Data/Evaluations/Number of principal components ' + str(self.number_of_principal_components) +'/Number of clusters ' + str(self.number_of_clusters) +'/custom_distances ' + track['id'][track.index.values[0]] + '.csv')
            arr_df.drop('Unnamed: 0', inplace=True, axis=1)
        except:
            if track.index.values[0] in self.X_train.index:
                X_labeled = self.X_labeled[track.label.values[0]]
                lyrics_tf_idf_matrix = self.lyrics_tf_idf_matrices[track.label.values[0]]#self.X.loc[Y==song.label.values[0]]#.loc[self.X.index!=song.index.values[0]]
            else:
                X_labeled = self.X_labeled[self.y_pred[track.index.values[0] - len(self.y_train)]]
                lyrics_tf_idf_matrix = self.lyrics_tf_idf_matrices[self.y_pred[track.index.values[0] - len(self.y_train)]]#self.X.loc[Y == self.y_pred[song.index.values[0] - len(self.y_train)]]

            position = X_labeled['lyrics'].index.get_loc(track.index.values[0])
            X_labeled =  X_labeled.drop(track.index.values[0])
            print("Current time passed at this point : " + str(perf_counter() - start_time) + " seconds.")

            print("Current time passed at this point after tf-idf transformation: " + str(perf_counter() - start_time) + " seconds.")
            cosine_similarities = pairwise_kernels(lyrics_tf_idf_matrix, lyrics_tf_idf_matrix[position], metric='cosine', n_jobs=-1)#cosine_similarity(lyrics_matrix.astype('float32'))
            cosine_similarities = np.delete(cosine_similarities, position, axis=0)
            print("Current time passed at this point after cosine similarity: " + str(perf_counter() - start_time) + " seconds.")

            array_dist = self.__compute_all_pca_based_distances(X_labeled.to_numpy(), track)

            print("Current time passed at this point after computing Euclidean distances : " + str(perf_counter() - start_time) + " seconds")
            arr = self.__computing_custom_distances(array_dist, X_labeled.reset_index(), cosine_similarities) #min_popularity, max_popularity)#.to_numpy(), cosine_similarities, min_popularity, max_popularity)
            print("Current time passed at this point after computing custom distances: " + str(perf_counter() - start_time) + " seconds.")
            arr.sort(key = lambda x: x.custom_distance, reverse=True)
            print("Current time passed at this point after sorting: " + str(perf_counter() - start_time) + " seconds.")
            arr_df = pd.DataFrame([t.__dict__ for t in arr])
            print(track['id'][track.index.values[0]])
            #arr_df.to_csv('../Data/Evaluations/Number of principal components ' + str(self.number_of_principal_components) + '/Number of clusters ' + str(self.number_of_clusters) +'/custom_distances ' + track['id'][track.index.values[0]] + '.csv')
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
            track = self.new_track_dataframe.loc[[song_number]]
        except:
            raise ValueError(str(song_number) + " is out of range.")
        #ans = self.song_recommendation(track, self.data,number_of_recommendations)
        ans = self.recommend_songs(track)
        self.song_print(track)
        j = 1
        for track in ans.itertuples():#[::-1]:
            if (j > number_of_recommendations):
                break
            print('Number:  ', j)
            print('Custom Distance Value:  ', track.custom_distance)
            print('Artist:  ', track.artist)
            print('Song Name:   ', track.song_name)
            print('Type:   ', track.type)
            print('Preview link:   ', track.preview)
            print('-' * 100)
            j += 1
        return

    def get_recommendations(self, song_number_or_id):
        if isinstance(song_number_or_id, int):
            try:
                song = self.new_track_dataframe.loc[[song_number_or_id]]
            except:
                raise ValueError(str(song_number_or_id) + " is out of range.")
        else:
            try:
                song = self.new_track_dataframe[self.new_track_dataframe['id'] == song_number_or_id]
            except:
                raise ValueError(str(song_number_or_id) + " doesn't exist as an id.")
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


def process_song(args):

    current_track_id, user_track_interactions_df, recommendations_filename, number_of_recommendations = args
    """ Computes the average mean of precision, recall and F1 score based on all users for a single track
    
        Args:
            current_track_id : the id of the current track from the set of evaluation tracks.
            user_track_interactions_df: the data set of user-track interactions.
            recommendations_filename : the file path of the recommendations based on each track from the set of evaluation tracks.
            number_of_recommendations: the number of recommendations.
        
        Returns:
            The average mean of precision, recall and F1 score based on all users for a single track       
    """

    # Read the recommended tracks based on the track with the current track id from the set of evaluation tracks
    recommendations = pd.read_csv(recommendations_filename + str(current_track_id) + ".csv")

    # Extract from the user-track interaction dataset the users who listened to the input track with current track id
    track_listened_by_users = user_track_interactions_df[user_track_interactions_df['song_id'] == current_track_id]
    
    # Filter out the rows where the 'track_id' is the same as the target song_id
    filtered_tracks = user_track_interactions_df[user_track_interactions_df['song_id'] != current_track_id]
  
    # Determine the total number of listened tracks by each user
    total_number_of_listened_tracks_by_user = filtered_tracks.groupby('user_id').size().reindex(
        filtered_tracks['user_id'].unique(),
        fill_value=0
    )
 
    # Determine all tracks listened by using set intersection and vectorized operations
    tracks_listened_by_all_users = filtered_tracks[
        filtered_tracks['user_id'].isin(track_listened_by_users['user_id'])
    ]

    # Convert the recommendations DataFrame to a set for faster look-up
    recommended_song_ids = set(recommendations['id'][:number_of_recommendations].values)

    # Find all tracks listened by all users in the set of recommendations
    track_hit_mask = tracks_listened_by_all_users['song_id'].isin(recommended_song_ids)

    # Number of tracks each user listened to and were provided as recommendations
    hits_by_user = tracks_listened_by_all_users[track_hit_mask].groupby('user_id').size().reindex(
        track_listened_by_users['user_id'].unique(),
        fill_value=0
    )

    # Calculate precision and recall using vectorized operations
    precision = hits_by_user / number_of_recommendations
    recall = hits_by_user / total_number_of_listened_tracks_by_user.reindex(
        track_listened_by_users['user_id'].unique(),
        fill_value=0
    )

    # Handle the case when both precision and recall are 0 in F1 score formula: F1 = 2*P*R/(P+R)
    zero_mask = (precision == 0) & (recall == 0)

    # Calculate average recall and average precision separately
    average_recall = np.nanmean(recall)
    average_precision = np.nanmean(precision)


    f1_score = compute_f1_score(precision, recall)

    # Set F1 score to 0 where both precision and recall are 0
    f1_score = np.where(zero_mask, 0, f1_score)

    # Compute the average F1 score, ignoring NaN values
    average_f1_score = np.nanmean(f1_score)

    return average_f1_score, average_precision, average_recall


#de modificat numele si eventual si metoda
def average_f1_score_evaluation_parallel(user_song_data_interactions_df,evaluation_songs, recommendations_filepath, number_of_recommendations=None):
    if number_of_recommendations is None:
        number_of_recommendations = 28510

    print("Number of processes : " + str(multiprocessing.cpu_count()))
    # Create a pool of processes with the maximum number of available CPU cores (-1)
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Prepare the arguments for each song ID to be processed
        song_args = [(current_song_id,user_song_data_interactions_df, recommendations_filepath, number_of_recommendations) for current_song_id in evaluation_songs]

        # Use the pool to map the tasks to the processes and get the results

        results = pool.map(process_song, song_args)
        one_song_average_f1_scores, one_song_average_precision_scores, one_song_average_recall_scores = zip(*results)


    average_f1_score = sum(one_song_average_f1_scores) / len(one_song_average_f1_scores)
    average_precision_score = sum(one_song_average_precision_scores) / len(one_song_average_precision_scores)
    average_recall_score = sum(one_song_average_recall_scores) / len(one_song_average_recall_scores)

    return average_f1_score, average_precision_score, average_recall_score




def find_optimal_number_of_recommendations_for_recommendations(lower_number,upper_number, recommendations_filepath):

    begin = lower_number
    end = upper_number
    user_song_data_interactions_df = pd.read_csv("../Data/new_song-user_interactions.csv")
    best_average_f1_list = []
    max_best_average_f1 = average_f1_score_evaluation_parallel(user_song_data_interactions_df, evaluation_songs, recommendations_filepath, end)
    result_dict = { "max_best_average_f1": max_best_average_f1,"number of recommendations": end}
    best_average_f1_list.append(result_dict)
    #print("Current time elapsed : " + str(perf_counter() - current_start_time) + " seconds for value " + str(end))
    while end>begin:
        current_start_time = perf_counter()
        mid = (begin + end)//2
        best_average_f1_mid = average_f1_score_evaluation_parallel(user_song_data_interactions_df, evaluation_songs, recommendations_filepath, mid)
        result_dict = {"max_best_average_f1": best_average_f1_mid,
                       "number of recommendations": mid}
        best_average_f1_list.append(result_dict)
        if best_average_f1_mid > max_best_average_f1:
            max_best_average_f1 = best_average_f1_mid
            end = mid
            current_optimal_no_recommendations = end
        else:
            begin = mid + 1
            current_optimal_no_recommendations = begin
        print("Current time elapsed : " + str(perf_counter() - current_start_time) + " seconds for value " + str(mid))
    print("Best average f1 score : " + str(max_best_average_f1))

    best_average_f1_df = pd.DataFrame( best_average_f1_list)
    output_file = "random_output_15.csv"
    best_average_f1_df.to_csv(output_file, index=False)
    return current_optimal_no_recommendations

if __name__ == '__main__':


    start_time = perf_counter()
    number_of_principal_components=6
    number_of_clusters = 6
    sr = SongReccomender('../Data/final_data_with_lyrics_101.xlsx', number_of_principal_components=number_of_principal_components, number_of_clusters=number_of_clusters)
    print("Time needed for constructor : " + str(perf_counter() - start_time) + " seconds.")
    recommendations = sr.print_song_reccomendation(song_number=180219, number_of_recommendations=5)
    sr.prepare_user_data_evaluation("../Data/new_song-user_interactions.csv")
    evaluation_songs = sr.get_user_data_evaluation()
    for current_id in evaluation_songs:
        current_time = perf_counter()
        if (os.path.exists('../Data/Evaluations/Number of principal components ' + str(number_of_principal_components)+ '/Number of clusters ' + str(number_of_clusters)+'/custom_distances ' + current_id + '.csv')):
            continue
        else :
            recommendations = sr.get_recommendations(current_id)
            recommendations.to_csv('../Data/Evaluations/Number of principal components ' + str(number_of_principal_components)+ '/Number of clusters ' + str(number_of_clusters)+'/custom_distances ' + current_id + '.csv')
            print("Time elapsed : " + str(perf_counter() - current_time) + " seconds.")

    user_track_interactions_df = pd.read_csv("../Data/new_song-user_interactions.csv")

    upper = 2
    lower = 1

    new_start_time = perf_counter()
    numbers = [1,2]
    #current_number = 6041
    for current_number in numbers:
        current_average_f1, current_precision, current_recall = average_f1_score_evaluation_parallel(user_track_interactions_df, evaluation_songs, "../Data/Evaluations/Number of principal components 5/Number of clusters 2/custom_distances ", number_of_recommendations=current_number)
        print("Current value : " + str(current_number))
        print("Current average f1 : " + str(current_average_f1))
        print("Current average precision : " + str(current_precision))
        print("Current average recall : " + str(current_recall))
        print("Time required for these values : "+ str(perf_counter()-new_start_time) + " seconds")