"""
This module contains functions for:
    - sorting clusters by size
    - sorting clusters by clumpiness and declumping
    - other cluster analyses
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from skimage import filters
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
from instapipeline import util


# list of declumping algs handled
declumping_algs = ['KMeans']


"""

Functions for sorting clusters by size

"""


def get_cluster_size_threshold(clusters):
    """
    Calculate a cluster size threshold for all clusters
    using K-means in 1D. Assumes a bimodal distribution.

    Parameters
    ----------
    clusters : pandas dataframe
        (centroid_x | centroid_y | members)

    Returns
    -------
    cluster size threshold
    """
    total_list = []
    for index, row in clusters.iterrows():
        members = row['members']
        worker_list = [member[3] for member in members]
        num_members = len(np.unique(worker_list))
        total_list.append(num_members)
    total_array = np.asarray(total_list)
    km = KMeans(n_clusters=2).fit(total_array.reshape(-1, 1))
    cluster_centers = km.cluster_centers_
    return (cluster_centers[0][0]+cluster_centers[1][0])/2


def plot_cluster_size_threshold(clusters, threshold):
    """
    Visualize cluster sizes in a histogram with threshold demarcated.

    Parameters
    ----------
    clusters : pandas dataframe (centroid_x | centroid_y | members)
        centroid_x = x coord of cluster centroid
        centroid_y = y coord of cluster centroid
        members = list of annotations belonging to the cluster
            each member is a list of properties of the annotation
            i.e. [x coord, y coord, time spent, worker ID]
    threshold : value to show threshold demarcation on histogram

    Returns
    -------
    figure
    axes
    """
    fig = plt.figure()
    hist_list = []
    for index, row in clusters.iterrows():
        members = row['members']
        worker_list = [member[3] for member in members]
        hist_list.append(len(np.unique(worker_list)))
    width = max(hist_list)
    plt.hist(hist_list, bins=np.arange(0, width+4, 2)-1)
    plt.axvline(x=threshold, color='b')
    plt.legend(handles=[Line2D([0], [0], color='b',
               label='cluster size threshold')])
    plt.title('Find Cluster Size Threshold')
    plt.xlabel('Number of unique annotators for cluster')
    plt.ylabel('Number of clusters')
    plt.show()

    return fig, fig.axes


def sort_clusters_by_size(clusters, threshold):
    """
    Sort clusters by quantity of unique annotators.

    Parameters
    ----------
    clusters : pandas dataframe
        (centroid_x | centroid_y | members)
    threshold : threshold quantity of unique annotators

    Returns
    -------
    small_clusters : pandas dataframe containing clusters
        for which num unique annotators < threshold
        (centroid_x | centroid_y | members)
    large_clusters : pandas dataframe containing clusters
        for which num unique annotators >= threshold
        (centroid_x | centroid_y | members)
    """
    small_clusters_list = []
    large_clusters_list = []
    for index, row in clusters.iterrows():
        members = row['members']
        centroid_x = row['centroid_x']
        centroid_y = row['centroid_y']

        worker_list = []
        for member in members:
            worker_list.append(member[3])
        num_members = len(np.unique(worker_list))

        if (num_members < threshold):
            small_clusters_list.append([centroid_x, centroid_y, members])
        else:
            large_clusters_list.append([centroid_x, centroid_y, members])

    column_list = ['centroid_x', 'centroid_y', 'members']
    small_clusters = pd.DataFrame(index=range(len(small_clusters_list)),
                                  columns=column_list)
    large_clusters = pd.DataFrame(index=range(len(large_clusters_list)),
                                  columns=column_list)

    for i, small_cluster in enumerate(small_clusters_list):
        small_clusters['centroid_x'][i] = small_cluster[0]
        small_clusters['centroid_y'][i] = small_cluster[1]
        small_clusters['members'][i] = small_cluster[2]

    for i, large_cluster in enumerate(large_clusters_list):
        large_clusters['centroid_x'][i] = large_cluster[0]
        large_clusters['centroid_y'][i] = large_cluster[1]
        large_clusters['members'][i] = large_cluster[2]

    return small_clusters, large_clusters


"""

Functions for sorting clusters by clumpiness and declumping

"""


def get_clumpiness_threshold(clusters):
    """
    Calculate a clumpiness threshold for all clusters
    by finding the value between the tail and the main mode.
    Assumes a left-skewed unimodal distribution.
    Protocol for finding threshold:
    Sort all clusters into bins.
        e.g. if bin_size = 0.1, then sort clusters into bins 100-95%, ..., 5-0%
        (% of contributors contributed only once to this cluster)
    Find all values between two adjacent bins where the number of
        clusters in the higher-value bin is at least cutoff_fraction
        times greater than the number of clusters in the lower-value bin,
        and neither bin contains zero clusters.
    threshold is the lowest of these values minus 0.1 (in order to move
        one bin to the left, to minimize the number of clusters which
        are actually single in the group of clusters detected as clumpy),
        or 0 if no such values exist.

    Parameters
    ----------
    clusters : pandas dataframe
        (centroid_x | centroid_y | members)
    bin_size : see protocol
    cutoff_fraction : see protocol

    Returns
    -------
    clumpiness threshold
    """
    single_fraction_list = []
    for index, row in clusters.iterrows():
        members = row['members']
        workers = [member[3] for member in members]
        uniques = np.unique(workers)
        num_instances_list = [workers.count(unique) for unique in uniques]
        single_fraction_list.append(num_instances_list.count(1)/len(uniques))

    plt.figure()
    (n, bins, patches) = plt.hist(single_fraction_list,
                                  bins=np.arange(0, 1.2, 0.1) - 0.05)
    plt.close()
    # calculate threshold
    total_counts_rev = list(reversed(n))
    threshold, prev_count, bin_width = 0, 0, 0.1
    for i in range(len(total_counts_rev)):
        count = total_counts_rev[i]
        if (count != 0):
            if((count < prev_count/3) and (count != 0) and (prev_count != 0)):
                threshold = bin_width*10-i*bin_width-bin_width/2
        prev_count = count

    return threshold


def plot_clumpiness_threshold(clusters):
    """
    Get cluster clumpiness threshold, visualize cluster clumpiness
    in a histogram with threshold demarcated.

    Parameters
    ----------
    clusters : pandas dataframe (centroid_x | centroid_y | members)
        centroid_x = x coord of cluster centroid
        centroid_y = y coord of cluster centroid
        members = list of annotations belonging to the cluster
            each member is a list of properties of the annotation
            i.e. [x coord, y coord, time spent, worker ID]

    Returns
    -------
    threshold : the fraction of workers who contribute 1x
    figure
    axes
    """
    single_fraction_list = []
    for index, row in clusters.iterrows():
        members = row['members']
        worker_list = [member[3] for member in members]
        uniques = np.unique(worker_list)
        num_instances_list = [worker_list.count(unique) for unique in uniques]
        single_fraction_list.append(num_instances_list.count(1)/len(uniques))

    fig = plt.figure()
    (n, bins, patches) = plt.hist(single_fraction_list,
                                  bins=np.arange(0, 1.2, 0.1) - 0.05)

    # calculate threshold
    total_counts_rev = list(reversed(n))
    threshold, prev_count, bin_width = 0, 0, 0.1
    for i in range(len(total_counts_rev)):
        count = total_counts_rev[i]
        if (count != 0):
            if((count < prev_count/3) and (count != 0) and (prev_count != 0)):
                threshold = bin_width*10-i*bin_width-bin_width/2
        prev_count = count

    threshold_line = Line2D([0], [0], color='orange',
                            label='clumpiness threshold')
    plt.legend(handles=[threshold_line])
    plt.axvline(x=threshold, color='orange')
    plt.xticks(np.arange(0, bin_width*11, bin_width))
    plt.xlabel('Fraction of contributing workers who contribute only once')
    plt.ylabel('Number of clusters')
    plt.title('Finding the Clumpiness Threshold')
    plt.show()
    return threshold, fig, fig.axes


def sort_clusters_by_clumpiness(clusters, threshold):
    """
    Sort clusters by fraction of contributors who contribute once
    to the cluster.

    Parameters
    ----------
    clusters : pandas dataframe
        (centroid_x | centroid_y | members)
    threshold : threshold fraction of contributors who only contribute 1x

    Returns
    -------
    clumpy_clusters : pandas dataframe containing clusters
        for which fraction of contributors who only contribute 1x < threshold
        (centroid_x | centroid_y | members)
    nonclumpy_clusters : pandas dataframe containing clusters
        for which fraction of contributors who only contribute 1x >= threshold
        (centroid_x | centroid_y | members)
    """
    clumpy_clusters_list = []
    nonclumpy_clusters_list = []
    clumpy_counter = 0
    nonclumpy_counter = 0
    for index, row in clusters.iterrows():
        members = row['members']
        centroid_x = row['centroid_x']
        centroid_y = row['centroid_y']

        workers = []
        for member in members:
            workers.append(member[3])
        unique_workers = np.unique(workers)

        num_instances_list = []
        for unique_worker in unique_workers:
            num_instances_list.append(workers.count(unique_worker))
        singles = num_instances_list.count(1)
        single_fraction = singles/len(unique_workers)

        if (single_fraction < threshold):
            clumpy_clusters_list.append([centroid_x, centroid_y, members])
            clumpy_counter += 1
        else:
            nonclumpy_clusters_list.append([centroid_x, centroid_y, members])
            nonclumpy_counter += 1

    column_list = ['centroid_x', 'centroid_y', 'members']
    clumpy_clusters = pd.DataFrame(index=range(clumpy_counter),
                                   columns=column_list)
    nonclumpy_clusters = pd.DataFrame(index=range(nonclumpy_counter),
                                      columns=column_list)

    for k in range(clumpy_counter):
        clumpy_clusters['centroid_x'][k] = clumpy_clusters_list[k][0]
        clumpy_clusters['centroid_y'][k] = clumpy_clusters_list[k][1]
        clumpy_clusters['members'][k] = clumpy_clusters_list[k][2]

    for m in range(nonclumpy_counter):
        nonclumpy_clusters['centroid_x'][m] = nonclumpy_clusters_list[m][0]
        nonclumpy_clusters['centroid_y'][m] = nonclumpy_clusters_list[m][1]
        nonclumpy_clusters['members'][m] = nonclumpy_clusters_list[m][2]

    return clumpy_clusters, nonclumpy_clusters


def declump(clusters, i, declumping_params):
    """
    Declump the cluster at the ith index of clusters,
    a df only containing clumpy clusters.

    Parameters
    ----------
    clusters : pandas dataframe (centroid_x | centroid_y | members)
        centroid_x = x coord of cluster centroid
        centroid_y = y coord of cluster centroid
        members = list of annotations belonging to the cluster
            each member is a list of properties of the annotation
            i.e. [x coord, y coord, time spent, worker ID]
    i : index of cluster in clusters to declump
    declumping_params : list of clustering parameters
        first element is string name of declumping algorithm
        subsequent elements are additional parameters

    Returns
    -------
    declumped_clusters : pandas df containing resulting declumped clusters
    """
    if (declumping_params[0] not in declumping_algs):
        raise ValueError('Invalid declumping algorithm name entered.')

    row = clusters.iloc[[i]]
    members = row.iloc[0]['members']
    x_coords = [member[0] for member in members]
    y_coords = [member[1] for member in members]
    timestamps = [member[2] for member in members]
    workers = [member[3] for member in members]
    coords = np.stack((x_coords, y_coords), axis=-1)

    if (declumping_params[0] == 'KMeans'):
        k = declumping_params[1]
        km = KMeans(n_clusters=k).fit(coords)
        centers = km.cluster_centers_
        labels = km.labels_
        num_subclusters = k

    members = np.stack((x_coords, y_coords, timestamps, workers), axis=-1)
    subclusters_list = [[center[0], center[1], []] for center in centers]
    for member, label in zip(members, labels):
        subclusters_list[label][2].append(member)

    subclusters = pd.DataFrame(index=range(num_subclusters),
                               columns=['centroid_x', 'centroid_y', 'members'])

    for ind in range(num_subclusters):
        subclusters['centroid_x'][ind] = subclusters_list[ind][0]
        subclusters['centroid_y'][ind] = subclusters_list[ind][1]
        subclusters['members'][ind] = subclusters_list[ind][2]

    return subclusters


"""

Functions for other cluster analyses

"""


def get_cluster_means(clusters):
    """
    Get the mean x and y of each cluster.
    (Different from cluster centroids, which are the exemplar
    annotation for each cluster.)

    Parameters
    ----------
    clusters : pandas dataframe
        (centroid_x | centroid_y | members)

    Returns
    -------
    numpy array of coords
    """
    mean_coords = []

    for index, row in clusters.iterrows():
        members = row['members']
        x_coords = []
        y_coords = []
        for member in members:
            x_coords.append(member[0])
            y_coords.append(member[1])
        mean_coord = [np.mean(x_coords), np.mean(y_coords)]
        mean_coords.append(mean_coord)
    return np.asarray(mean_coords)


def get_cluster_correctness(df, correctness_threshold):
    """
    Assemble a dataframe of centroids found with
    annotation and reference data consolidated.

    Parameters
    ----------
    centroid_and_ref_df : outputted by util.centroid_and_ref_df()
        centroid_x | centroid_y | x of nearest ref | y of nearest ref
            | NN_dist | members (x | y | time_spent | worker_id)
        * the index is the Centroid ID
    correctness_threshold: tolerance for correctness in pixels
        None if correctness will not be visualized
        for each centroid, if NN_dist <= threshold, centroid is "correct"

    Returns
    -------
    2-column array with a row for each centroid
        column 0 = Centroid ID
        column 1 = T if centroid is "correct" else F
    """
    num_centroids = df.shape[0]
    to_return = np.empty([num_centroids, 2])
    for i in range(num_centroids):
        to_return[i] = i
        nn_dist = df['NN_dist'][i]
        if (nn_dist <= correctness_threshold):
            to_return[i][1] = True
        else:
            to_return[i][1] = False
    return to_return


def get_pair_scores(df):
    """
    Calculate pair scores for each pair of workers in df.

    Parameters
    ----------
    df : pandas dataframe

    Returns
    -------
    pair_scores : pandas dataframe
        indices and columns of the dataframe are worker IDs
        contents are pair scores
        pair score between worker_A and worker_B
            = ((avg A->B NND) + (avg B->A NND))/2
    """
    worker_list = util.get_workers(df)
    pair_scores = pd.DataFrame(index=worker_list, columns=worker_list)
    for worker in worker_list:
        worker_df = util.slice_by_worker(df, worker)
        worker_coords = util.get_click_properties(worker_df)[:, :2]
        worker_kdt = KDTree(worker_coords, leaf_size=2, metric='euclidean')

        for other_worker in worker_list:
            if worker == other_worker:
                pair_scores[worker][other_worker] = 0
                continue

            other_worker_df = util.slice_by_worker(df, other_worker)
            click_properties = util.get_click_properties(other_worker_df)
            other_worker_coords = click_properties[:, :2]
            other_worker_kdt = KDTree(other_worker_coords, leaf_size=2,
                                      metric='euclidean')

            list_A = [None]*len(worker_coords)
            for i in range(len(worker_coords)):
                dist, ind = other_worker_kdt.query([worker_coords[i]], k=1)
                list_A[i] = dist[0][0]

            list_B = [None]*len(other_worker_coords)
            for j in range(len(other_worker_coords)):
                dist, ind = worker_kdt.query([other_worker_coords[j]], k=1)
                list_B[j] = dist[0][0]

            mean_A = np.mean(list_A)
            mean_B = np.mean(list_B)

            pair_scores[worker][other_worker] = (mean_A + mean_B)/2

    return pair_scores


def get_worker_pair_scores(df):
    """
    Calculate the total pairwise score for each workers in df.

    Parameters
    ----------
    df : pandas dataframe

    Returns
    -------
    worker_scores : pandas dataframe
        indices of the dataframe are worker IDs
        column header of dataframe is "score"
        "score" is the sum of the worker's pairwise scores
    """
    worker_list = util.get_workers(df)
    pair_scores = get_pair_scores(df)
    worker_scores = pd.DataFrame(index=worker_list, columns=["score"])
    for worker in worker_list:
        worker_scores["score"][worker] = sum(pair_scores[worker].values)
    return worker_scores


def get_worker_pair_score_threshold(df):
    """
    Calculate a pairwise score threshold for all workers in
    df using Otsu's method. Assumes a bimodal distribution.

    Parameters
    ----------
    df : pandas dataframe

    Returns
    -------
    pairwise score threshold value
    """
    # score workers based on pairwise matching
    worker_pairwise_scores = get_worker_pair_scores(df)
    # get IDs of all workers
    worker_scores_list = worker_pairwise_scores['score'].tolist()
    return filters.threshold_otsu(np.asarray(worker_scores_list))


def slice_by_worker_pair_score(df, threshold):
    """
    Drop all annotations in df by workers with average pairwise
    score greater than threshold

    Parameters
    ----------
    df : pandas dataframe
    threshold : pairwise score threshold

    Returns
    -------
    df : pandas dataframe
    """
    worker_pair_scores = get_worker_pair_scores(df)
    high_scores = worker_pair_scores[worker_pair_scores.score > threshold]
    high_scoring_workers = high_scores.index.values
    for worker in high_scoring_workers:
        df = df[df.worker_id != worker]
    return df
