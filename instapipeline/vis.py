"""
This module contains functions for visualizing annotations and clusters
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from matplotlib.lines import Line2D
from instapipeline import clus, util

# colors used by plotters
colors = ['#3399FF', '#CC33FF', '#FFFF00', '#FF33CC',
          '#9966FF', '#009999', '#99E3FF', '#B88A00',
          '#33FFCC', '#FF3366', '#F5B800', '#FF6633',
          '#FF9966', '#FF9ECE', '#CCFF33', '#FF667F',
          '#EB4E00', '#FFCC33', '#FF66CC', '#33CCFF',
          '#ACFF07', '#667FFF', '#FF99FF', '#FF1F8F',
          '#9999FF', '#99FFCC', '#FF9999', '#91FFFF',
          '#8A00B8', '#91BBFF', '#FFB71C', '#FF1C76']


def plot_annotations(df=None, show_workers=False,
                     show_correctness_workers=False, show_centroids=False,
                     show_correctness_centroids=False, show_ref_points=False,
                     show_NN_inc=False, centroid_and_ref_df=None,
                     corr_threshold=4, worker_marker_size=8,
                     cluster_marker_size=40, ref_marker_size=20,
                     img_filepath=None, csv_filepath=None,
                     bigger_window_size=True):
    """
    Quick visualization of worker annotations, clusters, and/or
    annotation and cluster "correctness."

    Parameters
    ----------
    df : pandas dataframe with annotation data for one crop only
    show_workers : bool whether to plot workers
    show_centroids : bool whether to plot cluster centroids
    show_ref_points : bool whether to plot reference annotations
    show_NN_inc : bool whether to show nearest neighbor for all
        "incorrect" centroids
    centroid_and_ref_df = pandas dataframe outputted by centroid_and_ref_df()
        centroid_x | centroid_y | x of nearest ref |
        y of nearest ref | NN_dist | members
    corr_threshold : int tolerance for correctness in pixels,
        None if correctness will not be visualized
    worker_marker_size, cluster_marker_size : int plot parameters
    img_filepath, csv_filepath : str paths to image and reference csv files
    bigger_window_size : bool whether to use bigger window size

    Returns
    -------
    none
    """
    plt.figure(figsize=(12, 7))
    if bigger_window_size:
        plt.figure(figsize=(14, 12))

    handle_list = []
    img_height = df['height'].values[0]

    if corr_threshold is not None:
        cluster_correctness = clus.get_cluster_correctness(centroid_and_ref_df,
                                                           corr_threshold)

    if show_workers:
        if show_correctness_workers:
            member_lists = centroid_and_ref_df['members'].values
            zipped = zip(member_lists, cluster_correctness)
            for member_list, correctness in zipped:
                if correctness[1]:
                    color = 'g'
                else:
                    color = 'm'
                for member in member_list:
                    coords = member[:2]
                    plt.scatter([coords[0]], util.flip([coords[1]],
                                img_height), s=worker_marker_size,
                                facecolors=color, alpha=0.5)
            line1 = Line2D([0], [0], marker='o', color='w',
                           markerfacecolor='g',
                           label='anno of correct cluster')
            handle_list.append(line1)
            line2 = Line2D([0], [0], marker='o', color='w',
                           markerfacecolor='m',
                           label='anno of incorrect cluster')
            handle_list.append(line2)
        else:
            worker_list = util.get_workers(df)
            for worker, color in zip(worker_list, colors):
                anno_one_worker = util.slice_by_worker(df, worker)
                coords = util.get_click_properties(anno_one_worker)[:, :2]
                x_coords = coords[:, 0]
                y_coords = coords[:, 1]
                y_coords_flipped = util.flip(y_coords, img_height)
                handle = plt.scatter(x_coords, y_coords_flipped,
                                     s=worker_marker_size, facecolors=color,
                                     alpha=0.5, label=worker)
                handle_list.append(handle)
        if not show_centroids:
            plt.title('Worker Annotations')

    if show_centroids:
        x_coords = centroid_and_ref_df['centroid_x'].values
        y_coords = centroid_and_ref_df['centroid_y'].values
        y_coords_flipped = util.flip(y_coords, img_height)
        color_index = 0
        if show_correctness_centroids:
            for i in range(len(centroid_and_ref_df.index)):
                if (cluster_correctness[i][1]):
                    color = 'g'
                else:
                    if show_NN_inc:
                        color = colors[color_index]
                        color_index = (color_index+1) % len(colors)
                        nnx = centroid_and_ref_df['NN_x']
                        nny = img_height-centroid_and_ref_df['NN_y']
                        plt.scatter([nnx.values[i]],
                                    [nny.values[i]],
                                    s=worker_marker_size*2, facecolors=color,
                                    edgecolors=color)
                    else:
                        color = 'm'
                plt.scatter(x_coords[i], y_coords_flipped[i],
                            s=cluster_marker_size, facecolors='none',
                            edgecolors=color)
            handle_list.append(Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=None, markeredgecolor='g',
                               label='centroid of correct cluster'))
            handle_list.append(Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=None, markeredgecolor='m',
                               label='centroid of incorrect cluster'))
        else:
            plt.scatter(x_coords, y_coords_flipped, s=cluster_marker_size,
                        facecolors='none', edgecolors='cyan')
            handle_list.append(Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=None, markeredgecolor='cyan',
                               label='cluster centroid'))
        if not show_workers:
            plt.title('Cluster Centroids')

    if show_workers and show_centroids:
        plt.title('Worker Annotations and Cluster Centroids')

    if show_ref_points:
        ref_df = pd.read_csv(csv_filepath)
        ref_points = ref_df.loc[:, ['col', 'row']].as_matrix()
        for point in ref_points:
            plt.scatter([point[0]], [point[1]],
                        s=ref_marker_size, facecolors='y')
        handle_list.append(Line2D([0], [0], marker='o', color='w',
                           markerfacecolor='y', label='reference points'))

    img = mpimg.imread(img_filepath)
    plt.imshow(img, cmap='gray')
    plt.legend(handles=handle_list, loc=9, bbox_to_anchor=(1.2, 1.015))
    plt.show()


def visualize_clusters(clusters=None, worker_marker_size=8,
                       cluster_marker_size=40, ref_marker_size=20,
                       csv_filepath=None, img_filepath=None,
                       img_height=None, x_bounds=None,
                       y_bounds=None, plot_title=None,
                       show_workers=False, show_centroids=False,
                       show_ref_points=False, bigger_window_size=True,
                       show_ticks=False, title_font_size=16):
    """
    Visualize clusters, each with a different color.

    Parameters
    ----------
    clusters : pandas dataframe (centroid_x | centroid_y | members)
        centroid_x = int x coord of cluster centroid
        centroid_y = int y coord of cluster centroid
        members = list of annotations belonging to the cluster
            each annotation is a numpy ndarray of properties:
            [int x coord, int y coord, int time spent, str worker ID]
    show_workers : bool whether to plot workers
    show_centroids : bool whether to plot cluster centroids
    worker_marker_size, cluster_marker_size : plot parameters
    img_filepath : str path to image file
    img_height : int height of image in pixels
    plot_title : str title of plot
    bigger_window_size : bool whether to use bigger window size

    Returns
    -------
    none
    """
    plt.figure(figsize=(12, 7))
    if bigger_window_size:
        plt.figure(figsize=(14, 12))
    if x_bounds:
        plt.xlim(x_bounds[0], x_bounds[1])
    if y_bounds:
        plt.ylim(y_bounds[0], y_bounds[1])
    img = mpimg.imread(img_filepath)
    plt.imshow(img, cmap='gray')

    legend_handles = []

    if show_workers:
        for color, member_list in zip(colors*10, clusters['members'].values):
            for member in member_list:
                x = int(member[0])
                y = int(img_height)-int(member[1])
                plt.scatter([x], [y], s=worker_marker_size,
                            facecolors=color, edgecolors='None')

    if show_ref_points:
        ref_df = pd.read_csv(csv_filepath)
        ref_points = ref_df.loc[:, ['col', 'row']].to_numpy()
        for point in ref_points:
            plt.scatter([point[0]], [point[1]],
                        s=ref_marker_size, facecolors='y')
        legend_handles += [Line2D([0], [0], marker='o', color='w',
                           markerfacecolor='y', label='reference points',
                           markersize=10)]

    if show_centroids:
        plt.scatter(clusters['centroid_x'].values,
                    util.flip(clusters['centroid_y'].values, img_height),
                    s=cluster_marker_size, facecolors='none',
                    edgecolors='c')
        legend_handles += [Line2D([0], [0], marker='o', color='w',
                                  markeredgecolor='c', markerfacecolor=None,
                                  label='centroids', markersize=10)]

    if not show_ticks:
        plt.xticks([])
        plt.yticks([])

    plt.legend(handles=legend_handles, loc=9,
               bbox_to_anchor=(1.3, 1.015), prop={'size': 15})

    plt.title(plot_title, fontsize=title_font_size)
    plt.show()
