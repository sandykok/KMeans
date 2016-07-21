__author__ = 'sandeep'

import argparse
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
import math

class kMeans_main(object):

    @staticmethod
    def get_distance(x1, y1, x2, y2):
        dist = math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
        return dist

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_file", help = "Location of Input File", required = True)
        parser.add_argument("--threshold_percentile", help = "Threshold percentile to find Outliers", required = True)
        parser.add_argument("--output_folder", help = "Location of Output Files", required = True)
        args = parser.parse_args()
        return(args)

    @staticmethod
    def get_natural_clusters(data, output_folder):
        plt.plot(data['Column1'],data['Column2'], 'ro')
        plt.savefig(output_folder+'given_data.png')
        plt.clf()

    @staticmethod
    def kMeans_clustering_main(data, output_folder):

        #After plotting the given data, it is clear that there are 3 natural clusters in data
        k_means = KMeans(n_clusters=3)
        k_means.fit(data)
        centers = k_means.cluster_centers_
        labels = k_means.labels_

        #Save the cluster centroids in CSV file
        np.savetxt(output_folder+"cluster_centers.csv", centers, delimiter=",")

        #Save the data with cluster assignment in CSV file
        data['assigned_cluster'] = labels.tolist()
        data.to_csv(output_folder+"data_with_cluster_assignment.csv", index=False)
      

        #Plot the data after cluster assignment

        pyplot.scatter(data['Column1'],data['Column2'], c=data['assigned_cluster'])
        plt.savefig(output_folder+'plot_with_cluster_assignment.png')
        plt.clf()
        return k_means, data

    @staticmethod
    def find_outliers(data, centers, threshold_percentile, output_folder):
        distance_from_centriods = []
        for i in range(len(data)):
            assigned_centriod = centers[data['assigned_cluster'][i]]
            temp_dist = kMeans_main.get_distance(data['Column1'][i], data['Column2'][i], assigned_centriod[0], assigned_centriod[1])
            distance_from_centriods.append(temp_dist)

        data['distance_from_assigned_cluster_centriod'] = distance_from_centriods
        percentile_value = np.percentile(distance_from_centriods,float(threshold_percentile))
        is_outlier = []
        for i in range(len(data)):
            if data['distance_from_assigned_cluster_centriod'][i] > percentile_value:
                is_outlier.append(1)
            else:
                is_outlier.append(0)
        data['is_outlier'] = is_outlier
        data.to_csv(output_folder+"final_data_with_outlier_info.csv", index=False)


    @staticmethod
    def main_function(input_file, threshold_percentile, output_folder):
        data = pd.read_csv(input_file, header=None)
        data.columns =['Column1', 'Column2']
        kMeans_main.get_natural_clusters(data, output_folder)
        k_means, data = kMeans_main.kMeans_clustering_main(data=data,output_folder=output_folder)
        kMeans_main.find_outliers(data=data,centers=k_means.cluster_centers_,threshold_percentile=threshold_percentile,output_folder=output_folder)

if __name__ == '__main__':
    args = kMeans_main.get_args()
    kMeans_main.main_function(input_file=args.input_file,threshold_percentile=args.threshold_percentile,output_folder=args.output_folder)
