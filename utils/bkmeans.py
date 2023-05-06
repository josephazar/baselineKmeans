#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 17:00:03 2023

@author: josephazar
"""
import numpy as np

class BaselineKMeans:
    def __init__(self, baseline_data, max_iterations=100, percentile=90, update_after_iteration=1):
        self.baseline_data = baseline_data
        self.anomalous_data = []
        self.max_iterations = max_iterations
        self.percentile = percentile
        self.baseline_mean = None
        self.baseline_cov = None
        self.anomalous_mean = None
        self.anomalous_cov = None
        self.threshold = None
        self.anomalous_centroid = None
        self.baseline_centroid = None
        self.clusters = [[] for _ in range(2)]
        self.dimension = baseline_data.shape[1]
        self.centroids = np.zeros((2, self.dimension))
        self.update_after_iteration = update_after_iteration
        self.update()
        
    def update(self):
      if (len(self.baseline_data)>0):
        # compute mean and covariance of the baseline data
        self.baseline_mean = np.mean(self.baseline_data, axis=0)
        self.baseline_cov = np.cov(self.baseline_data.T)
        self.baseline_centroid = self.baseline_mean
        linalgInv = np.linalg.pinv(self.baseline_cov + 0.001 * np.eye(self.dimension))
        # compute distances from baseline points to baseline centroid
        distances = np.array([np.sqrt(np.sum((np.reshape(point, (1, self.dimension)) - self.baseline_mean) 
                                             @ linalgInv * (np.reshape(point, (1, self.dimension)) - self.baseline_mean), axis=1))  
                              for point in self.baseline_data])
        # compute threshold based on percentile of distances
        self.threshold = np.percentile(distances, self.percentile)
        self.centroids[0] = self.baseline_centroid
      if (len(self.anomalous_data)>0):
        # compute mean and covariance of the anomalous data
        self.anomalous_mean = np.mean(self.anomalous_data, axis=0)
        self.anomalous_cov = np.cov(self.anomalous_data.T) 
        self.anomalous_centroid = self.anomalous_mean 
        self.centroids[1] = self.anomalous_centroid

    def dropData(self):
      if len(self.baseline_data < 10000):
        return
      linalgInv = np.linalg.pinv(self.baseline_cov + 0.001 * np.eye(self.dimension))
      # Calculate distances for all points in baseline_data
      distances = np.array([np.sqrt(np.sum((np.reshape(point, (1, self.dimension)) - self.baseline_mean) 
                                          @ linalgInv * (np.reshape(point, (1, self.dimension)) - self.baseline_mean), axis=1))  
                            for point in self.baseline_data])
      # Sort distances and get the indices of the sorted array
      sorted_indices = np.argsort(distances)
      # Select the elements with the smallest distances
      num_elements = int(len(self.baseline_data) / 2)
      selected_indices = sorted_indices[:num_elements]
      # Create a new array with only the selected elements
      new_baseline_data = np.array([self.baseline_data[i].squeeze() for i in selected_indices])
      # Update self.baseline_data
      self.baseline_data = new_baseline_data


    def append_baseline_anomalous(self, baseline_distance, point):
      if baseline_distance <= 0.95 * self.threshold:
        self.baseline_data = np.vstack((self.baseline_data,point))
        return
      if baseline_distance > 1.0 * self.threshold:
        if len(self.anomalous_data) == 0:
          self.anomalous_data = point
        else:
          self.anomalous_data = np.vstack((self.anomalous_data,point))
        return

    def fit(self, data):
        n = data.shape[0]  # number of data points
        d = data.shape[1]  # number of features
        num_iter = 0

        # iterate until convergence or max_iterations reached
        for iteration in range(self.max_iterations):
            num_iter += 1
            print("iteration # ",str(num_iter))
            fit_clusters = [[] for _ in range(2)]
            # update the baseline and anomalous statistics
            if (num_iter % self.update_after_iteration == 0):
                self.update()
            
            # assign each data point to the nearest centroid
            for i in range(len(data)):
                point = data[i,:]
                point = np.reshape(point, (1, d))
                # compute Mahalanobis distance to baseline centroid
                #linalgInv = np.linalg.pinv(self.baseline_cov + 0.001 * np.eye(self.dimension))
                identity_matrix = np.eye(self.dimension)
                regularization_term = 0.001 * identity_matrix
                covariance_matrix = self.baseline_cov + regularization_term
                linalgInv = np.linalg.pinv(covariance_matrix)
                baseline_distance = np.sqrt(np.sum((np.reshape(point, (1, self.dimension)) - self.baseline_mean) 
                                             @ linalgInv * (np.reshape(point, (1, self.dimension)) - self.baseline_mean), axis=1))
                self.append_baseline_anomalous(baseline_distance,point)
                # if there are info about the anomalous cluster
                if self.anomalous_centroid is not None:
                  # compute Mahalanobis distance to anomalous centroid
                  linalgInv = np.linalg.pinv(self.anomalous_cov + 0.001 * np.eye(self.dimension))
                  anomalous_distance = np.sqrt(np.sum((point - self.anomalous_mean) @ linalgInv * (point - self.anomalous_mean), axis=1)) 
                  # if distance to the anomalous distribution is smaller than to the baseline, then consider the point an anomaly
                  if anomalous_distance <= baseline_distance * 1.05:
                    fit_clusters[1].append(i) 
                    continue
              
                # consider point benign if distance to baseline centroid is below threshold
                if baseline_distance <= self.threshold:
                    fit_clusters[0].append(i) 
                else:
                    fit_clusters[1].append(i) 

            self.update()
            if num_iter > 5:
              if set(self.clusters[0]) == set(fit_clusters[0]) and set(self.clusters[1]) == set(fit_clusters[1]):
                print("Converged!")
                break
            self.clusters  = fit_clusters


    def predict(self, new_data):
        # classify new data as baseline or anomalous based on distances and baseline distance threshold
        baseline_distances=[]
        anomalous_distances=[]
        linalgInv = np.linalg.pinv(self.baseline_cov + 0.001 * np.eye(self.dimension))
        anom_linalgInv = None
        if self.anomalous_centroid is not None:
            anom_linalgInv = np.linalg.pinv(self.anomalous_cov + 0.001 * np.eye(self.dimension))
        for i in range(len(new_data)):
            point = new_data[i,:]
            point = np.reshape(point, (1, new_data.shape[1]))
            dist = np.sqrt(np.sum((point - self.baseline_mean) @ linalgInv * (point - self.baseline_mean), axis=1))
            if dist <= 0.97 * self.threshold:
              self.baseline_data = np.vstack((self.baseline_data,point))
            baseline_distances.append(dist)
            if self.anomalous_centroid is not None:
              anomalous_distances.append(np.sqrt(np.sum((point - self.anomalous_mean) @ anom_linalgInv * (point - self.anomalous_mean), axis=1)))
        baseline_distances = np.array(baseline_distances).flatten()
        if len(anomalous_distances) > 0:
          anomalous_distances = np.array(anomalous_distances).flatten()
        is_baseline = baseline_distances < self.threshold
        if len(anomalous_distances) > 0:
          is_baseline = np.logical_or(is_baseline,np.less(baseline_distances*1.05,anomalous_distances).astype(int))
        # assign each new data point to the nearest centroid
        closest_idxs = np.zeros(new_data.shape[0], dtype=int)
        closest_idxs[is_baseline] = 0
        closest_idxs[~is_baseline] = 1
        # return the assigned clusters (0 for benign, 1 for anomalous)
        return np.array([1 if idx != 0 else 0 for idx in closest_idxs])

    def export_stats(self):
      linalgInv_baseline = np.linalg.pinv(self.baseline_cov + 0.001 * np.eye(self.dimension))
      linalgInv_anomalous = np.linalg.pinv(self.anomalous_cov + 0.001 * np.eye(self.dimension))
      return self.threshold, self.baseline_mean, linalgInv_baseline, self.anomalous_mean, linalgInv_anomalous


    def merge(self, worker_baseline_means, worker_anomalous_means, distances_inline, distances_outline):
        self.update()
        self.baseline_mean =  np.mean([self.baseline_mean,worker_baseline_means], axis = 0) 
        self.anomalous_mean =  np.mean([self.anomalous_mean,worker_anomalous_means], axis = 0) 
        linalgInv = np.linalg.pinv(self.baseline_cov + 0.001 * np.eye(self.dimension))
        # compute distances from baseline points to baseline centroid
        distances = np.array([np.sqrt(np.sum((np.reshape(point, (1, self.dimension)) - self.baseline_mean) 
                                             @ linalgInv * (np.reshape(point, (1, self.dimension)) - self.baseline_mean), axis=1))  
                              for point in self.baseline_data])
        distances = np.vstack((distances, distances_inline))
        distances = distances.flatten()
        min_outline = np.min(distances_outline)
        # Create a boolean mask for the distances array
        mask = np.where(distances < min_outline)[0]
        # Filter the distances array using the boolean mask
        distances = distances[mask]
        # compute threshold based on percentile of distances
        self.threshold = np.percentile(distances, self.percentile)       

class WorkerKmeans:
      def __init__(self, threshold, baseline_mean, linalgInv_baseline, anomalous_mean, linalgInv_anomalous):
        self.threshold = threshold
        self.baseline_mean = baseline_mean
        self.linalgInv_baseline = linalgInv_baseline
        self.anomalous_mean = anomalous_mean
        self.linalgInv_anomalous = linalgInv_anomalous
        self.baseline_distances = []
        self.anomalous_distances = []
        self.baseline_data = np.empty((0, baseline_mean.shape[0]), float)
        self.anomalous_data = np.empty((0, anomalous_mean.shape[0]), float)
        self.dimension = baseline_mean.shape[0]

      def predict(self, new_data):
        # classify new data as baseline or anomalous based on distances and baseline distance threshold
        point_distances=[]
        for i in range(len(new_data)):
            point = new_data[i,:]
            point = np.reshape(point, (1, new_data.shape[1]))
            distance = np.sqrt(np.sum((point - self.baseline_mean) @ self.linalgInv_baseline * (point - self.baseline_mean), axis=1))
            if distance < self.threshold:
              self.baseline_data = np.vstack((self.baseline_data, point))
              self.baseline_distances.append(distance)
            else:
              self.anomalous_data = np.vstack((self.anomalous_data, point))
              self.anomalous_distances.append(distance)
            point_distances.append(distance)

        point_distances = np.array(point_distances).flatten()
        is_baseline = point_distances < self.threshold
        # assign each new data point to the nearest centroid
        closest_idxs = np.zeros(new_data.shape[0], dtype=int)
        closest_idxs[is_baseline] = 0
        closest_idxs[~is_baseline] = 1
        # return the assigned clusters (0 for benign, 1 for anomalous)
        return np.array([1 if idx != 0 else 0 for idx in closest_idxs])   


      def export_stats(self):
        baseline_data_mean = np.mean(self.baseline_data, axis=0)
        self.baseline_mean =  np.mean([self.baseline_mean,baseline_data_mean], axis = 0) 
        anomalous_data_mean = np.mean(self.anomalous_data, axis=0)
        self.anomalous_mean =  np.mean([self.anomalous_mean,anomalous_data_mean], axis = 0)        
        return self.baseline_mean, self.anomalous_mean, np.array(self.baseline_distances), np.array(self.anomalous_distances)


      def merge(self, threshold, coordinator_baseline_mean, linalgInv_baseline, coordinator_anomalous_mean, linalgInv_anomalous):
          self.baseline_mean =  np.mean([self.baseline_mean,coordinator_baseline_mean], axis = 0) 
          self.anomalous_mean =  np.mean([self.anomalous_mean,coordinator_anomalous_mean], axis = 0) 
          self.linalgInv_baseline = linalgInv_baseline
          self.linalgInv_anomalous = linalgInv_anomalous
          self.threshold = threshold