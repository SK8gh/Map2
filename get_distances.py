# Imports

from WazeRouteCalculator import logging
import WazeRouteCalculator

from datetime import datetime as datetime
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from time import sleep
import numpy as np
import threading
import shutup
import heapq
import time
import sys
import os

import configuration as conf
from heap import Heap

import warnings
warnings.filterwarnings("ignore")

dict_coordinates_load = np.load(conf.path_coordinates, allow_pickle=True).reshape(1)[0]
places_names = conf.places_sub_list

# Reducing the size of the coordinates dictionary
dict_coordinates = {place : dict_coordinates_load[place] for place in places_names}


class GetDistances:
    def __init__(self, coordinates, places):
        self.coordinates = coordinates
        self.places = places

        self.distances = np.zeros([len(places), len(places)])
        self.count = 0

    def add_attr_heap(self):
        heap = Heap()
        for i, place_i in enumerate(self.places):
            for j, place_j in enumerate(self.places):
                if i > j:
                    heap.heappush((i, j))
        self.heap = heap

    def thread_target(self):
        if not self.heap:
            return

        while self.heap:
            index1, index2 = self.heap.heappop()
            self.count += 1
            try:
                from_address = self.coordinates[self.places[index1]]
                to_address = self.coordinates[self.places[index2]]
                region = 'EU'
                route = WazeRouteCalculator.WazeRouteCalculator(from_address, to_address, region)

                self.distances[index1, index2] = route.calc_route_info()[0]

                print(f"Distance from {self.places[index1]} to {self.places[index2]} = {self.distances[index1, index2]}")

            except Exception:
                pass

            finally:
                time.sleep(1)

    def run_threads(self, nb_threads):
        threads = []

        self.add_attr_heap()

        for _ in range(nb_threads):
            t = threading.Thread(target=GetDistances.thread_target, args=(self,), daemon=True)
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()


def get_positions():
    times = np.load('/Users/miller/Desktop/MAP2/data/distances.npy', allow_pickle=True)
    len_times = len(times)

    def get_error(args):
        error = 0
        x, y = args[:len_times], args[len_times:]

        for i_1, x_val in enumerate(x):
            for i_2, y_val in enumerate(y):
                if times[i_1, i_2] != 0:
                    error += abs((x[i_1] - x[i_2]) ** 2 + (y[i_1] - y[i_2]) ** 2 - times[i_1, i_2] ** 2)
        return error

    args0 = np.zeros([2 * len_times])

    min_result = minimize(get_error, args0, method="Powell", tol=1e-6)
    return min_result


class PlotMap:
    @staticmethod
    def plot():
        pts_values = get_positions().x
        len_values = int(len(pts_values) / 2)

        x, y = pts_values[:len_values], pts_values[len_values:]

        fig, ax = plt.subplots()

        for i in range(len_values):
            ax.scatter(-x[i], -y[i], color='purple')

        for i in range(len_values):
            ax.annotate(places_names[i], (-x[i], -y[i]))

        plt.legend()
        plt.show()


if __name__ == '__main__':

    PlotMap.plot()



