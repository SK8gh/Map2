# Imports

from matplotlib.patches import Circle
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import WazeRouteCalculator
from heap import Heap
import numpy as np
import threading
import time


import configuration as conf

import warnings
warnings.filterwarnings("ignore")

dict_coordinates_load = np.load(conf.path_coordinates, allow_pickle=True).reshape(1)[0]
places_names = conf.places_sub_list

# Reducing the size of the coordinates dictionary
dict_coordinates = {place: dict_coordinates_load[place] for place in places_names}


class GetDistances:
    def __init__(self, coordinates, places):
        self.coordinates = coordinates
        self.places = places
        self.redo = {}

        self.distances = np.zeros([len(places), len(places)])

    def add_attr_heap(self):
        heap = Heap()
        for i, place_i in enumerate(self.places):
            for j, place_j in enumerate(self.places):
                if i > j:
                    heap.heappush((i, j))
                    self.redo[(i, j)] = 0
        self.heap = heap

    def thread_target(self):
        if not self.heap:
            return

        while self.heap:
            index1, index2 = self.heap.heappop()
            try:
                from_address = self.coordinates[self.places[index1]]
                to_address = self.coordinates[self.places[index2]]
                region = 'EU'
                route = WazeRouteCalculator.WazeRouteCalculator(from_address, to_address, region)

                self.distances[index1, index2] = route.calc_route_info()[0]
                self.redo[(index1, index2)] += 1

                print(f"Distance from {self.places[index1]} to {self.places[index2]} = {self.distances[index1, index2]}")

            except Exception:
                if self.redo[(index1, index2)] < 2:
                    self.heap.heappush((index1, index2))

            finally:
                time.sleep(2)

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


def plot_map(labelling=False, radius=None, place=None):
    if radius is not None and place is None:
        raise ValueError("Must provide a place name (str) if radius = True")

    pts_values = get_positions().x
    len_values = int(len(pts_values) / 2)
    x, y = pts_values[:len_values], pts_values[len_values:]

    # len_values = 26

    # x = [8.91254995, -12.17516123, -15.72134663, 9.28512626, 4.57139819,
    #         8.82307475, -1.05812682, -8.61957861, 7.39634822, 5.68787915,
    #         -8.21628939, -9.13702954, -7.79833336, 8.04354382, 0.26302326,
    #         -5.46894509, 0.70362746, 6.97329319, -11.94363191, -6.33307449,
    #         3.04327222, -3.61078547, -5.16059223, 7.53090863, 6.93802962,
    #         -1.9167224]

    # y = [4.80762397, -0.37056648, 1.66658083, 1.3119542, 0.79731625,
    #         -0.54921391, 10.69191471, 5.52590189, 11.51914474, 7.53897132,
    #         9.58941228, -7.98648997, -8.04476173, 3.77945517, 12.77069461,
    #         13.536569, -11.16410718, -6.88493783, -5.51220762, 1.09621636,
    #         11.23200394, -3.8676266, 11.54619351, 0.84790804, -5.26445966,
    #         -9.55128235]

    # Plot orientation, comparing the positions of Montrouge and Pasteur
    index_pasteur, index_montrouge = places_names.index('Pasteur'), places_names.index('Montrouge')

    if x[index_pasteur] < x[index_montrouge]:
        x = [- elt for elt in x]
    if y[index_pasteur] > y[index_montrouge]:
        y = [- elt for elt in y]

    fig, ax = plt.subplots()

    for i in range(len_values):
        ax.scatter(x[i], y[i], color='purple')

    if labelling:
        for i in range(len_values):
            ax.annotate(places_names[i], (x[i], y[i]))

    if radius:
        place_index = places_names.index(place)
        circle_plot = Circle((x[place_index], y[place_index]), 10, facecolor='purple', alpha=0.15, lw=1)
        ax.add_patch(circle_plot)
        ax.legend([circle_plot], ['10 minutes away from ' + place], loc='upper left')

    plt.show()


# plot_map(labelling=False)

plot_map(labelling=True)
