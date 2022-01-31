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
    def __init__(self, coordinates, places, vehicle_type):
        self.vehicle_type = vehicle_type
        self.coordinates = coordinates
        self.places = places
        self.heap = None
        self.redo = {}

        self.times = np.zeros([len(places), len(places)])
        self.speeds = np.zeros([len(places), len(places)])

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

        vehicle_type = self.vehicle_type

        while self.heap:
            index1, index2 = self.heap.heappop()
            try:
                from_address = self.coordinates[self.places[index1]]
                to_address = self.coordinates[self.places[index2]]
                region = 'EU'
                route = WazeRouteCalculator.WazeRouteCalculator(from_address, to_address, region, vehicle_type)

                self.times[index1, index2] = route.calc_route_info()[0]
                self.redo[(index1, index2)] += 1

                if route.calc_route_info()[0] != 0:
                    self.speeds[index1, index2] = route.calc_route_info()[1] / route.calc_route_info()[0]

                print(f"Distance from {self.places[index1]} to {self.places[index2]} = {self.times[index1, index2]}")

            except Exception:
                print(f"Exception raised for {self.places[index1]} to {self.places[index2]}")
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

    def average_speed(self):
        speed_values = [elt for elt in self.speeds.reshape(self.speeds.shape[0] ** 2) if elt != 0]
        return np.mean(speed_values)


def get_positions(vehicle_type):
    times = np.load(f"/Users/miller/Desktop/MAP2/data/times_{vehicle_type}.npy", allow_pickle=True)
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


def plot_map(vehicle_type, labelling=False, radius=None, place=None):
    if radius is not None and place is None:
        raise ValueError("Must provide a place name (str) if radius = True")

    pts_values = get_positions(vehicle_type).x
    len_values = int(len(pts_values) / 2)
    x, y = pts_values[:len_values], pts_values[len_values:]

    # Plot orientation, comparing the positions of Montrouge and Pasteur
    index_pasteur, index_montrouge = places_names.index('Pasteur'), places_names.index('Montrouge')

    if x[index_montrouge] < x[index_pasteur]:
        x = [- elt for elt in x]
    if y[index_montrouge] > y[index_pasteur]:
        y = [- elt for elt in y]

    fig, ax = plt.subplots()

    color = 'purple' if vehicle_type == "TAXI" else 'red'

    for i in range(len_values):
        ax.scatter(x[i], y[i], color=color)

    if labelling:
        for i in range(len_values):
            ax.annotate(places_names[i], (x[i], y[i]))

    if radius:
        place_index = places_names.index(place)
        circle_plot = Circle((x[place_index], y[place_index]), 10, facecolor='purple', alpha=0.15, lw=1)
        ax.add_patch(circle_plot)
        ax.legend([circle_plot], ['10 minutes away from ' + place], loc='upper left')

    plt.show()


# plot_map("TAXI", labelling=True)

# plot_map("MOTORCYCLE", labelling=True)

# plot_map("TAXI", labelling=False)

# plot_map("MOTORCYCLE", labelling=False)
