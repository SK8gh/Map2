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

dict_coordinates = np.load(conf.path_coordinates, allow_pickle=True).reshape(1)[0]
places_names = list(dict_coordinates.keys())


class GetDistances:
    def __init__(self, coordinates, places):
        self.heap_elt_tries = {}
        self.coordinates = coordinates
        self.places = places

        self.distances = np.zeros([len(places), len(places)])

    def add_attr_heap(self):
        heap = Heap()
        for i, place_i in enumerate(self.places):
            for j, place_j in enumerate(self.places):
                if i > j:
                    heap.heappush((i, j))
                    self.heap_elt_tries[(i, j)] = 0

        self.heap = heap

    def thread_target(self):
        if not self.heap:
            return

        while self.heap:
            index1, index2 = self.heap.heappop()
            try:
                self.heap_elt_tries[(index1, index2)] += 1

                logger = logging.getLogger('WazeRouteCalculator.WazeRouteCalculator')
                logger.setLevel(logging.DEBUG)
                handler = logging.StreamHandler()
                logger.addHandler(handler)

                from_address = self.coordinates[self.places[index1]]
                to_address = self.coordinates[self.places[index2]]
                region = 'EU'
                route = WazeRouteCalculator.WazeRouteCalculator(from_address, to_address, region)

                self.distances[index1, index2] = route.calc_route_info()[0]

            except Exception as e:
                print(f"An exception was raised for places : {self.places[index1], self.places[index2]}")
                if self.heap_elt_tries[(index1, index2)] < 3:
                    self.heap.heappush((index1, index2))
        time.sleep(0.1)

    def run_threads(self, nb_threads):
        threads = []

        self.add_attr_heap()

        print(type(self.heap))

        for _ in range(nb_threads):
            t = threading.Thread(target=GetDistances.thread_target, args=(self,), daemon=True)
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()


def go():
    number_threads = 25
    GetDistancesInstance = GetDistances(dict_coordinates, places_names)
    GetDistancesInstance.run_threads(number_threads)
    np.save('/Users/miller/Desktop/MAP2/data/distances.npy', GetDistancesInstance.distances)

go()

dist = np.load('/Users/miller/Desktop/MAP2/data/distances.npy', allow_pickle=True)

for row in dist:
    print(row, "\n")
