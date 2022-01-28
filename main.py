from get_distances import *


def go():
    number_threads = 50
    get_distance_instance = GetDistances(dict_coordinates, places_names)
    get_distance_instance.run_threads(number_threads)
    np.save('/Users/miller/Desktop/MAP2/data/distances.npy', get_distance_instance.distances)


if __name__ == '__main__':
    # go()

    dist = np.load('/Users/miller/Desktop/MAP2/data/distances.npy', allow_pickle=True)

    c = 0
    for i in range(26):
        for j in range(26):
            if dist[i,j] != 0:
                c += 1
    print(c)
