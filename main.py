from mapping import *


def create_distances_file(vehicle_type):
    print(f"Getting distances for vehicle_type = {vehicle_type}\n")
    number_threads = 50
    get_distance_instance = GetDistances(dict_coordinates, places_names, vehicle_type=vehicle_type)
    get_distance_instance.run_threads(number_threads)
    np.save(f"/Users/miller/Desktop/MAP2/data/distances_{vehicle_type}.npy", get_distance_instance.distances)


if __name__ == '__main__':
    create_distances_file("TAXI")
    create_distances_file("MOTORCYCLE")
