# This data structure is an append-only list, that has to be initialized as an empty list []

class Heap:
    """
    Heap class, to fix the 'attribute problem' in the GetDistances class
    """
    def __init__(self):
        self.list = []

    def __repr__(self):
        if len(self.list) != 0:
            string = "["
            for elt in self.list:
                string += (str(elt) + ', ')
            string = string[:-2]
            string += "]"
            return string
        else:
            return "[]"

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        return self.list[index]

    def heappush(self, elt):
        self.list.append(elt)

    def heappop(self):
        if not self.list:
            raise IndexError("Can't call the pop method on an empty heap")

        r = self.list[-1]
        self.list = self.list[:-1]
        return r

