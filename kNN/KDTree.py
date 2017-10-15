class KDTree:
    class Node:
        def __init__(self):
            self.left = None
            self.right = None

        def print(self):
            pass

        def search(self, value):
            pass

    class Leaf(Node):
        def __init__(self, values):
            super().__init__()
            self.values = values

        def print(self):
            return self.values

        def search(self, value):
            return self.values

    class InterLeaf(Node):
        def __init__(self, delimeter, value):
            super().__init__()
            self.delimeter = delimeter
            self.value = value

        def print(self):
            if self.delimeter == 'x':
                return '{} {}'.format(self.delimeter, self.value[0])
            else:
                return '{} {}'.format(self.delimeter, self.value[1])

        def search(self, value):
            if self.delimeter == 'x':
                if value[0] < self.value[0]:
                    return self.left.search(value)
                else:
                    return self.right.search(value)
            else:
                if value[1] < self.value[1]:
                    return self.left.search(value)
                else:
                    return self.right.search(value)


    def __init__(self, values, k):
        self.k = k
        self.root = self.split(values, 'x')

    def split(self, values, delimeter):
        if len(values) < self.k * 2:
            return KDTree.Leaf(values)
        else:
            if delimeter == 'x':
                values = sorted(values, key=lambda val: val[0])
            else:
                values = sorted(values, key=lambda val: val[1])

            middle_pos = int(len(values) / 2)
            middle = values[middle_pos]
            intermediate = KDTree.InterLeaf(delimeter, middle)

            if delimeter == 'x':
                new_delim = 'y'
            else:
                new_delim = 'x'

            intermediate.left = self.split(values[0: middle_pos], new_delim)
            intermediate.right = self.split(values[middle_pos:len(values)], new_delim)
            return intermediate

    def search(self, point):
        return self.root.search(point)

    def print(self):
        self.print_node(self.root, '', '')

    def print_node(self, node, prefix, side):
        print(prefix, side, node.print())
        prefix = prefix + '\t'
        if node.left is not None:
            self.print_node(node.left, prefix, 'left')
        if node.right is not None:
            self.print_node(node.right, prefix, 'right')
