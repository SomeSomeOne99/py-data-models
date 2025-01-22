from numpy import array, ndarray # Array
from numpy.linalg import inv # Matrix inversion function
class Matrix():
    def __init__(self, data):
        if type(data) == array or type(data) == ndarray:
            self.data = data # Already correct type
        elif type(data) == list:
            self.data = array(data) # Convert list to array
        elif type(data) == int or type(data) == float:
            self.data = array([data]) # Create 1x1 array
        else:
            raise TypeError("Data must be a list, number or array, type was " + str(type(data))) # Invalid type
    def __matmul__(self, other):
        return Matrix(self.data @ other.data)
    def __mul__(self, other):
        return Matrix(self.data * other.data)
    def __add__(self, other):
        return Matrix(self.data + other.data)
    def __sub__(self, other):
        return Matrix(self.data - other.data)
    def __repr__(self):
        return str(self.data)
    def __getitem__(self, key):
        return Matrix(self.data[key])
    def __setitem__(self, key, value):
        self.data[key] = value.data
    def __len__(self):
        return len(self.data)
    def T(self):
        return Matrix(self.data.T)
    def I(self):
        return Matrix(inv(self.data))