import numpy as np

class Modifier:
    def __init__(self):
        """
        Contains:
            name:
                name of modifier
            function:
                The function that will modify an image
            params:
                list of prameters. Each parameter is a tuple on the form:
                    (parameter_name, parameter_type, parameter_init_value)
                    Examples:
                    ("threshold",float,0.5)
                    ("image",np.ndarray,None)
        """
        self.name = "modifier"
        self.function = lambda x: 2*x
        self.params = [("source", int, 0)]
        self.values = [1]
        self.initDefaultValues()

    def initDefaultValues(self):
        self.values = []
        for i in self.params:
            self.values.append(i[2])

    def transform(self):
        # Check if no parameters contain none:
        if any(elem is None for elem in self.values[1:]):
            return self.values[0]
        else:
            return np.clip(self.function(*self.values), 0, 255).astype(np.uint8)
