
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
        self.name="modifier"
        self.function=lambda x : 2*x
        self.params=[ ("source",int,0) ]
        self.values=[1] #Just to make intelisense shut up about line 24 error
        self.initDefaultValues()
    def initDefaultValues(self):
        self.values=[]
        for i in self.params:
            self.values.append(i[2])
    def transform(self):
        return self.function( *self.values )