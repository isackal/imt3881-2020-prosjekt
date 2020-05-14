import numpy as np
import sys


# Enums Format:
FORMAT_RGBA = 0
FORMAT_RGB = 1
FORMAT_BW = 2
FORMAT_BOOL = 3
FORMAT_MASK = FORMAT_BOOL

# The following enums are prime numbers, such that
# res = product(enums), if ( res % enum == 0 ) => enum is set
FLAG_CAN_BE_NULL = 3


class Limit:
    def __init__(self, _type, _min, _max):
        self._type = _type
        self._min = _min
        self._max = _max

    def clamp(self, val):
        if val > self._max:
            return self._max
        elif val < self._min:
            return self._min
        else:
            return val


def clamp(lim, val):
    return lim.clamp(val)


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
        self.function = lambda: 1
        self.params = [("source", int, 0)]
        self.flags = [1]  # Flags for each parameter
        """
        If the second argument (index 1) is np.ndimage, a 4th argument is
        requiered. This arguments is an integers that tells
        the program what kind of format the input
        image is.
        The formats are:
        0: RGBA standard.
        1: RGB  RGB and ignores the alpha channel
        2: BW   black white images
        3: BOOL bolean images, or masks
        The same integer formats are used for the outputs.
        The default will be 1.
        """
        self.outputFormat = 1  # RGB format.
        self.values = [1]
        self.initDefaultValues()

    def initDefaultValues(self):
        self.values = []
        self.flags = []
        for i in self.params:
            self.values.append(i[2])
            self.flags.append(1)

    def formattedValues(self):
        """
        The functions takes its values and formats the np.ndimages so they
        will be in the wanted format, assuming the input always is RGBA8888 and
        puts the values between 0 - 1.
        """
        _values = []
        j = 0
        for i in self.params:
            if (i[1] is np.ndarray):
                # Start with the most common types:
                if self.values[j] is None:
                    _values.append(None)
                    j += 1
                    continue
                if i[3] == 1:  # RGB 3 channels
                    _values.append(self.values[j][:, :, :3].astype(float)/255)
                elif i[3] == 0:  # RGBA 4 channels
                    _values.append(self.values[j].astype(float)/255)
                elif i[3] == 2:  # BW 1 channel
                    # NB: The program will assume the image is all ready
                    #   Black and white, meaning it will not take the sum
                    #   of the channels and divide by 3, as it assumes these
                    #   have the same values. This means it only needs to take
                    #   the first channel.
                    _values.append(self.values[j][:, :, 0].astype(float)/255)
                else:
                    # Does the same assumtion as above.
                    _values.append(self.values[j][:, :, 0] > 127)
            elif type(i[1]) is Limit:
                self.values[j] = clamp(i[1], self.values[j])
                _values.append(self.values[j])
            else:
                _values.append(self.values[j])
            j += 1
        return _values

    def transform(self):
        global FLAG_CAN_BE_NULL
        # Check if no parameters contain none:
        for i in range(1, len(self.values)):
            if (
                (self.values[i] is None) and
                (self.flags[i] % FLAG_CAN_BE_NULL != 0)
            ):
                return self.values[0]  # return just source image
        # Format values:
        _values = self.formattedValues()
        ret = None
        # Function Result:
        fncRes = np.clip(
                self.function(*_values)*255, 0, 255
                ).astype(np.uint8)
        if self.outputFormat != 0:
            ret = np.ones(
                (fncRes.shape[0], fncRes.shape[1], 4), np.uint8
            )
            ret[:, :, 3] = ret[:, :, 3]*255  # Alpha channel
        if self.outputFormat == 1:  # RGB
            ret[:, :, :3] = fncRes
            return ret
        elif self.outputFormat == 0:  # RGBA
            return fncRes
        else:  # Grayscale and boolean / mask
            ret[:, :, 0] = fncRes
            ret[:, :, 1] = fncRes
            ret[:, :, 2] = fncRes
            return ret

    def setFlags(self, param, *args):
        """
        Set flags in the modifier

        Parameters
        ----------

        param   :   <int>
                    index of which parameter to set the flag
        *args   :   <int>
                    list of modifier flags starting with FLAG_
        """
        prd = 1
        for i in args:
            prd *= i
        self.flags[param] = prd
