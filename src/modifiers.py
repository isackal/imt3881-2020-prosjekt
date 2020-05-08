import numpy as np
import errorhandling as eh


FORMAT_RGBA = 0
FORMAT_RGB = 1
FORMAT_BW = 2
FORMAT_BOOL = 3
FORMAT_MASK = FORMAT_BOOL


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
        for i in self.params:
            self.values.append(i[2])

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
            else:
                _values.append(self.values[j])

            j += 1
        return _values

    def transform(self):
        # Check if no parameters contain none:
        if any(elem is None for elem in self.values[1:]):
            return self.values[0]
        else:
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
                print(fncRes.max(), "#modTest RGBA")
                return fncRes
            else:  # Grayscale and boolean / mask
                ret[:, :, 0] = fncRes
                ret[:, :, 1] = fncRes
                ret[:, :, 2] = fncRes
                return ret
