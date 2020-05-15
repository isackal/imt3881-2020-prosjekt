import sys

import numpy as np
import imageio as im
from math import log
import hdr

# Import GUI:
import PyQt5.QtWidgets as wd
import PyQt5.QtCore as cr
from PyQt5 import QtGui
import customWidgets as cst

# Import Image interactivity modifiers:
import imageMath

# Import Modifiers:
import bitcrusher  # testmodule, can be kept
import contrast
import blurring
import meanimage  # testmodule, delete later
import kantBevGlatting as kbg
import colortogray as ctg
import demosaic
import inpaint
# import anonymiser
import cloning
# import poisson

# List of modifiers that are being used
MODIFIERS = [
    bitcrusher.Bitcrusher,
    contrast.Contrast,
    blurring.Blurring,
    meanimage.Meanimage,
    kbg.KantbevarendeGlatting,
    ctg.ColorToGray,
    demosaic.Demosaic,
    inpaint.Inpaint,
    # anonymiser.Anonymisering,
    cloning.Cloning,
    imageMath.WeightedAddition,
    imageMath.FitSize,
    imageMath.Offset,
    imageMath.ColorToGrayWeighted,
    imageMath.VecotrGray,
    imageMath.Binary,
    imageMath.Invert,
    imageMath.Multiplication,
    imageMath.Exponent,
    imageMath.Diff,
    imageMath.Normalize,
    imageMath.ColorFilter,
    imageMath.Mosaic,
    imageMath.Noise,
    imageMath.FindEdges,
    imageMath.Crop
]

_NUMBERED_IMAGE = 0  # used for defualt naming of images

SELECTED = None  # pointer to current selected image

BIG_IMAGE = None  # pointer to the big screen image

GLOBAL_IMAGES = []  # List of all image groups

WINDOW = None  # Pointer to main window


def upDownDelToolbar(
    upFunc,
    downFunc,
    delFunc
):
    """
    This function creates a toolbar with buttons that move
    something up or down in a "hirarchy" or delete it.
    """
    toolbar = cst.Packlist(None, wd.QHBoxLayout)
    _up = cst.MiniButton(
        None,
        "../ui/up.png",
        "Move modifier up in the pipeline."
    )
    _down = cst.MiniButton(
        None,
        "../ui/down.png",
        "Move modifier down in the pipeline."
    )
    _del = cst.MiniButton(
        None,
        "../ui/delete.png",
        "Delete modifier"
    )

    # Sets the functions of each button:
    _up.mousePressEvent = upFunc
    _down.mousePressEvent = downFunc
    _del.mousePressEvent = delFunc

    # Adds them to the tool bar
    toolbar.addWidget(_up)
    toolbar.addWidget(_down)
    toolbar.addWidget(_del)

    return toolbar


def loadHDRDialog():
    """
    Loads a HDR image through a Qt Dialog

    Returns
    -------
    hdr_img : <numpy.ndarray> if successfull
    None if not successful
    """
    fd = wd.QFileDialog()
    fd.setFileMode(wd.QFileDialog.Directory)
    if fd.exec_() and hdr.validateSelection(fd.selectedFiles()[0]):
        images, times = hdr.loadImages(fd.selectedFiles()[0])
        _img = hdr.hdr(images, times, 100)
        return _img
    else:
        return None


class Collapser(wd.QWidget):
    """
    The collapser widget is used to hide and show widgets
    in the collapser section.
    """
    def __init__(self,  parent):
        wd.QWidget.__init__(self,  parent)
        layout = wd.QVBoxLayout()  # main layout
        top = wd.QGroupBox()
        self.topl = wd.QHBoxLayout()  # top layout of the collapser
        self.topl.setContentsMargins(8,  0,  8,  0)  # Reduse margins
        self.topl.setAlignment(cr.Qt.AlignLeft)  # Set left alignment
        self.topl.setStretch(0,  0)
        top.setLayout(self.topl)

        # Creating button and add them to layer:
        self.minimizeButton = wd.QPushButton('+',  self)
        self.minimizeButton.setFixedHeight(32)
        self.minimizeButton.setFixedWidth(32)
        self.minimizeButton.mousePressEvent = self.toggle
        self.topl.addWidget(self.minimizeButton)
        self.title = "Modify"  # default title
        self.label = wd.QLabel(self.title,  self)  # Displayed title
        self.topl.addWidget(self.label)  # Add the title to the top layer
        top.setFixedHeight(48)
        self.vgrp = wd.QGroupBox()  # hide/show part
        cLayout = wd.QVBoxLayout()  # Layout of hide/show part
        self.content = cst.Packlist(self.vgrp)  # content of the collapser
        cLayout.addWidget(self.content)  # Add the content to the layout
        self.vgrp.setLayout(cLayout)  # Set the layout of the hide/show part
        self.mainUI()  # Calls the "overrided function" to create the content
        self.setLayout(layout)  # Sets the main layout
        layout.addWidget(top)  # adds top part
        layout.setSpacing(0)   # set spacing between them to be small
        layout.addWidget(self.vgrp)  # adds bottom part
        # Sets the default to be hidden:
        self.open = False
        self.vgrp.setHidden(True)

    def toggle(self,  event):
        """
        Hides and shows content of the collapser.
        """
        if self.open:
            self.vgrp.setHidden(True)
            self.minimizeButton.text = "+"
            self.open = False
        else:
            self.vgrp.setHidden(False)
            self.minimizeButton.text = "-"
            self.open = True

    def mainUI(self):
        """
        Default test ui
        """
        for i in range(10):
            btn = wd.QPushButton("Button %2d" % i,  self)
            self.content.addWidget(btn)

    def addToTopUI(self, widget):
        self.topl.addWidget(widget)


class TypeInput(wd.QWidget):
    """
    A more sofisticated input widget that can be an image or
    a number.
    (Current version does not support booleans, so use an integer if needed)
    """
    def __init__(self,  parent,  _type):
        wd.QWidget.__init__(self, parent)
        self.playout = wd.QHBoxLayout()  # main Layout
        self.setLayout(self.playout)
        self.isImage = False  # Whether input is image or numeric
        self._type = _type  # input type, eg. float, int, ndimage
        self.widget = None  # ImageFrame or QtLineEdit (child of QtLineEdit)
        if _type is np.ndarray:
            self.isImage = True
            # Set widget to be an image
            self.widget = imageFrame(
                self,
                loadable=False,
                zoomable=False,
                toolsEnabled=False
            )
            # not possible to use as source, as it would just be a copy
            # of the original image:
            self.widget.sourceAble = False
            self.widget.setSize(128, 128)
            self.widget.mousePressEvent = self.setImage
        else:
            # Input is numeric
            self.widget = cst.NumericInput(_type)
        self.playout.addWidget(self.widget)

    def setText(self, txt):
        """
        Set the text of the line widget or image.
        """
        self.widget.setText(txt)

    def validate(self,  text):
        """
        Set the text of the widget to a valid text, if not
        valid. Else, set the text to the actual input.

        Parameters
        ----------
        text : parameter reserved for PyQt events. (Has to be there)
        """
        self.setText(str(self.getValue()))

    def getValue(self):
        """
        Get the value of the input. Numeric value if type is numeric,
        image data if the widget is an image.
        """
        val = self._type(0)
        if self.isImage:
            if self.widget.src is not None:
                return self.widget.src.picdata
            else:
                return None
        else:
            val = self.widget.getValue()
        return val

    def setImage(self, event=None):
        """
        Sets the image through a QtDialog
        """
        sel = selectImage()  # opens the Dialog
        if sel is not None:
            self.widget.disconnectParent()  # Disconnect from old input
            sel.connect(self.widget)  # Connect new image to this widget
            sel.pipeOnce()  # Get the new image data by from new input
            self.widget.pipe()  # Update pipeline through piping

    def setValue(self, val):
        """
        Set the value of the input.

        Parameters
        ----------

        val :   numpy.ndimage if input is an image
                numeric value if input is numeric
        """
        if self.isImage:
            self.widget.setData(val)
        else:
            self.widget.setText(str(val))

    def onDelete(self):
        """
        Recursively calls onDelete for connected widgets
        """
        if self.isImage:
            self.widget.onDelete()  # call widgets on Delete as well


class ModifierWidget(Collapser):
    """
    The ModifierWidget is the user interface for a
    modifier, eg. Inpaint, ColorToGray
    """
    def __init__(self,  parent,  modf,  reference):
        """
        Parameters
        ----------

        parent      : pointer to which PipelineWidget this belongs to

        modf        : <class : modifiers.Modifier> (child class)

        reference   : image input into this modifier
        """
        self.reference = reference  # image input
        self.modifier = modf()  # Create new instance of
        self.dtas = []  # List of inputs
        self.widgetPipeline = parent
        Collapser.__init__(self,  parent)
        tl = upDownDelToolbar(  # create a toolbar
            self.moveUp,
            self.moveDown,
            self.deleteThis
        )
        self.addToTopUI(tl)
        self.masterLayout = None  # Pointer to its "Packlist"

    def moveUp(self, event):
        """
        Moves this widget up in the hirarchy of its corresponding
        packlist (a widget containing the list of widgets which this widget
        belongs to)
        """
        if self.masterLayout is not None:
            # Return the last neighbour that used to be over
            # itself in the hirarchy:
            neighbour = self.masterLayout.moveUp(self)
            if neighbour is not None:  # If there was a neighbour
                self.swapPlaces(neighbour, self)  # Switch places
                self.reference.pipe()  # Update pipeline

    def moveDown(self, event):
        """
        Moves this widget down in the hirarchy of its corresponding
        packlist (a widget containing the list of widgets which this widget
        belongs to)
        """
        if self.masterLayout is not None:
            # Return the last neighbour that used to be under
            # itself in the hirarchy:
            neighbour = self.masterLayout.moveDown(self)
            if neighbour is not None:  # If there was a neighbour there
                self.swapPlaces(self, neighbour)
                neighbour.reference.pipe()

    def swapPlaces(self, first, second):
        """
        Swaps first DOWN and second UP in the masterlayout paclist
        hirarchy. Will not work in opposite order.
        """
        global BIG_IMAGE
        pf = first.img.src  # parent-image of "first". Assumed to not be None
        # parent second is assumed to me first
        cS = first.masterLayout.next(first)  # child second
        # A-- because it is assumed the order in the layout
        # is all ready changed due to moveUp / moveDown functions.

        # Swap connections:
        second.img.disconnectParent()
        first.img.disconnectParent()

        pf.connect(second.img)  # pf : parents-image of "first" parameter
        second.reference = pf
        second.img.connect(first.img)
        first.reference = second.img

        if cS is not None:
            cS.img.disconnectParent()
            first.img.connect(cS.img)
            cS.reference = first.img
        else:
            # Assuming bigImg is the same accross all images:
            BIG_IMAGE.disconnectParent()
            first.img.connect(BIG_IMAGE)
            self.widgetPipeline.at = first.img

    def onDelete(self):
        """
        Recursively calls onDelete for connected widgets.
        Disconnects connections.
        """
        self.img.disconnectParent()
        self.img.disconnectChildren()
        # Delete input connections:
        for _inp in self.dtas:
            _inp.onDelete()

    def deleteThis(self, event=None):
        """
        Deletes this object. Can be used with an event.
        """
        global BIG_IMAGE
        pimg = self.img.src  # parent-image. Assumed to not be None
        # parent second is assumed to me first
        self.img.disconnectParent()
        # Make sure inputs to clear inputs connections
        # to avoid deleted image reference:
        for _inp in self.dtas:
            _inp.onDelete()
        child = None  # Child next in the "pipeline"
        if self.masterLayout is not None:
            child = self.masterLayout.next(self)  # child second
            if child is not None:
                # reconnect the child to this objects parent image.
                child.img.disconnectParent()
                pimg.connect(child.img)
                child.reference = pimg
            else:
                # if there was no child, the last modifier in
                # the widget pipeline must be set to this ones parent
                # image:
                self.widgetPipeline.at = pimg
                # update the display image to be connected to the new
                # last element:
                BIG_IMAGE.disconnectParent()
                pimg.connect(BIG_IMAGE)
            self.masterLayout.removeWidget(self)  # remove self from its layout
        pimg.pipe()  # Update pipeline
        self.deleteLater()  # Delete this instance

    def mainUI(self):
        """
        Main UI of the Modifier Widget. This overrides its parent class
        mainUI and will be created as the parent class __init__ is called.
        """
        self.img = imageFrame(  # image containing transformed data
            self,
            loadable=False,
            zoomable=False,
            toolsEnabled=False
        )
        self.img.setFixedWidth(128)
        self.img.setFixedHeight(128)
        self.reference.connect(self.img)  # Connect its input image
        self.content.addWidget(self.img)
        self.title = self.modifier.name
        self.label.setText(self.title)
        grd = wd.QGridLayout()
        _row = 0
        self.modifier.values[0] = self.img.picdata  # Set the source value
        # Creates inputs:
        for param in self.modifier.params[1:]:
            lbl = wd.QLabel(param[0],  self)
            inp = TypeInput(self,  param[1])
            inp.setValue(param[2])
            if inp.isImage:
                inp.widget.onPipe = self.onUpdateData
            else:
                inp.widget.textChanged.connect(self.onUpdateText)
            grd.addWidget(lbl,  _row,  0)
            grd.addWidget(inp,  _row,  1)
            self.dtas.append(inp)
            _row += 1
        _add = wd.QWidget(self)
        _add.setLayout(grd)
        self.content.addWidget(_add)
        self.img.modifier = self.modifier  # set the imagemodifier

    def onUpdateData(self, pipeTrace=None):
        """
        Called when data is updated.

        Parameters
        ----------

        pipeTrace   :   <list>
                        list of pointers images all ready involved in the
                        same "pipe round". Used to prevent infinite
                        recursion.
        """
        if pipeTrace is None:
            # aka. new "pipe round:"
            pipeTrace = []  # Creates a new pipetrace if None is provided.
        for i in range(len(self.modifier.values)-1):
            # Get the values from each input:
            self.modifier.values[i+1] = self.dtas[i].getValue()
        self.reference.pipe(pipeTrace)

    def onUpdateText(self, text):
        """
        Called when Text is updated, eg. when text has been edited
        in one of the inputs.
        """
        self.onUpdateData()


class imageFrame(cst.DragableWidget):
    """
    Widget used to hold, transform and pipe image data.
    """
    def __init__(
        self, parent,
        loadable=True,  # Allow loading images
        zoomable=True,  # Allow zoom
        bakeable=True,  # Allow making copies of the result
        deleteable=True,
        toolsEnabled=True  # Allow any tools, such as zoom, load, etc
    ):
        """
        parameters:
        loadable    :   <bool>  whether you can load images into this
        zoomable    :   <bool>  if false, the image will be fixed zoomed.
        bakeable    :   <bool>  if this image can bake its result to new image
        deleteable  :   <bool>  if this image can be deleted
        toolsEnabled:   <bool>  whether any tools will be enabled at all
        """
        cst.DragableWidget.__init__(self, parent)
        self.title = "?"  # default title
        self.top = 0
        self.left = 0
        self.width = 384
        self.height = 240
        self.picdata = None  # <numpy.ndarray> main picture data (np.uint8)
        self.qimg = None  # QtImage
        self.piximg = None  # QtPixmap
        self.imgObject = wd.QLabel(self)  # Image widget
        self.parent = parent
        self.zoomStrength = 2
        self.zoom = 1
        self.z = 0
        self.zoomRange = (-4, 4)  # k^-n to k^m, where k = zoomStrength
        self.fname = ""  # File name origin
        self.zoomSensitivity = 128
        self.srcSet = False  # whether source is set
        self.loadable = loadable
        self.bakeable = bakeable
        self.deleteable = deleteable
        self.toolsEnabled = toolsEnabled
        self.zoomable = zoomable
        self.isSelected = False  # whether this image is selected
        self.recursion = True  # unused varaible
        self.pipes = []  # list of images of which this image is input
        self.src = None  # Which object is piping to this.
        # 0: Binary,  1: Grayscale [0, 1],  2: RGBA: [0, 255]
        # self.format = 2 # unused DELETE
        self.bigImg = None  # pointer to main display image
        self.modifier = None  # modifier used to transform input data
        self.widgetPipeline = None  # pointer to its pipeline widget
        self.showEvent = self.show  # called when the image is being un-hid
        self.mainUI()  # Sets up main UI
        self.init_image()  # initializes its image widget and data.
        self.group = [self]  # group of this and connected images (unused)
        self.inputAble = True  # Can be dragged and used as input (unfinished)
        self.onPipe = cst.noFunction  # function to be called when it is piping
        self.isDeleted = False  # a flag to check if the image is deleted
        self.sourceAble = True  # If true, this image can be selected as input.

    def pipeToGroup(self, grp):
        """
        Recursivly get the connected images.

        Parameters
        ----------
        grp :   <list>  list used to add images to
        """
        grp.append(self)
        for child in self.pipes:
            if child.inputAble:
                child.pipeToGroup(grp)

    def getImageGroup(self):
        ret = []
        self.pipeToGroup(ret)
        return ret

    def show(self, event):
        """
        Event function to be called when the image is un-hid
        """
        if not self.zoomable:
            self.autoFit()

    def setSize(self, w, h):
        """
        Set the size of widget

        Parameters
        ----------

        w   :   <float>
        h   :   <float>
        """
        self.setFixedHeight(w)
        self.setFixedWidth(h)

    def addToLibrary(self):
        """
        Add this image to the library of input selectable images
        """
        global GLOBAL_IMAGES,  _NUMBERED_IMAGE
        if self not in GLOBAL_IMAGES:
            if self.title == "?":
                self.title = "untiteled%d" % _NUMBERED_IMAGE
                _NUMBERED_IMAGE += 1
            GLOBAL_IMAGES.append(self)

    def load_image(self, fil):
        """
        Loads an image from file

        Parameters
        ----------

        fil :   <string>    filename
        """
        if self.loadable:
            o_image = im.imread(fil)
            self.setData(imageMath.rgbaFormat(o_image))

    def exportImageDialog(self, event):
        """
        Starts an export image Dialog
        """
        fil, _ = wd.QFileDialog.getSaveFileName(
            self,
            'Export image',
            './',
            'Images (*.png *.jpg)'
            )
        if fil:
            im.imwrite(fil, self.picdata)

    def update_image(self):
        """
        Updats the image to show its correct picdata
        """
        self.qimg = QtGui.QImage(
            self.picdata,
            self.picdata.shape[1],
            self.picdata.shape[0],
            # Format:
            QtGui.QImage.Format_RGBA8888
        )
        self.piximg = QtGui.QPixmap(self.qimg)
        self.piximg = self.piximg.scaledToWidth(
            self.picdata.shape[1] * self.zoom
            )
        self.imgObject.setPixmap(self.piximg)

    def init_image(self):
        """
        Initializes this image with some default values.
        """
        self.setData((np.ones((32, 32, 4)) * 255).astype(np.uint8))
        self.autoFit()

    def update(self):
        """
        Overrides widgets update function.
        """
        self.update_image()

    def __update__(self):
        """
        Widget update and updates image
        """
        self.update()
        self.update_image()

    def mainUI(self):
        """
        main UI of this widget
        """
        # Splitters used to divide into sections:
        splitter1 = wd.QSplitter(cr.Qt.Vertical)
        splitter2 = wd.QSplitter(cr.Qt.Horizontal)
        self.vlayout = wd.QVBoxLayout()
        self.vobject = wd.QScrollArea()
        self.vobject.setWidget(self.imgObject)
        self.vobject.setWidgetResizable(True)
        self.imgObject.mouseDoubleClickEvent = self.load_dialog
        self.mouseReleaseEvent = self.select
        splitter1.addWidget(self.vobject)

        # Tools:
        if self.toolsEnabled:
            toolbar = cst.Packlist(self, wd.QHBoxLayout)
            toolbar.setMaximumHeight(16)
            if self.loadable:
                loadBtn = cst.MiniButton(
                    self,
                    "../ui/openProject.png",
                    "Open Image"
                )
                loadBtn.mousePressEvent = self.load_dialog
                toolbar.addWidget(loadBtn)
            if self.zoomable:
                fitButton = cst.MiniButton(
                    self,
                    "../ui/autofit.png",
                    "Automatically set the zoom to fit the frame."
                )
                fitButton.mousePressEvent = self.autoFit
                toolbar.addWidget(fitButton)
                zinBtn = cst.MiniButton(
                    self, "../ui/zoomInDark.png", "Zoom in"
                )
                zutBtn = cst.MiniButton(
                    self, "../ui/zoomOutDark.png", "Zoom out"
                )
                zinBtn.mousePressEvent = self.zoomIn
                zutBtn.mousePressEvent = self.zoomOut
                toolbar.addWidget(zinBtn)
                toolbar.addWidget(zutBtn)
            if self.bakeable:
                bakeButton = cst.MiniButton(
                    self,
                    "../ui/bake.png",
                    "Bakes image into new image."
                )
                bakeButton.mousePressEvent = self.bakeImage
                toolbar.addWidget(bakeButton)
            expBtn = cst.MiniButton(self, "../ui/export.png", "Export image")
            expBtn.mousePressEvent = self.exportImageDialog
            toolbar.addWidget(expBtn)
            if self.deleteable:
                delBtn = cst.MiniButton(
                    self, "../ui/delete.png", "Delete image"
                )
                delBtn.mousePressEvent = self.deleteDialog
                toolbar.addWidget(delBtn)
            splitter2.addWidget(toolbar)

        splitter1.addWidget(splitter2)
        self.vlayout.addWidget(splitter1)
        self.setLayout(self.vlayout)

    def bakeImage(self, event=None):
        """
        Bake the image result into a new image "resource" with a new
        pipeline attatched to it.
        """
        imgData = np.copy(self.picdata)
        _add = WINDOW.addResource()
        _add.setData(imgData)
        _add.pipe()
        _add.autoFit()

    def updateZoom(self):
        """
        Updates the zoom
        """
        self.zoom = self.zoomStrength**(self.z)

    def zoomIn(self, event):
        """
        Zooms in on the image. Can be used with events.
        """
        self.z += 1
        if self.z > self.zoomRange[1]:
            self.z = self.zoomRange[1]
        self.updateZoom()
        self.update_image()

    def zoomOut(self, event):
        """
        Zooms out of the image. Can be used with events.
        """
        self.z -= 1
        if self.z < self.zoomRange[0]:
            self.z = self.zoomRange[0]
        self.updateZoom()
        self.update_image()

    def deleteDialog(self, event):
        """
        Dialog called to delete this image
        """
        msg = '\n'.join([
            "Would you like to delete this image?",
            "NB: This will change the results of any operators",
            "using this image, and delete its pipeline."
        ])
        dlg = cst.OptionsDialog(
            msg,
            [
                "Yes",
                "No"
            ],
            False
        )
        # 0 is index of yes:
        if dlg.exec_() and (dlg.selectedIndex.get() == 0):
            self.erase()

    def load_dialog(self, event):
        """
        Creates a dialog used to load an image into this image.
        """
        if self.loadable:
            fname = wd.QFileDialog.getOpenFileName(
                self.parent,
                'Sorce Image',
                './',
                'Image File (*.jpg *.jpeg *.gif *.png)'
                )
            if fname[0] != "" and fname[0] != self.fname:
                self.fname = fname[0]
                self.load_image(fname[0])
                self.autoFit()
                self.pipe()
            if self.bigImg is not None:
                self.bigImg.autoFit()

    def disconnectParent(self):
        """
        Disconnects itself from its input (parent)
        """
        if self.src is not None:
            self.src.pipes.remove(self)
            self.src = None

    def disconnectChild(self, child):
        """
        Disconnects all a specified child / pipe from this image.

        Parameters
        ----------

        child   :   pointer to child image
        """
        if child in self.pipes:
            self.pipes.remove(child)

    def disconnectChildren(self):
        """
        Disconnects all children / pipes from this image.
        """
        ret = []
        for child in self.pipes:
            child.src = None
            ret.append(child)
        self.pipes = []
        return ret

    def connect(self, child):
        """
        Sets "child"s input source to be this image
        """
        if (child is not None) and (child.src is None):
            child.src = self
            self.pipes.append(child)

    def connectList(self, children):
        """
        Adds its connections to the children list

        Parameters
        ----------

        children    :   <list>
        """
        for child in children:
            if child.src is None:
                child.src = self
                self.pipes.append(child)

    def zooming(self, event):
        """
        Can be called when eg. using the scrolling wheel on the mouse
        """
        dt = event.angleDelta()
        delta = int(round(dt.y()/self.zoomSensitivity))
        if delta != 0:
            self.z += delta
            if self.z < self.zoomRange[0]:
                self.z = self.zoomRange[0]
            if self.z > self.zoomRange[1]:
                self.z = self.zoomRange[1]
            self.updateZoom()
            self.update_image()

    def drag(self, event):
        """
        This functionality was planned but is still unused.
        """
        pass

    def setSource(self, _src):
        """
        Sets the source of the image
        """
        self.srcSet = True
        self.src = _src
        self.loadable = False
        self.update_image()

    def select(self, event):
        """
        Called when selecting this image
        """
        self.selectThis()

    def selectThis(self):
        """
        Selects this image, such that the main display image
        will show its pipeline result.
        """
        global SELECTED, WINDOW
        oldSel = SELECTED
        if (
            (self.bigImg is not None) and
            (self != self.bigImg) and
            (self.bigImg.src != self)
        ):
            if SELECTED is not None:
                SELECTED.widgetPipeline.toggleOff()
            self.bigImg.disconnectParent()
            self.widgetPipeline.at.connect(self.bigImg)
            self.widgetPipeline.at.pipe()
            self.parent.update_on_command()
            self.widgetPipeline.toggleOn()
            SELECTED = self
        self.repaint()
        if oldSel is not None:
            oldSel.repaint()

    def deselectThis(self):
        """
        Unselects this image
        """
        global SELECTED
        if (
            (SELECTED is self)
        ):
            self.widgetPipeline.toggleOff()
            SELECTED = None
            self.bigImg.disconnectParent()

    def pipeOnce(self, pipeTrace=None):
        """
        Pipes only once, not reccursivly.

        Parameters
        ----------

        pipeTrace   :   <list>  list of images involved in a "pipe round"
        """
        if pipeTrace is None:
            pipeTrace = []
        if self not in pipeTrace:
            pipeTrace.append(self)
            for child in self.pipes:
                child.setData(self.picdata)
            if self.onPipe is not None:
                self.onPipe(pipeTrace)
        return pipeTrace

    def pipe(self, pipeTrace=None):
        """
        Pipe data to connected children.

        Parameters
        ----------
        pipeTrace :     Trace what modifiers has been run through a pipeline.
                        This is done to make sure no modifier widget can pipe
                        more than once per pipe "round"

        """
        if pipeTrace is None:
            pipeTrace = []
        if self not in pipeTrace:
            pipeTrace.append(self)
            for child in self.pipes:
                child.setData(self.picdata)
                child.pipe(pipeTrace)
            if self.onPipe is not None:
                self.onPipe(pipeTrace)
        return pipeTrace

    def modify(self):
        """
        Uses its modifier to modify picData if any modifier is provided
        """
        if self.modifier is not None:
            self.modifier.values[0] = self.picdata
            self.picdata = self.modifier.transform()

    def setSizeInterval(self, minW, minH, maxW, maxH):
        """
        Specifies the min and max sizes of the widget
        """
        self.setMinimumWidth(minW)
        self.setMinimumHeight(minH)
        self.setMaximumHeight(maxH)
        self.setMaximumWidth(maxW)

    def delete(self):
        """
        Can be used when this gets deleted. TODO
        """
        self.disconnectParent()

    def autoFit(self, event=None):
        """
        Sets the zoom to make the image automatically fit its frame.
        """
        if not self.isHidden():
            w = self.picdata.shape[1] / self.vobject.width()
            h = self.picdata.shape[0] / self.vobject.height()
            if w > h:
                self.z = log(
                    self.vobject.width() / self.picdata.shape[1],
                    self.zoomStrength
                )
            else:
                self.z = log(
                    self.vobject.height() / self.picdata.shape[0],
                    self.zoomStrength
                )
            self.updateZoom()
            self.update_image()

    def setData(self, data):
        """
        Sets the picData of this image and updates it.
        """
        if data is not None:
            self.picdata = data
            self.modify()
            if not self.zoomable:
                self.autoFit()
            else:
                self.update_image()

    def erase(self):
        """
        Erases / deletes this image
        """
        global GLOBAL_IMAGES
        self.deselectThis()
        # Recursivly calls erase function of children:
        if self.widgetPipeline is not None:
            self.widgetPipeline.erase()
        GLOBAL_IMAGES.remove(self)  # Remove this from global images.
        self.deleteLater()  # Delete it

    def onDelete(self):
        """
        Called when the object if finally deleted by the garbage
        collector.
        """
        global GLOBAL_IMAGES
        self.disconnectParent()
        self.disconnectChildren()
        cst.removeFromPacklist(self)

    def __del__(self):
        self.onDelete()

    def paintEvent(self, event):
        global SELECTED
        """
        Override paintEvent to display whether this image is
        selected or not
        """
        wd.QWidget.paintEvent(self, event)
        if SELECTED is self:  # then this is selected
            # Set brush:
            painter = QtGui.QPainter(self)
            color = QtGui.QColor(127, 255, 127, 127)
            pen = QtGui.QPen(
                color,
                8,
                cr.Qt.SolidLine,
                cr.Qt.RoundCap,
                cr.Qt.RoundJoin
            )
            painter.setPen(pen)
            # Draw a rectangle around it
            painter.drawRect(self.rect())


class ReferenceImage(wd.QLabel):  # $rfi
    """
    A reference image used in the image selection dialog
    """
    def __init__(self, parent, reference, width=128, height=128):
        """
        Parameters
        ----------

        parent      :   <QWidget>Qt widget parent
        reference   :   <imageFrame>
        width       :   float
        height      :   float
        """
        wd.QLabel.__init__(self, parent)  # Inherit from QLabel
        # Set the image:
        self.piximg = QtGui.QPixmap(reference.qimg)
        self.piximg = self.piximg.scaledToWidth(
            width
        )
        # Set the referenc image
        self.reference = reference
        self.setPixmap(self.piximg)
        self._parent = parent  # Original parent / dialog / window

    def paintEvent(self, event):
        """
        Override paintEvent to display whether this image is
        selected or not
        """
        wd.QLabel.paintEvent(self, event)
        if self._parent.select is self:  # then this is selected
            # Set brush:
            painter = QtGui.QPainter(self)
            color = QtGui.QColor(127, 255, 127, 127)
            pen = QtGui.QPen(
                color,
                8,
                cr.Qt.SolidLine,
                cr.Qt.RoundCap,
                cr.Qt.RoundJoin
            )
            painter.setPen(pen)
            # Draw a rectangle around it
            painter.drawRect(self.rect())

    def mouseReleaseEvent(self, event):
        old = self._parent.select
        self._parent.select = self
        self._parent.register = self.reference
        # npx = self.piximg.copy().scaledToWidth(384)  # old
        npx = QtGui.QPixmap(self.reference.qimg)
        npx = npx.scaledToWidth(
            480
        )
        self._parent.mainImg.setPixmap(npx)
        self._parent.update()
        self.update()  # update to show selection
        if old is not None:
            old.update()  # Update to not draw "selection frame"


class PipelineWidget(wd.QScrollArea):
    """
    A widget containing imageFrames that pipe their result to one another
    """
    def __init__(self, parent, refr):
        """
        Parameters
        ----------

        parent  :   <QWidget>
        refr    :   <imageFrame>    the master image, root of the pipeline
        """
        wd.QScrollArea.__init__(self, parent)
        self.playout = cst.Packlist(self)
        self.setWidget(self.playout)
        self.setWidgetResizable(True)
        self.setMinimumWidth(256)
        self.setMaximumWidth(384)
        self.reference = refr
        self.at = self.reference
        self.pipes = []  # List of ModifierWidgets belonging to this pipeline
        self.visible = False
        self.setHidden(True)
        refr.widgetPipeline = self

    def addModifier(self, modifier):
        """
        Adds a Modifier to the pipeline.

        Parameters
        ----------

        modifier    :   <modifiers.Modifier>
                        (Child class of Modifier)

        Returns
        -------
        prev        :   <modifier.Modifier>
                        Returns the previous tail of the pipeline.
        """
        prev = self.at  # Previous "tail" (last modifier in pipe line)
        c1 = ModifierWidget(self, modifier, self.at)  # Created modifier
        self.pipes.append(c1)
        self.at = c1.img  # Update tail
        self.playout.addWidget(c1)
        return prev

    def toggle(self):
        """
        Hide / show pipeline
        """
        if self.visible:
            self.visible = False
            self.setHidden(True)
        else:
            self.visible = True
            self.setHidden(False)

    def toggleOn(self):
        """
        Show pipeline
        """
        self.visible = True
        self.setHidden(False)

    def toggleOff(self):
        """
        Hide pipeline
        """
        self.visible = False
        self.setHidden(True)

    def onDelete(self):
        """
        Recursively calls onDelete for connected widgets.
        """
        for modifier in self.pipes:
            modifier.onDelete()

    def erase(self):
        """
        Erases this and connected widgets.
        """
        self.onDelete()
        self.deleteLater()


class Window(wd.QDialog):
    """
    Main window
    """
    def __init__(self, parent=None):
        global WINDOW
        super().__init__()  # inherit from QDialog (aka. window class)
        WINDOW = self
        self.title = "QtRaccon"  # Name of the app
        self.top = 100
        self.left = 100
        self.width = 400
        self.height = 400
        self.setMinimumHeight(200)
        self.setMinimumWidth(200)
        self.initWindow()  # Initializing window function.
        self.imgSelected = False
        self.selected = None

    def initWindow(self):
        """
        Function that initializes the main windows and
        shows it.
        """
        self.setWindowIcon(QtGui.QIcon("../ui/raccoony.png"))
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.mainUI()
        self.show()

    def mainUI(self):
        """
        Creates the main UI of the window:
        """
        global BIG_IMAGE
        layout = wd.QHBoxLayout()
        pn0 = wd.QScrollArea()
        self.imagesSection = cst.Packlist(self)
        pn0.setWidget(self.imagesSection)
        pn0.setWidgetResizable(True)
        pn1 = wd.QScrollArea()
        pn0.setMinimumWidth(128)
        pn0.setMaximumWidth(192)
        sp0 = wd.QSplitter(cr.Qt.Horizontal)
        self.sp1 = wd.QSplitter(cr.Qt.Vertical)
        sp3 = wd.QSplitter(cr.Qt.Horizontal)
        self.sp1.addWidget(sp3)
        btnAddPic = wd.QPushButton("+", self)
        btnAddPic.setMaximumHeight(32)
        btnAddPic.mousePressEvent = self.addImage
        sp3.addWidget(btnAddPic)
        hdrBtn = wd.QPushButton("HDR", self)
        hdrBtn.mousePressEvent = self.loadHDRImage
        hdrBtn.setMaximumHeight(32)
        sp3.addWidget(hdrBtn)
        sp0.addWidget(self.sp1)
        layout.addWidget(sp0)
        sp0.addWidget(pn1)
        self.sp1.addWidget(pn0)
        self.setLayout(layout)

        # Main Image:
        self.mainImg = imageFrame(
            self,
            loadable=False,
            zoomable=True,
            bakeable=True,
            deleteable=False,
            toolsEnabled=True
        )
        BIG_IMAGE = self.mainImg  # Set the pointer to the main display
        self.mainImg.setSizeInterval(256, 256, 2560, 2560)  # Set size range
        self.mainImg.loadable = False  # Cant load directly into main display
        self.mainImg.recursion = False  # Turn off recursion
        self.mainImg.inputAble = False  # This image can not be used as input
        pn1.setWidgetResizable(True)
        pn1.setWidget(self.mainImg)

        # Pipelines:
        self.sp4 = wd.QSplitter(cr.Qt.Vertical)
        sp0.addWidget(self.sp4)
        pipeArea = cst.Packlist(self)
        pipeArea.setMaximumWidth(384)

        pipeTools = cst.Packlist(self, wd.QHBoxLayout)
        self.selectModifier = self.modifierWidgets()
        addModBtn = wd.QPushButton("Add Modifier")
        addModBtn.mousePressEvent = self.selectAndAddModifier
        pipeTools.addWidget(self.selectModifier)
        pipeTools.addWidget(addModBtn)

        self.pipelineWidgets = cst.Packlist(self)
        pipeArea.addWidget(pipeTools)
        pipeArea.addWidget(self.pipelineWidgets)
        self.sp4.addWidget(pipeArea)

    def update(self):
        pass

    def update_on_command(self):
        pass

    def __update__(self):
        self.update()

    def addResource(self, imgData=None):
        """
        Adds an image Resource to the images section
        of the window.

        Parameters
        ----------

        imgData :   <numpy.ndarray>
                    image data.
        """
        global GLOBAL_IMAGES
        _add = imageFrame(self)  # image to be added
        _add.setSize(192, 192)
        _add.bigImg = self.mainImg  # set its display reference
        GLOBAL_IMAGES.append(_add)  # add it to the list om images
        self.imagesSection.addWidget(_add)  # add it to the image section
        pl = PipelineWidget(self, _add)  # creates a new pipeline widget
        self.pipelineWidgets.addWidget(pl)  # add it to the list of pipelines
        if imgData is not None:
            _add.setData(imgData)
            _add.pipe()
        _add.selectThis()  # Set the newly added picture as selected
        return _add

    def loadHDRImage(self, event):
        """
        Uses a QDialog to load a folder containing images with
        different exposure of light and create a hdr image.
        """
        _img = loadHDRDialog()
        if _img is not None:
            self.addResource(imageMath.rgbaFormat(_img))

    def addImage(self, event):
        """
        Add a new empty image to images sections.
        """
        self.addResource()

    def selectAndAddModifier(self, event):
        """
        Adds a modifier from the MODIFIERS list
        to the currently selected image. Can be used with events.
        """
        global MODIFIERS
        self.addModifierToSelected(
            MODIFIERS[self.selectModifier.currentIndex()]
        )

    def addModifierToSelected(self, modf):
        """
        Adds a modifier to the currently selected image.

        Parameters
        ----------

        modf    :   <Class>
                    (Child class of Modifier)
        """
        global SELECTED
        if SELECTED is not None:
            # Add modifer to the selected pipeline and return its
            # previous "tail" / last element in pipeline:
            previous = SELECTED.widgetPipeline.addModifier(modf)
            # Connect display image to the new tail:
            self.mainImg.disconnectParent()
            SELECTED.widgetPipeline.at.connect(self.mainImg)
            previous.pipe()  # Update pipeline
            self.update_on_command()  # Calls update

    def modifierWidgets(self):
        """
        Returns a selection box where you can select a modifier
        based on the MODIFIERS list

        Returns
        -------

        QComboBox
        """
        global MODIFIERS

        comboBox = wd.QComboBox()
        for modf in MODIFIERS:
            comboBox.addItem(modf.__name__)
        return comboBox


class SelectImageDialog(wd.QDialog):  # $sid
    """
    Dialog used to select an image input
    """

    def __init__(self, parent=None):
        super().__init__()
        self.title = "select image"
        self.top = 100
        self.left = 100
        self.width = 640
        self.height = 480
        self.register = None
        self.select = None

        self.InitWindow()

    def mainUI(self):
        global GLOBAL_IMAGES
        self.imgSelected = False
        images = cst.Packlist(self)
        images.select = None  # Requiered
        self.mainImg = wd.QLabel()
        scrl2 = wd.QScrollArea()
        scrl2.setWidgetResizable(True)
        scrl2.setWidget(self.mainImg)
        playout = wd.QHBoxLayout()
        playout.setAlignment(cr.Qt.AlignLeft | cr.Qt.AlignTop)
        self.setLayout(playout)
        scrl = wd.QScrollArea()
        scrl.setWidgetResizable(True)
        scrl.setWidget(images)
        leftSide = cst.Packlist(self)
        leftSide.addWidget(wd.QLabel("Images"))
        leftSide.addWidget(scrl)
        playout.addWidget(leftSide)
        rightSide = wd.QVBoxLayout()
        rightSide.addWidget(scrl2)
        rightWidget = wd.QWidget(self)
        rightWidget.setLayout(rightSide)
        buttons = cst.Packlist(self, wd.QHBoxLayout)
        cancelBtn = wd.QPushButton("Cancel")
        selectBtn = wd.QPushButton("Select")
        buttons.addWidget(selectBtn)
        buttons.addWidget(cancelBtn)
        cancelBtn.clicked.connect(self.cancel)
        selectBtn.clicked.connect(self.selectImg)
        rightSide.addWidget(buttons)
        playout.addWidget(rightWidget)
        self.select = None
        # Get list om selectable images:
        for imgFrm in GLOBAL_IMAGES:  # For each image frame:
            imgs = imgFrm.getImageGroup()  # Get list of images in its pipeline
            for img in imgs:  # For each of these images:
                if img.sourceAble:
                    _add = ReferenceImage(self, img)
                    images.addWidget(_add)

    def InitWindow(self):
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, self.width, self.height)
        self.mainUI()

    def getValues(self):
        """
        Used to get the return value from the dialog
        """
        return self.register

    def selectImg(self):
        """
        If there is a selected image when this is called, the
        dialog is accepted and exec_() will return true.
        """
        if self.register is not None:
            self.accept()

    def cancel(self):
        """
        Reject the dialog.
        """
        self.reject()


def selectImage():
    """
    Uses a dialog to select an image

    Returns
    -------

    dlg.register    <numpy.ndarray>
                    if the dialog register is not empty
    None
                    if the dialog register is empty
    """
    GLOBAL_IMAGES
    dlg = SelectImageDialog()  # Dialog
    if dlg.exec_():
        return dlg.register
    else:
        return None


def uiStart():
    """
    Creates the main window and runs it.
    """
    App = wd.QApplication(sys.argv)
    Window()
    sys.exit(App.exec())


"""
MAIN
"""

if __name__ == "__main__":
    uiStart()
