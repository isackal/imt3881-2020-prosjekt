import sys

import numpy as np
import imageio as im
from math import log

# Import GUI:
import PyQt5.QtWidgets as wd
import PyQt5.QtCore as cr
from PyQt5 import QtGui
import customWidgets as cst
import errorhandling as eh  # used to display error and warnings

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
# import anonymiser  # temporary overlapping library bug on mac
import cloning
# import poisson

# Set Modifiers:
MODIFIERS = [
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
    bitcrusher.Bitcrusher,
    contrast.Contrast,
    blurring.Blurring,
    meanimage.Meanimage,
    kbg.KantbevarendeGlatting,
    ctg.ColorToGray,
    demosaic.Demosaic,
    inpaint.Inpaint,
    # anonymiser.Anonymisering,
    cloning.Cloning
]

_NUMBERED_IMAGE = 0  # used for defualt naming of images

SELECTED = None  # pointer to current selected image

BIG_IMAGE = None  # pointer to the big screen image

GLOBAL_IMAGES = []  # List of all image groups


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

    _up.mousePressEvent = upFunc
    _down.mousePressEvent = downFunc
    _del.mousePressEvent = delFunc

    toolbar.addWidget(_up)
    toolbar.addWidget(_down)
    toolbar.addWidget(_del)

    return toolbar


class Collapser(wd.QWidget):
    def __init__(self,  parent):
        wd.QWidget.__init__(self,  parent)
        layout = wd.QVBoxLayout()
        top = wd.QGroupBox()
        self.topl = wd.QHBoxLayout()
        self.topl.setContentsMargins(8,  0,  8,  0)
        self.topl.setAlignment(cr.Qt.AlignLeft)
        self.topl.setStretch(0,  0)
        top.setLayout(self.topl)
        self.minimizeButton = wd.QPushButton('+',  self)
        self.minimizeButton.setFixedHeight(32)
        self.minimizeButton.setFixedWidth(32)
        self.minimizeButton.mousePressEvent = self.toggle
        self.topl.addWidget(self.minimizeButton)
        self.title = "Modify"
        self.label = wd.QLabel(self.title,  self)
        self.topl.addWidget(self.label)
        top.setFixedHeight(48)
        self.vgrp = wd.QGroupBox()
        cLayout = wd.QVBoxLayout()
        self.content = cst.Packlist(self.vgrp)
        cLayout.addWidget(self.content)
        self.vgrp.setLayout(cLayout)
        self.mainUI()
        self.setLayout(layout)
        layout.addWidget(top)
        layout.setSpacing(0)
        layout.addWidget(self.vgrp)
        self.open = False
        self.vgrp.setHidden(True)

    def toggle(self,  event):
        if self.open:
            self.vgrp.setHidden(True)
            self.minimizeButton.text = "+"
            self.open = False
        else:
            self.vgrp.setHidden(False)
            self.minimizeButton.text = "-"
            self.open = True

    def mainUI(self):
        for i in range(10):
            btn = wd.QPushButton("Button %2d" % i,  self)
            self.content.addWidget(btn)

    def addToTopUI(self, widget):
        self.topl.addWidget(widget)


class TypeInput(wd.QWidget):
    """
    A more sofisticated input widget that can be an image or
    a number.
    """
    def __init__(self,  parent,  _type):
        wd.QWidget.__init__(self, parent)
        self.playout = wd.QHBoxLayout()
        self.setLayout(self.playout)
        self.isImage = False
        self._type = _type
        self.widget = None
        if _type is np.ndarray:
            self.isImage = True
            self.widget = imageFrame(
                self,
                loadable=False,
                zoomable=False,
                toolsEnabled=False
            )
            self.widget.setSize(64, 64)
            self.setFixedWidth(64)
            self.setFixedHeight(64)
            self.widget.mousePressEvent = self.setImage
        else:
            self.widget = cst.NumericInput(_type)
            self.playout.addWidget(self.widget)

    def setText(self, txt):
        self.widget.setText(txt)

    def validate(self,  text):
        self.setText(str(self.getValue()))

    def getValue(self):
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
        sel = selectImage()
        if sel is not None:
            self.widget.disconnectParent()
            sel.connect(self.widget)
            sel.pipeOnce()
            self.widget.pipe()  # To call onPipe

    def setValue(self, val):
        if self.isImage:
            self.widget.setData(val)
        else:
            self.widget.setText(str(val))

    def onDelete(self):
        if self.isImage:
            self.widget.onDelete()


class ModifierWidget(Collapser):
    """
    This is the user interface of a certain modifier.
    """
    def __init__(self,  parent,  modf,  reference):
        self.reference = reference  # image input
        print("#75")
        self.modifier = modf()
        print("#76")
        self.dtas = []
        self.widgetPipeline = parent
        Collapser.__init__(self,  parent)
        tl = upDownDelToolbar(
            self.moveUp,
            self.moveDown,
            self.deleteThis
        )
        self.addToTopUI(tl)
        self.masterLayout = None  # Pointer to its "Packlist"

    def moveUp(self, event):
        if self.masterLayout is not None:
            neighbour = self.masterLayout.moveUp(self)
            if neighbour is not None:
                self.swapPlaces(neighbour, self)
                self.reference.pipe()
                print("Moved up a bit")

    def moveDown(self, event):
        if self.masterLayout is not None:
            neighbour = self.masterLayout.moveDown(self)
            if neighbour is not None:
                self.swapPlaces(self, neighbour)
                neighbour.reference.pipe()
                print("Moved down a bit")

    def swapPlaces(self, first, second):
        global BIG_IMAGE
        """
        This swaps first down and second up. Will not work in
        opposite order.
        """
        pf = first.img.src  # parent-image first. Assumed to not be None
        # parent second is assumed to me first
        cS = first.masterLayout.next(first)  # child second
        # A-- because it is assumed the order in the layout changed
        # is all ready changed.
        second.img.disconnectParent()
        first.img.disconnectParent()

        pf.connect(second.img)
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

    def deleteThis(self, event=None):
        global BIG_IMAGE
        pimg = self.img.src  # parent-image. Assumed to not be None
        # parent second is assumed to me first
        self.img.disconnectParent()
        for _inp in self.dtas:
            _inp.onDelete()
        child = None
        if self.masterLayout is not None:
            child = self.masterLayout.next(self)  # child second
            if child is not None:
                child.img.disconnectParent()
                pimg.connect(child.img)
                child.reference = pimg
            else:
                self.widgetPipeline.at = pimg
                BIG_IMAGE.disconnectParent()
                pimg.connect(BIG_IMAGE)
            self.masterLayout.removeWidget(self)
        pimg.pipe()
        self.deleteLater()

    def mainUI(self):
        self.img = imageFrame(
            self,
            loadable=False,
            zoomable=False,
            toolsEnabled=False
        )
        self.img.setFixedWidth(128)
        self.img.setFixedHeight(128)
        self.reference.connect(self.img)
        self.content.addWidget(self.img)
        self.title = self.modifier.name
        self.label.setText(self.title)
        grd = wd.QGridLayout()
        _row = 0
        self.modifier.values[0] = self.img.picdata
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
        self.img.modifier = self.modifier

    def onUpdateData(self):
        for i in range(len(self.modifier.values)-1):
            self.modifier.values[i+1] = self.dtas[i].getValue()
        self.reference.pipe()

    def onUpdateText(self, text):
        self.onUpdateData()


class imageFrame(cst.DragableWidget):
    def __init__(
        self, parent,
        loadable=True,  # Allow loading images
        zoomable=True,  # Allow zoom
        toolsEnabled=True  # Allow any tools, such as zoom, load, etc
    ):
        cst.DragableWidget.__init__(self, parent)
        self.title = "?"
        self.top = 0
        self.left = 0
        self.width = 384
        self.height = 240
        self.picdata = None
        self.qimg = None
        self.piximg = None
        self.imgObject = wd.QLabel(self)
        self.parent = parent
        self.zoomStrength = 2
        self.zoom = 1
        self.z = 0
        self.zoomRange = (-4, 4)
        self.fname = ""
        self.zoomSensitivity = 128
        self.src = None
        self.srcSet = False
        self.loadable = loadable
        self.toolsEnabled = toolsEnabled
        self.zoomable = zoomable
        self.isSelected = False
        self.recursion = True
        self.pipes = []
        self.src = None  # Which object is piping to this.
        # 0: Binary,  1: Grayscale [0, 1],  2: RGBA: [0, 255]
        self.format = 2
        self.bigImg = None
        self.modifier = None
        self.widgetPipeline = None
        self.showEvent = self.show
        self.mainUI()
        self.init_image()
        self.group = [self]
        self.inputAble = True  # Can be dragged and used as input
        self.onPipe = cst.noFunction
        self.isDeleted = False

    def pipeToGroup(self, grp):
        grp.append(self)
        for child in self.pipes:
            if child.inputAble:
                child.pipeToGroup(grp)

    def getImageGroup(self):
        ret = []
        self.pipeToGroup(ret)
        return ret

    def show(self, event):
        if not self.zoomable:
            self.autoFit()

    def setSize(self, w, h):
        self.setFixedHeight(128)
        self.setFixedWidth(128)

    def addToLibrary(self):
        global GLOBAL_IMAGES,  _NUMBERED_IMAGE
        if self not in GLOBAL_IMAGES:
            if self.title == "?":
                self.title = "untiteled%d" % _NUMBERED_IMAGE
                _NUMBERED_IMAGE += 1
            GLOBAL_IMAGES.append(self)

    def load_image(self, fil):
        if self.loadable:
            o_image = im.imread(fil)
            _image = np.ones((o_image.shape[0], o_image.shape[1], 4)) * 255
            _image[:, :, 0] = o_image[:, :, 0]  # RED
            _image[:, :, 1] = o_image[:, :, 1]  # GREEN
            _image[:, :, 2] = o_image[:, :, 2]  # BLUE
            if o_image.shape[2] == 4:
                _image[:, :, 3] = o_image[:, :, 3]  # BLUE
            self.setData(_image.astype(np.uint8))

    def exportImageDialog(self, event):
        fil, _ = wd.QFileDialog.getSaveFileName(
            self,
            'Export image',
            './',
            'Images (*.png *.jpg)'
            )
        if fil:
            im.imwrite(fil, self.picdata)

    def update_image(self):
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
        self.setData((np.ones((32, 32, 4)) * 255).astype(np.uint8))
        self.autoFit()

    def update(self):
        self.update_image()

    def __update__(self):
        self.update()
        self.update_image()

    def mainUI(self):
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
            loadBtn = cst.MiniButton(
                self,
                "../ui/openProject.png",
                "Open Image"
            )
            loadBtn.mousePressEvent = self.load_dialog
            fitButton = cst.MiniButton(
                self,
                "../ui/autofit.png",
                "Automatically set the zoom to fit the frame."
            )
            fitButton.mousePressEvent = self.autoFit
            expBtn = cst.MiniButton(self, "../ui/export.png", "Export image")
            expBtn.mousePressEvent = self.exportImageDialog
            zinBtn = cst.MiniButton(self, "../ui/zoomInDark.png", "Zoom in")
            zutBtn = cst.MiniButton(self, "../ui/zoomOutDark.png", "Zoom out")
            delBtn = cst.MiniButton(self, "../ui/delete.png", "Delete image")
            delBtn.mousePressEvent = self.deleteDialog
            toolbar = cst.Packlist(self, wd.QHBoxLayout)
            toolbar.setMaximumHeight(16)
            toolbar.addWidget(loadBtn)
            toolbar.addWidget(fitButton)
            toolbar.addWidget(zinBtn)
            toolbar.addWidget(zutBtn)
            toolbar.addWidget(expBtn)
            toolbar.addWidget(delBtn)
            zinBtn.mousePressEvent = self.zoomIn
            zutBtn.mousePressEvent = self.zoomOut
            splitter2.addWidget(toolbar)

        splitter1.addWidget(splitter2)
        self.vlayout.addWidget(splitter1)
        """
        if self.width>0:
            self.setGeometry(0, 0, self.width, self.height)
            self.setMaximumWidth(self.height)
        if self.height>0:
            self.setMaximumHeight(self.height)
        """
        self.setLayout(self.vlayout)

    def updateZoom(self):
        self.zoom = self.zoomStrength**(self.z)

    def zoomIn(self, event):
        print("Zoom In")
        self.z += 1
        if self.z > self.zoomRange[1]:
            self.z = self.zoomRange[1]
        print(self.z)
        self.updateZoom()
        self.update_image()

    def zoomOut(self, event):
        print("Zoom Out")
        self.z -= 1
        if self.z < self.zoomRange[0]:
            self.z = self.zoomRange[0]
        print(self.z)
        self.updateZoom()
        print(self.zoom)
        self.update_image()

    def deleteDialog(self, event):
        msg = '\n'.join([
            "Would you like to delete this image?",
            "NB: This will change the results of any operators",
            "using this image."
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
        if self.src is not None:
            self.src.pipes.remove(self)
            self.src = None

    def disconnectChild(self, child):
        if child in self.pipes:
            self.pipes.remove(child)

    def disconnectChildren(self):
        ret = []
        for child in self.pipes:
            child.src = None
            ret.append(child)
        self.pipes = []
        return ret

    def connect(self, child):
        if (child is not None) and (child.src is None):
            print(child.src)
            child.src = self
            self.pipes.append(child)

    def connectList(self, children):
        for child in children:
            if child.src is None:
                print(child.src)
                child.src = self
                self.pipes.append(child)

    def zooming(self, event):
        dt = event.angleDelta()
        delta = int(round(dt.y()/self.zoomSensitivity))
        print(delta)
        if delta != 0:
            self.z += delta
            if self.z < self.zoomRange[0]:
                self.z = self.zoomRange[0]
            if self.z > self.zoomRange[1]:
                self.z = self.zoomRange[1]
            self.updateZoom()
            self.update_image()

    def drag(self, event):
        print(event.x())

    def setSource(self, _src):
        self.srcSet = True
        self.src = _src
        self.loadable = False
        self.update_image()

    def select(self, event):
        self.selectThis()

    def selectThis(self):
        global SELECTED
        if (
            (self.bigImg is not None) and
            (self != self.bigImg) and
            (self.bigImg.src != self)
        ):
            print("# 7")
            if SELECTED is not None:
                SELECTED.widgetPipeline.toggleOff()
            self.bigImg.disconnectParent()
            self.widgetPipeline.at.connect(self.bigImg)
            self.widgetPipeline.at.pipe()
            self.parent.update_on_command()
            self.widgetPipeline.toggleOn()
            SELECTED = self

    def deselectThis(self):
        global SELECTED
        if (
            (SELECTED is self)
        ):
            self.widgetPipeline.toggleOff()
            SELECTED = None
            self.bigImg.disconnectParent()

    def pipeOnce(self):
        for child in self.pipes:
            print("Sending data to %s" % child.title)
            child.setData(self.picdata)
        if self.onPipe is not None:
            self.onPipe()

    def pipe(self):
        if self.isDeleted:
            print("Why am I still alive?")
            sys.exit(1)
        print("\n\nPiping")
        print(self.zoomable)
        for child in self.pipes:
            print("Sending data to %s" % child.title)
            child.setData(self.picdata)
            child.pipe()
        if self.onPipe is not None:
            self.onPipe()

    def modify(self):
        print(")))")
        if self.modifier is not None:
            self.modifier.values[0] = self.picdata
            print("!!!")
            self.picdata = self.modifier.transform()

    def setSizeInterval(self, minW, minH, maxW, maxH):
        self.setMinimumWidth(minW)
        self.setMinimumHeight(minH)
        self.setMaximumHeight(maxH)
        self.setMaximumWidth(maxW)

    def delete(self):
        self.disconnectParent()

    def autoFit(self, event=None):
        if not self.isHidden():
            print("Autofitting")
            print(self.picdata.shape[1], self.vobject.width())
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
            print("New zoom:")
            print(self.z)

    def setData(self, data):
        if data is not None:
            print("Modified data")
            self.picdata = data
            self.modify()
            if not self.zoomable:
                print("Case 1")
                self.autoFit()
            else:
                print("Case 2")
                self.update_image()

    def erase(self):
        global GLOBAL_IMAGES
        self.deselectThis()
        if self.widgetPipeline is not None:
            self.widgetPipeline.erase()
        GLOBAL_IMAGES.remove(self)
        self.deleteLater()

    def onDelete(self):
        global GLOBAL_IMAGES
        self.disconnectParent()
        self.disconnectChildren()
        cst.removeFromPacklist(self)

    def __del__(self):
        self.onDelete()


class ReferenceImage(wd.QLabel):  # $rfi

    def __init__(self, parent, reference, width=128, height=128):
        wd.QLabel.__init__(self, parent)
        self.piximg = QtGui.QPixmap(reference.qimg)
        self.piximg = self.piximg.scaledToWidth(
            width
        )
        self.reference = reference
        self.setPixmap(self.piximg)
        self._parent = parent  # Original parent

    def paintEvent(self, event):
        wd.QLabel.paintEvent(self, event)
        if self._parent.select is self:
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
            painter.drawRect(self.rect())

    def mouseReleaseEvent(self, event):
        old = self._parent.select
        self._parent.select = self
        self._parent.register = self.reference
        npx = self.piximg.copy().scaledToWidth(384)
        self._parent.mainImg.setPixmap(npx)

        self._parent.update()
        self.update()
        if old is not None:
            old.update()


class operatorImage(imageFrame):

    def __init__(self, parent, opr):
        imageFrame.__init__(self, parent)
        self.setFixedHeight(128)
        self.setFixedWidth(128)
        self.opr = opr

    def modify(self):
        pass


class operator(wd.QWidget):

    def __init__(self, parent, func, name="operator"):
        wd.QWidget.__init__(self, parent)
        self.name = name
        grp = wd.QGroupBox(self.name)
        self.layout = wd.QGridLayout()
        grp.setLayout(self.layout)
        vlayout = wd.QVBoxLayout()
        vlayout.addWidget(grp)
        self.setLayout(vlayout)
        self.y = 0
        self.dataFields = []
        self.src = None
        self.srcSet = False
        self.pipes = []
        self.res = None

    def addData(self, dataName, dataType, defaultValue):
        lbl = wd.QLabel(dataName, self)
        dta = wd.QLineEdit(self)
        dta.setText(str(defaultValue))
        self.dataFields.append((dataName, defaultValue, dataType))
        self.layout.addWidget(lbl, self.y, 0)
        self.layout.addWidget(dta, self.y, 1)
        self.y += 1

    def calculate(self):
        self.res = self.src * 1
        self.updatePipeline()

    def updatePipeline(self):
        for child in self.pipes:
            child.src = self.res
            child.calculate()


class Canvas(wd.QWidget):

    def __init__(self, parent):
        wd.QWidget.__init__(self, parent)
        self.setGeometry(0, 0, 2000, 2000)
        self.setAttribute(cr.Qt.WA_TransparentForMouseEvents)

    def instantiate(self, widget):
        widget.setParent(self)
        widget.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setPen(cr.Qt.red)
        painter.drawLine(0, 0, 2000, 2000)


class MoveableBox(wd.QGroupBox):

    def __init__(self, parent, lbl):
        wd.QGroupBox.__init__(self, parent)
        self.title = lbl
        self.setFixedWidth(128)
        self.setFixedHeight(192)
        self.setAcceptDrops(True)
        self.mousePressEvent = self.mousePress
        self.mouseReleaseEvent = self.mouseRelease
        self.mouseMoveEvent = self.mouseMove
        self.ox = 0
        self.oy = 0
        self.isDragged = False

    def mousePress(self,  event):
        self.isDragged = True
        self.ox = event.x()-self.x()
        self.oy = event.y()-self.y()

    def mouseMove(self,  event):
        if (self.isDragged):
            self.move(event.x()-self.ox, event.y()-self.oy)

    def mouseRelease(self,  event):
        print(":(")


class PipelineWidget(wd.QScrollArea):
    def __init__(self, parent, refr):
        wd.QScrollArea.__init__(self, parent)
        self.playout = cst.Packlist(self)
        self.setWidget(self.playout)
        self.setWidgetResizable(True)
        self.setMinimumWidth(256)
        self.setMaximumWidth(384)
        self.reference = refr
        self.at = self.reference
        self.pipes = []
        self.visible = False
        self.setHidden(True)
        refr.widgetPipeline = self

    def addModifier(self, modifier):
        prev = self.at
        c1 = ModifierWidget(self, modifier, self.at)
        self.pipes.append(c1)
        self.at = c1.img
        self.playout.addWidget(c1)
        return prev

    def toggle(self):
        if self.visible:
            self.visible = False
            self.setHidden(True)
        else:
            self.visible = True
            self.setHidden(False)

    def toggleOn(self):
        self.visible = True
        self.setHidden(False)

    def toggleOff(self):
        self.visible = False
        self.setHidden(True)

    def onDelete(self):
        for modifier in self.pipes:
            modifier.deleteThis()

    def erase(self):
        self.onDelete()
        self.deleteLater()


class Window(wd.QDialog):

    def __init__(self, parent=None):
        super().__init__()  # inherit from QDialog (aka. window class)
        self.title = "defualt"
        self.top = 100
        self.left = 100
        self.width = 400
        self.height = 400
        self.setMinimumHeight(200)
        self.setMinimumWidth(200)
        # self.timer = cr.QTimer(self)
        # self.timer.timeout.connect(self.__update__)
        # self.fps = 1
        # self.timer.start(1000/self.fps)
        # Init window:
        self.initWindow()
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
        btnRempvePic = wd.QPushButton("-", self)
        btnRempvePic.setMaximumHeight(32)
        sp3.addWidget(btnRempvePic)
        sp0.addWidget(self.sp1)
        layout.addWidget(sp0)
        sp0.addWidget(pn1)
        self.sp1.addWidget(pn0)
        self.setLayout(layout)

        # Main Image:
        self.mainImg = imageFrame(self)
        BIG_IMAGE = self.mainImg
        self.mainImg.setSizeInterval(256, 256, 2560, 2560)
        self.mainImg.loadable = False
        self.mainImg.recursion = False  # Turn off recursive feedback
        self.mainImg.inputAble = False
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
        # addModBtn.mousePressEvent = self.selectImg
        addModBtn.mousePressEvent = self.selectAndAddModifier
        pipeTools.addWidget(self.selectModifier)
        pipeTools.addWidget(addModBtn)

        self.pipelineWidgets = cst.Packlist(self)
        pipeArea.addWidget(pipeTools)
        pipeArea.addWidget(self.pipelineWidgets)
        self.sp4.addWidget(pipeArea)

    def update(self):
        pass

    def selectImg(self, event):
        k = selectImage()
        print(k)

    def update_on_command(self):
        pass

    def __update__(self):
        self.update()

    def addImage(self, event):
        global GLOBAL_IMAGES
        _add = imageFrame(self)
        _add.setSize(128, 128)
        _add.bigImg = self.mainImg
        GLOBAL_IMAGES.append(_add)
        self.imagesSection.addWidget(_add)
        pl = PipelineWidget(self, _add)
        self.pipelineWidgets.addWidget(pl)
        _add.selectThis()

    def selectAndAddModifier(self, event):
        global MODIFIERS
        self.addModifierToSelected(
            MODIFIERS[self.selectModifier.currentIndex()]
        )

    def addModifierToSelected(self, modf):
        global SELECTED
        if SELECTED is not None:
            previous = SELECTED.widgetPipeline.addModifier(modf)
            self.mainImg.disconnectParent()
            SELECTED.widgetPipeline.at.connect(self.mainImg)
            previous.pipe()
            self.update_on_command()

    def modifierWidgets(self):
        global MODIFIERS

        comboBox = wd.QComboBox()
        for modf in MODIFIERS:
            comboBox.addItem(modf.__name__)
        return comboBox


class SelectImageDialog(wd.QDialog):  # $sid

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
        for imgFrm in GLOBAL_IMAGES:
            imgs = imgFrm.getImageGroup()
            for img in imgs:
                _add = ReferenceImage(self, img)
                images.addWidget(_add)

    def InitWindow(self):
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, self.width, self.height)
        self.mainUI()

    def getValues(self):
        return self.register

    def selectImg(self):
        if self.register is not None:
            self.accept()

    def cancel(self):
        self.reject()


def selectImage():
    GLOBAL_IMAGES
    dlg = SelectImageDialog()
    if dlg.exec_():
        return dlg.register
    else:
        return None


def selectOption(txt, options):
    pass
    # wd.Dial


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
