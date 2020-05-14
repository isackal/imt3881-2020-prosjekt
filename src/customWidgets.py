from PyQt5 import QtGui
import PyQt5.QtWidgets as wd
import PyQt5.QtCore as cr
from time import time
from threading import Lock


def noFunction(*args, **qwargs):  # This function is a place holder function.
    pass


class DragableWidget(wd.QWidget):
    """
    A DragAbleWidget can be dragged.
    """
    def __init__(self, parent):
        wd.QWidget.__init__(self, parent)
        self.dragStartPosition = self.pos()

    def mousePressEvent(self, event):
        if event.button() == cr.Qt.LeftButton:
            self.dragStartPosition = event.pos()

    def mouseMoveEvent(self, event):
        if not (event.buttons() & cr.Qt.LeftButton):
            return
        if (
            (event.pos() - self.dragStartPosition).manhattanLength() <
            wd.QApplication.startDragDistance()
        ):
            return
        pixmap = QtGui.QPixmap(self.size())
        painter = QtGui.QPainter(pixmap)
        painter.drawPixmap(self.rect(), self.grab())
        painter.end()
        mimedata = cr.QMimeData()
        mimedata.setText("???")
        drag = QtGui.QDrag(self)
        drag.setMimeData(mimedata)
        drag.setPixmap(pixmap)
        drag.setHotSpot(event.pos())
        drag.exec_(cr.Qt.CopyAction | cr.Qt.MoveAction)


class NumericInput(wd.QLineEdit):
    """
    A modified QLineEdit that is used for
    numeric input.
    """
    def __init__(self, _type, parent=None):
        wd.QLineEdit.__init__(self, parent)
        # Values used to determine value changes when dragged:
        self.dragStartPosition = self.pos()
        self.dragAtPosition = self.x()
        self.dragStartX = self.x()
        self.startValue = 0.
        self.addValue = 0.

        self._type = _type  # Type (float, int)
        self._time = 0.  # Used to determine value changes.
        self.sensitivity = 1.  # Value change sensitivity
        self.precision = 4  # Float precision
        if _type is int:
            self.sensitivity = 0.1
        else:
            self.sensitivity = 0.01
        if _type == int:
            self.setValidator(IntValidator)
        else:
            self.setValidator(FloatValidator)

    def getValue(self):
        """
        Get the value of the input field

        Returns
        -------

        val :   <int>, <float>
        """
        val = self._type(0)
        if self.text() != "":
            try:  # If casting to the right value works:
                val = self._type(self.text())  # returned value is input
            except Exception:  # if an error occured (casting didn't work)
                val = 0  # returned value is 0
        return val

    def setValue(self, val):
        """
        Sets the input value to val

        Parameters
        ----------

        val :   <int>, <float>
        """
        if self._type is float:
            self.setText("%.4f" % val)
        else:
            self.setText("%d" % round(val))

    def mousePressEvent(self, event):
        if event.button() == cr.Qt.LeftButton:  # Left clicked:
            self.dragStartPosition = event.pos()
            self.dragAtPosition = event.x()
            self._time = time()
            self.startValue = 1.*self.getValue()
            self.addValue = 0.
            self.dragStartX = event.x()

    def mouseMoveEvent(self, event):
        """
        Allows the user to click and drag the input to interactively change
        the value of the input.
        """
        if not (event.buttons() & cr.Qt.LeftButton):  # if not clicked first
            return
        if (
            (event.pos() - self.dragStartPosition).manhattanLength() <
            wd.QApplication.startDragDistance()
        ):  # If outside of scope
            return
        # Determine value change:
        newTime = time()
        dt = newTime - self._time
        dx = event.x() - self.dragAtPosition
        dxx = event.x() - self.dragStartX
        _add = self._type(dx / dt)
        self.addValue += _add
        self._time = newTime
        self.dragAtPosition = event.x()
        self.setValue(self._type(
            self.startValue + dxx * self.sensitivity
            ))


class MiniButton(wd.QLabel):
    """
    A button that takes an image. Is basically just an image.
    """
    def __init__(self, parent, imagesource, toolTip="what is this?"):
        wd.QLabel.__init__(self, parent)
        self.setToolTip(toolTip)
        self.piximg = QtGui.QPixmap(imagesource)
        self.setPixmap(self.piximg)


def removeFromPacklist(widget):
    """
    Remove widget from a packlist if a packlist (masterlayout)
    is assigned.
    """
    try:  # See if the widget has an instance of masterLayout
        widget.masterLayout
    except Exception:
        # Do nothing because then there is no masterLayout to disconnect
        pass
    else:  # If yes, remove it
        if widget.masterLayout is not None:
            widget.masterLayout.removeWidget(widget)


class Packlist(wd.QWidget):
    """
    A widget that packes widgets together in a list manner either
    horizontally or vertically
    """
    def __init__(self, parent, layout=wd.QVBoxLayout, spacing=1):
        """
        Parameters
        ----------

        parent  :   <QWidget>
                    Parent of the packlist

        layout  :   <QLayout>
                    Can be either QVBoxLayout or QHBoxLayout

        spacing :   <float>
                    the amount of space between the widgets
        """
        wd.QWidget.__init__(self, parent)
        self._layout = layout()
        self._layout.setAlignment(cr.Qt.AlignTop | cr.Qt.AlignLeft)
        self._layout.setSpacing(spacing)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)
        # self.widgets is a list used to keep track of widgets:
        self.widgets = []  # List of widgets in the list

    def addWidget(self, widget):
        """
        Adds a widget to the packlist

        Parameters
        ----------

        widget  :   <QWidget>
                    widget to be added
        """
        self._layout.addWidget(widget, 0, cr.Qt.AlignLeft)  # add to layout
        self.widgets.append(widget)  # add to list
        widget.masterLayout = self

    def removeWidget(self, widget):
        """
        Removes a widget from the packlist

        Parameters
        ----------

        widget  :   <QWidget>
                    widget to be removed
        """
        self.widgets.remove(widget)
        self._layout.removeWidget(widget)
        widget.masterLayout = None

    def remove(self, idx):
        """
        Removes a widget from the packlist
        given an index

        Parameters
        ----------

        idx  :  int
                index of the widget to be removed
        """
        wid = self.widgets[idx]
        self.removeWidget(wid)
        return wid

    def insert(self, wid, idx):
        """
        Inserts a widget into the packlist

        Parameters
        ----------

        wid :   <QWidget>
                widget to be added

        idx :   <int>
                index of where to insert the widget
        """
        self._layout.insertWidget(idx, wid)
        self.widgets.insert(idx, wid)
        wid.masterLayout = self

    def setMargins(self, left, top, right, bottom):
        """
        Sets the margins

        Parameters
        ----------

        left    :   <int>,  <float>
                    left margin
        top     :   <int>,  <float>
                    top margin
        right   :   <int>,  <float>
                    right margin
        bottom  :   <int>,  <float>
                    bottom margin
        """
        self._layout.setContentsMargins(left, top, right, bottom)

    def moveUp(self, wid):
        """
        moves the target "wid" up in the list and returns
        the widget it switched places with.

        Parameters
        ----------

        wid :   <QWidget>
                Widget to be moved
        """
        idxof = self.widgets.index(wid)
        ret = None
        if (idxof > 0):
            ret = self.widgets[idxof - 1]
            self.removeWidget(wid)
            self.insert(wid, idxof - 1)
        return ret

    def moveDown(self, wid):
        """
        moves the target "wid" down in the list and returns
        the widget it switched places with.

        Parameters
        ----------

        wid :   <QWidget>
                Widget to be moved
        """
        _len = len(self.widgets)
        idxof = self.widgets.index(wid)
        ret = None
        if (idxof < _len - 1):
            ret = self.widgets[idxof + 1]
            self.removeWidget(wid)
            self.insert(wid, idxof + 1)
        return ret

    def next(self, wid):
        """
        Get the next widget (after to wid) in the list.

        Parameters
        ----------

        wid :   <QWidget>

        Returns
        -------

        self.widgets[idxof+1]   :   <QWidget>
        """
        idxof = self.widgets.index(wid)
        if idxof < len(self.widgets)-1:
            return self.widgets[idxof+1]
        else:
            return None

    def previous(self, wid):
        """
        Get the previous widget (before wid) in the list.

        Parameters
        ----------

        wid :   <QWidget>

        Returns
        -------

        self.widgets[idxof-1]   :   <QWidget>
        """
        idxof = self.widgets.index(wid)
        if idxof > 0:
            return self.widgets[idxof-1]
        else:
            return None

    def swap(self, idx1, idx2):
        """
        Swap places of two widgets

        Parameters
        ----------

        idx1    :   <int>
                    index of the first widget

        idx2    :   <int>
                    index of the second widget
        """
        _left = -1
        _right = -1
        if idx1 > idx2:
            _left = idx2
            _right = idx1
        else:
            _left = idx1
            _right = idx2
        wid2 = self.remove(_right)
        self.insert(wid2, _left)
        wid1 = self.remove(_left)
        self.insert(wid1, _right)


class ReferenceValue:
    """
    Classes in python are always passed as reference. This
    class holds a value such it can be passed by reference.
    """
    def __init__(self, value):
        self.value = value

    def set(self, other):
        if type(other) is ReferenceValue:
            self.value = other.value
        else:
            self.value = other

    def get(self):
        return self.value


class IndexedButton(wd.QPushButton):
    """
    A button holding a value refering to an index.
    """
    def __init__(self, txt, _id, reference, parent=None):
        """
        Parameters
        ----------

        txt :   <string>
                label of the button

        _id :   <int>
                index of something

        reference   :   <ReferenceValue>
                        holding a reference Value

        parent      :   <QWidget>
                        Parent of button
        """
        wd.QPushButton.__init__(self, txt, parent)
        self.idx = _id
        self.refr = reference
        self.mousePressEvent = self.press
        self.onPress = noFunction

    def press(self, event):
        """
        Set the referenced value to be its own index value
        """
        self.refr.set(self.idx)
        self.onPress()


class OptionsDialog(wd.QDialog):  # $sid
    """
    A dialog for making choices
    """
    def __init__(self, txt, options, cancelButton=True, parent=None):
        """
        Parameters
        ----------

        txt     :   <string>
                    text to be displayed in the dialog

        options :   <list>
                    a list of strings, or options

        cancelButton    :   Whether a cancelbutton is provided

        parent          :   Parent of dialog
        """
        super().__init__()
        self.txt = txt
        self.options = options
        self.selectedIndex = ReferenceValue(-1)
        self.selectedWord = ""
        self.cb = cancelButton
        self.initUI()

    def initUI(self):
        vlayout = wd.QVBoxLayout()
        self.setLayout(vlayout)
        lbl = wd.QLabel(self.txt)
        lbl.setWordWrap(True)
        vlayout.addWidget(lbl)
        ans = wd.QWidget()  # possible answers widget
        hlayout = wd.QHBoxLayout()
        ans.setLayout(hlayout)
        vlayout.addWidget(ans)
        idxCount = 0
        for option in self.options:
            btn = IndexedButton(option, idxCount, self.selectedIndex)
            btn.onPress = self.press
            idxCount += 1
            hlayout.addWidget(btn)
        if self.cb:
            btn = IndexedButton("Cancel", -1, self.selectedIndex)
            btn.onPress = self.press
            hlayout.addWidget(btn)

    def press(self):
        """
        Function that either accepts or rejects the dialog.
        Called when pushing a button.
        """
        if self.selectedIndex.get() == -1:
            self.reject()
        else:
            self.accept()


class Progression(wd.QDialog):
    """
    A Progression dialog used to display progression of a
    function.
    """
    def __init__(self, _max=1, txt="Progress"):
        super().__init__()
        self.p = 0  # Progression
        self.pMax = _max  # max Progression
        self.finished = 0  # 0 to 1, how much is finished?
        self.lock = Lock()  # Lock (mutex) used when accessing its resources.
        self.txt = txt  # Text to be displayed
        self.retValue = None  # value to store a return value
        self.initUI()  # Creates its ui
        self.timer = cr.QTimer(self)  # timer to regulary call updates
        self.timer.timeout.connect(self._update)
        self.fps = 24  # We do not need more fps for a loading bar
        self.timer.start(1000/self.fps)

    def progressionRead(self):
        """
        Reads its progression and store it in the
        self.finished value.
        """
        self.lock.acquire()
        self.finished = self.p/self.pMax
        self.lock.release()

    def initUI(self):
        vlayout = wd.QVBoxLayout()
        self.setLayout(vlayout)
        self.pbar = wd.QProgressBar()
        self.pbar.setMaximum(100)
        lbl = wd.QLabel(self.txt)
        vlayout.addWidget(lbl)
        vlayout.addWidget(self.pbar)
        self.setGeometry(0, 0, 320, 240)

    def _update(self):
        """
        Called every time intervall.
        """
        self.progressionRead()
        self.pbar.text = "%.8f / 100" % self.finished
        self.pbar.setValue(self.finished*100)
        if (self.finished > 0.999):
            self.accept()


# Float validator using regex expression
FloatValidator = QtGui.QRegExpValidator(cr.QRegExp(r"[-]?\d+[.]?\d*"))

# Integer validator using regex expression
IntValidator = QtGui.QRegExpValidator(cr.QRegExp(r"[-]?\d+"))
