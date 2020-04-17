from PyQt5 import QtGui
import PyQt5.QtWidgets as wd
import PyQt5.QtCore as cr


def noFunction():  # This function is a place holder function.
    pass


class DragableWidget(wd.QWidget):

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


class MiniButton(wd.QLabel):

    def __init__(self, parent, imagesource, toolTip="what is this?"):
        wd.QLabel.__init__(self, parent)
        self.setToolTip(toolTip)
        self.piximg = QtGui.QPixmap(imagesource)
        self.setPixmap(self.piximg)


class Packlist(wd.QWidget):

    def __init__(self, parent, layout=wd.QVBoxLayout, spacing=1):
        wd.QWidget.__init__(self, parent)
        self._layout = layout()
        self._layout.setAlignment(cr.Qt.AlignTop | cr.Qt.AlignLeft)
        self._layout.setSpacing(spacing)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)
        self.widgets = []

    def addWidget(self, widget):
        self._layout.addWidget(widget, 0, cr.Qt.AlignLeft)
        self.widgets.append(widget)
        widget.masterLayout = self

    def removeWidget(self, widget):
        self.widgets.remove(widget)
        self._layout.removeWidget(widget)
        widget.masterLayout = None

    def remove(self, idx):
        wid = self.widgets[idx]
        self.removeWidget(wid)
        return wid

    def insert(self, wid, idx):
        self._layout.insertWidget(idx, wid)
        self.widgets.insert(idx, wid)
        wid.masterLayout = self

    def setMargins(self, left, top, right, bottom):
        self._layout.setContentsMargins(left, top, right, bottom)

    def moveUp(self, wid):
        """
        moves the target "wid" up in the list and returns
        the widget it switched places with.
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
        idxof = self.widgets.index(wid)
        if idxof < len(self.widgets)-1:
            return self.widgets[idxof+1]
        else:
            return None

    def previous(self, wid):
        idxof = self.widgets.index(wid)
        if idxof > 0:
            return self.widgets[idxof-1]
        else:
            return None

    def swap(self, idx1, idx2):
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


# The following class was made to use referenced values:
class ReferenceValue:

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

    def __init__(self, txt, _id, reference, parent=None):
        wd.QPushButton.__init__(self, txt, parent)
        self.idx = _id
        self.refr = reference
        self.mousePressEvent = self.press
        self.onPress = noFunction

    def press(self, event):
        self.refr.set(self.idx)
        self.onPress()


class OptionsDialog(wd.QDialog):  # $sid

    def __init__(self, txt, options, cancelButton=True, parent=None):
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
        if self.selectedIndex.get() == -1:
            self.reject()
        else:
            self.accept()
            print(self.selectedIndex.get())


FloatValidator = QtGui.QRegExpValidator(cr.QRegExp(r"[-]?\d+[.]?\d*"))
IntValidator = QtGui.QRegExpValidator(cr.QRegExp(r"[-]?\d+"))
