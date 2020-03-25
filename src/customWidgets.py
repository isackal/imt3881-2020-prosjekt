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

    def addWidget(self, widget):
        self._layout.addWidget(widget, 0, cr.Qt.AlignLeft)

    def setMargins(self, left, top, right, bottom):
        self._layout.setContentsMargins(left, top, right, bottom)

    def dragEnterEvent(self, e):
        print("Konnichiwa")


FloatValidator = QtGui.QRegExpValidator(cr.QRegExp(r"[-]?\d+[.]?\d*"))
IntValidator = QtGui.QRegExpValidator(cr.QRegExp(r"[-]?\d+"))
