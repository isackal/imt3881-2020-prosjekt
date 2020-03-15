import sys
import PyQt5.QtWidgets as wd
import PyQt5.QtCore as cr
from PyQt5 import QtGui

import numpy as np
import scipy as sp
import imageio as im
import modules.bitcrusher as bitcrusher

#TODO 2.9

_NUMBERED_IMAGE=0

SELECTED=None

GLOBAL_IMAGES=[]

mouse_x=0
mouse_y=0

class Collapser(wd.QWidget):
    def __init__(self,parent):
        wd.QWidget.__init__(self,parent)
        layout=wd.QVBoxLayout()
        top=wd.QGroupBox()
        topl=wd.QHBoxLayout()
        topl.setContentsMargins(8,0,8,0)
        topl.setAlignment(cr.Qt.AlignLeft)
        topl.setStretch(0,0)
        top.setLayout(topl)
        self.minimizeButton=wd.QPushButton('+',self)
        self.minimizeButton.setFixedHeight(32)
        self.minimizeButton.setFixedWidth(32)
        self.minimizeButton.mousePressEvent = self.toggle
        topl.addWidget(self.minimizeButton)
        self.title="Modify"
        self.label=wd.QLabel(self.title,self)
        topl.addWidget(self.label)
        top.setFixedHeight(48)
        self.vgrp=wd.QGroupBox()
        self.content=wd.QFormLayout()
        self.content.setSpacing(0)
        self.vgrp.setLayout(self.content)
        self.mainUI()
        self.setLayout(layout)
        layout.addWidget(top)
        layout.setSpacing(0)
        layout.addWidget(self.vgrp)
        self.open=False
        self.vgrp.setHidden(True)
    def toggle(self,event):
        if self.open:
            self.vgrp.setHidden(True)
            self.minimizeButton.text="+"
            self.open=False
        else:
            self.vgrp.setHidden(False)
            self.minimizeButton.text="-"
            self.open=True
    def mainUI(self):
        for i in range(10):
            btn=wd.QPushButton("Button %2d"%i,self)
            self.content.addWidget(btn)

class TypeInput(wd.QLineEdit):
    def __init__(self,parent,_type):
        wd.QLineEdit.__init__(self,parent)
        self._type=_type
        self.textEdited.connect(self.validate)
    def validate(self, text):
        self.setText( str(  self.getValue()  ) )
    def getValue(self):
        try:
            val=self._type( self.text() )
        except:
            val=0
        return val

class ModifierWidget(Collapser):
    def __init__(self,parent,modf,reference):
        self.reference=reference
        self.modifier=modf()
        self.dtas=[]
        Collapser.__init__(self,parent)
    def mainUI(self):
        self.img=imageFrame(self)
        self.img.setFixedWidth(128)
        self.img.setFixedHeight(128)
        self.reference.connect(self.img)
        self.content.addWidget(self.img)
        self.title=self.modifier.name
        self.label.setText(self.title)
        grd=wd.QGridLayout()
        _row=0
        self.modifier.values[0]=self.img.picdata
        for param in self.modifier.params[1:]:
            lbl=wd.QLabel(param[0],self)
            inp=TypeInput(self,int)
            inp.setText(str(param[2]))
            inp.textEdited.connect(self.onUpdateText)
            inp.dtatp=param[1]
            grd.addWidget(lbl,_row,0)
            grd.addWidget(inp,_row,1)
            self.dtas.append(inp)
            _row+=1
        _add=wd.QWidget(self)
        _add.setLayout(grd)
        self.content.addWidget(_add)
        self.img.modifier=self.modifier
    def onUpdateText(self,text):
        for i in range(len(self.modifier.values)-1):
            self.modifier.values[i+1]=self.dtas[i].getValue()
        self.reference.pipe()

class imageFrame( wd.QWidget ):
    def __init__(self,parent):
        wd.QWidget.__init__(self,parent)
        self.title="?"
        self.top=0
        self.left=0
        self.width=384
        self.height=240
        self.picdata=None
        self.qimg=None
        self.piximg=None
        self.imgObject=wd.QLabel(self)
        self.init_image()
        self.parent=parent
        self.mainUI()
        self.zoom=1
        self.z=0
        self.zoomRange=(-4,4)
        self.fname=""
        self.zoomSensitivity=128
        self.src=None
        self.srcSet=False
        self.loadable=True
        self.isSelected=False
        self.recursion=True
        self.pipes=[]
        self.src=None #Which object is piping to this.
        #0: Binary, 1: Grayscale [0,1], 2: RGBA: [0,255] #
        self.format=2
        self.bigImg=None
        self.modifier=None
        self.widgetPipeline=None
    def addToLibrary(self):
        global GLOBAL_IMAGES, _NUMBERED_IMAGE
        if self not in GLOBAL_IMAGES:
            if self.title=="?":
                self.title="untiteled%d" % _NUMBERED_IMAGE
                _NUMBERED_IMAGE+=1
            GLOBAL_IMAGES.append(self)
    def load_image(self,fil):
        o_image=im.imread(fil)
        _image=np.ones( (o_image.shape[0],o_image.shape[1],4) )*255
        _image[:,:,0]=o_image[:,:,0] #RED
        _image[:,:,1]=o_image[:,:,1] #GREEN
        _image[:,:,2]=o_image[:,:,2] #BLUE
        if o_image.shape[2]==4:
            _image[:,:,3]=o_image[:,:,3] #BLUE
        self.picdata=_image.astype(np.uint8)
        self.update_image()
    def update_image(self):
        #if self.srcSet:
        #    print("Inherited data")
        #    self.picdata=self.src.picdata
        self.qimg=QtGui.QImage(self.picdata,self.picdata.shape[1],
            self.picdata.shape[0],QtGui.QImage.Format_RGBA8888)
        self.piximg=QtGui.QPixmap( self.qimg )
        self.piximg=self.piximg.scaledToWidth( self.picdata.shape[1]*self.zoom )
        self.imgObject.setPixmap( self.piximg )
        #if self.recursion and(self.parent.selected==self):
        #    print("Updating parent")
        #    self.parent.mainImg.update()
    def init_image(self):
        self.picdata = (np.ones((32,32,4))*255).astype(np.uint8)
    def update(self):
        self.update_image()
    def __update__(self):
        self.update()
        self.update_image()
    def mainUI(self):
        splitter1=wd.QSplitter(cr.Qt.Vertical)
        splitter2=wd.QSplitter(cr.Qt.Horizontal)
        self.vlayout=wd.QVBoxLayout()
        self.vobject = wd.QScrollArea()
        self.vobject.setWidget(self.imgObject)
        self.vobject.setWidgetResizable(True)
        self.imgObject.mouseDoubleClickEvent = self.load_dialog
        self.imgObject.mousePressEvent = self.select
        #self.imgObject.wheelEvent = self.zooming
        splitter1.addWidget(self.vobject)
        zinBtn=wd.QPushButton("+",self)
        zutBtn=wd.QPushButton("-",self)
        zinBtn.setMaximumHeight(32)
        zutBtn.setMaximumHeight(32)
        zinBtn.mousePressEvent=self.zoomIn
        zutBtn.mousePressEvent=self.zoomOut
        splitter2.addWidget(zinBtn)
        splitter2.addWidget(zutBtn)
        splitter2.setMaximumHeight(64)

        splitter1.addWidget(splitter2)
        self.vlayout.addWidget(splitter1)
        """
        if self.width>0:
            self.setGeometry(0,0,self.width,self.height)
            self.setMaximumWidth(self.height)
        if self.height>0:
            self.setMaximumHeight(self.height)
        """
        self.setLayout(self.vlayout)
        #self.move(128,128)
        #self.setGeometry(0,0,self.width,self.height)
        #self.setMinimumWidth(self.width/4)
        #self.setMinimumHeight(self.height/4)
    def zoomIn(self,event):
        print("Zoom In")
        self.z+=1
        if self.z>self.zoomRange[1]:
            self.z=self.zoomRange[1]
        print(self.z)
        self.zoom=2**(self.z)
        self.update_image()
    def zoomOut(self,event):
        print("Zoom Out")
        self.z-=1
        if self.z<self.zoomRange[0]:
            self.z=self.zoomRange[0]
        print(self.z)
        self.zoom=2.**(self.z)
        print(self.zoom)
        self.update_image()
    def load_dialog(self,event):
        if self.loadable:
            fname=wd.QFileDialog.getOpenFileName(self.parent,'Sorce Image','./','Image Files (*.jpg *.jpeg *.gif *.png)')
            if fname[0]!="" and fname[0]!=self.fname:
                self.fname=fname[0]
                self.load_image(fname[0])
            self.pipe()
    def disconnectParent(self):
        if self.src is not None:
            self.src.pipes.remove(self)
            self.src=None
    def disconnectChild(self,child):
        if child in self.pipes:
            self.pipes.remove(child)
    def connect(self,child):
        if child.src is None:
            print(child.src)
            child.src=self
            self.pipes.append( child )
    def zooming(self,event):
        dt=event.angleDelta()
        delta = int( round(dt.y()/self.zoomSensitivity) )
        print(delta)
        if delta!=0:
            self.z+=delta
            if self.z<self.zoomRange[0]:
                self.z=self.zoomRange[0]
            if self.z>self.zoomRange[1]:
                self.z=self.zoomRange[1]
            self.zoom=2**(self.z)
            self.update_image()
    def drag(self,event):
        print(event.x())
    def setSource(self,_src):
        self.srcSet=True
        self.src=_src
        self.loadable=False
        self.update_image()
    def select(self,event):
        self.selectThis()
    def selectThis(self):
        global SELECTED
        if (self.bigImg is not None) and (self!=self.bigImg) and (self.bigImg.src!=self):
            print("#7")
            if SELECTED is not None:
                SELECTED.widgetPipeline.toggleOff()
            self.bigImg.disconnectParent()
            self.widgetPipeline.at.img.connect(self.bigImg)
            self.widgetPipeline.at.img.pipe()
            self.parent.update_on_command()
            self.widgetPipeline.toggleOn()
            SELECTED=self
    def pipe(self):
        for child in self.pipes:
            print("Sending data to %s"%child.title)
            child.picdata=self.picdata
            child.modify()
            child.update_image()
            child.pipe()
    def modify(self):
        print(")))")
        if self.modifier is not None:
            self.modifier.values[0]=self.picdata
            print("!!!")
            self.picdata=self.modifier.transform()
    def setSizeInterval(self,minW,minH,maxW,maxH):
        self.setMinimumWidth(minW)
        self.setMinimumHeight(minH)
        self.setMaximumHeight(maxH)
        self.setMaximumWidth(maxW)
    def delete(self):
        self.disconnectParent()

class operatorImage(imageFrame):
    def __init__(self,parent,opr):
        imageFrame.__init__(self,parent)
        self.setFixedHeight(128)
        self.setFixedWidth(128)
        self.opr=opr
    def modify(self):
        pass

class operator(wd.QWidget):
    def __init__(self,parent,func,name="operator"):
        wd.QWidget.__init__(self,parent)
        self.name=name
        grp=wd.QGroupBox(self.name)
        self.layout=wd.QGridLayout()
        grp.setLayout(self.layout)
        vlayout=wd.QVBoxLayout()
        vlayout.addWidget(grp)
        self.setLayout(vlayout)
        self.y=0
        self.dataFields=[]
        self.src=None
        self.srcSet=False
        self.pipes=[]
        self.res=None
    def addData(self,dataName,dataType,defaultValue):
        lbl = wd.QLabel(dataName,self)
        dta = wd.QLineEdit(self)
        dta.setText(str(defaultValue))
        self.dataFields.append( (dataName,defaultValue,dataType) )
        self.layout.addWidget(lbl,self.y,0)
        self.layout.addWidget(dta,self.y,1)
        self.y+=1
    def calculate(self):
        self.res = self.src*1
        self.updatePipeline()
    def updatePipeline(self):
        for child in self.pipes:
            child.src=self.res
            child.calculate()

class Canvas(wd.QWidget):
    def __init__(self,parent):
        wd.QWidget.__init__(self,parent)
        self.setGeometry(0,0,2000,2000)
        self.setAttribute(cr.Qt.WA_TransparentForMouseEvents)
    def instantiate(self,widget):
        widget.setParent(self)
        widget.update()
    def paintEvent(self,event):
        painter=QtGui.QPainter(self)
        painter.setPen(cr.Qt.red)
        painter.drawLine(0,0,2000,2000)

class MoveableBox(wd.QGroupBox):
    def __init__(self,parent,lbl):
        wd.QGroupBox.__init__(self,parent)
        self.title=lbl
        self.setFixedWidth(128)
        self.setFixedHeight(192)
        self.setAcceptDrops(True)
        self.mousePressEvent=self.mousePress
        self.mouseReleaseEvent=self.mouseRelease
        self.mouseMoveEvent=self.mouseMove
        self.ox=0
        self.oy=0
        self.isDragged=False
    def mousePress(self, event):
        self.isDragged=True
        self.ox=event.x()-self.x()
        self.oy=event.y()-self.y()
    def mouseMove(self, event):
        if (self.isDragged):
            self.move(event.x()-self.ox,event.y()-self.oy)
    def mouseRelease(self, event):
        print(":(")

class PipelineWidget(wd.QFrame):
    def __init__(self,parent,refr):
        wd.QFrame.__init__(self,parent)
        self.playout=wd.QVBoxLayout()
        self.playout.setAlignment(cr.Qt.AlignTop)
        self.setLayout(self.playout)
        self.reference=refr
        self.at=self.reference
        self.pipes=[]
        self.visible=False
        self.setHidden(True)
        refr.widgetPipeline=self
    def addModifier(self,modifier):
        c1=ModifierWidget(self,modifier,self.at)
        self.pipes.append(c1)
        self.at=c1
        self.playout.addWidget(c1)
    def toggle(self):
        if self.visible:
            self.visible=False
            self.setHidden(True)
        else:
            self.visible=True
            self.setHidden(False)
    def toggleOn(self):
        self.visible=True
        self.setHidden(False)
    def toggleOff(self):
        self.visible=False
        self.setHidden(True)

class Window(wd.QDialog):
    def __init__(self, parent=None):
        super().__init__() #inherit from QDialog (aka. window class)
        self.title="defualt"
        self.top=100
        self.left=100
        self.width=400
        self.height=400
        self.setMinimumHeight(200)
        self.setMinimumWidth(200)
        #self.timer=cr.QTimer(self)
        #self.timer.timeout.connect(self.__update__)
        #self.fps=1
        #self.timer.start(1000/self.fps)
        #Init window:
        self.initWindow()
        self.images=[]
        self.imgSelected=False
        self.selected=None
    def initWindow(self):
        """
        Function that initializes the main windows and
        shows it.
        """
        self.setWindowIcon(QtGui.QIcon("home.png"))
        self.setWindowTitle(self.title)
        self.setGeometry(self.left,self.top,self.width,self.height)
        self.mainUI()
        self.show()
    def mainUI(self):
        layout = wd.QHBoxLayout()
        pn0=wd.QScrollArea()
        grp=wd.QWidget()
        pn0.setWidget(grp)
        self.imagesSection=wd.QFormLayout()
        pn0.setWidgetResizable(True)
        grp.setLayout(self.imagesSection)
        pn1=wd.QScrollArea()
        pn0.setMinimumWidth(192)
        pn0.setMaximumWidth(256)
        sp0=wd.QSplitter(cr.Qt.Horizontal)
        self.sp1=wd.QSplitter(cr.Qt.Vertical)
        sp3=wd.QSplitter(cr.Qt.Horizontal)
        self.sp1.addWidget(sp3)
        btnAddPic = wd.QPushButton("+",self)
        btnAddPic.setMaximumHeight(32)
        btnAddPic.mousePressEvent = self.addImage
        sp3.addWidget(btnAddPic)
        btnRempvePic = wd.QPushButton("-",self)
        btnRempvePic.setMaximumHeight(32)
        sp3.addWidget(btnRempvePic)
        sp0.addWidget(self.sp1)
        layout.addWidget(sp0)
        sp0.addWidget(pn1)
        self.sp1.addWidget(pn0)
        self.setLayout(layout)

        #Main Image:
        self.mainImg = imageFrame(self)
        self.mainImg.setSizeInterval(256,256,2560,2560)
        self.mainImg.loadable=False
        self.mainImg.recursion=False #Turn off recursive feedback
        pn1.setWidgetResizable(True)
        pn1.setWidget(self.mainImg)

        #Pipelines:
        #panel=wd.QFrame()
        self.sp4 = wd.QSplitter(cr.Qt.Vertical)
        #playout=wd.QVBoxLayout()
        #playout.setAlignment(cr.Qt.AlignTop)
        #panel.setLayout(playout)
        sp0.addWidget(self.sp4)
        #c1=ModifierWidget(self,bitcrusher.Bitcrusher,self.mainImg)
        #playout.addWidget(c1)
        #sp4.addWidget(panel)


    def update(self):
        pass
    def update_on_command(self):
        print("#3")
        """
        if self.mainImg.srcSet:
            print("#4")
            self.mainImg.src=self.selected
            self.mainImg.update()
        """
    def __update__(self):
        self.update()
    def addImage(self,event):
        _add=imageFrame(self)
        _add.setFixedHeight(128)
        _add.setFixedWidth(128)
        _add.bigImg=self.mainImg
        self.images.append( _add )
        self.imagesSection.addWidget( _add )
        pl=PipelineWidget(self,_add)
        pl.addModifier(bitcrusher.Bitcrusher)
        self.sp4.addWidget(pl)
        _add.selectThis()

def uiStart():
    App = wd.QApplication(sys.argv)
    main = Window()
    sys.exit( App.exec() )

"""
MAIN
"""
if __name__=="__main__":
    uiStart()