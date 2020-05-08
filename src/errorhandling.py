import PyQt5.QtWidgets as wd
import matplotlib


LOGG = "logg.txt"


def loggText(txt):
    fil = open(LOGG, 'a')
    fil.write(txt)
    fil.close()


def displayError(error):
    loggText("ERROR:%s\n\n" % error)
    error_dialog = wd.QErrorMessage(error)
    error_dialog.showMessage(error)


def displayWarning(warning):
    loggText("WARNING:%s\n\n" % warning)
    warning_dialog = wd.QMessageBox()
    warning_dialog.setIcon(wd.QMessageBox.Warning)
    warning_dialog.setText(warning)
    warning_dialog.setStandardButtons(
        wd.QMessageBox.Ok
    )
    warning_dialog.exec_()


def showImageData(img, message="no message"):
    matplotlib.pyplot.title(message)
    matplotlib.pyplot.imshow(img)
    matplotlib.pyplot.show()
