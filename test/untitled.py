from PyQt5 import QtWidgets, uic, QtGui, QtCore
from PyQt5.QtCore import QPoint
import pyqtgraph as pg
import sys
import numpy as np
from typing import List
from algorithms import approximation, interpolation
from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "Point(" + str(self.x) + ";" + str(self.y) + ")"


class Data(list):
    def __init__(self):
        super(Data, self).__init__()
        self.distance_to_remove = 100

    def append(self, point: Point) -> None:
        super(Data, self).append(point)
        self.sort(key=lambda p: p.x)

    def x(self) -> List[float]:
        return [point.x for point in self]

    def y(self) -> List[float]:
        return [point.y for point in self]

    def remove_all(self):
        while len(self) != 0:
            self.pop(0)

    def check_to_remove(self, x, y, distance):
        to_remove = None
        for p in self:
            if ((x - p.x) ** 2 + (y - p.y) ** 2) ** 0.5 < distance / self.distance_to_remove:
                to_remove = p
        if to_remove is not None:
            self.remove(to_remove)

    def set_distance_to_remove(self, distance):
        self.distance_to_remove = distance


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        uic.loadUi('untitled.ui', self)
        self.pg: pg.PlotWidget = self.graphWidget
        self.vb: pg.ViewBox = self.pg.plotItem.vb
        self.approx: QtWidgets.QCheckBox = self.checkBoxApproximate
        self.interp: QtWidgets.QCheckBox = self.checkBoxInterpolate
        self.approx_type: QtWidgets.QSpinBox = self.spinBoxApproximate
        self.interp_type: QtWidgets.QComboBox = self.comboBoxInterpolate
        self.clear_button: QtWidgets.QPushButton = self.pushButtonClear

        self.data = Data()
        self.pg.setTitle("Your Title Here")
        self.pg.showGrid(x=True, y=True)
        self.pen = pg.mkPen(color=(255, 0, 0))

        self.pg.scene().sigMouseClicked.connect(self.mouse_clicked)
        self.approx.stateChanged.connect(self.changed_approx)
        self.interp.stateChanged.connect(self.changed_interp)
        self.approx_type.valueChanged.connect(self.changed_approx_type)
        self.interp_type.currentIndexChanged.connect(self.changed_interp_type)
        self.clear_button.pressed.connect(self.pressed_button_clear)

    def mouse_clicked(self, event: MouseClickEvent):
        point = self.vb.mapSceneToView(event.scenePos())
        x = point.x()
        y = point.y()
        if event.button() == 4:  # mouse wheel
            scale_x = self.vb.viewRange()[0][1] - self.vb.viewRange()[0][0]
            scale_y = self.vb.viewRange()[1][1] - self.vb.viewRange()[1][0]
            distance = (scale_x ** 2 + scale_y ** 2) ** 0.5
            self.data.check_to_remove(x, y, distance)
        elif event.button() == 1:  # left mouse key
            self.data.append(Point(x, y))
        self.plot()

    def changed_approx(self):
        self.plot()

    def changed_interp(self):
        self.plot()

    def changed_approx_type(self):
        self.plot()

    def changed_interp_type(self):
        self.plot()

    def pressed_button_clear(self):
        self.pg.clear()
        self.data.remove_all()

    def plot(self):
        self.pg.clear()
        if self.approx.isChecked():
            approx_func = approximation.least_square(self.data.x(), self.data.y(), self.approx_type.value())
            x = np.linspace(self.data.x()[0], self.data.x()[-1], 1000)
            y = approx_func(x)
            self.pg.plot(x, y, pen=pg.mkPen(color=(255, 255, 0)))
        if self.interp.isChecked():
            index = self.interp_type.currentIndex()
            if index == 0:  # linear
                interp_func = interpolation.linear(self.data.x(), self.data.y())
                x = np.linspace(self.data.x()[0], self.data.x()[-1], 1000)
                y = interp_func(x)
                self.pg.plot(x, y, pen=pg.mkPen(color=(0, 255, 255)))
            elif index == 1:  # quadratic
                interp_func = interpolation.quadratic(self.data.x(), self.data.y())
                x = np.linspace(self.data.x()[0], self.data.x()[-1], 1000)
                y = interp_func(x)
                self.pg.plot(x, y, pen=pg.mkPen(color=(0, 255, 255)))
            elif index == 2:  # spline
                interp_func = interpolation.spline(self.data.x(), self.data.y())
                x = np.linspace(self.data.x()[0], self.data.x()[-1], 1000)
                y = interp_func(x)
                self.pg.plot(x, y, pen=pg.mkPen(color=(0, 255, 255)))
            elif index == 3:  # lagrange
                interp_func = interpolation.lagrange(self.data.x(), self.data.y())
                x = np.linspace(self.data.x()[0], self.data.x()[-1], 1000)
                y = interp_func(x)
                self.pg.plot(x, y, pen=pg.mkPen(color=(0, 255, 255)))
        self.pg.plot(self.data.x(), self.data.y(), pen=None, symbol='o', symbolSize=5, symbolBrush=('b'))


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
