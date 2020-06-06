from PyQt5 import QtWidgets, uic, QtCore
import pyqtgraph as pg
import sys
import numpy as np
from typing import List
from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent
import scipy.interpolate as interp


class SmoothPointNumberError(Exception):
    pass


class Algorithms(object):
    class Approximation(object):
        @staticmethod
        def approximate_least_squares(x: List[float], y: List[float], n=1) -> callable:
            x = np.array(x, dtype=np.float64)
            y = np.array(y, dtype=np.float64)
            A = []
            b = []
            for i in range(n + 1):
                A.append([])
                b.append(sum(y[k] * x[k] ** i for k in range(len(x))))
                for j in range(n + 1):
                    if i == j == 0:
                        A[i].append(len(x))
                    else:
                        A[i].append(sum(x[k] ** (i + j) for k in range(len(x))))
            c = np.linalg.solve(np.array(A, dtype=np.float64), np.array(b, dtype=np.float64))
            return lambda x: sum(c[i] * x ** i for i in range(len(c)))

    class Interpolation(object):
        @staticmethod
        def interpolate(x: List[float], y: List[float], kind: str) -> callable:
            if kind == "lagrange":
                return interp.lagrange(x, y)
            else:
                return interp.interp1d(x, y, kind=kind)

    class Smoothing(object):
        @staticmethod
        def smooth(y: List[float], n=3) -> np.array:
            res = []
            if n == 3:
                if len(y) < 3:
                    raise SmoothPointNumberError("Points number must be >= 3")
                else:
                    res.append((5 * y[0] + 2 * y[1] - y[2]) / 6)
                    for i in range(1, len(y) - 1):
                        res.append((y[i - 1] + y[i] + y[i + 1]) / 3)
                    res.append((5 * y[-1] + 2 * y[-2] - y[-3]) / 6)
            elif n == 5:
                if len(y) < 5:
                    raise SmoothPointNumberError("Points number must be >= 5")
                else:
                    res.append((3 * y[0] + 2 * y[1] + y[2] - y[4]) / 5)
                    res.append((4 * y[0] + 3 * y[1] + 2 * y[2] + y[3]) / 10)
                    for i in range(2, len(y) - 2):
                        res.append((y[i - 2] + y[i - 1] + y[i] + y[i + 1] + y[i + 2]) / 5)
                    res.append((4 * y[-1] + 3 * y[-2] + 2 * y[-3] + y[-4]) / 10)
                    res.append((3 * y[-1] + 2 * y[-2] + y[-3] - y[-5]) / 5)
            elif n == 7:
                if len(y) < 7:
                    raise SmoothPointNumberError("Points number must be >= 7")
                else:
                    res.append((39 * y[0] + 8 * y[1] - 4 * (y[2] + y[3] - y[4]) + y[5] - 2 * y[6]) / 42)
                    res.append((8 * y[0] + 19 * y[1] + 16 * y[2] + 6 * y[3] - 4 * y[4] - 7 * y[5] + 4 * y[6]) / 42)
                    res.append((-4 * y[0] + 16 * y[1] + 19 * y[2] + 12 * y[3] + 2 * y[4] - 4 * y[5] + y[6]) / 42)
                    for i in range(3, len(y) - 3):
                        res.append((7 * y[i] + 6 * (y[i + 1] + y[i - 1]) + 3 * (y[i + 2] + y[i - 2]) - 2 * (
                                y[i + 3] + y[i - 3])) / 21)
                    res.append((-4 * y[-1] + 16 * y[-2] + 19 * y[-3] + 12 * y[-4] + 2 * y[-5] - 4 * y[-6] + y[-7]) / 42)
                    res.append(
                        (8 * y[-1] + 19 * y[-2] + 16 * y[-3] + 6 * y[-4] - 4 * y[-5] - 7 * y[-6] + 4 * y[-7]) / 42)
                    res.append((39 * y[-1] + 8 * y[-2] - 4 * y[-3] - 4 * y[-4] + y[-5] + 4 * y[-6] - 2 * y[-7]) / 42)
            else:
                raise SmoothPointNumberError("Unknown smooth point number. Available: 3, 5, 7")
            return np.array(res, dtype=np.float64)


class Point(object):
    def __init__(self, x, y):
        self.x = round(x, 3)
        self.y = round(y, 3)

    def __str__(self):
        return "Point(" + str(self.x) + ";" + str(self.y) + ")"


class Data(list):
    accuracy = 3

    def __init__(self):
        super(Data, self).__init__()
        self.distance_to_remove = 100

    def append(self, point: Point) -> None:
        point.x = round(point.x, Data.accuracy)
        point.y = round(point.y, Data.accuracy)
        for p in self:
            if p.x == point.x:
                return
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

    def set_accuracy(self, accuracy):
        Data.accuracy = accuracy
        for i in range(len(self)):
            self[i].x = round(self[i].x, Data.accuracy)
            self[i].y = round(self[i].y, Data.accuracy)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        uic.loadUi('untitled.ui', self)
        self.pg: pg.PlotWidget = self.graphWidget
        self.vb: pg.ViewBox = self.pg.plotItem.vb
        self.num: QtWidgets.QLabel = self.labelNum
        self.accuracy: QtWidgets.QLabel = self.labelAccuracy
        self.accuracy_slider: QtWidgets.QSlider = self.horizontalSlider
        self.approx: QtWidgets.QCheckBox = self.checkBoxApproximate
        self.interp: QtWidgets.QCheckBox = self.checkBoxInterpolate
        self.smooth: QtWidgets.QCheckBox = self.checkBoxSmooth
        self.approx_type: QtWidgets.QSpinBox = self.spinBoxApproximate
        self.interp_type: QtWidgets.QComboBox = self.comboBoxInterpolate
        self.smooth_type: QtWidgets.QComboBox = self.comboBoxSmooth
        self.clear_button: QtWidgets.QPushButton = self.pushButtonClear
        self.browse_button: QtWidgets.QPushButton = self.pushButtonBrowse
        self.save_button: QtWidgets.QPushButton = self.pushButtonSave
        self.add_button: QtWidgets.QPushButton = self.pushButtonAdd
        self.func: QtWidgets.QLineEdit = self.lineEditFunc
        self.func_a: QtWidgets.QLineEdit = self.lineEditA
        self.func_b: QtWidgets.QLineEdit = self.lineEditB
        self.func_step: QtWidgets.QLineEdit = self.lineEditStep
        self.func_num: QtWidgets.QLineEdit = self.lineEditNum
        self.add_x: QtWidgets.QLineEdit = self.lineEditX
        self.add_y: QtWidgets.QLineEdit = self.lineEditY

        self.data = Data()
        self.pg.setTitle("Computing App")
        self.pg.showGrid(x=True, y=True)
        self.pg.setLabel("bottom", "X")
        self.pg.setLabel("left", "Y")
        self.vb.addItem(pg.InfiniteLine(angle=0))
        self.vb.addItem(pg.InfiniteLine(angle=90))
        self.legend = self.pg.plotItem.addLegend()
        self.current_cords = pg.TextItem("(0; 0)", color=(255, 255, 255), anchor=(0, -0.7))
        self.current_cords.setPos(0, 0)
        self.h_line = pg.InfiniteLine(angle=0, pen=(0, 100, 100))
        self.v_line = pg.InfiniteLine(angle=90, pen=(0, 100, 100))
        self.vb.addItem(self.current_cords)
        self.vb.addItem(self.h_line)
        self.vb.addItem(self.v_line)
        self.vb.scaleBy(0, 0, 0, 0)

        self.pg.scene().sigMouseMoved.connect(self.mouse_moved)
        self.pg.scene().sigMouseClicked.connect(self.mouse_clicked)
        self.accuracy_slider.valueChanged.connect(self.slider_changed)
        self.approx.stateChanged.connect(self.changed_approx)
        self.interp.stateChanged.connect(self.changed_interp)
        self.smooth.stateChanged.connect(self.changed_smooth)
        self.approx_type.valueChanged.connect(self.changed_approx_type)
        self.interp_type.currentIndexChanged.connect(self.changed_interp_type)
        self.smooth_type.currentIndexChanged.connect(self.changed_smooth_type)
        self.clear_button.pressed.connect(self.pressed_button_clear)
        self.browse_button.pressed.connect(self.pressed_button_browse)
        self.save_button.pressed.connect(self.pressed_button_save)
        self.add_button.pressed.connect(self.pressed_button_add)

    def mouse_moved(self, event):
        point = self.vb.mapSceneToView(event)
        x = point.x()
        y = point.y()
        self.h_line.setPos(y)
        self.v_line.setPos(x)
        self.current_cords.setPos(x, y)
        self.current_cords.setText(f'(%0.{Data.accuracy}f; %0.{Data.accuracy}f)' % (x, y))

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

    def changed_smooth(self):
        self.plot()

    def changed_approx_type(self):
        self.plot()

    def changed_interp_type(self):
        self.plot()

    def changed_smooth_type(self):
        self.plot()

    def pressed_button_clear(self):
        self.clear_legend()
        self.pg.clear()
        self.data.remove_all()
        self.num.setText("Num: {0}".format(len(self.data)))

    def pressed_button_browse(self):
        file = QtWidgets.QFileDialog.getOpenFileName()[0]
        self.clear_legend()
        self.data.remove_all()
        with open(file) as f:
            line = f.readline()
            while line:
                x, y = map(float, line.split())
                self.data.append(Point(x, y))
                line = f.readline()
        self.plot()

    def pressed_button_save(self):
        file = QtWidgets.QFileDialog.getSaveFileName()[0]
        with open(file, 'w') as f:
            for p in self.data:
                f.write("%f %f\n" % (p.x, p.y))

    def pressed_button_add(self):
        x = float(self.add_x.text())
        y = float(self.add_y.text())
        self.data.append(Point(x, y))
        self.plot()

    def slider_changed(self):
        accuracy = self.accuracy_slider.value()
        self.accuracy.setText("Accuracy: {0}".format(accuracy))
        self.vb.setLimits(minXRange=1/10**accuracy, minYRange=1/10**accuracy)
        self.data.set_accuracy(accuracy)

    def clear_legend(self):
        while self.legend.items != []:
            self.legend.removeItem(self.legend.items[0][1].text)

    def plot(self):
        self.pg.clear()
        self.clear_legend()

        if self.approx.isChecked():
            approx_func = Algorithms.Approximation.approximate_least_squares(
                self.data.x(), self.data.y(), self.approx_type.value())
            x = np.linspace(self.data.x()[0], self.data.x()[-1], 1000)
            y = approx_func(x)
            pp = self.pg.plot(x, y, pen=pg.mkPen(color=(255, 255, 0)))
            self.legend.addItem(pp, "approximation line")

        if self.interp.isChecked():
            interp_func = Algorithms.Interpolation.interpolate(
                self.data.x(), self.data.y(), self.interp_type.currentText())
            x = np.linspace(self.data.x()[0], self.data.x()[-1], 1000)
            y = interp_func(x)
            pp = self.pg.plot(x, y, pen=pg.mkPen(color=(0, 255, 255)))
            self.legend.addItem(pp, "interpolation line")

        if self.smooth.isChecked():
            if self.smooth_type.currentText() == "3 points":
                n = 3
            elif self.smooth_type.currentText() == "5 points":
                n = 5
            elif self.smooth_type.currentText() == "7 points":
                n = 7
            smooth_y = Algorithms.Smoothing.smooth(self.data.y(), n)
            pp = self.pg.plot(self.data.x(), smooth_y, pen=pg.mkPen(color=(255, 0, 255)))
            self.legend.addItem(pp, "smooth points")

        pp = self.pg.plot(self.data.x(), self.data.y(), pen=None, symbol='o', symbolSize=5, symbolBrush=('b'))
        self.legend.addItem(pp, "basic points")
        self.pg.plot()
        self.num.setText("Num: {0}".format(len(self.data)))


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        sys.exit(app.exec_())


if __name__ == '__main__':
    main()
