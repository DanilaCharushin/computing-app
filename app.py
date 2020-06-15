from PyQt5 import QtWidgets, uic, QtCore
import pyqtgraph as pg
import sys
import numpy as np
from typing import List
from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent
import scipy.interpolate as interp
import scipy.integrate as integr
from sympy import *


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
        def interpolate_demo(x: List[float], y: List[float], kind: str) -> callable:
            if kind == "lagrange":
                return interp.lagrange(x, y)
            else:
                return interp.interp1d(x, y, kind=kind)

        @staticmethod
        def interpolate(x: List[float], y: List[float], n: int) -> callable:
            return interp.UnivariateSpline(x, y, s=0, k=n)

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

    class Derivative(object):
        @staticmethod
        def derivate(x: List[float], y: List[float], n: int, interp_order: int) -> callable:
            spl = interp.UnivariateSpline(x, y, s=0, k=interp_order)
            return spl.derivative(n)

    class Integration(object):
        @staticmethod
        def integrate(x: List[float], y: List[float], kind: str) -> callable:
            if kind == "trapz":
                return integr.trapz(y, x)
            else:
                return integr.simps(y, x)


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

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
        return point

    def x(self) -> List[float]:
        return [point.x for point in self]

    def y(self) -> List[float]:
        return [point.y for point in self]

    def remove_all(self):
        while len(self) != 0:
            self.pop(0)

    def check_to_remove(self, x, y, distance):
        to_remove = []
        for p in self:
            if ((x - p.x) ** 2 + (y - p.y) ** 2) ** 0.5 < distance / self.distance_to_remove:
                to_remove.append(p)
        if to_remove != []:
            for p in to_remove:
                self.remove(p)
        return to_remove

    def set_distance_to_remove(self, distance):
        self.distance_to_remove = distance

    def set_accuracy(self, accuracy):
        Data.accuracy = accuracy


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        uic.loadUi('app.ui', self)
        self.pg: pg.PlotWidget = self.graphWidget
        self.vb: pg.ViewBox = self.pg.plotItem.vb

        self.need_add_point_func = False
        self.need_add_point_deriv = False
        self.need_add_point_integral = False
        self.need_plot = False

        self.num: QtWidgets.QLabel = self.labelNum
        self.accuracy: QtWidgets.QLabel = self.labelAccuracy
        self.accuracy_slider: QtWidgets.QSlider = self.horizontalSlider
        self.step_slider = QtWidgets.QSlider = self.verticalSlider
        self.step = self.step_slider.value()
        self.logger = QtWidgets.QTextBrowser = self.textEditLog
        self.clear_log_button = QtWidgets.QPushButton = self.pushButtonClearLog

        self.approx: QtWidgets.QCheckBox = self.checkBoxApproximate
        self.interp: QtWidgets.QCheckBox = self.checkBoxInterpolate
        self.smooth: QtWidgets.QCheckBox = self.checkBoxSmooth

        self.approx_type: QtWidgets.QSpinBox = self.spinBoxApproximate
        self.interp_type: QtWidgets.QComboBox = self.comboBoxInterpolate
        self.smooth_n: QtWidgets.QSpinBox = self.spinBoxSmoothN
        self.smooth_interp_order: QtWidgets.QSpinBox = self.spinBoxSmoothInterp

        self.clear_button: QtWidgets.QPushButton = self.pushButtonClear
        self.browse_button: QtWidgets.QPushButton = self.pushButtonBrowse
        self.save_button: QtWidgets.QPushButton = self.pushButtonSave

        self.add_x: QtWidgets.QLineEdit = self.lineEditX
        self.add_y: QtWidgets.QLineEdit = self.lineEditY
        self.add_point: QtWidgets.QPushButton = self.pushButtonAdd

        self.func_a: QtWidgets.QLineEdit = self.lineEditA
        self.func_b: QtWidgets.QLineEdit = self.lineEditB
        self.func_step: QtWidgets.QLineEdit = self.lineEditStep
        self.func_num: QtWidgets.QLineEdit = self.lineEditNum

        self.func: QtWidgets.QCheckBox = self.checkBoxFunc
        self.func_text: QtWidgets.QLineEdit = self.lineEditFunc
        self.add_func: QtWidgets.QPushButton = self.pushButtonFunc

        self.deriv = QtWidgets.QCheckBox = self.checkBoxDerivative
        self.add_deriv = QtWidgets.QLineEdit = self.pushButtonDerivative
        self.deriv_text = QtWidgets.QLineEdit = self.lineEditDerivative
        self.deriv_order = QtWidgets.QSpinBox = self.spinBoxDerivative

        self.integral = QtWidgets.QCheckBox = self.checkBoxIntegral
        self.add_integral = QtWidgets.QLineEdit = self.pushButtonIntegral
        self.integral_text = QtWidgets.QCheckBox = self.lineEditIntegral
        self.integral_value = QtWidgets.QLabel = self.labelIntegral

        self.deriv_points = QtWidgets.QCheckBox = self.checkBoxDerivativePoints
        self.deriv_points_order = QtWidgets.QCheckBox = self.spinBoxDerivativePoints
        self.deriv_points_interp = QtWidgets.QCheckBox = self.spinBoxInterpPoints
        self.integral_points = QtWidgets.QCheckBox = self.checkBoxIntegralPoints
        self.integral_points_value = QtWidgets.QLabel = self.labelIntegralPoints

        self.in_time = QtWidgets.QCheckBox = self.checkBoxInTime
        self.plot_button = QtWidgets.QPushButton = self.pushButtonPlot
        self.to_zero = QtWidgets.QPushButton = self.pushButtonToZero
        self.cursor_lines = QtWidgets.QCheckBox = self.checkBoxCursorLines
        self.func_lines = QtWidgets.QCheckBox = self.checkBoxFuncLines
        self.points_lines = QtWidgets.QCheckBox = self.checkBoxPointsLines

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

        self.cursor_hline = pg.InfiniteLine(angle=0, pen=(0, 100, 100))
        self.cursor_vline = pg.InfiniteLine(angle=90, pen=(0, 100, 100))
        self.func_aline = pg.InfiniteLine(angle=90, pen=(147, 90, 118))
        self.func_bline = pg.InfiniteLine(angle=90, pen=(147, 90, 118))
        self.points_aline = pg.InfiniteLine(angle=90, pen=(118, 255, 147))
        self.points_bline = pg.InfiniteLine(angle=90, pen=(118, 255, 147))

        self.vb.addItem(self.current_cords)
        self.vb.addItem(self.cursor_hline)
        self.vb.addItem(self.cursor_vline)
        self.vb.scaleBy(0, 0, 0, 0)

        self.pg.scene().sigMouseMoved.connect(self.mouse_moved)
        self.pg.scene().sigMouseClicked.connect(self.mouse_clicked)

        self.accuracy_slider.valueChanged.connect(self.accuracy_slider_changed)
        self.step_slider.valueChanged.connect(self.step_slider_changed)

        self.approx.stateChanged.connect(self.changed_approx)
        self.interp.stateChanged.connect(self.changed_interp)
        self.smooth.stateChanged.connect(self.changed_smooth)

        self.approx_type.valueChanged.connect(self.changed_approx_type)
        self.interp_type.currentIndexChanged.connect(self.changed_interp_type)
        self.smooth_n.valueChanged.connect(self.changed_smooth_n)
        self.smooth_interp_order.valueChanged.connect(self.changed_smooth_interp_order)

        self.clear_log_button.pressed.connect(self.pressed_button_clear_log)
        self.clear_button.pressed.connect(self.pressed_button_clear)
        self.browse_button.pressed.connect(self.pressed_button_browse)
        self.save_button.pressed.connect(self.pressed_button_save)
        self.to_zero.pressed.connect(self.pressed_button_to_zero)
        self.cursor_lines.stateChanged.connect(self.changed_cursor_lines)
        self.func_lines.stateChanged.connect(self.changed_func_lines)
        self.points_lines.stateChanged.connect(self.changed_points_lines)
        self.in_time.stateChanged.connect(self.changed_in_time)
        self.plot_button.pressed.connect(self.pressed_button_plot)

        self.func_a.textChanged.connect(self.func_a_changed)
        self.func_b.textChanged.connect(self.func_b_changed)
        self.func_step.textChanged.connect(self.func_step_changed)
        self.func_num.textChanged.connect(self.func_num_changed)

        self.add_point.pressed.connect(self.pressed_button_add_point)
        self.add_func.pressed.connect(self.pressed_button_add_func)
        self.add_deriv.pressed.connect(self.pressed_button_add_deriv)
        self.add_integral.pressed.connect(self.pressed_button_add_integral)

        self.func.stateChanged.connect(self.changed_show)
        self.func_text.textChanged.connect(self.func_changed)

        self.deriv.stateChanged.connect(self.changed_deriv)
        self.deriv_order.valueChanged.connect(self.changed_deriv_order)
        self.integral.stateChanged.connect(self.changed_integral)

        self.deriv_points.stateChanged.connect(self.changed_deriv_points)
        self.deriv_points_order.valueChanged.connect(self.changed_deriv_points_order)
        self.deriv_points_interp.valueChanged.connect(self.changed_deriv_points_interp)
        self.integral_points.stateChanged.connect(self.radio_changed)
        self.radioButtonSimps.toggled.connect(self.radio_changed)
        self.radioButtonTrapz.toggled.connect(self.radio_changed)

    def pressed_button_plot(self):
        self.need_plot = True
        self.plot()

    def changed_in_time(self):
        if self.in_time.isChecked():
            self.plot_button.setEnabled(False)
        else:
            self.plot_button.setEnabled(True)

    def radio_changed(self):
        if self.integral_points.isChecked():
            try:
                if self.radioButtonSimps.isChecked():
                    self.integral_points_value.setText(
                        "Value: {0}".format(
                            round(Algorithms.Integration.integrate(self.data.x(), self.data.y(), "simps"),
                                  self.accuracy_slider.value())))
                else:
                    self.integral_points_value.setText(
                        "Value: {0}".format(
                            round(Algorithms.Integration.integrate(self.data.x(), self.data.y(), "trapz"),
                                  self.accuracy_slider.value())))
            except Exception as ex:
                self.log(ex)

    def pressed_button_clear_log(self):
        self.logger.setText("")

    def changed_cursor_lines(self):
        if self.cursor_lines.isChecked():
            self.vb.addItem(self.cursor_hline)
            self.vb.addItem(self.cursor_vline)
            self.vb.addItem(self.current_cords)
        else:
            self.vb.removeItem(self.cursor_hline)
            self.vb.removeItem(self.cursor_vline)
            self.vb.removeItem(self.current_cords)

    def changed_func_lines(self):
        try:
            if self.func_lines.isChecked() and (
                    self.func.isChecked() or self.deriv.isChecked() or self.integral.isChecked()):
                self.func_aline.setPos(float(self.func_a.text()))
                self.func_bline.setPos(float(self.func_b.text()))
                self.vb.addItem(self.func_aline)
                self.vb.addItem(self.func_bline)
            else:
                self.vb.removeItem(self.func_aline)
                self.vb.removeItem(self.func_bline)
        except Exception as ex:
            self.log(ex)

    def changed_points_lines(self):
        try:
            if self.points_lines.isChecked():
                self.points_aline.setPos(self.data.x()[0])
                self.points_bline.setPos(self.data.x()[-1])
                self.vb.addItem(self.points_aline)
                self.vb.addItem(self.points_bline)
            else:
                self.vb.removeItem(self.points_aline)
                self.vb.removeItem(self.points_bline)
        except Exception as ex:
            self.log(ex)

    def changed_deriv_points_order(self):
        self.plot()

    def changed_deriv_points_interp(self):
        self.plot()

    def pressed_button_to_zero(self):
        self.vb.scaleBy(0, 0, 0, 0)

    def changed_deriv_order(self):
        self.plot()

    def pressed_button_add_func(self):
        self.need_add_point_func = True
        self.plot()

    def pressed_button_add_deriv(self):
        self.need_add_point_deriv = True
        self.plot()

    def pressed_button_add_integral(self):
        self.need_add_point_integral = True
        self.plot()

    def mouse_moved(self, event):
        point = self.vb.mapSceneToView(event)
        x = point.x()
        y = point.y()
        self.cursor_hline.setPos(y)
        self.cursor_vline.setPos(x)
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
            removed = self.data.check_to_remove(x, y, distance)
            for p in removed:
                self.log("removed " + str(p))
        elif event.button() == 1:  # left mouse key
            p = self.data.append(Point(x, y))
            if p is not None:
                self.log("added " + str(p))
            self.changed_points_lines()
        self.plot()

    def changed_approx(self):
        self.plot()

    def changed_interp(self):
        self.plot()

    def changed_smooth(self):
        self.plot()

    def changed_show(self):
        self.plot()

    def changed_approx_type(self):
        self.plot()

    def changed_interp_type(self):
        self.plot()

    def changed_smooth_n(self):
        self.plot()

    def changed_smooth_interp_order(self):
        self.plot()

    def func_changed(self):
        self.changed_func_lines()
        self.plot()

    def func_a_changed(self):
        self.changed_func_lines()
        self.func_step_changed()
        self.plot()

    def func_b_changed(self):
        self.changed_func_lines()
        self.func_step_changed()
        self.plot()

    def func_step_changed(self):
        try:
            a = float(self.func_a.text())
            b = float(self.func_b.text())
            step = float(self.func_step.text())
            self.func_num.setText(str(int((b - a) / step)))
            self.plot()
        except Exception as ex:
            self.log(ex)

    def func_num_changed(self):
        try:
            a = float(self.func_a.text())
            b = float(self.func_b.text())
            num = int(self.func_num.text())
            self.func_step.setText(str((b - a) / num))
            self.plot()
        except Exception as ex:
            self.log(ex)

    def pressed_button_clear(self):
        self.clear_legend()
        self.pg.clear()
        self.data.remove_all()
        self.num.setText("Num: {0}".format(len(self.data)))
        self.integral_value.setText("Value: ")
        self.integral_points_value.setText("Value: ")

    def pressed_button_browse(self):
        try:
            file = QtWidgets.QFileDialog.getOpenFileName()[0]
            self.clear_legend()
            self.data.remove_all()
            with open(file) as f:
                line = f.readline()
                if line.startswith("FUNC"):
                    line = f.readline()
                    self.func_text.setText(line.rstrip())
                    line = f.readline()
                    self.func_a.setText(line.rstrip())
                    line = f.readline()
                    self.func_b.setText(line.rstrip())
                    line = f.readline()
                    self.func_step.setText(line.rstrip())
                    line = f.readline()
                    self.func_num.setText(line.rstrip())
                    line = f.readline()
                while line:
                    x, y = map(float, line.split())
                    self.data.append(Point(x, y))
                    line = f.readline()
        except Exception as ex:
            self.log(ex)
        self.changed_points_lines()
        self.changed_func_lines()
        self.plot()

    def pressed_button_save(self):
        try:
            file = QtWidgets.QFileDialog.getSaveFileName()[0]
            with open(file, 'w') as f:
                if self.func_text.text() != "":
                    f.write("FUNC\n")
                    f.write(self.func_text.text() + "\n")
                    f.write(self.func_a.text() + "\n")
                    f.write(self.func_b.text() + "\n")
                    f.write(self.func_step.text() + "\n")
                    f.write(self.func_num.text() + "\n")
                for p in self.data:
                    f.write("%f %f\n" % (p.x, p.y))
        except Exception as ex:
            self.log(ex)

    def pressed_button_add_point(self):
        x = float(self.add_x.text())
        y = float(self.add_y.text())
        p = self.data.append(Point(x, y))
        if p is not None:
            self.log("added " + str(p))
        self.plot()

    def accuracy_slider_changed(self):
        accuracy = self.accuracy_slider.value()
        self.accuracy.setText("Accuracy: {0}".format(accuracy))
        self.vb.setLimits(minXRange=1 / 10 ** (accuracy - 1), minYRange=1 / 10 ** (accuracy - 1))
        self.data.set_accuracy(accuracy)

    def step_slider_changed(self):
        self.step = self.step_slider.value()
        self.plot()

    def changed_deriv(self):
        self.plot()

    def changed_deriv_points(self):
        self.plot()

    def changed_integral(self):
        self.plot()

    def changed_integral_points(self):
        self.plot()

    def clear_legend(self):
        while self.legend.items != []:
            self.legend.removeItem(self.legend.items[0][1].text)

    def log(self, log_text):
        self.logger.setText(self.logger.toPlainText() + "\n" + str(log_text))

    def get_func(self):
        x = symbols('x')
        f = sympify(self.func_text.text())
        a = float(self.func_a.text())
        b = float(self.func_b.text())
        num = int(self.func_num.text())
        return x, f, a, b, num

    def plot(self):
        if self.in_time.isChecked() or self.need_plot:
            try:
                self.pg.clear()
                self.clear_legend()
                self.radio_changed()
                if self.approx.isChecked():
                    approx_func = Algorithms.Approximation.approximate_least_squares(
                        self.data.x(), self.data.y(), self.approx_type.value())
                    x = np.linspace(self.data.x()[0], self.data.x()[-1], self.step)
                    y = approx_func(x)
                    pp = self.pg.plot(x, y, pen=pg.mkPen(color=(255, 255, 0)))
                    self.legend.addItem(pp, "approximation line")

                if self.interp.isChecked():
                    interp_func = Algorithms.Interpolation.interpolate_demo(
                        self.data.x(), self.data.y(), self.interp_type.currentText())
                    x = np.linspace(self.data.x()[0], self.data.x()[-1], self.step)
                    y = interp_func(x)
                    pp = self.pg.plot(x, y, pen=pg.mkPen(color=(0, 255, 255)))
                    self.legend.addItem(pp, "interpolation line")

                if self.smooth.isChecked():
                    smooth_y = Algorithms.Smoothing.smooth(self.data.y(), self.smooth_n.value())
                    smooth_func = Algorithms.Interpolation.interpolate(self.data.x(), smooth_y,
                                                                       self.smooth_interp_order.value())
                    x = np.linspace(self.data.x()[0], self.data.x()[-1], self.step)
                    pp = self.pg.plot(x, smooth_func(x), pen=pg.mkPen(color=(255, 255, 255)))
                    self.legend.addItem(pp, "smooth points")

                if self.deriv_points.isChecked():
                    x = np.linspace(self.data.x()[0], self.data.x()[-1], self.step)
                    deriv_func = Algorithms.Derivative.derivate(self.data.x(), self.data.y(),
                                                                self.deriv_points_order.value(),
                                                                self.deriv_points_interp.value() + 1)
                    pp = self.pg.plot(x, deriv_func(x), pen=pg.mkPen(color=(204, 0, 204)))
                    self.legend.addItem(pp, "derivative points")

                if self.func.isChecked():
                    x, f, a, b, num = self.get_func()
                    X = np.linspace(a, b, num)
                    Y = []
                    for arg in X:
                        Y.append(float(f.evalf(subs={x: arg})))
                        if self.need_add_point_func:
                            self.data.append(Point(arg, float(f.evalf(subs={x: arg}))))
                    pp = self.pg.plot(X, Y, pen=pg.mkPen(color=(255, 50, 100)))
                    self.legend.addItem(pp, "f(x)")
                    self.need_add_point_func = False

                if self.deriv.isChecked():
                    x, f, a, b, num = self.get_func()
                    self.deriv_text.setText(str(f.diff(x, self.deriv_order.value())))
                    X = np.linspace(a, b, num)
                    Y = []
                    for arg in X:
                        Y.append(float(f.diff(x, self.deriv_order.value()).evalf(subs={x: arg})))
                        if self.need_add_point_deriv:
                            self.data.append(
                                Point(arg, float(f.diff(x, self.deriv_order.value()).evalf(subs={x: arg}))))
                    pp = self.pg.plot(X, Y, pen=pg.mkPen(color=(100, 150, 255)))
                    shtrih = "'" * self.deriv_order.value()
                    self.legend.addItem(pp, f'f{shtrih}(x)')
                    self.need_add_point_deriv = False

                if self.integral.isChecked():
                    x, f, a, b, num = self.get_func()
                    self.integral_text.setText(str(f.integrate(x)))
                    self.integral_value.setText(
                        "Value: {0}".format(round(float(integrate(f, (x, a, b))), self.accuracy_slider.value())))
                    X = np.linspace(a, b, num)
                    Y = []
                    for arg in X:
                        Y.append(float(f.integrate(x).evalf(subs={x: arg})))
                        if self.need_add_point_integral:
                            self.data.append(Point(arg, float(f.integrate(x).evalf(subs={x: arg}))))
                    pp = self.pg.plot(X, Y, pen=pg.mkPen(color=(0, 153, 0)))
                    self.legend.addItem(pp, "F(x)")
                    self.need_add_point_integral = False
            except Exception as ex:
                self.log(ex)

        pp = self.pg.plot(self.data.x(), self.data.y(), pen=None, symbol='o', symbolSize=5, symbolBrush=('b'))
        self.legend.removeItem("basic points")
        self.legend.addItem(pp, "basic points")
        self.pg.plot()
        self.num.setText("Num: {0}".format(len(self.data)))
        self.need_plot = False


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.showFullScreen()
    main.show()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        sys.exit(app.exec_())


if __name__ == '__main__':
    main()
