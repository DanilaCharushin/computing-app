from PyQt5 import QtWidgets, uic, QtGui
from pyqtgraph import PlotWidget
import pyqtgraph as pg
import sys
from typing import List


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "Point(" + str(self.x) + ";" + str(self.y) + ")"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        uic.loadUi('untitled.ui', self)
        self.pg = self.graphWidget
        self.pg.setTitle("Your Title Here")
        self.pg.showGrid(x=True, y=True)
        self.pen = pg.mkPen(color=(255, 0, 0))
        self.data: List[Point] = []
        self.vb = self.pg.plotItem.vb
        self.pg.scene().sigMouseClicked.connect(self.mouse_clicked)

    def mouse_clicked(self, event):
        point = self.vb.mapSceneToView(event.scenePos())
        x = point.x()
        y = point.y()
        if event.button() == 4:
            dx = self.vb.viewRange()[0][1] - self.vb.viewRange()[0][0]
            dy = self.vb.viewRange()[1][1] - self.vb.viewRange()[1][0]
            d = (dx ** 2 + dy ** 2) ** 0.5
            to_del = None
            for p in self.data:
                if ((x - p.x) ** 2 + (y - p.y) ** 2) ** 0.5 < d / 100:
                    to_del = p
            if to_del is not None:
                self.data.remove(to_del)
        elif event.button() == 1:
            self.data.append(Point(x, y))
        self.data.sort(key=lambda p: p.x)
        self.plot()

    def plot(self):
        x = [point.x for point in self.data]
        y = [point.y for point in self.data]
        self.pg.clear()
        self.pg.plot(x, y, pen=self.pen, symbol='o', symbolSize=5, symbolBrush=('b'))


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
