# Created by Shlyankin Nickolay & Vladimir Michailov & Alena Zahodyakina
from PyQt5 import QtWidgets
from mydesign import Ui_MainWindow
from util import TaskCrankNicholson, TaskImplicit, TaskExplicit
import pyqtgraph as pg
import sys

class mywindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(mywindow, self).__init__()
        self.tasks = []
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.graphWidget = pg.PlotWidget()
        self.ui.layoutGraph.addWidget(self.graphWidget)
        self.graphWidget.setBackground('w')
        self.graphWidget.setLabel('left', 'Температура (К)', color='red', size=30)
        self.graphWidget.setLabel('bottom', 'Радиус (см)', color='red', size=30)
        self.ui.label_current_time.setText("Индекс времени k = ")
        self.ui.sliderImage.valueChanged.connect(self.plotNextGraph)
        self.ui.buttonCaluclate.clicked.connect(self.calculate)
        self.ui.buttonClear.clicked.connect(self.clear)
        self.l = pg.LegendItem((160,60), offset=(430,10))
        self.l.setParentItem(self.graphWidget.graphicsItem())


    def clear(self):
        self.ui.edit_R.setText("")
        self.ui.edit_l.setText("")
        self.ui.edit_k.setText("")
        self.ui.edit_c.setText("")
        self.ui.edit_T.setText("")
        self.ui.edit_Uc.setText("")
        self.ui.edit_alpha.setText("")
        self.ui.edit_K.setText("")
        self.ui.edit_I.setText("")
        self.graphWidget.clear()
        self.tasks = []
        self.ui.label_gridInfo.setStyleSheet("color: rgb(0, 0, 0);")
        self.ui.label_gridInfo.setText("")
        self.ui.label_max_t.setText("0")
        self.ui.label_current_time.setText("Индекс времени k = ")
        self.ui.label_current_time_2.setText("Время t = ")

    def calculate(self):
        self.ui.label_gridInfo.setStyleSheet("color: rgb(0, 0, 0);")
        self.ui.label_gridInfo.setText("")
        try:
            R =     float(self.ui.edit_R.text())
            l =     float(self.ui.edit_l.text())
            k =     float(self.ui.edit_k.text())
            c =     float(self.ui.edit_c.text())
            T =     float(self.ui.edit_T.text())
            Uc =    float(self.ui.edit_Uc.text())
            alpha = float(self.ui.edit_alpha.text())
            K =     int(  self.ui.edit_K.text())
            I =     int(  self.ui.edit_I.text())
            self.ui.label_gridInfo.setStyleSheet("color: rgb(255, 0, 0);")
            self.tasks = []
            self.tasks.append(TaskCrankNicholson(R, l, k, c, alpha, T, Uc, K, I))
            self.tasks.append(TaskImplicit(R, l, k, c, alpha, T, Uc, K, I))
            self.tasks.append(TaskExplicit(R, l, k, c, alpha, T, Uc, K, I))
            self.ui.label_gridInfo.setText("ht: " + str(self.tasks[0].ht) + " hr: " + str(self.tasks[0].hr))
            self.graphWidget.clear()
            self.legend_del()
            self.ui.label_current_time.setText("Индекс времени k = " + str(0))
            self.ui.label_current_time_2.setText("Время t = " + str(0) + " c")
            for task in self.tasks:
                answer = task.calculate()
                answer_analytic = task.analytic_decision()
                y = answer[0]
                x = task.r
                self.ui.label_gridInfo.setText(self.ui.label_gridInfo.text() +
                                               "\n" + task.name + " absolute error: " + str(task.calculateAbsError()) +
                                               " isStable: " + str(task.isStable()))
                self.plotGraph(x, y, task.name, task.color)
            self.ui.sliderImage.setValue(0)
            self.ui.label_max_t.setText(str(len(answer) - 1))
            self.ui.sliderImage.setMaximum(len(answer) - 1)
            self.plotGraph(x, answer_analytic[0], "Аналитическое решение", 'b')

        except ValueError:
            self.ui.label_gridInfo.setStyleSheet("color: rgb(255, 0, 0);")
            self.ui.label_gridInfo.setText("Проверьте поля!")


    def legend_del(self):
        while(len(self.l.items)):
            item, label = self.l.items[0]
            self.l.items.remove((item, label))  # удалить линию
            self.l.layout.removeItem(item)
            item.close()
            self.l.layout.removeItem(label)  # удалить надпись
            label.close()
            self.l.updateSize()

    def plotNextGraph(self):
        if len(self.tasks) == 0: return
        self.legend_del()
        t = self.ui.sliderImage.value()
        self.ui.label_current_time.setText("Индекс времени k = " + str(t))
        self.ui.label_current_time_2.setText("Время t = " + str(round(t*self.tasks[0].ht, 2)) + " c")
        self.graphWidget.clear()
        for task in self.tasks:
            y = task.answer[t]
            x = task.r
            self.plotGraph(x, y, task.name, task.color)
        self.plotGraph(x, self.tasks[0].answer_analytic[t], "Аналитическое решение", 'b')


    def plotGraph(self, x, y, plotname, color):
        self.graphWidget.showGrid(x=True, y=True)
        pen = pg.mkPen(color=color, width=3)
        self.l.addItem(self.graphWidget.plot(x, y, name=plotname, pen=pen), plotname)



app = QtWidgets.QApplication([])
application = mywindow()
application.show()

sys.exit(app.exec())
