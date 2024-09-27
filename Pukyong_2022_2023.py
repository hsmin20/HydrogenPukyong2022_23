import sys
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflowjs as tfjs
from PyQt5.QtWidgets import QApplication, QMainWindow, QDesktopWidget, QAction, QFileDialog, \
    QVBoxLayout, QWidget, QPushButton, QGridLayout, QLabel, QInputDialog, \
    QLineEdit, QComboBox, QMessageBox, QCheckBox, QProgressBar, QHBoxLayout, QTableWidget, QTableWidgetItem, \
    QAbstractItemView, QHeaderView, QDialogButtonBox, QDialog, QGroupBox, QRadioButton, QButtonGroup
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import time

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def R_squared(y, y_pred):
    MAX_LINE = 8185
    total_len = len(y)
    no_of_loop = (int)(total_len / MAX_LINE)

    resi = 0.0
    tota = 0.0
    for i in range(no_of_loop):
        start_index = i * MAX_LINE
        y_sub = y[start_index:start_index+MAX_LINE]
        y_pred_sub = y_pred[start_index:start_index+MAX_LINE]

        residual = tf.reduce_sum(tf.square(tf.subtract(y_sub, y_pred_sub)))
        total = tf.reduce_sum(tf.square(tf.subtract(y_sub, tf.reduce_mean(y_sub))))
        # r2_sub = tf.subtract(1.0, tf.divide(residual, total))

        resi += residual.numpy()
        tota += total.numpy()

    if tota == 0:
        r2 = 0
    else:
        r2 = 1.0 - (resi / tota)

    return r2

class PukyongMachineLearner:
    def __init__(self):
        self.modelLoaded = False

    def set(self, nnList, batchSize, epoch, learningRate, splitPercentage, earlyStopping, verb, callback):
        self.batchSize = batchSize
        self.epoch = epoch
        self.learningRate = learningRate
        self.splitPercentage = splitPercentage
        self.earlyStopping = earlyStopping
        self.verbose = verb
        self.callback = callback
        self.model = self.createModel(nnList)
        self.modelLoaded = True

    def isModelLoaded(self):
        return self.modelLoaded

    def fit(self, x_data, y_data):
        _callbacks = [self.callback]
        if self.earlyStopping == True:
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=500)
            _callbacks.append(early_stopping)

        if self.splitPercentage > 0:
            training_history = self.model.fit(x_data, y_data, batch_size=self.batchSize, epochs=self.epoch,
                                              validation_split=self.splitPercentage, verbose=self.verbose,
                                              callbacks=[self.callback])
        else:
            training_history = self.model.fit(x_data, y_data, batch_size=self.batchSize, epochs=self.epoch,
                                              verbose=self.verbose, callbacks=_callbacks)

        return training_history

    def fitWithValidation(self, x_train_data, y_train_data, x_valid_data, y_valid_data):
        _callbacks = [self.callback]
        if self.earlyStopping == True:
            early_stopping = tf.keras.callbacks.EarlyStopping()
            _callbacks.append(early_stopping)

        training_history = self.model.fit(x_train_data, y_train_data, batch_size=self.batchSize, epochs=self.epoch,
                                          verbose=self.verbose, validation_data=(x_valid_data, y_valid_data),
                                          callbacks=_callbacks)

        return training_history

    def predict(self, x_data):
        y_predicted = self.model.predict(x_data)

        return y_predicted

    def createModel(self, nnList):
        adamOpt = tf.keras.optimizers.Adam(learning_rate=self.learningRate)
        model = tf.keras.Sequential()

        firstLayer = True
        for nn in nnList:
            noOfNeuron = nn[0]
            activationFunc = nn[1]
            if firstLayer:
                model.add(tf.keras.layers.Dense(units=noOfNeuron, activation=activationFunc, input_shape=[noOfNeuron]))
                firstLayer = False
            else:
                model.add(tf.keras.layers.Dense(units=noOfNeuron, activation=activationFunc))

        model.compile(loss='mse', optimizer=adamOpt)

        if self.verbose:
            model.summary()

        return model

    def saveModel(self, foldername):
        if self.modelLoaded == True:
            self.model.save(foldername)

    def saveModelJS(self, filename):
        if self.modelLoaded == True:
            tfjs.converters.save_keras_model(self.model, filename)

    def loadModel(self, foldername):
        self.model = keras.models.load_model(foldername)
        self.modelLoaded = True

    def showResult(self, y_data, training_history, y_predicted, sensor_name, height):
        # max_val = max(y_predicted)
        index_at_max = max(range(len(y_predicted)), key=y_predicted.__getitem__)
        index_at_zero = index_at_max
        for i in range(index_at_max, len(y_predicted)-1):
            val = y_predicted[i]
            val_next = y_predicted[i+1]
            if val >= 0 and val_next <=0:
                index_at_zero = i
                break;

        fig, axs = plt.subplots(2, 1, figsize=(12, 12))
        title = sensor_name + height + ' / zeroIndexTime=' + str(index_at_zero * 0.000002)
        fig.suptitle(title)

        datasize = len(y_data)
        x_display = np.zeros((datasize, 1))
        for j in range(datasize):
            x_display[j][0] = j

        axs[0].scatter(x_display, y_data, color="red", s=1)
        axs[0].plot(x_display, y_predicted, color='blue')
        axs[0].grid()

        lossarray = training_history.history['loss']
        axs[1].plot(lossarray, label='Loss')
        axs[1].grid()

        plt.show()

    def showResultValid(self, y_data, training_history, y_predicted, y_train_data, y_train_pred,
                                          y_valid_data, y_valid_pred, sensor_name, height):
        r2All = R_squared(y_data, y_predicted)
        r2Train = R_squared(y_train_data, y_train_pred)
        r2Valid = R_squared(y_valid_data, y_valid_pred)

        r2AllValue = r2All  # .numpy()
        r2TrainValue = r2Train  # .numpy()
        r2ValidValue = r2Valid  # .numpy()

        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        title = sensor_name + height
        fig.suptitle(title)

        datasize = len(y_data)
        x_display = np.zeros((datasize, 1))
        for j in range(datasize):
            x_display[j][0] = j

        datasize = len(y_train_data)
        x_display2 = np.zeros((datasize, 1))
        for j in range(datasize):
            x_display2[j][0] = j

        datasize = len(y_valid_data)
        x_display3 = np.zeros((datasize, 1))
        for j in range(datasize):
            x_display3[j][0] = j

        axs[0, 0].scatter(x_display, y_data, color="red", s=1)
        axs[0, 0].plot(x_display, y_predicted, color='blue')
        title = f'All Data (R2 = {r2AllValue})'
        axs[0, 0].set_title(title)
        axs[0, 0].grid()

        lossarray = training_history.history['loss']
        axs[0, 1].plot(lossarray, label='Loss')
        axs[0, 1].set_title('Loss')
        axs[0, 1].grid()

        axs[1, 0].scatter(x_display2, y_train_data, color="red", s=1)
        axs[1, 0].scatter(x_display2, y_train_pred, color='blue', s=1)
        title = f'Train Data (R2 = {r2TrainValue})'
        axs[1, 0].set_title(title)
        axs[1, 0].grid()

        axs[1, 1].scatter(x_display3, y_valid_data, color="red", s=1)
        axs[1, 1].scatter(x_display3, y_valid_pred, color='blue', s=1)
        title = f'Validation Data (R2 = {r2ValidValue})'
        axs[1, 1].set_title(title)
        axs[1, 1].grid()

        plt.show()

class LayerDlg(QDialog):
    def __init__(self, unit='128', af='relu'):
        super().__init__()
        self.initUI(unit, af)

    def initUI(self, unit, af):
        self.setWindowTitle('Machine Learning Curve Fitting/Interpolation')

        label1 = QLabel('Units', self)
        self.tbUnits = QLineEdit(unit, self)
        self.tbUnits.resize(100, 40)

        label2 = QLabel('Activation f', self)
        self.cbActivation = QComboBox(self)
        self.cbActivation.addItem(af)
        self.cbActivation.addItem('swish')
        self.cbActivation.addItem('relu')
        self.cbActivation.addItem('selu')
        self.cbActivation.addItem('sigmoid')
        self.cbActivation.addItem('softmax')

        if af == 'linear':
            self.cbActivation.setEnabled(False)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        layout = QGridLayout()
        layout.addWidget(label1, 0, 0)
        layout.addWidget(self.tbUnits, 0, 1)

        layout.addWidget(label2, 1, 0)
        layout.addWidget(self.cbActivation, 1, 1)

        layout.addWidget(self.buttonBox, 2, 1)

        self.setLayout(layout)

class PukyongMLWindow(QMainWindow):
    N_FEATURE = 6 # time, sensor_x, sensor_y, barrier_pos, tank_pressure, tank_volume
    NUM_DATA = 4681

    def __init__(self):
        super().__init__()

        self.distArrayName = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
        self.distList = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

        self.heightArrayName = ['H2.0m', 'H1.0m']
        self.heightList = [2.0, 1.0]

        self.barrierLocName = ['2m', '5m', '8m', '1.18m', '2m', '5m']
        self.barrierLocList = [2.0, 5.0, 8.0, 1.18, 2, 5]

        self.distArray2m = []
        self.heightArray2m = []
        self.cbArray2m = []

        self.distArray5m = []
        self.heightArray5m = []
        self.cbArray5m = []

        self.distArray8m = []
        self.heightArray8m = []
        self.cbArray8m = []

        self.distArray1_18m = []
        self.heightArray1_18m = []
        self.cbArray1_18m = []

        self.distArray2m2 = []
        self.heightArray2m2 = []
        self.cbArray2m2 = []

        self.distArray5m2 = []
        self.heightArray5m2 = []
        self.cbArray5m2 = []

        self.initUI()

        self.time_data = None
        self.time_data_n = None
        self.indexijs2m = []
        self.indexijs5m = []
        self.indexijs8m = []
        self.indexijs1_18m = []
        self.indexijs2m2 = []
        self.indexijs5m2 = []

        self.southSensors2m = []
        self.southSensors5m = []
        self.southSensors8m = []
        self.southSensors1_18m = []
        self.southSensors2m2 = []
        self.southSensors5m2 = []

        self.dataLoaded = False
        self.modelLearner = PukyongMachineLearner()

    def initMenu(self):
        # Menu
        openNN = QAction(QIcon('open.png'), 'Open NN', self)
        openNN.setStatusTip('Open Neural Network Structure from a File')
        openNN.triggered.connect(self.showNNFileDialog)

        saveNN = QAction(QIcon('save.png'), 'Save NN', self)
        saveNN.setStatusTip('Save Neural Network Structure in a File')
        saveNN.triggered.connect(self.saveNNFileDialog)

        exitMenu = QAction(QIcon('exit.png'), 'Exit', self)
        exitMenu.setStatusTip('Exit')
        exitMenu.triggered.connect(self.close)

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openNN)
        fileMenu.addAction(saveNN)
        fileMenu.addSeparator()
        fileMenu.addAction(exitMenu)

        self.statusBar().showMessage('Welcome to Machine Learning for Pukyong CFD 2022~2023')
        self.central_widget = QWidget()  # define central widget
        self.setCentralWidget(self.central_widget)  # set QMainWindow.centralWidget

    def initCSVFileReader(self):
        layout = QHBoxLayout()

        fileLabel = QLabel('csv file (This program is for horizontaly expanded data)')
        self.editFile = QLineEdit('Please load /DataRepositoy/pukyong_2022_2023.csv')
        self.editFile.setFixedWidth(700)
        openBtn = QPushButton('...')
        openBtn.clicked.connect(self.showFileDialog)

        layout.addWidget(fileLabel)
        layout.addWidget(self.editFile)
        layout.addWidget(openBtn)

        return layout

    def initSensors(self, label, bpos, distArray, heightArray, cbArray, distClicked, heightClicked):
        layout = QGridLayout()

        rows = len(self.heightArrayName)
        cols = len(self.distArrayName)

        for i in range(cols):
            cbDist = QCheckBox(self.distArrayName[i])
            cbDist.stateChanged.connect(distClicked)
            distArray.append(cbDist)

        for i in range(rows):
            cbHeight = QCheckBox(self.heightArrayName[i])
            cbHeight.stateChanged.connect(heightClicked)
            heightArray.append(cbHeight)

        for i in range(rows):
            col = []
            for j in range(cols):
                if j == bpos:
                    cbTemp = QCheckBox('||')
                    col.append(cbTemp)
                else:
                    cbTemp = QCheckBox('')
                    col.append(cbTemp)
            cbArray.append(col)

        barrierPos = QLabel(label)
        layout.addWidget(barrierPos, 0, 0)

        for i in range(len(distArray)):
            cbDist = distArray[i]
            layout.addWidget(cbDist, 0, i + 1)

        for i in range(len(heightArray)):
            cbHeight = heightArray[i]
            layout.addWidget(cbHeight, i + 1, 0)

        for i in range(rows):
            for j in range(cols):
                qcheckbox = cbArray[i][j]
                layout.addWidget(qcheckbox, i + 1, j + 1)

        return layout

    def initReadOption(self):
        layout = QHBoxLayout()

        cbCheckAll = QCheckBox('CheckAll')
        cbCheckAll.clicked.connect(self.allClicked)
        loadButton = QPushButton('Load Data')
        loadButton.clicked.connect(self.loadData)
        showPressureButton = QPushButton('Show Pressure Graph')
        showPressureButton.clicked.connect(self.showPressureGraphs)
        showOverPressureButton = QPushButton('Show Overpressure Graph')
        showOverPressureButton.clicked.connect(self.showOverpressureGraphs)
        showImpulseButton = QPushButton('Show Impulse Graph')
        showImpulseButton.clicked.connect(self.showImpulseGraphs)

        layout.addWidget(cbCheckAll)
        layout.addWidget(loadButton)
        layout.addWidget(showPressureButton)
        layout.addWidget(showOverPressureButton)
        layout.addWidget(showImpulseButton)

        return layout

    def initNNTable(self):
        layout = QGridLayout()

        # NN Table
        self.tableNNWidget = QTableWidget()
        self.tableNNWidget.setColumnCount(2)
        self.tableNNWidget.setHorizontalHeaderLabels(['Units', 'Activation'])

        # read default layers
        DEFAULT_LAYER_FILE = 'defaultPukyong2022_2023.nn'
        self.updateNNList(DEFAULT_LAYER_FILE)

        self.tableNNWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableNNWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableNNWidget.setSelectionBehavior(QAbstractItemView.SelectRows)

        # Button of NN
        btnAdd = QPushButton('Add')
        btnAdd.setToolTip('Add a Hidden Layer')
        btnAdd.clicked.connect(self.addLayer)
        btnEdit = QPushButton('Edit')
        btnEdit.setToolTip('Edit a Hidden Layer')
        btnEdit.clicked.connect(self.editLayer)
        btnRemove = QPushButton('Remove')
        btnRemove.setToolTip('Remove a Hidden Layer')
        btnRemove.clicked.connect(self.removeLayer)
        btnLoad = QPushButton('Load')
        btnLoad.setToolTip('Load a NN File')
        btnLoad.clicked.connect(self.showNNFileDialog)
        btnSave = QPushButton('Save')
        btnSave.setToolTip('Save a NN File')
        btnSave.clicked.connect(self.saveNNFileDialog)
        btnMakeDefault = QPushButton('Make default')
        btnMakeDefault.setToolTip('Make this as a default NN layer')
        btnMakeDefault.clicked.connect(self.makeDefaultNN)

        layout.addWidget(self.tableNNWidget, 0, 0, 9, 6)
        layout.addWidget(btnAdd, 9, 0)
        layout.addWidget(btnEdit, 9, 1)
        layout.addWidget(btnRemove, 9, 2)
        layout.addWidget(btnLoad, 9, 3)
        layout.addWidget(btnSave, 9, 4)
        layout.addWidget(btnMakeDefault, 9, 5)

        return layout

    def initMLOption(self):
        layout = QGridLayout()

        batchLabel = QLabel('Batch Size')
        batchLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editBatch = QLineEdit('64')
        self.editBatch.setFixedWidth(100)
        epochLabel = QLabel('Epoch')
        epochLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editEpoch = QLineEdit('50')
        self.editEpoch.setFixedWidth(100)
        lrLabel = QLabel('Learning Rate')
        lrLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editLR = QLineEdit('0.0003')
        self.editLR.setFixedWidth(100)
        self.cbVerbose = QCheckBox('Verbose')
        self.cbVerbose.setChecked(True)

        splitLabel = QLabel('Split for Validation (0 means no split-data for validation)')
        splitLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editSplit = QLineEdit('0.2')
        self.editSplit.setFixedWidth(100)
        self.cbSKLearn = QCheckBox('use sklearn split')
        self.cbSKLearn.setChecked(True)
        self.cbEarlyStop = QCheckBox('Use Early Stopping (validation data)')
        multiplierLabel = QLabel('Multiplier for Barrier Position (0 is Normalization)')
        multiplierLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editMultiplier = QLineEdit('17')
        self.editMultiplier.setFixedWidth(100)

        self.cbMinMax = QCheckBox('Use Min/Max of  All data')
        self.cbMinMax.setChecked(True)
        self.cbValidData = QCheckBox('Use Partially checked  sensors as validation')
        self.cbValidData.stateChanged.connect(self.validDataClicked)
        self.epochPbar = QProgressBar()

        layout.addWidget(batchLabel, 0, 0, 1, 1)
        layout.addWidget(self.editBatch, 0, 1, 1, 1)
        layout.addWidget(epochLabel, 0, 2, 1, 1)
        layout.addWidget(self.editEpoch, 0, 3, 1, 1)
        layout.addWidget(lrLabel, 0, 4, 1, 1)
        layout.addWidget(self.editLR, 0, 5, 1, 1)
        layout.addWidget(self.cbVerbose, 0, 6, 1, 1)

        layout.addWidget(splitLabel, 1, 0, 1, 2)
        layout.addWidget(self.editSplit, 1, 2, 1, 1)
        layout.addWidget(self.cbSKLearn, 1, 3, 1, 1)
        layout.addWidget(self.cbEarlyStop, 1, 4, 1, 1)
        layout.addWidget(multiplierLabel, 1, 5, 1, 1)
        layout.addWidget(self.editMultiplier, 1, 6, 1, 1)

        layout.addWidget(self.cbMinMax, 2, 0, 1, 2)
        layout.addWidget(self.cbValidData, 2, 2, 1, 2)
        layout.addWidget(self.epochPbar, 2, 4, 1, 4)

        return layout

    def initCommand(self):
        layout = QGridLayout()

        mlWithDataBtn = QPushButton('ML with Data')
        mlWithDataBtn.clicked.connect(self.doMachineLearningWithData)
        self.cbResume = QCheckBox('Resume Learning')
        # self.cbResume.setChecked(True)
        saveModelBtn = QPushButton('Save Model')
        saveModelBtn.clicked.connect(self.saveModel)
        loadModelBtn =  QPushButton('Load Model')
        loadModelBtn.clicked.connect(self.loadModel)
        self.cbH5Format = QCheckBox('H5')
        checkValBtn = QPushButton('Check Trained')
        checkValBtn.clicked.connect(self.checkVal)
        saveModelJSBtn = QPushButton('Save Model for JS')
        saveModelJSBtn.clicked.connect(self.saveModelJS)

        layout.addWidget(mlWithDataBtn, 0, 0, 1, 1)
        layout.addWidget(self.cbResume, 0, 1, 1, 1)
        layout.addWidget(saveModelBtn, 0, 2, 1, 1)
        layout.addWidget(loadModelBtn, 0, 3, 1, 1)
        layout.addWidget(self.cbH5Format, 0, 4, 1, 1)
        layout.addWidget(checkValBtn, 0, 5, 1, 1)
        layout.addWidget(saveModelJSBtn, 0, 6, 1, 1)

        return layout

    def initGridTable(self):
        layout = QGridLayout()

        self.tableGridWidget = QTableWidget()

        self.tableGridWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # self.tableGridWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # self.tableGridWidget.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.tableGridWidget.setColumnCount(1)
        item = QTableWidgetItem('')
        self.tableGridWidget.setHorizontalHeaderItem(0, item)

        # Buttons
        btnAddDist = QPushButton('Add Distance')
        btnAddDist.setToolTip('Add a Distance')
        btnAddDist.clicked.connect(self.addDistance)
        btnAddHeight = QPushButton('Add Height')
        btnAddHeight.setToolTip('Add a Height')
        btnAddHeight.clicked.connect(self.addHeight)
        btnRemoveDist = QPushButton('Remove Distance')
        btnRemoveDist.setToolTip('Remove a Distance')
        btnRemoveDist.clicked.connect(self.removeDistance)
        btnRemoveHeight = QPushButton('Remove Height')
        btnRemoveHeight.setToolTip('Remove a Height')
        btnRemoveHeight.clicked.connect(self.removeHeight)
        btnLoadDistHeight = QPushButton('Load')
        btnLoadDistHeight.setToolTip('Load predefined Distance/Height structure')
        btnLoadDistHeight.clicked.connect(self.loadDistHeight)

        tankPressure = QLabel('Tank Pressure(MPa)')
        self.editTankPressure = QLineEdit('103')
        self.editTankPressure.setFixedWidth(200)
        tankVolume = QLabel('Tank Volume(l)')
        self.cbTankVolume = QComboBox()
        self.cbTankVolume.addItem('175')
        self.cbTankVolume.addItem('184')
        self.cbTankVolume.addItem('721')
        self.cbTankVolume.setCurrentIndex(2)
        barrierPos = QLabel('Barrier Position(m)')
        barrierPos.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editBarrierPos = QLineEdit('3')
        self.editBarrierPos.setFixedWidth(100)
        predictBtn = QPushButton('Predict')
        predictBtn.clicked.connect(self.predict)
        self.cbFixToBarrierPos = QCheckBox('Fix to Barrier Pos')

        layout.addWidget(self.tableGridWidget, 0, 0, 9, 8)
        layout.addWidget(btnAddDist, 9, 0, 1, 1)
        layout.addWidget(btnAddHeight, 9, 1, 1, 1)
        layout.addWidget(btnRemoveDist, 9, 2, 1, 1)
        layout.addWidget(btnRemoveHeight, 9, 3, 1, 1)
        layout.addWidget(btnLoadDistHeight, 9, 4, 1, 1)
        layout.addWidget(tankPressure, 10, 0, 1, 1)
        layout.addWidget(self.editTankPressure, 10, 1, 1, 1)
        layout.addWidget(tankVolume, 10, 2, 1, 1)
        layout.addWidget(self.cbTankVolume, 10, 3, 1, 1)
        layout.addWidget(barrierPos, 10, 4, 1, 1)
        layout.addWidget(self.editBarrierPos, 10, 5, 1, 1)
        layout.addWidget(predictBtn, 10, 6, 1, 1)
        layout.addWidget(self.cbFixToBarrierPos, 10, 7, 1, 1)

        return layout

    def initUI(self):
        self.setWindowTitle('Machine Learning for Pukyong CFD')
        self.setWindowIcon(QIcon('web.png'))

        self.initMenu()

        layout = QVBoxLayout()

        sensorLayout2m = self.initSensors('Barrier 2m', 1, self.distArray2m, self.heightArray2m, self.cbArray2m, self.distClicked2m, self.heightClicked2m)
        sensorLayout5m = self.initSensors('Barrier 5m', 4, self.distArray5m, self.heightArray5m, self.cbArray5m, self.distClicked5m, self.heightClicked5m)
        sensorLayout8m = self.initSensors('Barrier 8m', 7, self.distArray8m, self.heightArray8m, self.cbArray8m, self.distClicked8m, self.heightClicked8m)
        sensorLayout1_18m = self.initSensors('Barrier 1.18m', 0, self.distArray1_18m, self.heightArray1_18m, self.cbArray1_18m, self.distClicked1_18m, self.heightClicked1_18m)
        sensorLayout2m2 = self.initSensors('Barrier 2m2', 1, self.distArray2m2, self.heightArray2m2, self.cbArray2m2, self.distClicked2m2, self.heightClicked2m2)
        sensorLayout5m2 = self.initSensors('Barrier 5m2', 4, self.distArray5m2, self.heightArray5m2, self.cbArray5m2, self.distClicked5m2, self.heightClicked5m2)

        readOptLayout = self.initReadOption()
        fileLayout = self.initCSVFileReader()
        cmdLayout = self.initCommand()
        nnLayout = self.initNNTable()
        mlOptLayout = self.initMLOption()
        tableLayout = self.initGridTable()

        layout.addLayout(fileLayout)
        layout.addLayout(sensorLayout2m)
        layout.addLayout(sensorLayout5m)
        layout.addLayout(sensorLayout8m)
        layout.addLayout(sensorLayout1_18m)
        layout.addLayout(sensorLayout2m2)
        layout.addLayout(sensorLayout5m2)
        layout.addLayout(readOptLayout)
        layout.addLayout(mlOptLayout)
        layout.addLayout(nnLayout)
        layout.addLayout(cmdLayout)
        layout.addLayout(tableLayout)

        self.centralWidget().setLayout(layout)

        self.resize(1200, 1000)
        self.center()
        self.show()

    def showNNFileDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open NN file', './', filter="NN file (*.nn);;All files (*)")
        if fname[0] != '':
            self.updateNNList(fname[0])

    def updateNNList(self, filename):
        self.tableNNWidget.setRowCount(0)
        with open(filename, "r") as f:
            lines = f.readlines()
            count = 0
            for line in lines:
                strAr = line.split(',')
                self.tableNNWidget.insertRow(count)
                self.tableNNWidget.setItem(count, 0, QTableWidgetItem(strAr[0]))
                self.tableNNWidget.setItem(count, 1, QTableWidgetItem(strAr[1].rstrip()))
                count += 1

    def saveNNFileDialog(self):
        fname = QFileDialog.getSaveFileName(self, 'Save NN file', './', filter="NN file (*.nn)")
        if fname[0] != '':
            self.saveNNFile(fname[0])

    def makeDefaultNN(self):
        filename = 'default.nn'
        self.saveNNFile(filename)
        QMessageBox.information(self, 'Saved', 'Neural Network Default Layers are set')

    def saveNNFile(self, filename):
        with open(filename, "w") as f:
            count = self.tableNNWidget.rowCount()
            for row in range(count):
                unit = self.tableNNWidget.item(row, 0).text()
                af = self.tableNNWidget.item(row, 1).text()
                f.write(unit + "," + af + "\n")

    def getNNLayer(self):
        nnList = []
        count = self.tableNNWidget.rowCount()
        for row in range(count):
            unit = int(self.tableNNWidget.item(row, 0).text())
            af = self.tableNNWidget.item(row, 1).text()
            nnList.append((unit, af))

        return nnList

    def addLayer(self):
        dlg = LayerDlg()
        rc = dlg.exec()
        if rc == 1: # ok
            unit = dlg.tbUnits.text()
            af = dlg.cbActivation.currentText()
            size = self.tableNNWidget.rowCount()
            self.tableNNWidget.insertRow(size-1)
            self.tableNNWidget.setItem(size-1, 0, QTableWidgetItem(unit))
            self.tableNNWidget.setItem(size-1, 1, QTableWidgetItem(af))

    def editLayer(self):
        row = self.tableNNWidget.currentRow()
        if row == -1 or row == (self.tableNNWidget.rowCount() - 1):
            return

        unit = self.tableNNWidget.item(row, 0).text()
        af = self.tableNNWidget.item(row, 1).text()
        dlg = LayerDlg(unit, af)
        rc = dlg.exec()
        if rc == 1: # ok
            unit = dlg.tbUnits.text()
            af = dlg.cbActivation.currentText()
            self.tableNNWidget.setItem(row, 0, QTableWidgetItem(unit))
            self.tableNNWidget.setItem(row, 1, QTableWidgetItem(af))

    def removeLayer(self):
        row = self.tableNNWidget.currentRow()
        if row > 0 and row < (self.tableNNWidget.rowCount() - 1):
            self.tableNNWidget.removeRow(row)

    def addDistance(self):
        sDist, ok = QInputDialog.getText(self, 'Input Distance', 'Distance to add:')
        if ok:
            cc = self.tableGridWidget.columnCount()
            self.tableGridWidget.setColumnCount(cc + 1)
            item = QTableWidgetItem('S' + sDist + 'm')
            self.tableGridWidget.setHorizontalHeaderItem(cc, item)

    def addHeight(self):
        sHeight, ok = QInputDialog.getText(self, 'Input Height', 'Height to add:')
        if ok:
            rc = self.tableGridWidget.rowCount()
            self.tableGridWidget.setRowCount(rc + 1)
            item = QTableWidgetItem('H' + sHeight + 'm')
            self.tableGridWidget.setVerticalHeaderItem(rc, item)

    def removeDistance(self):
        col = self.tableGridWidget.currentColumn()
        if col == -1:
            QMessageBox.warning(self, 'Warning', 'Select any cell')
            return
        if col == 0:
            QMessageBox.warning(self, 'Warning', 'First column cannot be removed')
            return

        self.tableGridWidget.removeColumn(col)

    def removeHeight(self):
        row = self.tableGridWidget.currentRow()
        if row == -1:
            QMessageBox.warning(self, 'Warning', 'Select any cell')
            return
        self.tableGridWidget.removeRow(row)

    def loadDistHeight(self):
        fname = QFileDialog.getOpenFileName(self, 'Open distance/height data file', '/srv/MLData',
                                            filter="CSV file (*.csv);;All files (*)")

        if fname[0] != '':
            with open(fname[0], "r") as f:
                self.tableGridWidget.setRowCount(0)
                self.tableGridWidget.setColumnCount(0)

                lines = f.readlines()

                distAr = lines[0].split(',')
                for sDist in distAr:
                    cc = self.tableGridWidget.columnCount()
                    self.tableGridWidget.setColumnCount(cc + 1)
                    item = QTableWidgetItem('S' + sDist + 'm')
                    self.tableGridWidget.setHorizontalHeaderItem(cc, item)

                heightAr = lines[1].split(',')
                for sHeight in heightAr:
                    rc = self.tableGridWidget.rowCount()
                    self.tableGridWidget.setRowCount(rc + 1)
                    item = QTableWidgetItem('H' + sHeight + 'm')
                    self.tableGridWidget.setVerticalHeaderItem(rc, item)

    def _allClicked(self, state, cbArray):
        for row in cbArray:
            for qcheckbox in row:
                if state == Qt.Unchecked:
                    qcheckbox.setCheckState(Qt.Unchecked)
                else:
                    qcheckbox.setCheckState(Qt.Checked)

    def allClicked(self, state):
        self._allClicked(state, self.cbArray2m)
        self._allClicked(state, self.cbArray5m)
        self._allClicked(state, self.cbArray8m)
        self._allClicked(state, self.cbArray1_18m)
        self._allClicked(state, self.cbArray2m2)
        self._allClicked(state, self.cbArray5m2)

    def distClicked2m(self, state):
        senderName = self.sender().text()
        col = self.distArrayName.index(senderName)
        rows = len(self.cbArray2m)
        for i in range(rows):
            qcheckbox = self.cbArray2m[i][col]
            if state == Qt.Unchecked:
                qcheckbox.setCheckState(Qt.Unchecked)
            elif state == Qt.Checked:
                qcheckbox.setCheckState(Qt.Checked)
            else:
                qcheckbox.setCheckState(Qt.PartiallyChecked)

    def heightClicked2m(self, state):
        senderName = self.sender().text()
        row = self.heightArrayName.index(senderName)
        cols = len(self.cbArray2m[0])
        for i in range(cols):
            qcheckbox = self.cbArray2m[row][i]
            if state == Qt.Unchecked:
                qcheckbox.setCheckState(Qt.Unchecked)
            elif state == Qt.Checked:
                qcheckbox.setCheckState(Qt.Checked)
            else:
                qcheckbox.setCheckState(Qt.PartiallyChecked)

    def distClicked5m(self, state):
        senderName = self.sender().text()
        col = self.distArrayName.index(senderName)
        rows = len(self.cbArray5m)
        for i in range(rows):
            qcheckbox = self.cbArray5m[i][col]
            if state == Qt.Unchecked:
                qcheckbox.setCheckState(Qt.Unchecked)
            elif state == Qt.Checked:
                qcheckbox.setCheckState(Qt.Checked)
            else:
                qcheckbox.setCheckState(Qt.PartiallyChecked)

    def heightClicked5m(self, state):
        senderName = self.sender().text()
        row = self.heightArrayName.index(senderName)
        cols = len(self.cbArray5m[0])
        for i in range(cols):
            qcheckbox = self.cbArray5m[row][i]
            if state == Qt.Unchecked:
                qcheckbox.setCheckState(Qt.Unchecked)
            elif state == Qt.Checked:
                qcheckbox.setCheckState(Qt.Checked)
            else:
                qcheckbox.setCheckState(Qt.PartiallyChecked)

    def distClicked8m(self, state):
        senderName = self.sender().text()
        col = self.distArrayName.index(senderName)
        rows = len(self.cbArray8m)
        for i in range(rows):
            qcheckbox = self.cbArray8m[i][col]
            if state == Qt.Unchecked:
                qcheckbox.setCheckState(Qt.Unchecked)
            elif state == Qt.Checked:
                qcheckbox.setCheckState(Qt.Checked)
            else:
                qcheckbox.setCheckState(Qt.PartiallyChecked)

    def heightClicked8m(self, state):
        senderName = self.sender().text()
        row = self.heightArrayName.index(senderName)
        cols = len(self.cbArray8m[0])
        for i in range(cols):
            qcheckbox = self.cbArray8m[row][i]
            if state == Qt.Unchecked:
                qcheckbox.setCheckState(Qt.Unchecked)
            elif state == Qt.Checked:
                qcheckbox.setCheckState(Qt.Checked)
            else:
                qcheckbox.setCheckState(Qt.PartiallyChecked)

    def distClicked1_18m(self, state):
        senderName = self.sender().text()
        col = self.distArrayName.index(senderName)
        rows = len(self.cbArray1_18m)
        for i in range(rows):
            qcheckbox = self.cbArray1_18m[i][col]
            if state == Qt.Unchecked:
                qcheckbox.setCheckState(Qt.Unchecked)
            elif state == Qt.Checked:
                qcheckbox.setCheckState(Qt.Checked)
            else:
                qcheckbox.setCheckState(Qt.PartiallyChecked)

    def heightClicked1_18m(self, state):
        senderName = self.sender().text()
        row = self.heightArrayName.index(senderName)
        cols = len(self.cbArray1_18m[0])
        for i in range(cols):
            qcheckbox = self.cbArray1_18m[row][i]
            if state == Qt.Unchecked:
                qcheckbox.setCheckState(Qt.Unchecked)
            elif state == Qt.Checked:
                qcheckbox.setCheckState(Qt.Checked)
            else:
                qcheckbox.setCheckState(Qt.PartiallyChecked)

    def distClicked2m2(self, state):
        senderName = self.sender().text()
        col = self.distArrayName.index(senderName)
        rows = len(self.cbArray2m2)
        for i in range(rows):
            qcheckbox = self.cbArray2m2[i][col]
            if state == Qt.Unchecked:
                qcheckbox.setCheckState(Qt.Unchecked)
            elif state == Qt.Checked:
                qcheckbox.setCheckState(Qt.Checked)
            else:
                qcheckbox.setCheckState(Qt.PartiallyChecked)

    def heightClicked2m2(self, state):
        senderName = self.sender().text()
        row = self.heightArrayName.index(senderName)
        cols = len(self.cbArray2m2[0])
        for i in range(cols):
            qcheckbox = self.cbArray2m2[row][i]
            if state == Qt.Unchecked:
                qcheckbox.setCheckState(Qt.Unchecked)
            elif state == Qt.Checked:
                qcheckbox.setCheckState(Qt.Checked)
            else:
                qcheckbox.setCheckState(Qt.PartiallyChecked)

    def distClicked5m2(self, state):
        senderName = self.sender().text()
        col = self.distArrayName.index(senderName)
        rows = len(self.cbArray5m2)
        for i in range(rows):
            qcheckbox = self.cbArray5m2[i][col]
            if state == Qt.Unchecked:
                qcheckbox.setCheckState(Qt.Unchecked)
            elif state == Qt.Checked:
                qcheckbox.setCheckState(Qt.Checked)
            else:
                qcheckbox.setCheckState(Qt.PartiallyChecked)

    def heightClicked5m2(self, state):
        senderName = self.sender().text()
        row = self.heightArrayName.index(senderName)
        cols = len(self.cbArray5m2[0])
        for i in range(cols):
            qcheckbox = self.cbArray5m2[row][i]
            if state == Qt.Unchecked:
                qcheckbox.setCheckState(Qt.Unchecked)
            elif state == Qt.Checked:
                qcheckbox.setCheckState(Qt.Checked)
            else:
                qcheckbox.setCheckState(Qt.PartiallyChecked)

    def validDataClicked(self, state):
        if state == Qt.Checked:
            self.editSplit.setEnabled(False)
            self.cbSKLearn.setEnabled(False)
        else:
            self.editSplit.setEnabled(True)
            self.cbSKLearn.setEnabled(True)

    def showFileDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open data file', '/srv/MLData', filter="CSV file (*.csv);;All files (*)")
        if fname[0]:
            self.editFile.setText(fname[0])
            self.df = pd.read_csv(fname[0], dtype=float)

    def loadData(self):
        filename = self.editFile.text()
        if filename == '' or filename.startswith('Please'):
            QMessageBox.about(self, 'Warining', 'No CSV Data')
            return

        rows = len(self.heightArrayName)
        cols = len(self.distArrayName)

        listSelected2m = []
        for i in range(rows):
            for j in range(cols):
                qcheckbox = self.cbArray2m[i][j]
                if qcheckbox.checkState() != Qt.Unchecked:
                    listSelected2m.append((i, j))

        listSelected5m = []
        for i in range(rows):
            for j in range(cols):
                qcheckbox = self.cbArray5m[i][j]
                if qcheckbox.checkState() != Qt.Unchecked:
                    listSelected5m.append((i, j))

        listSelected8m = []
        for i in range(rows):
            for j in range(cols):
                qcheckbox = self.cbArray8m[i][j]
                if qcheckbox.checkState() != Qt.Unchecked:
                    listSelected8m.append((i, j))

        listSelected1_18m = []
        for i in range(rows):
            for j in range(cols):
                qcheckbox = self.cbArray1_18m[i][j]
                if qcheckbox.checkState() != Qt.Unchecked:
                    listSelected1_18m.append((i, j))

        listSelected2m2 = []
        for i in range(rows):
            for j in range(cols):
                qcheckbox = self.cbArray2m2[i][j]
                if qcheckbox.checkState() != Qt.Unchecked:
                    listSelected2m2.append((i, j))

        listSelected5m2 = []
        for i in range(rows):
            for j in range(cols):
                qcheckbox = self.cbArray5m2[i][j]
                if qcheckbox.checkState() != Qt.Unchecked:
                    listSelected5m2.append((i, j))

        if len(listSelected2m) == 0 and len(listSelected5m) == 0 and len(listSelected8m) == 0\
                and len(listSelected1_18m) == 0 and len(listSelected2m2) == 0 and len(listSelected5m2) == 0:
            QMessageBox.information(self, 'Warning', 'Select sensor(s) first..')
            return

        try:
            self.readCSV(listSelected2m, listSelected5m, listSelected8m, listSelected1_18m, listSelected2m2, listSelected5m2)

        except ValueError:
            QMessageBox.information(self, 'Error', 'There is some error...')
            return

        self.dataLoaded = True
        QMessageBox.information(self, 'Done', 'Data is Loaded')

    def readCSV(self, listSelected2m, listSelected5m, listSelected8m, listSelected1_18m, listSelected2m2, listSelected5m2):
        self.indexijs2m.clear()
        self.indexijs5m.clear()
        self.indexijs8m.clear()
        self.indexijs1_18m.clear()
        self.indexijs2m2.clear()
        self.indexijs5m2.clear()
        self.time_data = None
        self.time_data_n = []
        self.southSensors2m.clear()
        self.southSensors5m.clear()
        self.southSensors8m.clear()
        self.southSensors1_18m.clear()
        self.southSensors2m2.clear()
        self.southSensors5m2.clear()

        self.time_data = self.df.values[:, 0:1].flatten()
        max_time = max(self.time_data)
        min_time = min(self.time_data)

        for i in range(len(self.time_data)):
            t = self.time_data[i]
            self.time_data_n.append((t - min_time) / (max_time - min_time))

        for index_ij in listSelected2m:
            sensorName, data = self.getPressureData(index_ij, 1)
            self.indexijs2m.append(index_ij)
            self.southSensors2m.append(data)

        for index_ij in listSelected5m:
            sensorName, data = self.getPressureData(index_ij, 1 + 24)
            self.indexijs5m.append(index_ij)
            self.southSensors5m.append(data)

        for index_ij in listSelected8m:
            sensorName, data = self.getPressureData(index_ij, 1 + 24 * 2)
            self.indexijs8m.append(index_ij)
            self.southSensors8m.append(data)

        for index_ij in listSelected1_18m:
            sensorName, data = self.getPressureData(index_ij, 1 + 24 * 3)
            self.indexijs1_18m.append(index_ij)
            self.southSensors1_18m.append(data)

        for index_ij in listSelected2m2:
            sensorName, data = self.getPressureData(index_ij, 1 + 24 * 4)
            self.indexijs2m2.append(index_ij)
            self.southSensors2m2.append(data)

        for index_ij in listSelected5m2:
            sensorName, data = self.getPressureData(index_ij, 1 + 24 * 5)
            self.indexijs5m2.append(index_ij)
            self.southSensors5m2.append(data)

    def _drawScatter(self, southSensors, indexijs):
        numSensors = len(southSensors)
        for index in range(numSensors):
            t_data = self.time_data
            s_data = southSensors[index]

            index_ij = indexijs[index]
            i = index_ij[0]
            j = index_ij[1]

            sensorName = 'S' + self.distArrayName[j] + 'm' + self.heightArrayName[i] + 'Barrier2m'
            plt.scatter(t_data, s_data, label=sensorName, s=1)

    def showPressureGraphs(self):
        if self.dataLoaded == False:
            QMessageBox.information(self, 'Warning', 'Load Data First')
            return

        plt.figure()

        self._drawScatter(self.southSensors2m, self.indexijs2m)
        self._drawScatter(self.southSensors5m, self.indexijs5m)
        self._drawScatter(self.southSensors8m, self.indexijs8m)
        self._drawScatter(self.southSensors1_18m, self.indexijs1_18m)
        self._drawScatter(self.southSensors2m2, self.indexijs2m2)
        self._drawScatter(self.southSensors5m2, self.indexijs5m2)

        plt.title('Pressure Graph')
        plt.xlabel('time (ms)')
        plt.ylabel('pressure (kPa)')
        plt.legend(loc='upper right', markerscale=4.)
        plt.grid()

        plt.show()

    def _appendOPData(self, southSensors, indexijs, xdata, ydata, name):
        numSensors = len(southSensors)
        for index in range(numSensors):
            s_data = southSensors[index]
            overpressure = max(s_data)

            index_ij = indexijs[index]
            i = index_ij[0]
            j = index_ij[1]

            sensorName = 'S' + self.distArrayName[j] + 'm' + self.heightArrayName[i] + name

            xdata.append(sensorName)
            ydata.append(overpressure)

    def showOverpressureGraphs(self):
        if self.dataLoaded == False:
            QMessageBox.information(self, 'Warning', 'Load Data First')
            return

        plt.figure()

        xdata = []
        ydata = []

        self._appendOPData(self.southSensors2m, self.indexijs2m, xdata, ydata, 'Barrier2m')
        self._appendOPData(self.southSensors5m, self.indexijs5m, xdata, ydata, 'Barrier5m')
        self._appendOPData(self.southSensors8m, self.indexijs8m, xdata, ydata, 'Barrier8m')
        self._appendOPData(self.southSensors1_18m, self.indexijs1_18m, xdata, ydata, 'Barrier1.18m')
        self._appendOPData(self.southSensors2m2, self.indexijs2m2, xdata, ydata, 'Barrier2m2')
        self._appendOPData(self.southSensors5m2, self.indexijs5m2, xdata, ydata, 'Barrier5m2')

        plt.plot(xdata, ydata, '-o')

        plt.title('Overpressure Graph')
        plt.xlabel('Sensors')
        plt.ylabel('pressure (kPa)')
        plt.legend(loc='upper right', markerscale=4.)
        plt.xticks(rotation=60)
        plt.grid()
        plt.tight_layout()

        plt.show()

    def _addImpData(self, southSensors, indexijs, xdata, ydata, name):
        numSensors = len(southSensors)
        for index in range(numSensors):
            s_data = southSensors[index]
            impulse = self.getImpulse(s_data)

            index_ij = indexijs[index]
            i = index_ij[0]
            j = index_ij[1]

            sensorName = 'S' + self.distArrayName[j] + 'm' + self.heightArrayName[i] + name

            xdata.append(sensorName)
            ydata.append(impulse)

    def showImpulseGraphs(self):
        if self.dataLoaded == False:
            QMessageBox.information(self, 'Warning', 'Load Data First')
            return

        plt.figure()

        xdata = []
        ydata = []

        self._addImpData(self.southSensors2m, self.indexijs2m, xdata, ydata, 'Barrier2m')
        self._addImpData(self.southSensors5m, self.indexijs5m, xdata, ydata, 'Barrier5m')
        self._addImpData(self.southSensors8m, self.indexijs8m, xdata, ydata, 'Barrier8m')
        self._addImpData(self.southSensors1_18m, self.indexijs1_18m, xdata, ydata, 'Barrier1.18m')
        self._addImpData(self.southSensors2m2, self.indexijs2m2, xdata, ydata, 'Barrier2m2')
        self._addImpData(self.southSensors5m2, self.indexijs5m2, xdata, ydata, 'Barrier5m2')

        plt.bar(xdata, ydata)

        plt.title('Impulse Graph')
        plt.xlabel('Sensors')
        plt.ylabel('Impulse(kgm)/s')
        plt.legend(loc='upper right', markerscale=4.)
        plt.xticks(rotation=60)
        plt.grid()
        plt.tight_layout()

        plt.show()

    def getPressureData(self, index_ij, columnToSkip):
        i = index_ij[0]
        j = index_ij[1]
        len_height = len(self.heightArrayName)
        len_dist = len(self.distArrayName)
        col_no = columnToSkip + len_dist * (len_height - 1 - i) + j
        sensorName = self.distArrayName[j] + self.heightArrayName[i]

        data_raw = self.df.values[:, col_no:col_no+1].flatten()

        # index_at_max = max(range(len(data_raw)), key=data_raw.__getitem__)
        data = self.df.values[:, col_no:col_no+1].flatten()

        return sensorName, data

    def _prepareForMachineLearning(self):
        heightList_n, distList_n, barrierPositionList, pressureList_n, volumeList_n = self._normalize()


        trainArray2m, trainIndex2m, validArray2m, validIndex2m = self.getArrayFromSensors(self.southSensors2m,
                                                                                          self.indexijs2m,
                                                                                          self.cbArray2m)
        trainArray5m, trainIndex5m, validArray5m, validIndex5m = self.getArrayFromSensors(self.southSensors5m,
                                                                                          self.indexijs5m,
                                                                                          self.cbArray5m)
        trainArray8m, trainIndex8m, validArray8m, validIndex8m = self.getArrayFromSensors(self.southSensors8m,
                                                                                          self.indexijs8m,
                                                                                          self.cbArray8m)
        trainArray1_18m, trainIndex1_18m, validArray1_18m, validIndex1_18m = self.getArrayFromSensors(
            self.southSensors1_18m, self.indexijs1_18m, self.cbArray1_18m)
        trainArray2m2, trainIndex2m2, validArray2m2, validIndex2m2 = self.getArrayFromSensors(self.southSensors2m2,
                                                                                              self.indexijs2m2,
                                                                                              self.cbArray2m2)
        trainArray5m2, trainIndex5m2, validArray5m2, validIndex5m2 = self.getArrayFromSensors(self.southSensors5m2,
                                                                                              self.indexijs5m2,
                                                                                              self.cbArray5m2)

        trainArray = trainArray2m + trainArray5m + trainArray8m + trainArray1_18m + trainArray2m2 + trainArray5m2
        trainIndex = trainIndex2m + trainIndex5m + trainIndex8m + trainIndex1_18m + trainIndex2m2 + trainIndex5m2
        trainBarrierPos = []
        pressureList = []
        volumeList = []
        for i in range(len(trainArray2m)):
            trainBarrierPos.append(barrierPositionList[0])
            pressureList.append(pressureList_n[0])
            volumeList.append(volumeList_n[0])
        for i in range(len(trainArray5m)):
            trainBarrierPos.append(barrierPositionList[1])
            pressureList.append(pressureList_n[0])
            volumeList.append(volumeList_n[0])
        for i in range(len(trainArray8m)):
            trainBarrierPos.append(barrierPositionList[2])
            pressureList.append(pressureList_n[0])
            volumeList.append(volumeList_n[0])
        for i in range(len(trainArray1_18m)):
            trainBarrierPos.append(barrierPositionList[3])
            pressureList.append(pressureList_n[1])
            volumeList.append(volumeList_n[1])
        for i in range(len(trainArray2m2)):
            trainBarrierPos.append(barrierPositionList[4])
            pressureList.append(pressureList_n[2])
            volumeList.append(volumeList_n[2])
        for i in range(len(trainArray5m2)):
            trainBarrierPos.append(barrierPositionList[5])
            pressureList.append(pressureList_n[2])
            volumeList.append(volumeList_n[2])

        if len(trainArray) == 0:
                QMessageBox.warning(self, 'warning', 'No Training Data!')
                return

        x_train_data = np.zeros(shape=(self.NUM_DATA * len(trainArray), self.N_FEATURE))
        y_train_data = np.zeros(shape=(self.NUM_DATA * len(trainArray) , 1))

        # get maxp/minp from all of the training data
        maxp = -1000000
        minp = 100000
        if self.cbMinMax.isChecked() == True:
            maxp, minp = self.getMinMaxPresssureOfAllData()
        else:
            maxp, minp = self.getMinMaxPressureOfLoadedData(trainArray)

        for i in range(len(trainArray)):
            index_ij = trainIndex[i]
            s_data = trainArray[i]
            barrierPos = trainBarrierPos[i]
            pressure = pressureList[i]
            volume = volumeList[i]
            x_data_1, y_data_1 = self.prepareOneSensorData(heightList_n[index_ij[0]], distList_n[index_ij[1]], s_data,
                                                           maxp, minp, barrierPos, pressure, volume)

            for j in range(self.NUM_DATA):
                x_train_data[i*self.NUM_DATA + j] = x_data_1[j]
                y_train_data[i*self.NUM_DATA + j] = y_data_1[j]

        return x_train_data, y_train_data

    def _prepareForMachineLearningSKLearn(self, splitPecentage):
        heightList_n, distList_n, barrierPositionList_n, pressureList_n, volumeList_n = self._normalize()

        allArray, allIndex, barrierArray, pressureArray, volumeArray = self.getAllArray(barrierPositionList_n, pressureList_n, volumeList_n)

        if len(allArray) == 0:
            QMessageBox.warning(self, 'warning', 'No Training Data!')
            return

        x_all_data = np.zeros(shape=(self.NUM_DATA * len(allArray), self.N_FEATURE))
        y_all_data = np.zeros(shape=(self.NUM_DATA * len(allArray) , 1))

        # get maxp/minp from all of the training data
        if self.cbMinMax.isChecked() == True:
            maxp, minp = self.getMinMaxPresssureOfAllData()
        else:
            maxp, minp = self.getMinMaxPressureOfLoadedData(allArray)

        for i in range(len(allArray)):
            index_ij = allIndex[i]
            s_data = allArray[i]
            barrierPos = barrierArray[i]
            pressure = pressureArray[i]
            volume = volumeArray[i]
            x_data_1, y_data_1 = self.prepareOneSensorData(heightList_n[index_ij[0]], distList_n[index_ij[1]], s_data,
                                                           maxp, minp, barrierPos, pressure, volume)

            for j in range(self.NUM_DATA):
                x_all_data[i*self.NUM_DATA + j] = x_data_1[j]
                y_all_data[i*self.NUM_DATA + j] = y_data_1[j]

        x_train_data, x_valid_data, y_train_data, y_valid_data = train_test_split(x_all_data, y_all_data,
                                                                                test_size=splitPecentage,
                                                                                random_state=42)

        return x_all_data, y_all_data, x_train_data, y_train_data, x_valid_data, y_valid_data

    def _prepareForMachineLearningManually(self):
        heightList_n, distList_n, barrierPositionList_n, pressureList_n, volumeList_n = self._normalize()

        trainArray2m, trainIndex2m, validArray2m, validIndex2m = self.getArrayFromSensors(self.southSensors2m, self.indexijs2m, self.cbArray2m)
        trainArray5m, trainIndex5m, validArray5m, validIndex5m = self.getArrayFromSensors(self.southSensors5m, self.indexijs5m, self.cbArray5m)
        trainArray8m, trainIndex8m, validArray8m, validIndex8m = self.getArrayFromSensors(self.southSensors8m, self.indexijs8m, self.cbArray8m)
        trainArray1_18m, trainIndex1_18m, validArray1_18m, validIndex1_18m = self.getArrayFromSensors(self.southSensors1_18m, self.indexijs1_18m, self.cbArray1_18m)
        trainArray2m2, trainIndex2m2, validArray2m2, validIndex2m2 = self.getArrayFromSensors(self.southSensors2m2, self.indexijs2m2, self.cbArray2m2)
        trainArray5m2, trainIndex5m2, validArray5m2, validIndex5m2 = self.getArrayFromSensors(self.southSensors5m2, self.indexijs5m2, self.cbArray5m2)

        trainArray = trainArray2m + trainArray5m + trainArray8m + trainArray1_18m + trainArray2m2 + trainArray5m2
        trainIndex = trainIndex2m + trainIndex5m + trainIndex8m + trainIndex1_18m + trainIndex2m2 + trainIndex5m2
        trainBarrierPos = []
        trainPressure = []
        trainVolume = []
        for i in range(len(trainArray2m)):
            trainBarrierPos.append(barrierPositionList_n[0])
            trainPressure.append(pressureList_n[0])
            trainVolume.append(volumeList_n[0])
        for i in range(len(trainArray5m)):
            trainBarrierPos.append(barrierPositionList_n[1])
            trainPressure.append(pressureList_n[0])
            trainVolume.append(volumeList_n[0])
        for i in range(len(trainArray8m)):
            trainBarrierPos.append(barrierPositionList_n[2])
            trainPressure.append(pressureList_n[0])
            trainVolume.append(volumeList_n[0])
        for i in range(len(trainArray1_18m)):
            trainBarrierPos.append(barrierPositionList_n[3])
            trainPressure.append(pressureList_n[1])
            trainVolume.append(volumeList_n[1])
        for i in range(len(trainArray2m2)):
            trainBarrierPos.append(barrierPositionList_n[4])
            trainPressure.append(pressureList_n[2])
            trainVolume.append(volumeList_n[2])
        for i in range(len(trainArray5m2)):
            trainBarrierPos.append(barrierPositionList_n[5])
            trainPressure.append(pressureList_n[2])
            trainVolume.append(volumeList_n[2])

        validArray = validArray2m + validArray5m + validArray8m + validArray1_18m + validArray2m2 + validArray5m2
        validIndex = validIndex2m + validIndex5m + validIndex8m + validIndex1_18m + validIndex2m2 + validIndex5m2
        validBarrierPos = []
        validPressure = []
        validVolume = []
        for i in range(len(validArray2m)):
            validBarrierPos.append(barrierPositionList_n[0])
            validPressure.append(pressureList_n[0])
            validVolume.append(volumeList_n[0])
        for i in range(len(validArray5m)):
            validBarrierPos.append(barrierPositionList_n[1])
            validPressure.append(pressureList_n[0])
            validVolume.append(volumeList_n[0])
        for i in range(len(validArray8m)):
            validBarrierPos.append(barrierPositionList_n[2])
            validPressure.append(pressureList_n[0])
            validVolume.append(volumeList_n[0])
        for i in range(len(validArray1_18m)):
            validBarrierPos.append(barrierPositionList_n[3])
            validPressure.append(pressureList_n[1])
            validVolume.append(volumeList_n[1])
        for i in range(len(validArray2m2)):
            validBarrierPos.append(barrierPositionList_n[4])
            validPressure.append(pressureList_n[2])
            validVolume.append(volumeList_n[2])
        for i in range(len(validArray5m2)):
            validBarrierPos.append(barrierPositionList_n[5])
            validPressure.append(pressureList_n[2])
            validVolume.append(volumeList_n[2])

        if len(trainArray) == 0:
            QMessageBox.warning(self, 'warning', 'No Training Data!')
            return

        x_train_data = np.zeros(shape=(self.NUM_DATA * len(trainArray), self.N_FEATURE))
        y_train_data = np.zeros(shape=(self.NUM_DATA * len(trainArray) , 1))

        x_valid_data = np.zeros(shape=(self.NUM_DATA * len(validArray), self.N_FEATURE))
        y_valid_data = np.zeros(shape=(self.NUM_DATA * len(validArray), 1))

        # get maxp/minp from all of the training data
        if self.cbMinMax.isChecked() == True:
            maxp, minp = self.getMinMaxPresssureOfAllData()
        else:
            maxp, minp = self.getMinMaxPressureOfLoadedData(trainArray)

        for i in range(len(trainArray)):
            index_ij = trainIndex[i]
            s_data = trainArray[i]
            barrierPos = trainBarrierPos[i]
            pressure = trainPressure[i]
            volume = trainVolume[i]
            x_data_1, y_data_1 = self.prepareOneSensorData(heightList_n[index_ij[0]], distList_n[index_ij[1]], s_data,
                                                           maxp, minp, barrierPos, pressure, volume)

            for j in range(self.NUM_DATA):
                x_train_data[i*self.NUM_DATA + j] = x_data_1[j]
                y_train_data[i*self.NUM_DATA + j] = y_data_1[j]

        for i in range(len(validArray)):
            index_ij = validIndex[i]
            s_data = validArray[i]
            barrierPos = validBarrierPos[i]
            pressure = validPressure[i]
            volume = validVolume[i]
            x_data_1, y_data_1 = self.prepareOneSensorData(heightList_n[index_ij[0]], distList_n[index_ij[1]], s_data,
                                                           maxp, minp, barrierPos, pressure, volume)

            for j in range(self.NUM_DATA):
                x_valid_data[i*self.NUM_DATA + j] = x_data_1[j]
                y_valid_data[i*self.NUM_DATA + j] = y_data_1[j]

        allArray, allIndex, barrierArray, pressureArray, volumeArray = self.getAllArray(barrierPositionList_n, pressureList_n, volumeList_n)

        if len(allArray) == 0:
            QMessageBox.warning(self, 'warning', 'No Training Data!')
            return

        x_all_data = np.zeros(shape=(self.NUM_DATA * len(allArray), self.N_FEATURE))
        y_all_data = np.zeros(shape=(self.NUM_DATA * len(allArray), 1))

        for i in range(len(allArray)):
            index_ij = allIndex[i]
            s_data = allArray[i]
            barrierPos = barrierArray[i]
            pressure = pressureArray[i]
            volume = volumeArray[i]
            x_data_1, y_data_1 = self.prepareOneSensorData(heightList_n[index_ij[0]], distList_n[index_ij[1]], s_data,
                                                           maxp, minp, barrierPos, pressure, volume)

            for j in range(self.NUM_DATA):
                x_all_data[i * self.NUM_DATA + j] = x_data_1[j]
                y_all_data[i * self.NUM_DATA + j] = y_data_1[j]

        return x_all_data, y_all_data, x_train_data, y_train_data, x_valid_data, y_valid_data

    def getMinMaxPresssureOfAllData(self):
        maxPressure = -1000000
        minPressure = 1000000
        for i in range(len(self.distList)*6 + 1): # 12 is right value but 6 is better?
            if i == 0:
                continue

            one_data = self.df.values[::, i:i+1].flatten()
            maxp_local = max(one_data)
            minp_local = min(one_data)
            if maxp_local > maxPressure:
                maxPressure = maxp_local
            if minp_local < minPressure:
                minPressure = minp_local

        return maxPressure, minPressure

    def getMinMaxPressureOfLoadedData(self, trainArray):
        maxPressure = -100000
        minPressure = 100000

        numSensors = len(trainArray)
        for i in range(numSensors):
            s_data = trainArray[i]
            maxp_local = max(s_data)
            minp_local = min(s_data)
            if maxp_local > maxPressure:
                maxPressure = maxp_local
            if minp_local < minPressure:
                minPressure = minp_local

        return maxPressure, minPressure

    def doMachineLearningWithData(self):
        if (not self.indexijs2m and not self.indexijs5m and not self.indexijs8m and not self.indexijs1_18m
            and not self.indexijs2m2 and not self.indexijs5m2) or \
                (not self.southSensors2m and not self.southSensors5m and not self.southSensors8m
                 and not self.southSensors1_18m and not self.southSensors2m2 and not self.southSensors5m2):
            QMessageBox.about(self, 'Warining', 'Load Data First')
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)

        batchSize = int(self.editBatch.text())
        epoch = int(self.editEpoch.text())
        if epoch < 1:
            QMessageBox.warning(self, 'warning', 'Epoch shall be greater than 0')
            return

        learningRate = float(self.editLR.text())
        verbose = self.cbVerbose.isChecked()

        # useValidation = self.cbValidData.isChecked()
        splitPercentage = float(self.editSplit.text())
        useSKLearn = self.cbSKLearn.isChecked()
        earlyStopping = self.cbEarlyStop.isChecked()
        useValidation = self.cbValidData.isChecked()

        if useValidation:
            x_all_data, y_all_data, x_train_data, y_train_data, x_valid_data, y_valid_data = \
                self._prepareForMachineLearningManually()
        else:
            if splitPercentage > 0.0 and useSKLearn:
                x_all_data, y_all_data, x_train_data, y_train_data, x_valid_data, y_valid_data = \
                    self._prepareForMachineLearningSKLearn(splitPercentage)
            else:
                x_train_data, y_train_data = self._prepareForMachineLearning()

        if useValidation or (splitPercentage > 0.0 and useSKLearn):
            self.doMachineLearningWithValidation(x_all_data, y_all_data, x_train_data, y_train_data, x_valid_data,
                                                 y_valid_data, batchSize, epoch, learningRate, splitPercentage,
                                                 earlyStopping, verbose)
        else:
            self.doMachineLearning(x_train_data, y_train_data, batchSize, epoch, learningRate, splitPercentage,
                                   earlyStopping, verbose)

        QApplication.restoreOverrideCursor()

    def saveData(self):
        if (not self.indexijs2m and not self.indexijs5m and not self.indexijs8m) or \
                (not self.southSensors2m and not self.southSensors5m and not self.southSensors8m):
            QMessageBox.about(self, 'Warining', 'Load Data First')
            return

        numSensors = len(self.southSensors)
        for i in range(numSensors):
            index_ij = self.indexijs[i]
            t_data = self.time_data
            s_data = self.southSensors[i]

            suggestion = '/srv/MLData/' + self.distArrayName[index_ij[1]] + self.heightArrayName[index_ij[0]] + '.csv'
            filename = QFileDialog.getSaveFileName(self, 'Save File', suggestion, "CSV Files (*.csv)")

            if filename[0] != '':
                file = open(filename[0], 'w')

                column1 = 'Time, ' + self.distArrayName[index_ij[1]] + self.heightArrayName[index_ij[0]] + '\n'
                file.write(column1)

                numData = len(s_data)
                for j in range(numData):
                    line = str(t_data[j]) + ',' + str(s_data[j])
                    file.write(line)
                    file.write('\n')

                file.close()

                QMessageBox.information(self, "Save", filename[0] + " is saved successfully")

    def saveDataAll(self):
        suggestion = '/srv/MLData/Changed.csv'
        # suggestion = '../MLData/' + self.distArrayName[index_ij[1]] + self.heightArrayName[index_ij[0]] + '.csv'
        filename = QFileDialog.getSaveFileName(self, 'Save File', suggestion, "CSV Files (*.csv)")
        if filename[0] != '':
            file = open(filename[0], 'w')

            column1 = 'Time, '
            numSensors = len(self.southSensors)
            for i in range(numSensors):
                index_ij = self.indexijs[i]
                column1 += self.distArrayName[index_ij[1]] + self.heightArrayName[index_ij[0]]
                if i != (numSensors - 1):
                    column1 += ','
            column1 += '\n'

            file.write(column1)

            t_data = self.time_data

            numData = len(t_data)
            for j in range(numData):
                line = str(t_data[j]) + ','

                for i in range(numSensors):
                    s_data = self.southSensors[i]
                    line += str(s_data[j])
                    if i != (numSensors - 1):
                        line += ','

                file.write(line)
                file.write('\n')

            file.close()

            QMessageBox.information(self, "Save", filename[0] + " is saved successfully")

    def saveModel(self):
        suggestion = ''
        filename = ''
        h5checked = self.cbH5Format.isChecked()
        if h5checked == True:
            suggestion = '/srv/MLData/newModel.h5'
            filename = QFileDialog.getSaveFileName(self, 'Save Model File', suggestion, filter="h5 file (*.h5)")
        else:
            suggestion = '/srv/MLData'
            filename = QFileDialog.getSaveFileName(self, 'Save File', suggestion, "Model")

        if filename[0] != '':
            self.modelLearner.saveModel(filename[0])

            QMessageBox.information(self, 'Saved', 'Model is saved.')

    def saveModelJS(self):
        suggestion = '/srv/MLData'
        filename = QFileDialog.getSaveFileName(self, 'Save File', suggestion, "Model")
        if filename[0] != '':
            self.modelLearner.saveModelJS(filename[0])

            QMessageBox.information(self, 'Saved', 'Model is saved for Javascript.')

    def loadModel(self):
        if self.cbH5Format.isChecked():
            fname = QFileDialog.getOpenFileName(self, 'Open data file', '/srv/MLData',
                                                filter="CSV file (*.h5);;All files (*)")
            if fname[0] != '':
                QApplication.setOverrideCursor(Qt.WaitCursor)
                self.modelLearner.loadModel(fname[0])
                QApplication.restoreOverrideCursor()

                QMessageBox.information(self, 'Loaded', 'Model is loaded.')

        else:
            fname = QFileDialog.getExistingDirectory(self, 'Select Folder', "/srv/MLData",
                                                      QFileDialog.ShowDirsOnly)

            if fname != '':
                QApplication.setOverrideCursor(Qt.WaitCursor)
                self.modelLearner.loadModel(fname)
                QApplication.restoreOverrideCursor()

                QMessageBox.information(self, 'Loaded', 'Model is loaded.')

    def checkVal(self):
        if (not self.indexijs2m and not self.indexijs5m and not self.indexijs8m and
        not self.indexijs1_18m and not self.indexijs2m2 and not self.indexijs5m2) or \
                (not self.southSensors2m and not self.southSensors5m and not self.southSensors8m and
                not self.southSensors1_18m and not self.southSensors2m2 and not self.southSensors5m2):
            QMessageBox.about(self, 'Warining', 'Load Data First')
            return

        if self.modelLearner.modelLoaded == False:
            QMessageBox.about(self, 'Warning', 'Model is not created/loaded')
            return

        start_time = time.time()

        QApplication.setOverrideCursor(Qt.WaitCursor)

        x_data, y_data = self._prepareForMachineLearning()
        y_predicted = self.modelLearner.predict(x_data)

        QApplication.restoreOverrideCursor()

        print("--- %s seconds ---" % (time.time() - start_time))

        datasize = len(y_data)
        x_display = np.zeros((datasize, 1))
        for j in range(datasize):
            x_display[j][0] = j

        plt.figure()
        plt.scatter(x_display, y_data, label='original data', color="red", s=1)
        plt.scatter(x_display, y_predicted, label='predicted', color="blue", s=1)
        plt.title('Machine Learning Prediction')
        plt.xlabel('time (ms)')
        plt.ylabel('pressure (kPa)')
        plt.legend(loc='upper right')
        plt.grid()

        plt.show()

    def predict(self):
        if self.time_data is None:
            QMessageBox.warning(self, 'Warning', 'For timedata, load at least one data')
            return

        if self.modelLearner.modelLoaded == False:
            QMessageBox.about(self, 'Warning', 'Model is not created/loaded')
            return

        maxp, minp = self.getMinMaxPresssureOfAllData()

        sBarrierPosition = self.editBarrierPos.text()
        barrierList = sBarrierPosition.split(',')

        sTankPressure = self.editTankPressure.text()
        tankPressure = float(sTankPressure)

        sTankVolume = self.cbTankVolume.currentText()
        tankVolume = float(sTankVolume)

        if self.cbFixToBarrierPos.isChecked():
            self.predictToBarrierPos(maxp, minp, barrierList, tankPressure, tankVolume)
        else:
            self.predictPressure(maxp, minp, barrierList, tankPressure, tankVolume)

    def predictToBarrierPos(self, maxp, minp, barrierList, tankPressure, tankVolume):
        QApplication.setOverrideCursor(Qt.WaitCursor)

        y_array = []
        for sBarrier in barrierList:
            barrierPosition = float(sBarrier)

            distHeight = (barrierPosition, 1.0) # height is 1.0m fixed now
            x_data = self._prepareOneSensorForPredict(distHeight, barrierPosition, tankPressure, tankVolume)

            y_predicted = self.modelLearner.predict(x_data)
            self.unnormalize(y_predicted, maxp, minp)
            y_array.append(y_predicted)

            # self.savePressureData(distHeightArray, y_array)

        resultArray = self.showOverpressurePredictionGraphs(barrierList, y_array)

        QApplication.restoreOverrideCursor()

    def predictPressure(self, maxp, minp, barrierList, tankPressure, tankVolume):
        distCount = self.tableGridWidget.columnCount()
        heightCount = self.tableGridWidget.rowCount()
        if distCount < 1 or heightCount < 1:
            QMessageBox.warning(self, 'Warning', 'You need to add distance or height to predict')
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)

        sDistHeightArray = []
        distHeightArray = []

        # put sensors in order of original Pukyong Excel file
        for j in range(heightCount):
            rownum = heightCount - (j + 1)
            height = self.tableGridWidget.verticalHeaderItem(rownum).text()
            heightOnly = height[1:len(height) - 1]

            for i in range(distCount):
                dist = self.tableGridWidget.horizontalHeaderItem(i).text()
                distOnly = dist[1:len(dist) - 1]

                distf = float(distOnly)
                heightf = float(heightOnly)

                sDistHeightArray.append(dist+height)
                distHeightArray.append((distf, heightf))

        for sBarrier in barrierList:
            barrierPosition = float(sBarrier)

            y_array = []
            for distHeight in distHeightArray:
                x_data = self._prepareOneSensorForPredict(distHeight, barrierPosition, tankPressure, tankVolume)

                y_predicted = self.modelLearner.predict(x_data)
                self.unnormalize(y_predicted, maxp, minp)

                y_array.append(y_predicted)

            # self.savePressureData(distHeightArray, y_array)

            resultArray = self.showPredictionGraphs(sDistHeightArray, distHeightArray, y_array)

        QApplication.restoreOverrideCursor()

    def saveOPImpulse(self, resultArray, barrierPosition):
        # reply = QMessageBox.question(self, 'Message', 'Do you want to save overpressure and impulse to a file?',
        #                              QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        reply = QMessageBox.Yes
        if reply == QMessageBox.Yes:
            sMultiplier = self.editMultiplier.text()

            suggestion = '/srv/MLData/opAndImpulse_pukyong_B' + str(barrierPosition) + 'm_' + sMultiplier + 'xB.csv'
            # filename = QFileDialog.getSaveFileName(self, 'Save File', suggestion, "CSV Files (*.csv)")
            #
            # if filename[0] != '':
            if suggestion != '':
                QApplication.setOverrideCursor(Qt.WaitCursor)
                # file = open(filename[0], 'w')
                file = open(suggestion, 'w')

                column1 = 'distance,height,indexAtMax,overpressure,indexAtZero,impulse\n'
                file.write(column1)

                for col in resultArray:
                    file.write(col+'\n')

                QApplication.restoreOverrideCursor()

    def savePressureData(self, distHeightArray, y_array):
        reply = QMessageBox.question(self, 'Message', 'Do you want to save Pressure data to files?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            suggestion = '/srv/MLData/predicted_barrier' + str(barrierPosition) + 'm.csv'
            filename = QFileDialog.getSaveFileName(self, 'Save File', suggestion, "CSV Files (*.csv)")

            if filename[0] != '':
                QApplication.setOverrideCursor(Qt.WaitCursor)

                file = open(filename[0], 'w')

                column1 = 'Time,Barrier_Dist,'
                for distHeight in distHeightArray:
                    distance = distHeight[0]
                    height = distHeight[1]
                    colname = 'S' + str(distance) + 'mH' + str(height) + 'm,'
                    column1 += colname

                column1 = column1[:-1]
                column1 += '\n'
                file.write(column1)

                dataSize = len(y_array[0])
                for index in range(dataSize):
                    time = self.time_data[index]
                    nextColumn = str(time) + ',' + self.editBarrierPos.text() + ','

                    for yindex in range(len(y_array)):
                        one_y = y_array[yindex]
                        onep = one_y[index][0]
                        nextColumn += str(onep) + ','

                    nextColumn = nextColumn[:-1]
                    nextColumn += '\n'
                    file.write(nextColumn)

                QApplication.restoreOverrideCursor()

    def unnormalize(self, data, max, min):
        for i in range(len(data)):
            data[i] = data[i] * (max - min) + min

    def _prepareOneSensorForPredict(self, distHeight, barrierPosition, tankPressure, tankVolume):
        distList_n = np.copy(self.distList)
        maxDist = max(distList_n)
        minDist = min(distList_n)

        heightList_n = np.copy(self.heightList)
        maxHeight = max(heightList_n)
        minHeight = min(heightList_n)

        distance = (distHeight[0] - minDist) / (maxDist - minDist)
        height = (distHeight[1] - minHeight) / (maxHeight - minHeight)

        # check for barrierPosition normalization
        multiplier = float(self.editMultiplier.text())
        if multiplier < 0.1:
            barrierPosition = (barrierPosition - 1.18) / (8 - 1.18)
        else:
            barrierPosition *= multiplier

        # pressureList = [95, 76.269, 103]
        tankPressure_n = (tankPressure - 76.269) / (103 - 76.269)

        # volumeList = [184, 175, 721]
        tankVolume_n = (tankVolume - 175) / (721 - 175)

        x_data = np.zeros(shape=(self.NUM_DATA, self.N_FEATURE))

        for i in range(self.NUM_DATA):
            x_data[i][0] = self.time_data[i]
            x_data[i][1] = height
            x_data[i][2] = distance
            x_data[i][3] = barrierPosition
            x_data[i][4] = tankPressure_n
            x_data[i][5] = tankVolume_n

        return x_data

    def showPredictionGraphs(self, sDistHeightArray, distHeightArray, y_array):
        # numSensors = len(y_array)
        resultArray = []

        plt.figure()
        for i in range(len(y_array)):
            t_data = self.time_data
            s_data = y_array[i]

            distHeight = distHeightArray[i]
            lab = sDistHeightArray[i]

            distance = distHeight[0]
            height = distHeight[1]

            index_at_max = max(range(len(s_data)), key=s_data.__getitem__)
            overpressure = max(s_data)
            impulse, index_at_zero = self.getImpulseAndIndexZero(s_data)

            dispLabel = lab + '/op=' + format(overpressure[0], '.2f') + '/impulse=' + format(impulse, '.2f')

            resultArray.append(str(distance) + ',' + str(height) + ',' + str(index_at_max) + ',' +
                               format(overpressure[0], '.6f') + ',' + str(index_at_zero) + ',' + format(impulse, '.6f'))

            plt.scatter(t_data, s_data, label=dispLabel, s=1)

        plt.title('Pressure Graph')
        plt.xlabel('time (ms)')
        plt.ylabel('pressure (kPa)')
        plt.legend(loc='upper right', markerscale=4.)
        plt.grid()

        plt.show()

        return resultArray

    def showOverpressurePredictionGraphs(self, barrierList, y_array):
        xdata = []
        opdata = []
        impdata = []

        for i in range(len(y_array)):
            s_data = y_array[i]

            barrierDist = float(barrierList[i])

            # index_at_max = max(range(len(s_data)), key=s_data.__getitem__)
            overpressure = max(s_data)
            impulse, index_at_zero = self.getImpulseAndIndexZero(s_data)

            opdata.append(overpressure)
            impdata.append(impulse)

            dispLabel = format(barrierDist, '.1f')
            xdata.append(dispLabel)

        # fig = plt.figure()
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(xdata, opdata, '-o', color='red', label='overpressure')
        ax2.bar(xdata, impdata, color='limegreen', label='impulse', alpha=0.5)

        plt.title('Overpressure/Impulse Graph')
        ax1.set_xlabel('Barrier Position (m)')
        ax1.set_ylabel('pressure (kPa)')
        ax2.set_ylabel('impulse (kg m/s)')

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc='upper right')

        plt.show()

    def getImpulse(self, data):
        sumImpulse = 0
        impulseArray = []
        for i in range(len(data)):
            cur_p = data[i]
            sumImpulse += cur_p
            impulseArray.append(sumImpulse)

        impulse = max(impulseArray)

        return impulse

    def getImpulseAndIndexZero(self, data):
        index_at_max = max(range(len(data)), key=data.__getitem__)

        sumImpulse = 0
        impulseArray = []
        initP = data[0][0]
        for i in range(len(data)):
            cur_p = data[i][0]
            if cur_p > 0 and cur_p <= initP :
                cur_p = 0

            sumImpulse += cur_p * 0.000002
            impulseArray.append(sumImpulse)

        # index_at_zero = max(range(len(impulseArray)), key=impulseArray.__getitem__)
        impulse = max(impulseArray)
        index_at_zero = impulseArray.index(impulse)

        return impulse, index_at_zero

    def checkDataGraph(self, sensorName, time_data, rawdata, filetered_data, data_label, iterNum):
        impulse_original = self.getImpulse(rawdata)
        impulse_filtered = self.getImpulse(filetered_data)

        plt.figure()

        rawLabel = 'Raw-Normalized (impulse=' + format(impulse_original, '.4f') + ')'
        plt.scatter(time_data, rawdata, label=rawLabel, color="red", s=1)
        filterLabel = data_label + ' (iter=' + str(iterNum) + ', impulse=' + format(impulse_filtered, '.4f') + ')'
        plt.scatter(time_data, filetered_data, label=filterLabel, color="blue", s=1)
        plt.title(sensorName)
        plt.xlabel('time (ms)')
        plt.ylabel('pressure (kPa)')
        plt.legend(loc='upper right')
        plt.grid()

        plt.show()

    def checkDataGraph2(self, sensorName, time_data, rawdata, filetered_data, data_label, index_at_max, overpressure, index_at_zero):
        impulse_original = self.getImpulse(rawdata)
        impulse_filtered = self.getImpulse(filetered_data)

        plt.figure()

        rawLabel = 'Raw-Normalized (impulse=' + format(impulse_original, '.4f') + ')'
        plt.scatter(time_data, rawdata, label=rawLabel, color="red", s=1)
        filterLabel = data_label + ', impulse=' + format(impulse_filtered, '.4f') + ',indexMax=' + str(index_at_max) \
                      + ',Overpressure=' + str(overpressure) + ',indexZero=' + str(index_at_zero)
        plt.scatter(time_data, filetered_data, label=filterLabel, color="blue", s=1)
        plt.title(sensorName)
        plt.xlabel('time (ms)')
        plt.ylabel('pressure (kPa)')
        plt.legend(loc='upper right')
        plt.grid()

        plt.show()

    def prepareOneSensorData(self, height, dist, data, maxp, minp, barrierPos, pressure, volume):
        datasize = len(data)
        cur_data = list(data)

        # don't remember why I didn't normalize pressure....
        for j in range(datasize):
            cur_data[j] = (cur_data[j] - minp) / (maxp - minp)

        x_data = np.zeros((datasize, self.N_FEATURE))

        # fill the data
        for j in range(datasize):
            x_data[j][0] = self.time_data[j]
            x_data[j][1] = height
            x_data[j][2] = dist
            x_data[j][3] = barrierPos
            x_data[j][4] = pressure
            x_data[j][5] = volume

        y_data = np.zeros((datasize, 1))
        for j in range(datasize):
            y_data[j][0] = cur_data[j]

        return x_data, y_data

    def _appendData(self, southSensors, indexijs, allArray, allIndex):
        numSensors = len(southSensors)
        for i in range(numSensors):
            indexij = indexijs[i]
            one_data = southSensors[i]
            allArray.append(one_data)
            allIndex.append(indexij)

    def getAllArray(self, barrierPositionList_n, pressureList_n, volumeList_n):
        allArray = []
        allIndex = []
        barrierArray = []
        pressureArray = []
        volumeArray = []

        self._appendData(self.southSensors2m, self.indexijs2m, allArray, allIndex)
        for i in range(len(self.southSensors2m)):
            barrierArray.append(barrierPositionList_n[0])
            pressureArray.append(pressureList_n[0])
            volumeArray.append(volumeList_n[0])
        self._appendData(self.southSensors5m, self.indexijs5m, allArray, allIndex)
        for i in range(len(self.southSensors5m)):
            barrierArray.append(barrierPositionList_n[1])
            pressureArray.append(pressureList_n[0])
            volumeArray.append(volumeList_n[0])
        self._appendData(self.southSensors8m, self.indexijs8m, allArray, allIndex)
        for i in range(len(self.southSensors8m)):
            barrierArray.append(barrierPositionList_n[2])
            pressureArray.append(pressureList_n[0])
            volumeArray.append(volumeList_n[0])
        self._appendData(self.southSensors1_18m, self.indexijs1_18m, allArray, allIndex)
        for i in range(len(self.southSensors1_18m)):
            barrierArray.append(barrierPositionList_n[3])
            pressureArray.append(pressureList_n[1])
            volumeArray.append(volumeList_n[1])
        self._appendData(self.southSensors2m2, self.indexijs2m2, allArray, allIndex)
        for i in range(len(self.southSensors2m2)):
            barrierArray.append(barrierPositionList_n[4])
            pressureArray.append(pressureList_n[2])
            volumeArray.append(volumeList_n[2])
        self._appendData(self.southSensors5m2, self.indexijs5m2, allArray, allIndex)
        for i in range(len(self.southSensors5m2)):
            barrierArray.append(barrierPositionList_n[5])
            pressureArray.append(pressureList_n[2])
            volumeArray.append(volumeList_n[2])

        return allArray, allIndex, barrierArray, pressureArray, volumeArray

    def getArrayFromSensors(self, southSensors, indexijs, cbArray):
        numSensors = len(southSensors)

        # separate data to train & validation
        trainArray = []
        trainIndex = []
        validArray = []
        validIndex = []

        useValidation = self.cbValidData.isChecked()
        if useValidation:
            for i in range(numSensors):
                indexij = indexijs[i]
                row_num = indexij[0]
                col_num = indexij[1]
                one_data = self.southSensors2m[i]

                if cbArray[row_num][col_num].checkState() == Qt.PartiallyChecked:
                    validArray.append(one_data)
                    validIndex.append(indexij)
                elif cbArray[row_num][col_num].checkState() == Qt.Checked:
                    trainArray.append(one_data)
                    trainIndex.append(indexij)
        else:
            for i in range(numSensors):
                indexij = indexijs[i]
                # row_num = indexij[0]
                one_data = southSensors[i]

                trainArray.append(one_data)
                trainIndex.append(indexij)

        return trainArray, trainIndex, validArray, validIndex

    def _normalize(self):
        heightList_n = np.copy(self.heightList)
        maxHeight = max(heightList_n)
        minHeight = min(heightList_n)
        for j in range(len(heightList_n)):
            heightList_n[j] = (heightList_n[j] - minHeight) / (maxHeight - minHeight)

        distList_n = np.copy(self.distList)
        maxDist = max(distList_n)
        minDist = min(distList_n)
        for j in range(len(distList_n)):
            distList_n[j] = (distList_n[j] - minDist) / (maxDist - minDist)

        # separate data to train & validation
        barrierPositionList = [2, 5, 8, 1.18, 2, 5]
        # Normalized check
        multiplier = float(self.editMultiplier.text())
        for i in range(len(barrierPositionList)):
            if multiplier < 0.1:
                barrierPositionList[i] = (barrierPositionList[i] - 1.18) / (8 - 1.18)
            else:
                barrierPositionList[i] *= multiplier

        pressureList = [95, 76.269, 103]
        for i in range(len(pressureList)):
            pressureList[i] = (pressureList[i] - 76.269) / (103 - 76.269)

        volumeList = [184, 175, 721]
        for i in range(len(volumeList)):
            volumeList[i] = (volumeList[i] - 175) / (721 - 175)

        return heightList_n, distList_n, barrierPositionList, pressureList, volumeList

    def doMachineLearning(self, x_data, y_data, batchSize, epoch, learningRate, splitPercentage, earlyStopping, verb):
        self.epochPbar.setMaximum(epoch)

        if self.cbResume.isChecked() == False or self.modelLearner.modelLoaded == False:
            nnList = self.getNNLayer()
            self.modelLearner.set(nnList, batchSize, epoch, learningRate, splitPercentage, earlyStopping, verb,
                                  TfCallback(self.epochPbar))

        training_history = self.modelLearner.fit(x_data, y_data)

        y_predicted = self.modelLearner.predict(x_data)
        self.modelLearner.showResult(y_data, training_history, y_predicted, 'Sensors', 'Height')

    def doMachineLearningWithValidation(self, x_all_data, y_all_data, x_train_data, y_train_data, x_valid_data,
                                        y_valid_data, batchSize, epoch, learningRate, splitPercentage, earlyStopping,
                                        verb):
        self.epochPbar.setMaximum(epoch)

        if self.cbResume.isChecked() == False or self.modelLearner.modelLoaded == False:
            nnList = self.getNNLayer()
            self.modelLearner.set(nnList, batchSize, epoch, learningRate, splitPercentage, earlyStopping, verb,
                                  TfCallback(self.epochPbar))

        training_history = self.modelLearner.fitWithValidation(x_train_data, y_train_data, x_valid_data, y_valid_data)

        y_predicted = self.modelLearner.predict(x_all_data)
        y_train_pred = self.modelLearner.predict(x_train_data)
        y_valid_pred = self.modelLearner.predict(x_valid_data)

        self.modelLearner.showResultValid(y_all_data, training_history, y_predicted, y_train_data, y_train_pred,
                                          y_valid_data, y_valid_pred, 'Sensors', 'Height')

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

class TfCallback(tf.keras.callbacks.Callback):
    def __init__(self, pbar):
        super().__init__()
        self.pbar = pbar
        self.curStep = 1

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.setValue(self.curStep)
        self.curStep += 1

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PukyongMLWindow()
    sys.exit(app.exec_())