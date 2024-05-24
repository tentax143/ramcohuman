from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
import cv2
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import os

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1942, 978)
        MainWindow.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))

        # Main widget
        self.counter = QtWidgets.QWidget(MainWindow)
        self.counter.setObjectName("counter")
        
        # Total in LCD display
        self.total_in = QtWidgets.QLCDNumber(self.counter)
        self.total_in.setGeometry(QtCore.QRect(1630, 130, 251, 51))
        self.total_in.setFont(QtGui.QFont("Yu Gothic UI Semibold", 9, QtGui.QFont.Bold))
        self.total_in.setObjectName("total_in")
        self.total_in.setStyleSheet("color: green;")
        
        # Total out LCD display
        self.total_out = QtWidgets.QLCDNumber(self.counter)
        self.total_out.setGeometry(QtCore.QRect(1630, 410, 251, 51))
        self.total_out.setObjectName("total_out")
        self.total_out.setStyleSheet("color: red;")
        
        # Total inside LCD display
        self.total_inside = QtWidgets.QLCDNumber(self.counter)
        self.total_inside.setGeometry(QtCore.QRect(1630, 270, 251, 51))
        self.total_inside.setObjectName("total_inside")
        self.total_inside.setStyleSheet("color: #E3651D;")
        
        # Car count LCD display
        self.car_count = QtWidgets.QLCDNumber(self.counter)
        self.car_count.setGeometry(QtCore.QRect(1630, 550, 251, 51))
        self.car_count.setObjectName("car_count")
        self.car_count.setStyleSheet("color: blue;")
        
        # Live play view
        self.LIVE_PLAY = QtWidgets.QGraphicsView(self.counter)
        self.LIVE_PLAY.setGeometry(QtCore.QRect(40, 80, 1561, 871))
        self.LIVE_PLAY.setObjectName("LIVE_PLAY")
        
        # Title label
        self.TITLE = QtWidgets.QLabel(self.counter)
        self.TITLE.setGeometry(QtCore.QRect(690, 10, 471, 71))
        self.TITLE.setFont(QtGui.QFont("Poppins ExtraBold", 28, QtGui.QFont.Bold))
        self.TITLE.setObjectName("TITLE")
        
        # Labels
        self.create_label(self.counter, "TOTAL ENTRY", QtCore.QRect(1630, 100, 251, 16))
        self.create_label(self.counter, "TRANSPORT ENTRY", QtCore.QRect(1630, 520, 251, 16))
        self.create_label(self.counter, "TOTAL INSIDE", QtCore.QRect(1630, 380, 251, 16))
        self.create_label(self.counter, "TOTAL EXIT", QtCore.QRect(1630, 230, 251, 16))
        self.create_label(self.counter, "TOTAL IN TRANSPORT", QtCore.QRect(1630, 660, 251, 16))
        self.create_label(self.counter, "TOTAL OUT TRANSPORT", QtCore.QRect(1630, 760, 251, 16))
        
        # Text edits for transport entry
        self.TOTALTRANNSPORTENTRY = QtWidgets.QTextEdit(self.counter)
        self.TOTALTRANNSPORTENTRY.setGeometry(QtCore.QRect(1630, 690, 241, 31))
        self.TOTALTRANNSPORTENTRY.setObjectName("TOTALTRANNSPORTENTRY")
        
        self.TOTALTRANNSPORTENTRY_2 = QtWidgets.QTextEdit(self.counter)
        self.TOTALTRANNSPORTENTRY_2.setGeometry(QtCore.QRect(1630, 790, 241, 31))
        self.TOTALTRANNSPORTENTRY_2.setObjectName("TOTALTRANNSPORTENTRY_2")
        
        # Set central widget and status bar
        MainWindow.setCentralWidget(self.counter)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def create_label(self, parent, text, geometry):
        """ Helper method to create a label with specified text and geometry. """
        label = QtWidgets.QLabel(parent)
        label.setGeometry(geometry)
        label.setFont(QtGui.QFont("", 14))
        label.setText(text)
        label.setObjectName(text.replace(" ", "").upper())

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.TITLE.setText(_translate("MainWindow", "HUMAN DETECTOR"))

class MainApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Ensure the YOLO model file is available
        if not os.path.exists("yolov8n.pt"):
            print("Downloading YOLOv8 model file...")
            YOLO("yolov8n.pt")  # This will download the file if not present

        # Initialize YOLO model and video capture
        self.model = YOLO("yolov8n.pt")
        self.cap = cv2.VideoCapture("rtsp://admin:admin123%23@172.16.3.21:554")
        assert self.cap.isOpened(), "Error reading video file"
        self.counter = object_counter.ObjectCounter()
        self.counter.set_args(view_img=True,
                              reg_pts=[(260, 300), (1100, 250)],
                              classes_names=self.model.names,
                              draw_tracks=True,
                              line_thickness=2)
        self.in_count = 0
        self.out_count = 0
        self.inside_count = 0
        self.car_count_val = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update frame every 30 ms

        # Connect the text edit to the function for updating the entry count
        self.TOTALTRANNSPORTENTRY.textChanged.connect(self.update_transport_entry)

    def update_transport_entry(self):
        try:
            value = int(self.TOTALTRANNSPORTENTRY.toPlainText())
            self.in_count += value
            self.TOTALTRANNSPORTENTRY.clear()
            self.total_in.display(self.in_count)
            self.total_inside.display(self.in_count - self.out_count)
        except ValueError:
            pass  # Ignore if the value is not an integer

    def update_frame(self):
        success, im0 = self.cap.read()
        if not success:
            self.timer.stop()
            print("Video frame is empty or video processing has been successfully completed.")
            return

        tracks = self.model.track(im0, persist=True, show=False, classes=[0, 2])
        im0 = self.counter.start_counting(im0, tracks)

        # Check the current in and out counts
        current_in_count = self.counter.in_counts
        current_out_count = self.counter.out_counts
        current_car_count = self.counter.in_counts  # Update as per logic for transport entry

        # Ensure out count is not negative and only increases
        if current_out_count > self.out_count:
            self.out_count = current_out_count

        # Ensure in count only increases
        if current_in_count > self.in_count:
            self.in_count = current_in_count

        # Ensure out_count is not negative
        if self.out_count < 0:
            self.out_count = 0
            print("Out count adjusted to 0 to prevent negative value.")

        self.inside_count = self.in_count - self.out_count

        # Update GUI with counts
        self.total_in.display(self.in_count)
        self.total_out.display(self.out_count)
        self.total_inside.display(self.inside_count)

        # Update car count
        if current_car_count != self.car_count_val:
            self.car_count_val = current_car_count
            self.car_count.display(self.car_count_val)

        # Convert frame to QImage for display in QLabel
        rgb_image = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # Display the image in the LIVE_PLAY widget
        scene = QtWidgets.QGraphicsScene()
        scene.addPixmap(pixmap)
        self.LIVE_PLAY.setScene(scene)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainApp()
    mainWindow.show()
    sys.exit(app.exec_())
