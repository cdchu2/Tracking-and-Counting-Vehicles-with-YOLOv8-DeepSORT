import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog

from Vehicles_Counting import VehicleCounter

class VehicleCounterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Car Counter")
        self.setGeometry(100, 100, 400, 200)

        layout = QHBoxLayout()

        #left_layout

        left_layout = QVBoxLayout()

        self.video_path_label = QLabel("VIDEO_PATH:")
        self.video_path_input = QLineEdit()
        self.video_path_button = QPushButton("Browse")
        self.video_path_button.clicked.connect(self.browse_video_path)

        self.res_path_label = QLabel("RES_PATH:")
        self.res_path_input = QLineEdit()
        self.res_path_button = QPushButton("Browse")
        self.res_path_button.clicked.connect(self.browse_res_path)
        
        self.yolo_weights_path_label = QLabel("YOLO_WEIGHT:")
        self.yolo_weights_path_input = QLineEdit()
        self.yolo_weights_path_button = QPushButton("Browse")
        self.yolo_weights_path_button.clicked.connect(self.browse_yolo_weights_path)
        
        self.model_path_label = QLabel("MODEL_PATH:")
        self.model_path_input = QLineEdit()
        self.model_path_button = QPushButton("Browse")
        self.model_path_button.clicked.connect(self.browse_model_path)
        
        self.classes_path_label = QLabel("CLASSES_PATH:")
        self.classes_path_input = QLineEdit()
        self.classes_path_button = QPushButton("Browse")
        self.classes_path_button.clicked.connect(self.browse_classes_path)

        left_layout.addWidget(self.video_path_label)
        left_layout.addWidget(self.video_path_input)
        left_layout.addWidget(self.video_path_button)
        
        left_layout.addWidget(self.res_path_label)
        left_layout.addWidget(self.res_path_input)
        left_layout.addWidget(self.res_path_button)
        
        left_layout.addWidget(self.yolo_weights_path_label)
        left_layout.addWidget(self.yolo_weights_path_input)
        left_layout.addWidget(self.yolo_weights_path_button)
        
        left_layout.addWidget(self.model_path_label)
        left_layout.addWidget(self.model_path_input)
        left_layout.addWidget(self.model_path_button)
        
        left_layout.addWidget(self.classes_path_label)
        left_layout.addWidget(self.classes_path_input)
        left_layout.addWidget(self.classes_path_button)


        layout.addLayout(left_layout)

        #right_layout

        right_layout = QVBoxLayout()
        right_layout.addStretch(1)
        layout.addLayout(right_layout)
        button_layout = QVBoxLayout()

        self.start_button = QPushButton("Start")
        self.start_button.setFixedSize(200, 200)
        self.start_button.clicked.connect(self.start_vehicles_counter)

        self.end_button = QPushButton("End")
        self.end_button.setFixedSize(200, 200)
        self.end_button.clicked.connect(self.end_vehicles_counter)

        button_layout.addStretch(1)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.end_button)

        right_layout.addStretch(1)
        right_layout.addLayout(button_layout)
        right_layout.addStretch(1)

        widget = QWidget(self)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.vehicles_counter = None

    def browse_video_path(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        path, _ = file_dialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi)")
        self.video_path_input.setText(path)

    def browse_res_path(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        path, _ = file_dialog.getOpenFileName(self, "Select RES File", "", "RES Files (*.mp4 *.avi)")
        self.res_path_input.setText(path)
        
    def browse_yolo_weights_path(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        path, _ = file_dialog.getOpenFileName(self, "Select YOLO Weights File", "", "Weights Files (*.pt)")
        self.yolo_weights_path_input.setText(path)
        
    def browse_model_path(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        path, _ = file_dialog.getOpenFileName(self, "Select Model File", "", "Model Files (*.pb)")
        self.model_path_input.setText(path)
        
    def browse_classes_path(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        path, _ = file_dialog.getOpenFileName(self, "Select Classes File", "", "Classes Files (*.names)")
        self.classes_path_input.setText(path)

    def start_vehicles_counter(self):
        video_path = self.video_path_input.text()
        res_path = self.res_path_input.text()
        yolo_weights_path = self.yolo_weights_path_input.text()
        model_path = self.model_path_input.text()
        classes_path = self.classes_path_input.text()


        self.vehicles_counter = VehicleCounter(video_path, res_path, yolo_weights_path, model_path, classes_path)
        self.vehicles_counter.initialize()
        self.vehicles_counter.process_video()

    def end_vehicles_counter(self):
        if self.vehicles_counter:
            self.vehicles_counter.end_processing()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VehicleCounterWindow()
    window.show()
    sys.exit(app.exec())
