"""
Main GUI Application for Fatigue Detection System
"""
import sys
import cv2
import yaml
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QGroupBox, QTableWidget,
    QTableWidgetItem, QProgressBar
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from deployment.detector import FatigueDetector
from alerts.voice_engine import VoiceAlertEngine
from database.db_manager import DatabaseManager


class VideoThread(QThread):
    """Thread for video capture and processing"""
    change_pixmap_signal = pyqtSignal(np.ndarray)
    detection_result_signal = pyqtSignal(dict)

    def __init__(self, detector, camera_id=0):
        super().__init__()
        self.detector = detector
        self.camera_id = camera_id
        self.running = False

    def run(self):
        """Main video processing loop"""
        cap = cv2.VideoCapture(self.camera_id)
        self.running = True

        while self.running:
            ret, frame = cap.read()
            if ret:
                # Detect fatigue
                result = self.detector.detect(frame)

                # Draw landmarks and info on frame
                if result['status'] == 'success':
                    frame = self._draw_info(frame, result)

                # Emit signals
                self.change_pixmap_signal.emit(frame)
                self.detection_result_signal.emit(result)

        cap.release()

    def _draw_info(self, frame, result):
        """Draw detection info on frame"""
        # Draw landmarks
        if 'landmarks' in result and result['landmarks'] is not None:
            for point in result['landmarks']:
                cv2.circle(frame, tuple(point), 1, (0, 255, 0), -1)

        # Draw text info
        h, w = frame.shape[:2]

        # Status box
        status_color = (0, 255, 0)  # Green
        if result['class'] == 'Drowsy':
            status_color = (0, 165, 255)  # Orange
        elif result['class'] == 'Very Drowsy':
            status_color = (0, 0, 255)  # Red

        cv2.rectangle(frame, (10, 10), (w - 10, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (w - 10, 120), status_color, 2)

        # Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Status: {result['class']}", (20, 40),
                    font, 0.8, status_color, 2)
        cv2.putText(frame, f"Confidence: {result['confidence']:.2f}", (20, 70),
                    font, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"EAR: {result['ear']:.3f}  MAR: {result['mar']:.3f}",
                    (20, 100), font, 0.6, (255, 255, 255), 1)

        return frame

    def stop(self):
        """Stop the thread"""
        self.running = False
        self.wait()


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fatigue Detection System")
        self.setGeometry(100, 100, 1400, 900)

        # Load configuration
        with open('config/train_config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.detector = None
        self.voice_engine = VoiceAlertEngine()
        self.db_manager = DatabaseManager()
        self.session_id = None

        self.video_thread = None
        self.is_monitoring = False

        # Setup UI
        self.init_ui()

        # FPS counter
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.update_fps)
        self.frame_count = 0
        self.fps = 0

    def init_ui(self):
        """Initialize user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Left panel - Video display
        left_panel = self._create_video_panel()
        main_layout.addWidget(left_panel, 2)

        # Right panel - Controls and logs
        right_panel = self._create_control_panel()
        main_layout.addWidget(right_panel, 1)

    def _create_video_panel(self):
        """Create video display panel"""
        group = QGroupBox("Live Video Feed")
        layout = QVBoxLayout()

        # Video label
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)

        # Status bar
        status_layout = QHBoxLayout()

        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setFont(QFont("Arial", 12))
        status_layout.addWidget(self.fps_label)

        status_layout.addStretch()

        self.status_label = QLabel("Status: Ready")
        self.status_label.setFont(QFont("Arial", 12, QFont.Bold))
        status_layout.addWidget(self.status_label)

        layout.addLayout(status_layout)

        group.setLayout(layout)
        return group

    def _create_control_panel(self):
        """Create control panel"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Control buttons
        control_group = QGroupBox("Controls")
        control_layout = QVBoxLayout()

        self.start_btn = QPushButton("Start Monitoring")
        self.start_btn.clicked.connect(self.start_monitoring)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        control_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Monitoring")
        self.stop_btn.clicked.connect(self.stop_monitoring)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                padding: 10px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        control_layout.addWidget(self.stop_btn)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # Detection info
        info_group = QGroupBox("Detection Information")
        info_layout = QVBoxLayout()

        self.ear_label = QLabel("EAR: --")
        self.mar_label = QLabel("MAR: --")
        self.class_label = QLabel("Class: --")
        self.confidence_label = QLabel("Confidence: --")

        for label in [self.ear_label, self.mar_label,
                      self.class_label, self.confidence_label]:
            label.setFont(QFont("Arial", 11))
            info_layout.addWidget(label)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Event log
        log_group = QGroupBox("Event Log")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)

        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        layout.addStretch()

        widget.setLayout(layout)
        return widget

    def start_monitoring(self):
        """Start fatigue monitoring"""
        try:
            # Initialize detector if not already done
            if self.detector is None:
                model_path = 'models/checkpoints/best_model.pth'
                if not Path(model_path).exists():
                    self.log_message("Error: Model file not found. Please train the model first.")
                    return

                self.detector = FatigueDetector(model_path, self.config)
                self.log_message("Detector initialized successfully")

            # Start database session
            self.session_id = self.db_manager.start_session()

            # Start video thread
            self.video_thread = VideoThread(self.detector)
            self.video_thread.change_pixmap_signal.connect(self.update_image)
            self.video_thread.detection_result_signal.connect(self.handle_detection)
            self.video_thread.start()

            # Start FPS timer
            self.fps_timer.start(1000)

            # Update UI
            self.is_monitoring = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.status_label.setText("Status: Monitoring")
            self.status_label.setStyleSheet("color: green;")

            self.log_message("Monitoring started")

        except Exception as e:
            self.log_message(f"Error starting monitoring: {str(e)}")

    def stop_monitoring(self):
        """Stop fatigue monitoring"""
        if self.video_thread:
            self.video_thread.stop()

        # Stop FPS timer
        self.fps_timer.stop()

        # End database session
        if self.session_id:
            self.db_manager.end_session(self.session_id)

        # Update UI
        self.is_monitoring = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Status: Stopped")
        self.status_label.setStyleSheet("color: red;")

        self.log_message("Monitoring stopped")

    def update_image(self, frame):
        """Update video display"""
        self.frame_count += 1

        # Convert to Qt format
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Scale to fit label
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def handle_detection(self, result):
        """Handle detection results"""
        if result['status'] != 'success':
            return

        # Update info labels
        self.ear_label.setText(f"EAR: {result['ear']:.3f}")
        self.mar_label.setText(f"MAR: {result['mar']:.3f}")
        self.class_label.setText(f"Class: {result['class']}")
        self.confidence_label.setText(f"Confidence: {result['confidence']:.2f}")

        # Handle alerts
        if result['alert']:
            if result['class'] == 'Drowsy':
                self.voice_engine.alert_drowsy()
                self.log_message("⚠️ Drowsy state detected!")
            elif result['class'] == 'Very Drowsy':
                self.voice_engine.alert_very_drowsy()
                self.log_message("🚨 Severe fatigue detected!")

            # Log to database
            self.db_manager.log_event(
                result['class'],
                result['confidence'],
                result['ear'],
                result['mar']
            )

    def update_fps(self):
        """Update FPS counter"""
        self.fps = self.frame_count
        self.frame_count = 0
        self.fps_label.setText(f"FPS: {self.fps}")

    def log_message(self, message):
        """Add message to log"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def closeEvent(self, event):
        """Handle window close event"""
        if self.is_monitoring:
            self.stop_monitoring()

        if self.detector:
            self.detector.release()

        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
