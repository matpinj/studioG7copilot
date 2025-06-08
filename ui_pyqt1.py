from PyQt5.QtWidgets import (
    QApplication, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QCheckBox, QComboBox, QSizePolicy
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPalette, QColor
import sys
import requests

class RequestWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, endpoint, payload):
        super().__init__()
        self.endpoint = endpoint
        self.payload = payload

    def run(self):
        try:
            r = requests.post(self.endpoint, json=self.payload, timeout=10)
            data = r.json()
            self.finished.emit(data)
        except Exception as e:
            self.error.emit(str(e))

class ChatTab(QWidget):
    def __init__(self, endpoint, extra_fields=None):
        super().__init__()
        self.endpoint = endpoint
        self.extra_fields = extra_fields or {}
        self.conversation_history = []

        layout = QVBoxLayout()
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display)

        # General tab only UI elements
        if "general_question" in self.endpoint:
            geom_layout = QHBoxLayout()

            # Show All Building Geometry button (far left)
            self.show_all_btn = QPushButton("Show All Building Geometry")
            self.show_all_btn.clicked.connect(self.show_all_geometry)
            geom_layout.addWidget(self.show_all_btn)

            # Add stretch to push dropdowns and "Show" button to the right
            geom_layout.addStretch(1)

            # Level label + dropdown
            level_box = QHBoxLayout()
            level_box.setSpacing(4)
            level_label = QLabel("Level:")
            level_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
            level_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.level_dropdown = QComboBox()
            self.level_dropdown.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self.level_dropdown.addItems(["1", "2", "3"])
            level_box.addWidget(level_label)
            level_box.addWidget(self.level_dropdown)
            geom_layout.addLayout(level_box)

            # Space Info label + dropdown
            space_info_box = QHBoxLayout()
            space_info_box.setSpacing(4)
            space_info_label = QLabel("Space Info:")
            space_info_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
            space_info_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.space_info_dropdown = QComboBox()
            self.space_info_dropdown.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self.space_info_dropdown.addItems([
                "Activity",
                "Area",
                "UTCI",
                "Wind",
                "Orientation"
            ])
            space_info_box.addWidget(space_info_label)
            space_info_box.addWidget(self.space_info_dropdown)
            geom_layout.addLayout(space_info_box)

            # Apartment Info label + dropdown
            apt_info_box = QHBoxLayout()
            apt_info_box.setSpacing(4)
            apt_info_label = QLabel("Apartment Info:")
            apt_info_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
            apt_info_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.apt_info_dropdown = QComboBox()
            self.apt_info_dropdown.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self.apt_info_dropdown.addItems([
                "Key",
                "Residents"
            ])
            apt_info_box.addWidget(apt_info_label)
            apt_info_box.addWidget(self.apt_info_dropdown)
            geom_layout.addLayout(apt_info_box)

            # "Show" button (to the right of dropdowns)
            self.show_geom_btn = QPushButton("Show")
            self.show_geom_btn.clicked.connect(self.send_geometry_command)
            geom_layout.addWidget(self.show_geom_btn)

            layout.addLayout(geom_layout)

        input_layout = QHBoxLayout()
        self.input_box = QLineEdit()
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(self.input_box)
        input_layout.addWidget(self.send_btn)
        layout.addLayout(input_layout)

        self.setLayout(layout)

    def send_message(self):
        user_text = self.input_box.text()
        if not user_text:
            return
        self.chat_display.append(f"<b>You:</b> {user_text}")
        payload = {
            "question": user_text,
            "conversation_history": self.conversation_history
        }
        payload.update(self.extra_fields)

        self.send_btn.setEnabled(False)
        self.worker = RequestWorker(self.endpoint, payload)
        self.worker.finished.connect(self.handle_response)
        self.worker.error.connect(self.handle_error)
        self.worker.start()

    def handle_response(self, data):
        answer = data.get("response", "No response")
        self.conversation_history = data.get("conversation_history", [])
        self.chat_display.append(f"<b>Bot:</b> {answer}")
        self.send_btn.setEnabled(True)
        self.input_box.clear()

    def handle_error(self, error_msg):
        self.chat_display.append(f"<b>Error:</b> {error_msg}")
        self.send_btn.setEnabled(True)
        self.input_box.clear()

    def send_geometry_command(self):
        # Get the selected level as int (1, 2, 3)
        level_option = int(self.level_dropdown.currentText())
        # Convert to 0-based index for Grasshopper (0, 1, 2)
        level_value = level_option - 1

        # Map space info
        space_info_map = {
            "Activity": 20,
            "Area": 4,
            "UTCI": 8,
            "Wind": 7,
            "Orientation": 2
        }
        space_info_option = self.space_info_dropdown.currentText()
        space_info_value = space_info_map.get(space_info_option, -1)
        # Map apartment info
        apt_info_map = {
            "Key": 0,
            "Residents": 1
        }
        apt_info_option = self.apt_info_dropdown.currentText()
        apt_info_value = apt_info_map.get(apt_info_option, -1)

        payload = {
            "level": level_value,  # Now 0, 1, or 2
            "space_info": space_info_value,
            "apt_info": apt_info_value
        }
        try:
            r = requests.post("http://localhost:5000/set_geometry", json=payload)
            if r.status_code == 200:
                self.chat_display.append(
                    f"<b>Geometry Command sent:</b> Level: {level_option} (GH: {level_value}), Space Info: {space_info_option} ({space_info_value}), Apartment Info: {apt_info_option} ({apt_info_value})"
                )
            else:
                self.chat_display.append(f"<b>Error sending geometry command</b>")
        except Exception as e:
            self.chat_display.append(f"<b>Error:</b> {e}")

    def show_all_geometry(self):
        try:
            r = requests.post("http://localhost:5000/set_geometry", json={"geometry_command": "toggle_all"})
            if r.status_code == 200:
                state = r.json().get("visible", False)
                if state:
                    self.show_all_btn.setText("Hide All Building Geometry")
                    msg = "Show All Building Geometry"
                else:
                    self.show_all_btn.setText("Show All Building Geometry")
                    msg = "Hide All Building Geometry"
                self.chat_display.append(f"<b>{msg}</b>")
            else:
                self.chat_display.append("<b>Error toggling Show All command</b>")
        except Exception as e:
            self.chat_display.append(f"<b>Error:</b> {e}")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Activity Copilot for Residentials")
        self.resize(1200, 800)  # Set the window size here
        layout = QVBoxLayout()
        tabs = QTabWidget()

        # Adjust endpoints as needed
        tabs.addTab(ChatTab("http://localhost:5000/general_question"), "General")
        tabs.addTab(ChatTab("http://localhost:5001/space_question"), "Space Q&A")
        tabs.addTab(ChatTab("http://localhost:5002/geometry_suggestion"), "Geometry/Negotiation")

        layout.addWidget(tabs)
        self.setLayout(layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Modern built-in style

    # Black and white (grayscale) palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, QColor(240, 240, 240))
    palette.setColor(QPalette.Base, QColor(20, 20, 20))
    palette.setColor(QPalette.Text, QColor(240, 240, 240))
    palette.setColor(QPalette.Button, QColor(40, 40, 40))
    palette.setColor(QPalette.ButtonText, QColor(240, 240, 240))
    palette.setColor(QPalette.Highlight, QColor(180, 180, 180))
    palette.setColor(QPalette.HighlightedText, QColor(30, 30, 30))
    app.setPalette(palette)

    window = MainWindow()

    # Black and white style sheet
    window.setStyleSheet("""
        QWidget {
            font-size: 14px;
            font-family: 'Segoe UI', 'Arial', 'Helvetica Neue', 'sans-serif';
        }
        QTabWidget::pane {
            border: 1px solid #888;
            border-radius: 10px;
            margin: 4px;
        }
        QTabBar::tab {
            background: #222;
            color: #eee;
            border-radius: 8px 8px 0 0;
            padding: 6px 12px;           /* Slightly smaller padding */
            min-width: 120px;            /* Minimum width for tabs */
            margin-right: 2px;
            font-weight: 500;
            letter-spacing: 0.5px;
        }
        QTabBar::tab:selected {
            background: #fff;
            color: #111;
        }
        QTextEdit, QLineEdit, QComboBox {
            border-radius: 6px;
            padding: 6px;
            background-color: #111;
            color: #eee;
            border: 1px solid #888;
            font-family: 'Segoe UI', 'Arial', 'Helvetica Neue', 'sans-serif';
            font-size: 15px;
        }
        QPushButton {
            border-radius: 8px;
            padding: 8px 18px;
            background-color: #fff;
            color: #222;
            font-weight: bold;
            font-family: 'Segoe UI', 'Arial', 'Helvetica Neue', 'sans-serif';
            font-size: 15px;
            border: 2px solid #bbb;
        }
        QPushButton:hover {
            background-color: #e0e0e0;
            color: #111;
            border: 2px solid #888;
        }
        QCheckBox {
            spacing: 8px;
            color: #eee;
            font-family: 'Segoe UI', 'Arial', 'Helvetica Neue', 'sans-serif';
            font-size: 15px;
        }
    """)

    window.show()
    sys.exit(app.exec_())