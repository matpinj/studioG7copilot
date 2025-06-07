from PyQt5.QtWidgets import (
    QApplication, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QCheckBox, QComboBox
)
from PyQt5.QtCore import QThread, pyqtSignal
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

        # Geometry toggle example
        self.toggle_geom = QCheckBox("Show Geometry")
        layout.addWidget(self.toggle_geom)

        # General tab only UI elements
        if "general_question" in self.endpoint:
            self.geometry_dropdown = QComboBox()
            self.geometry_dropdown.addItems([
                "None",
                "Show Building",
                "Show Outdoor Spaces",
                "Show Activity Spaces"
            ])
            layout.addWidget(self.geometry_dropdown)

            # Add a button to trigger geometry update
            self.show_geom_btn = QPushButton("Show Geometry")
            self.show_geom_btn.clicked.connect(self.send_geometry_command)
            layout.addWidget(self.show_geom_btn)

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
        geom_option = self.geometry_dropdown.currentText()
        # Map dropdown text to a command integer
        command_map = {
            "None": 0,
            "Show Building": 1,
            "Show Outdoor Spaces": 2,
            "Show Activity Spaces": 3
        }
        command = command_map.get(geom_option, 0)
        try:
            r = requests.post("http://localhost:5000/set_geometry", json={"geometry_command": command})
            if r.status_code == 200:
                self.chat_display.append(f"<b>Geometry Command sent:</b> {geom_option}")
            else:
                self.chat_display.append(f"<b>Error sending geometry command</b>")
        except Exception as e:
            self.chat_display.append(f"<b>Error:</b> {e}")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GH LLM Chat UI")
        self.resize(1000, 800)  # Set the window size here
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
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())