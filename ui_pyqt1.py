from PyQt5.QtWidgets import (
    QApplication, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QCheckBox, QComboBox, QSizePolicy
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPalette, QColor
import sys

#for gh_server_geometry
import subprocess # For running the server script
import os # For path manipulation
import atexit # To terminate the server on UI exit
import json # Added for json.dumps in GeometryWorkflowTab
#for gh_server_geometry
import requests


class RequestWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, endpoint, payload, timeout=None): # Default to no timeout
        super().__init__()
        self.endpoint = endpoint
        self.payload = payload
        self.timeout = timeout

    def run(self):
        try:
            r = requests.post(self.endpoint, json=self.payload, timeout=self.timeout) # If self.timeout is None, it waits indefinitely
            if r.status_code == 204: # No Content
                # Emit a specific dictionary or handle as an error/empty response
                self.finished.emit({"message": "No content from server.", "status_code": 204})
            else:
                r.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
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

            # Hide Selected Geometry button
            self.hide_specific_btn = QPushButton("Hide")
            self.hide_specific_btn.clicked.connect(self.hide_specific_geometry)
            geom_layout.addWidget(self.hide_specific_btn)

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


    #for gh_server_script
    #region
    def _get_server_script_path(self):
        # Assuming ui_pyqt1.py is in the project root,
        # and gh_server_geometry.py is in geometry_mod/
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # If ui_pyqt1.py is in the root:
        server_script_path = os.path.join(current_dir, "geometry_mod", "gh_server_geometry.py")
        # If ui_pyqt1.py is also in a subdirectory, adjust accordingly.
        # For example, if ui_pyqt1.py is in 'ui_files' and server in 'server_files':
        # server_script_path = os.path.join(os.path.dirname(current_dir), "server_files", "gh_server_geometry.py")
        return server_script_path

    def start_flask_server(self):
        global flask_server_process
        server_script = self._get_server_script_path()
        if os.path.exists(server_script):
            try:
                # sys.executable is the path to the current Python interpreter
                flask_server_process = subprocess.Popen([sys.executable, server_script])
                print(f"Flask server '{server_script}' started with PID: {flask_server_process.pid}")
            except Exception as e:
                print(f"Failed to start Flask server: {e}")
        else:
            print(f"Error: Server script not found at {server_script}")
    #endregion
    #for gh_server_script


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

    def hide_all_geometry(self):
        try:
            r = requests.post("http://localhost:5000/set_geometry", json={"geometry_command": "hide_all"})
            if r.status_code == 200:
                self.chat_display.append("<b>All geometry hidden.</b>")
            else:
                self.chat_display.append("<b>Error hiding all geometry</b>")
        except Exception as e:
            self.chat_display.append(f"<b>Error:</b> {e}")

    def hide_specific_geometry(self):
        try:
            r = requests.post("http://localhost:5000/set_geometry", json={"geometry_command": "hide_specific"})
            if r.status_code == 200:
                self.chat_display.append("<b>Selected geometry hidden.</b>")
            else:
                self.chat_display.append("<b>Error hiding selected geometry</b>")
        except Exception as e:
            self.chat_display.append(f"<b>Error:</b> {e}")

class WelcomeTab(QWidget):
    def __init__(self, info_text):
        super().__init__()
        layout = QVBoxLayout()
        label = QLabel(info_text)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        label.setStyleSheet("font-size: 16px; padding: 20px;")
        layout.addWidget(label)
        self.setLayout(layout)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Activity Copilot for Residentials")
        self.resize(1200, 800)  # Set the window size here

        #for gh_server_geometry
        self.start_flask_server_if_needed() # Start the server
        #for gh_server_geometry

        layout = QVBoxLayout()
        tabs = QTabWidget()

        # Add Welcome tab first
        welcome_text = (
            "Welcome to Copilot for Residents!\n\n"
            "This tool helps you explore, interact with, and ask questions about your building and its spaces.\n\n"
            "• You can ask questions about general info, coliving, climate, and thermal comfort in the chat tabs.\n"
            "• Use the General tab to show/hide all geometry or select specific levels and info.\n"
            "• Use Space Q&A for space-related questions.\n"
            "• Use Geometry/Negotiation for geometry suggestions.\n\n"
            "Instructions:\n"
            "1. Ask your questions in the chat box to get information about the building, coliving, or comfort.\n"
            "2. Select options from the dropdowns and click 'Show' to display specific geometry.\n"
            "3. Use 'Show All Building Geometry' to toggle all geometry on/off.\n"
            "4. Use 'Hide' to hide selected geometry.\n"
            "Enjoy exploring and learning about your building!"
        )
        tabs.addTab(WelcomeTab(welcome_text), "Welcome")

        # Existing tabs
        tabs.addTab(ChatTab("http://localhost:5000/general_question"), "General")
        tabs.addTab(ChatTab("http://localhost:5001/space_question"), "Space Q&A")
        tabs.addTab(ChatTab("http://localhost:5002/geometry_suggestion"), "Geometry/Negotiation")
        tabs.addTab(GeometryWorkflowTab("http://localhost:5004/initiate_gh_workflow"), "Geometry Workflow")

        layout.addWidget(tabs)
        self.setLayout(layout)

    #for gh_server_geometry
    def start_flask_server_if_needed(self):
        global flask_server_process
        server_script = _get_server_script_path_main() # Call as global function
        if os.path.exists(server_script):
            try:
                # Check if server is already running (simple check, not foolproof)
                # A more robust check would involve trying to connect to the port
                flask_server_process = subprocess.Popen([sys.executable, server_script])
                print(f"Flask server '{server_script}' started with PID: {flask_server_process.pid}")
                atexit.register(stop_flask_server) # Register cleanup function
            except Exception as e:
                print(f"Failed to start Flask server: {e}")
        else:
            print(f"Error: Server script not found at {server_script}")
    #for gh_server_geometry

# Geometry workflow tab

class GeometryWorkflowTab(QWidget):
    def __init__(self, endpoint):
        super().__init__()
        self.endpoint = endpoint

        layout = QVBoxLayout()

        # Input fields
        self.space_id_input = QLineEdit()
        self.space_id_input.setPlaceholderText("Space ID (e.g., O2)")
        layout.addWidget(QLabel("Space ID:"))
        layout.addWidget(self.space_id_input)

        self.resident_key_input = QLineEdit()
        self.resident_key_input.setPlaceholderText("Resident Key (e.g., H23)")
        layout.addWidget(QLabel("Resident Key:"))
        layout.addWidget(self.resident_key_input)

        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText("Your question or request")
        layout.addWidget(QLabel("Question/Request:"))
        layout.addWidget(self.question_input)

        self.desired_activity_input = QComboBox()
        layout.addWidget(QLabel("Desired Activity:"))
        activities = [
            "Sitting",
            "Offline Retreat",
            "Sunbath",
            "Healing Garden",
            "Playground",
            "Sports",
            "Outdoor Cinema/Event Space",
            "Community Pool/BBQ",
            "Flexible Space",
            "Creative Corridor",
            "Outdoor Meeting Room",
            "Green Corridor",
            "Biodiversity balcony",
            "Urban Agriculture Garden",
            "Viewpoint",
            "Storage & Technical Space"
        ]
        self.desired_activity_input.addItems(activities)
        self.desired_activity_input.setPlaceholderText("Select Desired Activity") # Optional for QComboBox
        self.desired_activity_input.setCurrentIndex(-1) # Start with no selection
        layout.addWidget(self.desired_activity_input)

        self.send_btn = QPushButton("Submit Job to GH")
        self.send_btn.clicked.connect(self.send_request)
        layout.addWidget(self.send_btn) # Add the button directly

        self.response_display = QTextEdit()
        self.response_display.setReadOnly(True)
        layout.addWidget(self.response_display)

        self.setLayout(layout)

    def send_request(self):
        payload = {
            "space_id": self.space_id_input.text(),
            "resident_key": self.resident_key_input.text(),
            "question": self.question_input.text(),
            "desired_activity": self.desired_activity_input.currentText()
        }

        if not payload["space_id"] or not payload["resident_key"]:
            self.response_display.append("<b>Error:</b> Space ID and Resident Key are required.")
            return

        self.response_display.append(f"<b>Submitting Job to GH:</b> {json.dumps(payload)}")
        self.send_btn.setEnabled(False)
        
        # Use a QThread for network requests to keep UI responsive
        self.worker = RequestWorker(self.endpoint, payload) # Assuming RequestWorker is defined as in your script
        self.worker.finished.connect(self.handle_submit_response)
        self.worker.error.connect(self.handle_submit_error)
        self.worker.start()
    def handle_submit_response(self, data):
        # This now handles the direct LLM response or status from the server
        self.send_btn.setEnabled(True)

        if "error" in data:
            self.response_display.append(f"<b>Server Error:</b> {data.get('error')}")
            if "details" in data:
                self.response_display.append(f"<i>Details: {data.get('details')}</i>")
            return

        # Attempt to display a more focused answer
        user_question = data.get("user_question_for_suggestion", "").lower()
        suggestions = data.get("suggestions", [])
        summary = data.get("summary_reasoning", "No summary provided.")
        
        display_text = f"<b>LLM Response:</b>\n"

        if suggestions and isinstance(suggestions, list) and len(suggestions) > 0:
            suggestion = suggestions[0] # Assuming one suggestion
            display_text += f"<b>Suggestion:</b> {suggestion.get('variation_name', 'N/A')}\n"
            display_text += f"<i>Description:</i> {suggestion.get('description', 'N/A')}\n"

            if "other householders" in user_question or "who else" in user_question or "other beneficiaries" in user_question:
                beneficiaries = suggestion.get("other_beneficiaries")
                if beneficiaries:
                    display_text += f"<b>Other Beneficiaries:</b> {', '.join(beneficiaries)}\n"
        
        display_text += f"\n<b>Summary Reasoning:</b> {summary}"
        self.response_display.append(display_text)
        # For debugging, you can still print the full JSON to the console or a log
        # print(f"Full server response: {json.dumps(data, indent=2)}")

    def handle_submit_error(self, error_msg):
        self.response_display.append(f"<b>Error Submitting Job:</b> {error_msg}")
        self.send_btn.setEnabled(True)

    def handle_submit_error(self, error_msg):
        self.response_display.append(f"<b>Error Submitting Job:</b> {error_msg}")
        self.send_btn.setEnabled(True)

# Global variable to hold the server process
flask_server_process = None

#for gh_server_geometry
#region
def _get_server_script_path_main():
    # Path relative to ui_pyqt1.py assuming it's in the project root
    # and gh_server_geometry.py is in geometry_mod/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "geometry_mod", "gh_server_geometry.py")

def stop_flask_server():
    global flask_server_process
    if flask_server_process:
        print(f"Stopping Flask server with PID: {flask_server_process.pid}...")
        flask_server_process.terminate() # Send SIGTERM
        flask_server_process.wait(timeout=60) # Wait for a bit
        if flask_server_process.poll() is None: # If still running
            print("Server did not terminate gracefully, killing...")
            flask_server_process.kill() # Force kill
        print("Flask server stopped.")
#endregion
#for gh_server_geometry
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
        QTextEdit, QLineEdit {
            border-radius: 6px;
            padding: 6px;
            background-color: #111;
            color: #eee;
            border: 1px solid #888;
            font-family: 'Segoe UI', 'Arial', 'Helvetica Neue', 'sans-serif';
            font-size: 15px;
        }

        QComboBox {
            border-radius: 6px;
            padding: 2px 24px 2px 8px; /* top right bottom left, more space for arrow */
            background-color: #111;
            color: #eee;
            border: 1px solid #888;
            font-family: 'Segoe UI', 'Arial', 'Helvetica Neue', 'sans-serif';
            font-size: 15px;
            min-width: 0px;
            max-width: 140px; /* optional: limit width */
        }

        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 22px;
            border-left: 1px solid #888;
            border-top-right-radius: 6px;
            border-bottom-right-radius: 6px;
            background: #222;
        }

        QComboBox QAbstractItemView {
            border-radius: 6px;
            background: #222;
            color: #eee;
            selection-background-color: #444;
            selection-color: #fff;
        }

        QPushButton {
            border-radius: 8px;
            padding: 8px 15px;
            background-color: #fff;
            color: #222;
            font-weight: bold;
            font-family: 'Segoe UI', 'Arial', 'Helvetica Neue', 'sans-serif';
            font-size: 15px;
        }
        QPushButton:hover {
            background-color: #e0e0e0;
            padding: 8px 15px;
            color: #111;
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