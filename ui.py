from PyQt5.QtWidgets import (
    QApplication, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QTextBrowser, QLineEdit, QPushButton, QLabel, QCheckBox, QComboBox, QSizePolicy, QScrollArea, QGridLayout
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QFileSystemWatcher, QObject
from PyQt5.QtGui import QPalette, QColor, QPixmap
import sys
from html import escape
import re


#for gh_server_geometry
import subprocess # For running the server script
import os # For path manipulation
import atexit # To terminate the server on UI exit
import json # Added for json.dumps in GeometryWorkflowTab
#for gh_server_geometry
import requests

import socket

#added for automated LLM reasoning activity assignments
from llm_reasoning_test import generate_llm_assignments

def send_udp_command(command: str, port: int = 5004, host: str = "127.0.0.1"):
    print("Sending UDP command...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(command.encode("utf-8"), (host, port))

def send_udp_command2(message, port=6001):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(message.encode("utf-8"), ("127.0.0.1", port))

def send_udp_command3(message, port=6002):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(message.encode("utf-8"), ("127.0.0.1", port))

class CsvWatcher(QObject):
    def __init__(self, csv_path, callback):
        super().__init__()
        self.csv_path = csv_path
        self.callback = callback
        self.watcher = QFileSystemWatcher([csv_path])
        self.watcher.fileChanged.connect(self.on_file_changed)
        self.last_content = None
        self.read_and_send()  # Optionally send on startup

    def on_file_changed(self, path):
        # QFileSystemWatcher sometimes emits multiple times, so debounce by content
        self.read_and_send()

    def read_and_send(self):
        try:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content and content != self.last_content:
                self.last_content = content
                self.callback(content)
        except Exception as e:
            print(f"Error reading CSV: {e}")

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
            r = requests.post(self.endpoint, json=self.payload, timeout=self.timeout)
            if r.status_code == 204:
                self.finished.emit({"message": "No content from server.", "status_code": 204})
            else:
                try:
                    r.raise_for_status()
                    data = r.json()
                    self.finished.emit(data)
                except requests.exceptions.HTTPError as http_err:
                    # Try to get JSON error from response
                    try:
                        error_json = r.json()
                        self.error.emit(json.dumps(error_json))
                    except Exception:
                        # Fallback: emit the error as string
                        self.error.emit(str(http_err))
                except Exception as e:
                    self.error.emit(str(e))
        except Exception as e:
            self.error.emit(str(e))

class ChatTab(QWidget):
    def __init__(self, endpoint, extra_fields=None):
        super().__init__()
        self.endpoint = endpoint
        self.extra_fields = extra_fields or {}
        self.conversation_history = []
        self.geometry_shown = False  # Track geometry state

        layout = QVBoxLayout()

        # QTextBrowser
        self.chat_display = QTextBrowser()
        self.chat_display.setOpenExternalLinks(True)
        self.chat_display.setStyleSheet("""
            background-color: #111;
            color: #eee;
            border: none;
            font-family: 'Segoe UI', 'Arial', 'Helvetica Neue', sans-serif;
            font-size: 15px;
        """)
        layout.addWidget(self.chat_display)

        # CSS Styles for bubbles (only once!)
        self.chat_style = """
        <style>
        .chat-user {
            color: white;
            margin: 12px 12px 12px 12px;
            text-align: right;
            max-width: 95%;
            min-width: 40px;
            width: fit-content;
            font-size: 20px;
            display: inline-block;
            clear: both;
            float: right;
            word-break: break-word;
            padding: 6px 16px 0px 16px;   
            line-height: 1.5;             /* reduce line height */
            vertical-align: middle;
        }
        .chat-bot {
            color: white;
            padding: 10px 16px;
            margin: 12px 200px 12px 12px;
            text-align: left;
            font-size: 16px;
            display: inline-block;
            clear: both;
            float: left;
            word-break: break-word;
        }
        .chat-bot.error {
            color: #b00;
            border: 1px solid #b00;
            border-radius: 20px;
        }
        </style>
        """
        self.chat_history_html = []  # Store all chat bubbles as HTML

        # Set initial content with CSS (empty div to keep layout)
        self.chat_display.setHtml(self.chat_style + "<div></div>")


        # General tab only UI elements
        if "general_question" in self.endpoint:
            geom_layout = QHBoxLayout()

            # Show All Building Geometry button (far left)
            self.show_all_btn = QPushButton("Show All Building Geometry")
            # self.show_all_btn.clicked.connect(self.show_all_geometry)
            self.show_all_btn.clicked.connect(self.toggle_all_geometry)
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
            self.level_dropdown.addItems(["1", "2", "3", "4"])
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
        self.input_box.returnPressed.connect(self.send_message)  # <-- Add this line
        input_layout.addWidget(self.input_box)
        input_layout.addWidget(self.send_btn)
        layout.addLayout(input_layout)

        self.setLayout(layout)

    def update_chat_display(self):
        # Rebuild the full HTML with style and all bubbles
        full_html = self.chat_style + ''.join(self.chat_history_html)
        self.chat_display.setHtml(full_html)
        self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())

    def send_message(self):
        user_text = self.input_box.text()
        if not user_text:
            return
        escaped_user_text = escape(user_text)
        user_bubble = f'<div class="chat-user">{escaped_user_text}</div>'
        self.chat_history_html.append(user_bubble)
        self.update_chat_display()

        # UI-driven geometry intent extraction
        if self.parse_and_trigger_geometry(user_text):
            self.input_box.clear()
            return  # Do NOT send to LLM if geometry action was triggered

        # Only send to LLM for text answer (no geometry intent flag)
        payload = {
            "question": user_text,
            "conversation_history": self.conversation_history
        }
        payload.update(self.extra_fields)

        self.send_btn.setEnabled(False)
        self.worker = RequestWorker(self.endpoint, payload)
        self.worker.error.connect(self.handle_error)
        self.worker.finished.connect(self.handle_response)
        self.worker.start()

        self.input_box.clear() 

    def trigger_geometry_display(self, level, info_type):
        """
        Sets dropdowns based on parsed level and info_type, then sends geometry command.
        """
        # Set level dropdown
        if hasattr(self, "level_dropdown") and level and str(level).isdigit():
            idx = self.level_dropdown.findText(str(int(level)))
            if idx != -1:
                self.level_dropdown.setCurrentIndex(idx)
        # Set info dropdowns
        info_type = (info_type or "").capitalize()
        if hasattr(self, "space_info_dropdown") and info_type in [self.space_info_dropdown.itemText(i) for i in range(self.space_info_dropdown.count())]:
            idx = self.space_info_dropdown.findText(info_type)
            if idx != -1:
                self.space_info_dropdown.setCurrentIndex(idx)
        elif hasattr(self, "apt_info_dropdown") and info_type in [self.apt_info_dropdown.itemText(i) for i in range(self.apt_info_dropdown.count())]:
            idx = self.apt_info_dropdown.findText(info_type)
            if idx != -1:
                self.apt_info_dropdown.setCurrentIndex(idx)
        # Send the geometry command with the selected options
        self.send_geometry_command()

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
        # Get the selected level and convert to 0-based index
        level_option = int(self.level_dropdown.currentText())
        level_value = level_option - 1

        # Encode space info to numeric value
        space_info_map = {
            "Activity": 23,
            "Area": 4,
            "UTCI": 8,
            "Wind": 10,
            "Orientation": 2
        }
        space_info_option = self.space_info_dropdown.currentText()
        space_info_value = space_info_map.get(space_info_option, -1)

        # Encode apartment info to numeric value
        apt_info_map = {
            "Key": 0,
            "Residents": 1
        }
        apt_info_option = self.apt_info_dropdown.currentText()
        apt_info_value = apt_info_map.get(apt_info_option, -1)

        # Send encoded values as a string like "1|8|0"
        payload = f"{level_value}|{space_info_value}|{apt_info_value}"

        try:
            print(f"Sending encoded payload: {payload}")
            send_udp_command2(payload, port=6001)
            self.chat_display.append(
                f"<b>Encoded Geometry Command sent:</b> {payload}"
            )
        except Exception as e:
            self.chat_display.append(f"<b>Error:</b> {e}")

    def toggle_all_geometry(self):
        if not self.geometry_shown:
            self.show_all_geometry(force=True)
        else:
            self.hide_all_geometry(force=True)

    def show_all_geometry(self, force=False):
        try:
            send_udp_command("show_all")
            self.geometry_shown = True
            self.show_all_btn.setText("Hide All Building Geometry")
            if force or not self.chat_display.toPlainText().endswith("All geometry shown."):
                self.chat_display.append("<b>All geometry shown.</b>")
        except Exception as e:
            self.chat_display.append(f"<b>Error:</b> {e}")

    def hide_all_geometry(self, force=False):
        try:
            send_udp_command("hide_all")
            self.geometry_shown = False
            self.show_all_btn.setText("Show All Building Geometry")
            if force or not self.chat_display.toPlainText().endswith("All geometry hidden."):
                self.chat_display.append("<b>All geometry hidden.</b>")
        except Exception as e:
            self.chat_display.append(f"<b>Error:</b> {e}")


    def hide_specific_geometry(self):
        # Send a "null|null|null" string to port 6001 to hide all geometry
        try:
            payload = "null|null|null"
            send_udp_command2(payload, port=6001)
            self.chat_display.append("<b>Selected geometry hidden (no geometry previewed).</b>")
        except Exception as e:
            self.chat_display.append(f"<b>Error:</b> {e}")

    # ...existing code...

    def parse_and_trigger_geometry(self, user_text):
        """
        Detects geometry-related commands in user_text and triggers the appropriate geometry actions.
        Returns True if a geometry action was triggered, else False.
        """
        text = user_text.lower().strip()

        # Show all geometry
        if re.search(r"\b(show|display|reveal)\s+(all\s+)?(building|geometry|spaces?)\b", text):
            self.show_all_geometry(force=True)
            return True

        # Hide all geometry
        if re.search(r"\b(hide|remove)\s+all(\s+(geometry|spaces?|apartments?))?\b", text):
            self.hide_all_geometry(force=True)
            return True

        # Show specific level (e.g., "show level 1", "display level 2 apartments")
        match = re.search(r"\b(show|display|reveal)\s+level\s+(\d+)", text)
        if match:
            level = int(match.group(2))
            # Set dropdown if available
            if hasattr(self, "level_dropdown"):
                idx = self.level_dropdown.findText(str(level))
                if idx != -1:
                    self.level_dropdown.setCurrentIndex(idx)
            # Optionally parse for apartment/space info
            if "apartment" in text or "resident" in text:
                if hasattr(self, "apt_info_dropdown"):
                    idx = self.apt_info_dropdown.findText("Residents")
                    if idx != -1:
                        self.apt_info_dropdown.setCurrentIndex(idx)
            self.send_geometry_command()
            return True

        # Show specific apartment (e.g., "show apartment key", "show residents")
        if re.search(r"\b(show|display|reveal)\s+(apartment|resident|key|residents)\b", text):
            if hasattr(self, "apt_info_dropdown"):
                idx = self.apt_info_dropdown.findText("Residents")
                if idx != -1:
                    self.apt_info_dropdown.setCurrentIndex(idx)
            self.send_geometry_command()
            return True

        # Hide selected geometry
        if re.search(r"\b(hide|remove)\s+(current|this)\s+(geometry|space|apartment)\b", text):
            self.hide_specific_geometry()
            return True

        # Patterns for level and info type (improved)
        level_pattern = r"(?:level\s*|lvl\s*|floor\s*)?(\d+)"
        info_pattern = r"(area|activity|wind|orientation|comfort|temperature|occupancy|residents|spaces|apartments|building|all)"

        # Try to match both orders: "show level 3 wind" or "show wind level 3"
        match = re.search(
            rf"(?:show|display|what are|what is|give me|can you show|visualize|see|plot|present)[^\n]*?(?:{info_pattern})?[^\n]*?(?:in|of|for)?[^\n]*?{level_pattern}(?:[^\n]*?{info_pattern})?",
            user_text, re.IGNORECASE
        )

        info = None
        level = None
        if match:
            # Find all occurrences of info_pattern and level_pattern
            info_matches = re.findall(info_pattern, user_text, re.IGNORECASE)
            level_matches = re.findall(level_pattern, user_text, re.IGNORECASE)
            # Pick the first found, or default
            info = info_matches[0] if info_matches else "all"
            level = level_matches[0] if level_matches else None
            if level:
                self.trigger_geometry_display(level=level, info_type=info)
                return True

        # Fallback: match "show all building" or "show all"
        if re.search(r"show all (building|apartments|spaces)?", user_text, re.IGNORECASE):
            self.trigger_geometry_display(level="all", info_type="all")
            return True

        # Add more patterns as needed...

        return False

    def handle_error(self, error_msg):
        """
        Handles errors from network/LLM requests and displays them in the chat.
        """
        error_html = f'<div class="chat-bot error"><b>Error:</b> {escape(error_msg)}</div><br>'
        self.chat_history_html.append(error_html)
        self.update_chat_display()
        self.send_btn.setEnabled(True)
        self.input_box.setEnabled(True)

    def handle_response(self, data):
        # Get the LLM response text
        response_text = data.get("response", "No response from server.")
        escaped_response = escape(response_text)
        bot_bubble = f'<div class="chat-bot">{escaped_response}</div>'
        self.chat_history_html.append(bot_bubble)
        self.update_chat_display()
        self.send_btn.setEnabled(True)
        self.input_box.setEnabled(True)

class WelcomeTab(QWidget):
    def __init__(self, info_text, tab_widget=None):
        super().__init__()
        layout = QVBoxLayout()
        label = QLabel(info_text)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        label.setStyleSheet("font-size: 16px; padding: 20px;")
        layout.addWidget(label)

        # Add "Fill Survey" button
        self.fill_survey_btn = QPushButton("Fill Survey")
        self.fill_survey_btn.clicked.connect(self.go_to_survey_tab)
        layout.addWidget(self.fill_survey_btn)

        self.setLayout(layout)
        self.tab_widget = tab_widget  # Store reference for navigation

    def go_to_survey_tab(self):
        if self.tab_widget:
            # Survey tab is at index 1 (after Welcome)
            self.tab_widget.setCurrentIndex(1)

class SurveyTab(QWidget):
    def __init__(self, tab_widget=None):
        super().__init__()
        layout = QVBoxLayout()

        # Resident Key
        self.resident_key_input = QLineEdit()
        self.resident_key_input.setPlaceholderText("Enter Resident Key")
        layout.addWidget(QLabel("Resident Key:"))
        layout.addWidget(self.resident_key_input)

        # Resident Persona
        self.resident_persona_input = QLineEdit()
        self.resident_persona_input.setPlaceholderText("Enter Resident Persona")
        layout.addWidget(QLabel("Resident Persona:"))
        layout.addWidget(self.resident_persona_input)

        # Resident Population
        self.resident_population_input = QLineEdit()
        self.resident_population_input.setPlaceholderText("Enter Resident Population")
        layout.addWidget(QLabel("Resident Population:"))
        layout.addWidget(self.resident_population_input)

        # Level
        self.level_input = QLineEdit()
        self.level_input.setPlaceholderText("Enter Level")
        layout.addWidget(QLabel("Level:"))
        layout.addWidget(self.level_input)

        # Age
        self.age_input = QLineEdit()
        self.age_input.setPlaceholderText("Enter Age")
        layout.addWidget(QLabel("Age:"))
        layout.addWidget(self.age_input)

        # Tenant/Owner
        self.tenant_owner_input = QLineEdit()
        self.tenant_owner_input.setPlaceholderText("Enter Tenant/Owner")
        layout.addWidget(QLabel("Tenant/Owner:"))
        layout.addWidget(self.tenant_owner_input)

        # Add "Submit Survey" button (fake, does nothing)
        self.submit_survey_btn = QPushButton("Submit Survey")
        # No action connected
        layout.addWidget(self.submit_survey_btn)

        # Navigation button
        self.ask_general_btn = QPushButton("Ask general questions")
        self.ask_general_btn.clicked.connect(self.go_to_general_tab)
        layout.addWidget(self.ask_general_btn)

        layout.addStretch(1)
        self.setLayout(layout)
        self.tab_widget = tab_widget  # Store reference for navigation

    def go_to_general_tab(self):
        if self.tab_widget:
            # Assuming General tab is at index 2 (after Welcome and Survey)
            self.tab_widget.setCurrentIndex(2)

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

        # Add Welcome tab first, pass tabs for navigation
        welcome_text = (
            "Welcome to Copilot for Residents!\n\n"
            "This tool helps you explore, interact with, and ask questions about your building, its spaces, and your neighbors. "
            "Learn about coliving concepts, understand your climate conditions, negotiate activity changes in shared spaces, or even swap apartments. "
            "Follow the steps below to build a happier, more connected community:\n\n"
            "1. Fill out the survey to set up your profile.\n"
            "2. Ask general questions, explore the building, and uncover hidden data about spaces and residents.\n"
            "3. Get to know your closest neighbors and shared spaces — negotiate, book, or swap!\n"
            "4. Suggest or make changes to the geometry of your building.\n"
            "5. View rendered images of your building to see it from new perspectives.\n\n"
            "Enjoy exploring and shaping your community!"
        )
        welcome_tab = WelcomeTab(welcome_text, tab_widget=tabs)
        tabs.addTab(welcome_tab, "Welcome")

        # Add Survey tab second, pass tabs for navigation
        survey_tab = SurveyTab(tab_widget=tabs)
        tabs.addTab(survey_tab, "Survey")

        # Existing tabs
        tabs.addTab(ChatTab("http://localhost:5000/general_question"), "General")

        # Add Q&A + Negotiate Tab from ui_pyqt_spaceqna.py
        from ui_pyqt_spaceqna import SpaceQnAUI
        self.qna_neg_tab = SpaceQnAUI()
        tabs.addTab(self.qna_neg_tab, "Q&A + Negotiate")

        # tabs.addTab(ChatTab("http://localhost:5002/geometry_suggestion"), "Geometry/Negotiation")
        tabs.addTab(GeometryWorkflowTab("http://localhost:5004/initiate_gh_workflow"), "Geometry Workflow")

        images_tab = ImagesTab(images_folder="images")
        tabs.addTab(images_tab, "Images")
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
        self.send_btn.setEnabled(True)
        if "error" in data:
            self.response_display.append(f"<b>Server Error:</b> {data.get('error')}")
            if "details" in data:
                self.response_display.append(f"<i>Details: {data.get('details')}</i>")
            return

        user_question = data.get("user_question_for_suggestion", "").lower()
        suggestions = data.get("suggestions", [])
        summary = data.get("summary_reasoning", "No summary provided.")

        display_text = f"<b>LLM Response:</b>\n"

        if suggestions and isinstance(suggestions, list) and len(suggestions) > 0:
            suggestion = suggestions[0] # Assuming one suggestion
            display_text += f"<b>Suggestion:</b> {suggestion.get('variation_name', 'N/A')}\n"
            display_text += f"<i>Description:</i> {suggestion.get('description', 'N/A')}\n"
            # Send suggestion description to Grasshopper via UDP
            send_udp_command(json.dumps(data))

        display_text += f"\n<b>Summary Reasoning:</b> {summary}"
        self.response_display.append(display_text)
            # For debugging, you can still print the full JSON to the console or a log
            # print(f"Full server response: {json.dumps(data, indent=2)}")

    def handle_submit_error(self, error_msg):
        try:
            import re, json
            # Look for JSON in the error message
            json_match = re.search(r'(\{.*\})', error_msg)
            if json_match:
                try:
                    error_json = json.loads(json_match.group(1))
                    # Show the detailed error if present
                    if "error" in error_json:
                        self.response_display.append(f"<b>Server Error:</b> {error_json['error']}")
                        if "details" in error_json:
                            self.response_display.append(f"<i>Details: {error_json['details']}</i>")
                        return
                except Exception:
                    pass
            # Fallback: show the raw error message
            self.response_display.append(f"<b>Error Submitting Job:</b> {error_msg}")
        finally:
            self.send_btn.setEnabled(True)


class ImagesTab(QWidget):
    def __init__(self, images_folder="images"):
        super().__init__()
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        grid = QGridLayout()
        content.setLayout(grid)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        self.setLayout(layout)

        # Load images from the folder
        import os
        from glob import glob

        # Accept common image extensions
        image_files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"):
            image_files.extend(glob(os.path.join(images_folder, ext)))
        image_files.sort()  # Sort for consistent order

        # Display images in a grid (3 per row)
        for idx, img_path in enumerate(image_files):
            label = QLabel()
            pixmap = QPixmap(img_path)
            if not pixmap.isNull():
                pixmap = pixmap.scaledToWidth(300, Qt.SmoothTransformation)
                label.setPixmap(pixmap)
                label.setAlignment(Qt.AlignCenter)
                grid.addWidget(label, idx // 3, idx % 3)
            else:
                label.setText(f"Failed to load {os.path.basename(img_path)}")
                grid.addWidget(label, idx // 3, idx % 3)

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
    csv_path = os.path.join(os.path.dirname(__file__), "llm_reasoning", "llm_activity_assignments.csv")
    watcher = CsvWatcher(csv_path, send_udp_command3)

    window.show()
    sys.exit(app.exec_())