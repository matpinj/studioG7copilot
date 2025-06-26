from PyQt5.QtWidgets import (
    QApplication, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QTextBrowser, QLineEdit, QPushButton, QLabel, QCheckBox, QComboBox, QSizePolicy, QScrollArea, QGridLayout, QDesktopWidget, QDialog
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QFileSystemWatcher, QObject
from PyQt5.QtGui import QPalette, QColor, QPixmap, QFont
import sys
from html import escape
import re
from PyQt5.QtWidgets import QListView


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

def send_udp_command(command: str, port: int = 6000, host: str = "127.0.0.1"):
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
        layout.setContentsMargins(8, 8, 8, 8)  # Reduced margins
        layout.setSpacing(8)                   # Reduced spacing

        # QTextBrowser with enhanced styling
        self.chat_display = QTextBrowser()
        self.chat_display.setOpenExternalLinks(True)
        self.chat_display.setStyleSheet("""
            QTextBrowser {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0f0f23, stop:1 #1a1a2e);
                color: #e8e8f0;
                border: none;
                border-radius: 8px;
                font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
                font-size: 13px;
                padding: 8px;
                selection-background-color: rgba(99, 102, 241, 0.3);
            }
            QScrollBar:vertical {
                background: rgba(255, 255, 255, 0.05);
                width: 8px;
                border-radius: 4px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 0.2);
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(255, 255, 255, 0.3);
            }
        """)
        layout.addWidget(self.chat_display)

        # Enhanced CSS Styles for chat bubbles
        self.chat_style = """
        <style>
        body { 
            margin: 0; 
            padding: 0; 
            font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
            background: transparent;
        }
        .chat-user {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin: 16px 0px 16px 80px;
            padding: 16px 20px;
            border-radius: 20px 20px 8px 20px;
            text-align: right;
            font-size: 15px;
            font-weight: 1400;
            line-height: 1.5;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
            word-wrap: break-word;
            max-width: calc(100% - 100px);
            float: right;
            clear: both;
        }
        .chat-bot {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 30%, #4facfe 100%);
            color: white;
            padding: 16px 20px;
            margin: 16px 80px 16px 0px;
            border-radius: 20px 20px 20px 8px;
            text-align: left;
            font-size: 15px;
            font-weight: 400;
            line-height: 1.5;
            box-shadow: 0 4px 12px rgba(240, 147, 251, 0.3);
            word-wrap: break-word;
            max-width: calc(100% - 100px);
            float: left;
            clear: both;
        }
        .chat-bot.error {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            border: 1px solid rgba(255, 107, 107, 0.5);
            box-shadow: 0 4px 12px rgba(255, 107, 107, 0.3);
        }
        .chat-container {
            padding: 10px 0;
            overflow: hidden;
        }
        </style>
        """
        self.chat_history_html = []  # Store all chat bubbles as HTML

        # Set initial content with CSS
        self.chat_display.setHtml(self.chat_style + '<div class="chat-container"></div>')

        # General tab only UI elements with enhanced styling
        if "general_question" in self.endpoint:
            # Main geometry controls container
            geom_container = QWidget()
            geom_container.setStyleSheet("""
                QWidget {
                    background: rgba(255, 255, 255, 0.02);
                    border-radius: 8px;
                    padding: 6px;
                }
            """)
            geom_layout = QHBoxLayout(geom_container)
            geom_layout.setSpacing(6)  # Reduced spacing

            # Show All Building Geometry button (enhanced)
            self.show_all_btn = QPushButton("Show All")
            self.show_all_btn.clicked.connect(self.toggle_all_geometry)
            self.show_all_btn.setStyleSheet(self.get_primary_button_style())
            geom_layout.addWidget(self.show_all_btn)

            # Add stretch to push dropdowns and "Show" button to the right
            geom_layout.addStretch(1)

            # Enhanced dropdown styling with responsive design
            dropdown_style = """
                QComboBox {
                    background: rgba(255, 255, 255, 0.08);
                    border: 1px solid rgba(255, 255, 255, 0.15);
                    border-radius: 12px;
                    padding: 10px 32px 10px 16px;
                    color: #e8e8f0;
                    font-size: 13px;
                    font-weight: 500;
                    min-width: 100px;
                    /* max-width: 150px;  REMOVE THIS LINE */
                }
                QComboBox:hover {
                    background: rgba(255, 255, 255, 0.12);
                    border-color: rgba(255, 255, 255, 0.25);
                }
                QComboBox::drop-down {
                    border: none;
                    width: 30px;
                }
                QComboBox QAbstractItemView {
                    background: #2a2a3e;
                    border: 1px solid rgba(255, 255, 255, 0.15);
                    border-radius: 8px;
                    padding: 8px 0;
                    color: #e8e8f0;
                    selection-background-color: rgba(99, 102, 241, 0.4);
                    font-size: 13px;
                }
            """

            # Level controls
            level_container = QWidget()
            level_box = QHBoxLayout(level_container)
            level_box.setSpacing(8)
            level_label = QLabel("Level:")
            level_label.setStyleSheet("color: #b8b8c8; font-weight: 500; font-size: 13px;")
            self.level_dropdown = QComboBox()
            self.level_dropdown.setView(QListView())
            self.level_dropdown.addItems(["1", "2", "3", "4"])
            self.level_dropdown.setStyleSheet(dropdown_style)
            self.level_dropdown.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            level_box.addWidget(level_label)
            level_box.addWidget(self.level_dropdown)
            geom_layout.addWidget(level_container, stretch=1)

            # Space Info controls
            space_container = QWidget()
            space_info_box = QHBoxLayout(space_container)
            space_info_box.setSpacing(8)
            space_info_label = QLabel("Space Info:")
            space_info_label.setStyleSheet("color: #b8b8c8; font-weight: 500; font-size: 13px;")
            self.space_info_dropdown = QComboBox()
            self.space_info_dropdown.setView(QListView())
            self.space_info_dropdown.addItems([
                "Activity", "Area", "UTCI", "Wind", "Orientation"
            ])
            self.space_info_dropdown.setStyleSheet(dropdown_style)
            self.space_info_dropdown.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            space_info_box.addWidget(space_info_label)
            space_info_box.addWidget(self.space_info_dropdown)
            geom_layout.addWidget(space_container, stretch=1)

            # Apartment Info controls
            apt_container = QWidget()
            apt_info_box = QHBoxLayout(apt_container)
            apt_info_box.setSpacing(8)
            apt_info_label = QLabel("Apartment Info:")
            apt_info_label.setStyleSheet("color: #b8b8c8; font-weight: 500; font-size: 13px;")
            self.apt_info_dropdown = QComboBox()
            self.apt_info_dropdown.setView(QListView())
            self.apt_info_dropdown.addItems(["Key", "Residents"])
            self.apt_info_dropdown.setStyleSheet(dropdown_style)
            self.apt_info_dropdown.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            apt_info_box.addWidget(apt_info_label)
            apt_info_box.addWidget(self.apt_info_dropdown)
            geom_layout.addWidget(apt_container, stretch=1)

            
            # Action buttons
            self.show_geom_btn = QPushButton("Show")
            self.show_geom_btn.clicked.connect(self.send_geometry_command)
            self.show_geom_btn.setStyleSheet(self.get_secondary_button_style())

            self.hide_specific_btn = QPushButton("Hide")
            self.hide_specific_btn.clicked.connect(self.hide_specific_geometry)
            self.hide_specific_btn.setStyleSheet(self.get_secondary_button_style())

            geom_layout.addWidget(self.show_geom_btn)
            geom_layout.addWidget(self.hide_specific_btn)

            layout.addWidget(geom_container)

        # Enhanced input section
        input_container = QWidget()
        input_container.setStyleSheet("""
            QWidget {
                background: rgba(255, 255, 255, 0.02);
                border-radius: 8px;
                padding: 6px;
            }
        """)
        input_layout = QHBoxLayout(input_container)
        input_layout.setSpacing(6)

        self.input_box = QLineEdit()
        self.input_box.setStyleSheet("""
            QLineEdit {
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 14px;
                padding: 14px 18px;
                color: #e8e8f0;
                font-size: 14px;
                font-weight: 400;
                font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
            }
            QLineEdit:focus {
                border-color: rgba(99, 102, 241, 0.6);
                background: rgba(255, 255, 255, 0.12);
                outline: none;
            }
            QLineEdit::placeholder {
                color: rgba(232, 232, 240, 0.5);
            }
        """)
        self.input_box.setPlaceholderText("Type your message here...")

        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_message)
        self.input_box.returnPressed.connect(self.send_message)
        self.send_btn.setStyleSheet(self.get_primary_button_style())

        input_layout.addWidget(self.input_box)
        input_layout.addWidget(self.send_btn)
        layout.addWidget(input_container)

        self.setLayout(layout)

    def get_primary_button_style(self):
        return """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #667eea, stop:0.5 #764ba2, stop:1 #f093fb);
                color: white;
                border: none;
                border-radius: 14px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 600;
                font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
                min-height: 20px;
                transition: all 0.3s ease;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #5a6fd8, stop:0.5 #6a4190, stop:1 #e081e9);
                transform: translateY(-1px);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #4e5fc6, stop:0.5 #5e377e, stop:1 #ce6fd7);
                transform: translateY(0px);
            }
            QPushButton:disabled {
                background: rgba(255, 255, 255, 0.1);
                color: rgba(255, 255, 255, 0.5);
            }
        """

    def get_secondary_button_style(self):
        return """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(102, 126, 234, 0.3), stop:1 rgba(240, 147, 251, 0.3));
                color: #e8e8f0;
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 12px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: 500;
                font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
                min-height: 16px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(102, 126, 234, 0.5), stop:1 rgba(240, 147, 251, 0.5));
                border-color: rgba(255, 255, 255, 0.3);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(102, 126, 234, 0.7), stop:1 rgba(240, 147, 251, 0.7));
            }
        """

    def update_chat_display(self):
        # Rebuild the full HTML with style and all bubbles
        full_html = self.chat_style + '<div class="chat-container">' + ''.join(self.chat_history_html) + '</div>'
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
            success_bubble = '<div class="chat-bot"><b>Geometry Command sent:</b> ' + escape(payload) + '</div>'
            self.chat_history_html.append(success_bubble)
            self.update_chat_display()
        except Exception as e:
            error_bubble = f'<div class="chat-bot error"><b>Error:</b> {escape(str(e))}</div>'
            self.chat_history_html.append(error_bubble)
            self.update_chat_display()

    def toggle_all_geometry(self):
        if not self.geometry_shown:
            self.show_all_geometry(force=True)
        else:
            self.hide_all_geometry(force=True)

    def show_all_geometry(self, force=False):
        try:
            send_udp_command("show_all")
            self.geometry_shown = True
            self.show_all_btn.setText("Hide All")
            success_bubble = '<div class="chat-bot"><b>All geometry shown.</b></div>'
            self.chat_history_html.append(success_bubble)
            self.update_chat_display()
        except Exception as e:
            error_bubble = f'<div class="chat-bot error"><b>Error:</b> {escape(str(e))}</div>'
            self.chat_history_html.append(error_bubble)
            self.update_chat_display()

    def hide_all_geometry(self, force=False):
        try:
            send_udp_command("hide_all")
            self.geometry_shown = False
            self.show_all_btn.setText("Show All")
            success_bubble = '<div class="chat-bot"><b>All geometry hidden.</b></div>'
            self.chat_history_html.append(success_bubble)
            self.update_chat_display()
        except Exception as e:
            error_bubble = f'<div class="chat-bot error"><b>Error:</b> {escape(str(e))}</div>'
            self.chat_history_html.append(error_bubble)
            self.update_chat_display()

    def hide_specific_geometry(self):
        # Send a "null|null|null" string to port 6001 to hide all geometry
        try:
            payload = "null|null|null"
            send_udp_command2(payload, port=6001)
            success_bubble = '<div class="chat-bot"><b>Selected geometry hidden (no geometry previewed).</b></div>'
            self.chat_history_html.append(success_bubble)
            self.update_chat_display()
        except Exception as e:
            error_bubble = f'<div class="chat-bot error"><b>Error:</b> {escape(str(e))}</div>'
            self.chat_history_html.append(error_bubble)
            self.update_chat_display()

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
        error_html = f'<div class="chat-bot error"><b>Error:</b> {escape(error_msg)}</div>'
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

        # Main layout with enhanced spacing and styling
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 8, 12, 8)  # Reduced margins
        layout.setSpacing(10)                    # Reduced spacing

        # Modern gradient background container
        content_container = QWidget()
        content_container.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 255, 255, 0.03), stop:1 rgba(255, 255, 255, 0.01));
                border-radius: 10px;
                padding: 12px;
            }
        """)
        content_layout = QVBoxLayout(content_container)
        content_layout.setSpacing(8)

        # Stylized header with gradient text effect
        header = QLabel("Welcome to the Co-creator App")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("""
            QLabel {
                font-size: 36px;  /* Slightly smaller for compact layout */
                font-weight: 700;
                color: #e8e8f0;
                padding: 24px 16px;  /* Reduced padding */
                border: none;
                background: transparent;
                font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
                letter-spacing: -0.5px;
            }
        """)
        content_layout.addWidget(header)

        # Enhanced info section with modern card styling
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        info_label.setStyleSheet("""
            QLabel {
                font-size: 15px;  /* Slightly smaller text */
                line-height: 26px;  /* Reduced line height */
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 255, 255, 0.06), stop:1 rgba(255, 255, 255, 0.02));
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 20px;
                padding: 24px;  /* Reduced padding */
                color: #c8c8d8;
                font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
                font-weight: 400;
            }
        """)
        content_layout.addWidget(info_label)

        # Enhanced button section
        btn_container = QWidget()
        btn_layout = QHBoxLayout(btn_container)
        btn_layout.setSpacing(24)
        btn_layout.addStretch(1)

        # Modern Fill Survey button
        self.fill_survey_btn = QPushButton("Fill Survey")
        self.fill_survey_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #667eea, stop:0.5 #764ba2, stop:1 #f093fb);
                color: white;
                border: none;
                border-radius: 16px;
                padding: 16px 40px;
                font-size: 16px;
                font-weight: 600;
                font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
                min-height: 24px;
                min-width: 160px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #5a6fd8, stop:0.5 #6a4190, stop:1 #e081e9);
                transform: translateY(-2px);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #4e5fc6, stop:0.5 #5e377e, stop:1 #ce6fd7);
                transform: translateY(0px);
            }
        """)
        self.fill_survey_btn.clicked.connect(self.go_to_survey_tab)
        btn_layout.addWidget(self.fill_survey_btn)
        btn_layout.addStretch(1)

        content_layout.addWidget(btn_container)
        content_layout.addStretch(1)

        layout.addWidget(content_container)
        self.setLayout(layout)
        self.tab_widget = tab_widget

    def go_to_survey_tab(self):
        if self.tab_widget:
            self.tab_widget.setCurrentIndex(1)

class SurveyTab(QWidget):
    def __init__(self, tab_widget=None):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 8, 12, 8)  # Reduced margins
        layout.setSpacing(10)

        # Create a modern container for the form
        form_container = QWidget()
        form_container.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 255, 255, 0.03), stop:1 rgba(255, 255, 255, 0.01));
                border-radius: 10px;
                padding: 12px;
            }
        """)
        form_layout = QVBoxLayout(form_container)
        form_layout.setSpacing(8)  # Reduced spacing for compact form

        # Enhanced input styling
        input_style = """
            QLineEdit {
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 12px;
                padding: 14px 18px;
                color: #e8e8f0;
                font-size: 14px;
                font-weight: 400;
                font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
                min-height: 20px;
            }
            QLineEdit:focus {
                border-color: rgba(99, 102, 241, 0.6);
                background: rgba(255, 255, 255, 0.12);
                outline: none;
            }
            QLineEdit::placeholder {
                color: rgba(232, 232, 240, 0.5);
            }
        """

        label_style = """
            QLabel {
                color: #b8b8c8;
                font-weight: 600;
                font-size: 14px;
                font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
                margin-bottom: 6px;
            }
        """

        # Form fields with enhanced styling
        fields = [
            ("Resident Key:", "Enter Resident Key", "resident_key_input"),
            ("Resident Persona:", "Enter Resident Persona", "resident_persona_input"),
            ("Resident Population:", "Enter Resident Population", "resident_population_input"),
            ("Level:", "Enter Level", "level_input"),
            ("Age:", "Enter Age", "age_input"),
            ("Tenant/Owner:", "Enter Tenant/Owner", "tenant_owner_input")
        ]

        for label_text, placeholder, attr_name in fields:
            label = QLabel(label_text)
            label.setStyleSheet(label_style)
            form_layout.addWidget(label)
            
            input_field = QLineEdit()
            input_field.setPlaceholderText(placeholder)
            input_field.setStyleSheet(input_style)
            setattr(self, attr_name, input_field)
            form_layout.addWidget(input_field)

        # Enhanced buttons
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setSpacing(16)

        self.submit_survey_btn = QPushButton("Submit Survey")
        self.submit_survey_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(102, 126, 234, 0.4), stop:1 rgba(240, 147, 251, 0.4));
                color: #e8e8f0;
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 14px;
                padding: 14px 28px;
                font-size: 14px;
                font-weight: 600;
                font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
                min-height: 20px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(102, 126, 234, 0.6), stop:1 rgba(240, 147, 251, 0.6));
                border-color: rgba(255, 255, 255, 0.3);
            }
        """)

        self.ask_general_btn = QPushButton("Ask general questions")
        self.ask_general_btn.clicked.connect(self.go_to_general_tab)
        self.ask_general_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #667eea, stop:0.5 #764ba2, stop:1 #f093fb);
                color: white;
                border: none;
                border-radius: 14px;
                padding: 14px 28px;
                font-size: 14px;
                font-weight: 600;
                font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
                min-height: 20px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #5a6fd8, stop:0.5 #6a4190, stop:1 #e081e9);
            }
        """)

        button_layout.addWidget(self.submit_survey_btn)
        button_layout.addWidget(self.ask_general_btn)
        form_layout.addWidget(button_container)

        layout.addWidget(form_container)
        layout.addStretch(1)
        self.setLayout(layout)
        self.tab_widget = tab_widget

    def go_to_general_tab(self):
        if self.tab_widget:
            self.tab_widget.setCurrentIndex(2)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Activity Copilot for Residentials")

        # Get screen size and set window to 1/3 of screen width, centered
        screen = QDesktopWidget().screenGeometry()
        width = int(screen.width() / 3)
        height = int(screen.height() * 0.8)
        self.resize(width, height)
        self.setMinimumSize(600, 500)  # Prevent too small

        # Optionally center the window
        self.move(
            screen.left() + (screen.width() - width) // 2,
            screen.top() + (screen.height() - height) // 2
        )

        # Enable window resizing (default, but explicit)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        #for gh_server_geometry
        self.start_flask_server_if_needed()
        #for gh_server_geometry

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0f0f23, stop:1 #1a1a2e);
            }
            QTabBar::tab {
                background: rgba(255, 255, 255, 0.05);
                color: #b8b8c8;
                border: none;
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
                padding: 14px 24px;
                margin-right: 2px;
                font-weight: 500;
                font-size: 14px;
                font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
                min-width: 120px;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(102, 126, 234, 0.2), stop:1 rgba(240, 147, 251, 0.2));
                color: #e8e8f0;
                border-bottom: 2px solid #667eea;
            }
            QTabBar::tab:hover:!selected {
                background: rgba(255, 255, 255, 0.08);
                color: #d8d8e8;
            }
        """)

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

        tabs.addTab(GeometryWorkflowTab("http://localhost:5004/initiate_gh_workflow"), "Geometry")

        images_tab = ImagesTab(images_folder="images")
        tabs.addTab(images_tab, "Images")
        
        layout.addWidget(tabs)
        self.setLayout(layout)

    def start_flask_server_if_needed(self):
        global flask_server_process
        server_script = _get_server_script_path_main()
        if os.path.exists(server_script):
            try:
                flask_server_process = subprocess.Popen([sys.executable, server_script])
                print(f"Flask server '{server_script}' started with PID: {flask_server_process.pid}")
                atexit.register(stop_flask_server)
            except Exception as e:
                print(f"Failed to start Flask server: {e}")
        else:
            print(f"Error: Server script not found at {server_script}")

# Geometry workflow tab
class GeometryWorkflowTab(QWidget):
    def __init__(self, endpoint):
        super().__init__()
        self.endpoint = endpoint

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 8, 12, 8)  # Reduced margins
        layout.setSpacing(10)

        # Create modern container
        container = QWidget()
        container.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 255, 255, 0.03), stop:1 rgba(255, 255, 255, 0.01));
                border-radius: 10px;
                padding: 12px;
            }
        """)
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(8)

        # Enhanced input styling with responsive design
        input_style = """
            QLineEdit {
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 12px;
                padding: 14px 18px;
                color: #e8e8f0;
                font-size: 14px;
                font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
                min-height: 20px;
                max-width: none;  /* Allow full width expansion */
            }
            QLineEdit:focus {
                border-color: rgba(99, 102, 241, 0.6);
                background: rgba(255, 255, 255, 0.12);
            }
        """

        combobox_style = """
            QComboBox {
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 12px;
                padding: 14px 32px 14px 18px;
                color: #e8e8f0;
                font-size: 14px;
                font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
                min-height: 20px;
                min-width: 180px;  /* Responsive minimum width */
            }
            QComboBox:hover {
                background: rgba(255, 255, 255, 0.12);
                border-color: rgba(255, 255, 255, 0.25);
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox QAbstractItemView {
                background: #2a2a3e;
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 8px;
                padding: 8px 0;
                color: #e8e8f0;
                selection-background-color: rgba(99, 102, 241, 0.4);
            }
        """

        label_style = """
            QLabel {
                color: #b8b8c8;
                font-weight: 600;
                font-size: 14px;
                font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
                margin-bottom: 6px;
            }
        """

        # Input fields
        fields = [
            ("Space ID:", "Space ID (e.g., O2)", "space_id_input", "input"),
            ("Resident Key:", "Resident Key (e.g., H23)", "resident_key_input", "input"),
            ("Question/Request:", "Your question or request", "question_input", "input")
        ]

        for label_text, placeholder, attr_name, field_type in fields:
            label = QLabel(label_text)
            label.setStyleSheet(label_style)
            container_layout.addWidget(label)
            
            if field_type == "input":
                field = QLineEdit()
                field.setPlaceholderText(placeholder)
                field.setStyleSheet(input_style)
            
            setattr(self, attr_name, field)
            container_layout.addWidget(field)

        # Desired Activity dropdown
        activity_label = QLabel("Desired Activity:")
        activity_label.setStyleSheet(label_style)
        container_layout.addWidget(activity_label)

        self.desired_activity_input = QComboBox()
        activities = [
            "Sitting", "Offline Retreat", "Sunbath", "Healing Garden",
            "Playground", "Sports", "Outdoor Cinema/Event Space",
            "Community Pool/BBQ", "Flexible Space", "Creative Corridor",
            "Outdoor Meeting Room", "Green Corridor", "Biodiversity balcony",
            "Urban Agriculture Garden", "Viewpoint", "Storage & Technical Space"
        ]
        self.desired_activity_input.addItems(activities)
        self.desired_activity_input.setCurrentIndex(-1)
        self.desired_activity_input.setStyleSheet(combobox_style)
        container_layout.addWidget(self.desired_activity_input)

        # Submit button
        self.send_btn = QPushButton("Submit Job to GH")
        self.send_btn.clicked.connect(self.send_request)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #667eea, stop:0.5 #764ba2, stop:1 #f093fb);
                color: white;
                border: none;
                border-radius: 14px;
                padding: 16px 32px;
                font-size: 16px;
                font-weight: 600;
                font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
                min-height: 24px;
                margin-top: 16px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #5a6fd8, stop:0.5 #6a4190, stop:1 #e081e9);
            }
            QPushButton:disabled {
                background: rgba(255, 255, 255, 0.1);
                color: rgba(255, 255, 255, 0.5);
            }
        """)
        container_layout.addWidget(self.send_btn)

        # Response display
        self.response_display = QTextEdit()
        self.response_display.setReadOnly(True)
        self.response_display.setStyleSheet("""
            QTextEdit {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                padding: 16px;
                color: #e8e8f0;
                font-size: 13px;
                font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
                min-height: 200px;
            }
        """)
        container_layout.addWidget(self.response_display)

        layout.addWidget(container)
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
        
        self.worker = RequestWorker(self.endpoint, payload)
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
            suggestion = suggestions[0]
            display_text += f"<b>Suggestion:</b> {suggestion.get('variation_name', 'N/A')}\n"
            display_text += f"<i>Description:</i> {suggestion.get('description', 'N/A')}\n"
            send_udp_command(json.dumps(data))

        display_text += f"\n<b>Summary Reasoning:</b> {summary}"
        self.response_display.append(display_text)

    def handle_submit_error(self, error_msg):
        try:
            import re, json
            json_match = re.search(r'(\{.*\})', error_msg)
            if json_match:
                try:
                    error_json = json.loads(json_match.group(1))
                    if "error" in error_json:
                        self.response_display.append(f"<b>Server Error:</b> {error_json['error']}")
                        if "details" in error_json:
                            self.response_display.append(f"<i>Details: {error_json['details']}</i>")
                        return
                except Exception:
                    pass
            self.response_display.append(f"<b>Error Submitting Job:</b> {error_msg}")
        finally:
            self.send_btn.setEnabled(True)

class ImagesTab(QWidget):
    def __init__(self, images_folder="images"):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 8, 12, 8)  # Reduced margins
        
        # Create modern container
        container = QWidget()
        container.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 255, 255, 0.03), stop:1 rgba(255, 255, 255, 0.01));
                border-radius: 10px;
                padding: 8px;
            }
        """)
        container_layout = QVBoxLayout(container)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                background: rgba(255, 255, 255, 0.05);
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 0.2);
                border-radius: 4px;
                min-height: 20px;
            }
        """)
        
        content = QWidget()
        grid = QGridLayout()
        content.setLayout(grid)
        scroll.setWidget(content)
        container_layout.addWidget(scroll)

        # Load images from the folder
        import os
        from glob import glob

        image_files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"):
            image_files.extend(glob(os.path.join(images_folder, ext)))
        image_files.sort()

        # Display images in a grid with modern styling
        for idx, img_path in enumerate(image_files):
            image_container = QWidget()
            image_container.setStyleSheet("""
                QWidget {
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 12px;
                    padding: 12px;
                    margin: 4px;
                }
            """)
            image_layout = QVBoxLayout(image_container)
            
            label = ClickableLabel(img_path)
            pixmap = QPixmap(img_path)
            if not pixmap.isNull():
                pixmap = pixmap.scaledToWidth(280, Qt.SmoothTransformation)
                label.setPixmap(pixmap)
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet("border-radius: 8px;")
            else:
                label.setText(f"Failed to load {os.path.basename(img_path)}")
                label.setStyleSheet("color: #ff6b6b; text-align: center;")
            
            image_layout.addWidget(label)
            grid.addWidget(image_container, idx // 3, idx % 3)

        layout.addWidget(container)
        self.setLayout(layout)

class ClickableLabel(QLabel):
    def __init__(self, img_path, parent=None):
        super().__init__(parent)
        self.img_path = img_path

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.show_full_image()

    def show_full_image(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Image Viewer")
        vbox = QVBoxLayout(dlg)
        lbl = QLabel()
        pixmap = QPixmap(self.img_path)
        if not pixmap.isNull():
            # Scale to fit screen but not larger than original
            screen = QApplication.primaryScreen().availableGeometry()
            max_w = int(screen.width() * 0.8)
            max_h = int(screen.height() * 0.8)
            pixmap = pixmap.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            lbl.setPixmap(pixmap)
        vbox.addWidget(lbl)
        dlg.setLayout(vbox)
        dlg.exec_()

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

    # Set modern dark theme palette
    palette = QPalette()
    
    # Dark theme with subtle blue tints
    palette.setColor(QPalette.Window, QColor(15, 15, 35))           # Deep dark blue-black
    palette.setColor(QPalette.Base, QColor(26, 26, 46))             # Slightly lighter dark blue
    palette.setColor(QPalette.WindowText, QColor(232, 232, 240))    # Light blue-white text
    palette.setColor(QPalette.Text, QColor(232, 232, 240))          # Light blue-white text
    palette.setColor(QPalette.ButtonText, QColor(232, 232, 240))    # Light blue-white text
    palette.setColor(QPalette.Button, QColor(42, 42, 62))           # Dark blue-gray buttons
    palette.setColor(QPalette.Highlight, QColor(99, 102, 241))      # Bright blue highlight
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255)) # White highlighted text
    palette.setColor(QPalette.Link, QColor(99, 102, 241))           # Blue links
    palette.setColor(QPalette.Light, QColor(60, 60, 80))            # Light accents

    app.setPalette(palette)
    
    window = MainWindow()
    
    # Apply the enhanced modern stylesheet
    window.setStyleSheet("""
        /* Main window background with gradient */
        QWidget {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #0f0f23, stop:1 #1a1a2e);
            color: #e8e8f0;
            font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
            font-size: 13px;
        }

        /* Enhanced Tab Styling */
        QTabWidget::pane {
            border: none;
            background: transparent;
        }

        QTabBar::tab {
            background: rgba(255, 255, 255, 0.05);
            color: #b8b8c8;
            border: none;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            padding: 6px 10px;
            margin-right: 2px;
            font-weight: 500;
            font-size: 13px;
            font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
            min-width: 80px;
        }

        QTabBar::tab:selected {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 rgba(102, 126, 234, 0.2), stop:1 rgba(240, 147, 251, 0.2));
            color: #e8e8f0;
            border-bottom: 2px solid #667eea;
        }

        QTabBar::tab:hover:!selected {
            background: rgba(255, 255, 255, 0.08);
            color: #d8d8e8;
        }

        /* Modern Text Inputs */
        QTextEdit, QTextBrowser, QLineEdit {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 6px;
            padding: 6px 8px;
            color: #e8e8f0;
            font-size: 13px;
            font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
            selection-background-color: rgba(99, 102, 241, 0.3);
        }

        QTextEdit:focus, QLineEdit:focus {
            border-color: rgba(99, 102, 241, 0.6);
            background: rgba(255, 255, 255, 0.12);
            outline: none;
        }

        /* Enhanced ComboBox */
        QComboBox {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 6px;
            padding: 6px 24px 6px 8px;
            color: #e8e8f0;
            font-size: 13px;
            font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
            min-height: 20px;
            min-width: 80px;
        }

        QComboBox:hover {
            background: rgba(255, 255, 255, 0.12);
            border-color: rgba(255, 255, 255, 0.25);
        }

        QComboBox::drop-down {
            border: none;
            width: 30px;
        }

        QComboBox QAbstractItemView {
            background: #2a2a3e;
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 8px;
            padding: 8px 0;
            color: #e8e8f0;
            selection-background-color: rgba(99, 102, 241, 0.4);
        }

        /* Enhanced Buttons with Gradients */
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #667eea, stop:0.5 #764ba2, stop:1 #f093fb);
            color: white;
            border: none;
            border-radius: 7px;
            padding: 6px 12px;
            font-size: 13px;
            font-weight: 600;
            font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
            min-height: 20px;
            min-width: 60px;
        }

        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #5a6fd8, stop:0.5 #6a4190, stop:1 #e081e9);
            transform: translateY(-1px);
        }

        QPushButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #4e5fc6, stop:0.5 #5e377e, stop:1 #ce6fd7);
            transform: translateY(0px);
        }

        QPushButton:disabled {
            background: rgba(255, 255, 255, 0.1);
            color: rgba(255, 255, 255, 0.5);
        }

        /* Enhanced Labels */
        QLabel {
            color: #e8e8f0;
            font-size: 14px;
            font-weight: 400;
        }

        /* Enhanced Checkboxes */
        QCheckBox {
            spacing: 8px;
            font-size: 14px;
            color: #e8e8f0;
        }

        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border-radius: 4px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            background: rgba(255, 255, 255, 0.05);
        }

        QCheckBox::indicator:checked {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #667eea, stop:1 #764ba2);
            border-color: #667eea;
        }

        /* Enhanced Scrollbars */
        QScrollBar:vertical {
            background: rgba(255, 255, 255, 0.05);
            width: 8px;
            border-radius: 4px;
            margin: 0;
        }

        QScrollBar::handle:vertical {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            min-height: 20px;
        }

        QScrollBar::handle:vertical:hover {
            background: rgba(255, 255, 255, 0.3);
        }

        QScrollBar:horizontal {
            background: rgba(255, 255, 255, 0.05);
            height: 8px;
            border-radius: 4px;
            margin: 0;
        }

        QScrollBar::handle:horizontal {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            min-width: 20px;
        }

        QScrollBar::handle:horizontal:hover {
            background: rgba(255, 255, 255, 0.3);
        }

        QScrollBar::add-line, QScrollBar::sub-line {
            border: none;
            background: none;
        }
    """) 

    # Start CSV watcher
    csv_path = os.path.join(os.path.dirname(__file__), "llm_reasoning", "llm_activity_assignments.csv")
    watcher = CsvWatcher(csv_path, send_udp_command3)

    window.show()
    sys.exit(app.exec_())