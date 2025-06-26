import sys
import requests
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QLabel, QLineEdit, QPushButton, QTextBrowser,
    QHBoxLayout, QComboBox, QFrame, QTextEdit, QTabWidget
)
from PyQt5.QtCore import Qt, pyqtSignal
import socket
import re



#new gh connection ghowl for show and hide space keys automatically
def send_udp_command_1(message, port=7000):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(message.encode("utf-8"), ("127.0.0.1", port))
    sock.close()

def extract_space_keys(text):
    """Extract all unique Hxx and Oxx keys from a string."""
    return set(re.findall(r'\b[HO]\d+\b', text))


class EnterTextEdit(QTextEdit):
    enterPressed = pyqtSignal()
    
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not (event.modifiers() & Qt.ShiftModifier):
            self.enterPressed.emit()
            event.accept()  # Accept the event to prevent default behavior
        else:
            super().keyPressEvent(event)

class SpaceQnAUI(QMainWindow):
    # Enhanced CSS Styles for chat bubbles (user right, bot left)
    chat_style = """
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
        font-weight: 600;
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

    def append_chat(self, browser, text, sender="bot", error=False):
        if browser == self.qna_display:
            history = self.qna_chat_history
        else:
            history = self.neg_chat_history

        if sender == "user":
            html = f'<div class="chat-user">{text}</div>'
        elif error:
            html = f'<div class="chat-bot error">{text}</div>'
        else:
            html = f'<div class="chat-bot">{text}</div>'
        history.append(html)
        self.update_chat_display(browser)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nearby Space QnA")
        self.setGeometry(200, 200, 800, 800)

        # Tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # --- Tab 1: Q&A ---
        qna_tab = QWidget()
        qna_layout = QVBoxLayout(qna_tab)
        qna_layout.setContentsMargins(24, 24, 24, 24)
        qna_layout.setSpacing(20)

        container = QFrame()
        container.setObjectName("container")
        qna_layout.addWidget(container)
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(20)

        header = QLabel("Nearby Space QnA")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: 700;
                color: #e8e8f0;
                background: transparent;
                font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
            }
        """)
        main_layout.addWidget(header)

        row = QHBoxLayout()
        label = QLabel("Resident Key:")
        label.setStyleSheet("color: #b8b8c8; font-weight: 500; font-size: 14px;")
        self.house_key_input = QLineEdit()
        self.house_key_input.setPlaceholderText("Enter your resident key (e.g. H1)")
        self.house_key_input.setMinimumHeight(32)
        self.house_key_input.setStyleSheet("""
            QLineEdit {
                background: rgba(255,255,255,0.08);
                border: 1px solid rgba(255,255,255,0.15);
                border-radius: 14px;
                padding: 14px 18px;
                color: #e8e8f0;
                font-size: 14px;
                font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
            }
        """)
        row.addWidget(label)
        row.addWidget(self.house_key_input, stretch=1)
        main_layout.addLayout(row)

        self.qna_display = QTextBrowser()
        self.qna_display.setReadOnly(True)
        self.qna_display.setStyleSheet("""
            QTextBrowser {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0f0f23, stop:1 #1a1a2e);
                color: #e8e8f0;
                border: none;
                border-radius: 16px;
                font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
                font-size: 14px;
                padding: 20px;
                selection-background-color: rgba(99, 102, 241, 0.3);
            }
        """)
        main_layout.addWidget(self.qna_display, stretch=1)

        input_row = QHBoxLayout()
        self.input_field = EnterTextEdit()
        self.input_field.setPlaceholderText("Type your question about nearby spaces... (Press Enter to send, Shift+Enter for new line)")
        self.input_field.setMinimumHeight(44)
        self.input_field.setMaximumHeight(60)
        self.input_field.setStyleSheet("""
            QTextEdit {
                background: rgba(255,255,255,0.08);
                color: #e8e8f0;
                border-radius: 16px;
                padding: 16px 20px;
                font-size: 15px;
                font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
                font-weight: 400;
            }
        """)
        self.input_field.enterPressed.connect(self.send_qna)

        self.ask_button = QPushButton("Ask")
        self.ask_button.setMinimumHeight(44)
        self.ask_button.setMaximumHeight(60)
        self.ask_button.setMinimumWidth(120)
        self.ask_button.setStyleSheet("""
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
        """)
        self.ask_button.clicked.connect(self.send_qna)

        input_row.addWidget(self.input_field, stretch=1)
        input_row.addWidget(self.ask_button)
        main_layout.addLayout(input_row)

        # # --- Show Geometry by Key ---
        # geom_row = QHBoxLayout()
        # geom_label = QLabel("Show Geometry Key:")
        # geom_label.setStyleSheet("color: #b8b8c8; font-weight: 500; font-size: 14px;")
        # self.geom_key_input = QLineEdit()
        # self.geom_key_input.setPlaceholderText("Enter space or resident key (e.g. O1, H10)")
        # self.geom_key_input.setMinimumHeight(32)
        # self.geom_key_input.setStyleSheet("""
        #     QLineEdit {
        #         background: rgba(255,255,255,0.08);
        #         border: 1px solid rgba(255,255,255,0.15);
        #         border-radius: 14px;
        #         padding: 14px 18px;
        #         color: #e8e8f0;
        #         font-size: 14px;
        #         font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
        #     }
        # """)
        # self.show_geom_btn = QPushButton("Show Geometry")
        # self.show_geom_btn.setMinimumHeight(36)
        # self.show_geom_btn.setStyleSheet("""
        #     QPushButton {
        #         background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        #             stop:0 #667eea, stop:1 #f093fb);
        #         color: white;
        #         border: none;
        #         border-radius: 14px;
        #         padding: 12px 24px;
        #         font-size: 14px;
        #         font-weight: 600;
        #         font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
        #         min-width: 60px;
        #         min-height: 20px;
        #     }
        #     QPushButton:hover {
        #         background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        #             stop:0 #5a6fd8, stop:1 #e081e9);
        #     }
        # """)
        # self.show_geom_btn.clicked.connect(self.toggle_show_hide_geometry)
        # geom_row.addWidget(geom_label)
        # geom_row.addWidget(self.geom_key_input, stretch=1)
        # geom_row.addWidget(self.show_geom_btn)
        # main_layout.addLayout(geom_row)

        self.tabs.addTab(qna_tab, "Nearby Space Q&A")


        # --- Tab 2: Negotiate ---
        negotiate_tab = QWidget()
        negotiate_layout = QVBoxLayout(negotiate_tab)
        negotiate_layout.setContentsMargins(24, 24, 24, 24)
        negotiate_layout.setSpacing(20)

        negotiate_container = QFrame()
        negotiate_container.setObjectName("container")
        negotiate_layout.addWidget(negotiate_container)
        neg_layout = QVBoxLayout(negotiate_container)
        neg_layout.setContentsMargins(0, 0, 0, 0)
        neg_layout.setSpacing(20)

        neg_header = QLabel("Negotiate")
        neg_header.setAlignment(Qt.AlignCenter)
        neg_header.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: 700;
                color: #e8e8f0;
                background: transparent;
                font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
            }
        """)
        neg_layout.addWidget(neg_header)

        neg_row = QHBoxLayout()
        neg_label = QLabel("Resident Key:")
        neg_label.setStyleSheet("color: #b8b8c8; font-weight: 500; font-size: 14px;")
        self.neg_house_key_input = QLineEdit()
        self.neg_house_key_input.setPlaceholderText("Enter your resident key (e.g. H1)")
        self.neg_house_key_input.setMinimumHeight(32)
        self.neg_house_key_input.setStyleSheet("""
            QLineEdit {
                background: rgba(255,255,255,0.08);
                border: 1px solid rgba(255,255,255,0.15);
                border-radius: 14px;
                padding: 14px 18px;
                color: #e8e8f0;
                font-size: 14px;
                font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
            }
        """)
        neg_row.addWidget(neg_label)
        neg_row.addWidget(self.neg_house_key_input, stretch=1)
        neg_layout.addLayout(neg_row)

        

        self.neg_display = QTextBrowser()
        self.neg_display.setReadOnly(True)
        self.neg_display.setStyleSheet("""
            QTextBrowser {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0f0f23, stop:1 #1a1a2e);
                color: #e8e8f0;
                border: none;
                border-radius: 16px;
                font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
                font-size: 14px;
                padding: 20px;
                selection-background-color: rgba(99, 102, 241, 0.3);
            }
        """)
        neg_layout.addWidget(self.neg_display, stretch=1)

        # Create input field with Enter support
        self.neg_input_field = EnterTextEdit()
        self.neg_input_field.setPlaceholderText("Type your negotiation query... (Press Enter to send, Shift+Enter for new line)")
        self.neg_input_field.setMinimumHeight(44)
        self.neg_input_field.setMaximumHeight(60)
        self.neg_input_field.setStyleSheet("""
            QTextEdit {
                background: rgba(255,255,255,0.08);
                color: #e8e8f0;
                border-radius: 16px;
                padding: 16px 20px;
                font-size: 15px;
                font-family: 'SF Pro Display', 'Segoe UI', 'Inter', sans-serif;
                font-weight: 400;
            }
        """)
        self.neg_input_field.enterPressed.connect(self.send_negotiate)
        neg_layout.addWidget(self.neg_input_field)

        self.neg_ask_button = QPushButton("Negotiate")
        self.neg_ask_button.setMinimumHeight(44)
        self.neg_ask_button.setMaximumHeight(60)
        self.neg_ask_button.setMinimumWidth(120)
        self.neg_ask_button.setStyleSheet("""
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
        """)
        self.neg_ask_button.clicked.connect(self.send_negotiate)
        neg_layout.addWidget(self.neg_ask_button)

        # Store last suggestions and context for multi-turn negotiation
        self.last_negotiation_suggestions = []
        self.last_negotiation_context = {}
        self.current_detected_keys = set()  # Initialize the detected keys set

        # NEW: Add these for enhanced features
        self.pending_geometry_changes = {}
        self.current_geometry_space = None

        # NEW: Initialize chat history lists
        self.qna_chat_history = []
        self.neg_chat_history = []

        self.tabs.addTab(negotiate_tab, "Negotiate")

    # NEARBY SPACE Q&A
    def send_qna(self):
        house_key = self.house_key_input.text().strip()
        question = self.input_field.toPlainText().strip()
        if not house_key or not question:
            self.append_chat(self.qna_display, "Please enter a resident key and a question.", sender="bot", error=True)
            return

        self.input_field.clear()
        self.append_chat(self.qna_display, f"<b>You ({house_key}):</b> {question}", sender="user")
        try:
            resp = requests.post(
                "http://127.0.0.1:5000/llm_nearby_space_qna",
                json={"house_key": house_key, "question": question},
                timeout=30
            )
            if resp.status_code == 200:
                answer = resp.json().get("response", "<No response>")
                self.append_chat(self.qna_display, answer, sender="bot")
                keys_str = self.update_detected_keys(question, answer)
                if keys_str:
                    self.append_chat(self.qna_display, f"<b>Detected keys:</b> {keys_str}", sender="bot")
                else:
                    self.append_chat(self.qna_display, f"<b>Detected keys:</b> None", sender="bot")
                send_udp_command_1(keys_str, port=7000)
            else:
                self.append_chat(self.qna_display, f"Server error: {resp.status_code}", sender="bot", error=True)
        except Exception as e:
            self.append_chat(self.qna_display, f"Error connecting to server: {e}", sender="bot", error=True)

    # NEGOTIATE
    def send_negotiate(self):
        # Clear any previous keys first
        send_udp_command_1("", port=7000)

        house_key = self.neg_house_key_input.text().strip()
        query = self.neg_input_field.toPlainText().strip()
        if not house_key or not query:
            self.append_chat(self.neg_display, "Please enter a resident key and a negotiation query.", sender="bot", error=True)
            return

        self.append_chat(self.neg_display, f"<b>You ({house_key}):</b> {query}", sender="user")

        # Check if this is a choice confirmation (number or "yes")
        if self.last_negotiation_suggestions and (query.isdigit() or query.lower() in ['yes', 'y']):
            if query.lower() in ['yes', 'y']:
                choice_idx = 0
            else:
                try:
                    choice_idx = int(query) - 1
                except:
                    self.append_chat(self.neg_display, "Please enter a valid number for your choice.", sender="bot", error=True)
                    self.neg_input_field.clear()
                    return

            if choice_idx < 0 or choice_idx >= len(self.last_negotiation_suggestions):
                self.append_chat(self.neg_display, "Choice out of range. Please enter a valid number.", sender="bot", error=True)
                self.neg_input_field.clear()
                return

            self.execute_negotiation_choice(choice_idx, house_key, query)
            self.neg_input_field.clear()
            return

        # This is a new negotiation query
        self.neg_input_field.clear()
        try:
            payload = {"house_key": house_key, "query": query}
            if hasattr(self, 'last_negotiation_context') and self.last_negotiation_context:
                payload["last_context"] = self.last_negotiation_context
            resp = requests.post(
                "http://127.0.0.1:5000/llm_negotiate",
                json=payload,
                timeout=30
            )
            if resp.status_code == 200:
                data = resp.json()
                context = data.get("context", {})
                suggestions = data.get("suggestions", [])
                self.last_negotiation_suggestions = suggestions  # Save for confirm
                self.last_negotiation_context = context  # Save for multi-turn
                # Build the suggestions HTML separately
                if suggestions:
                    suggestions_html = "<b>Suggestions:</b><ol>"
                    for s in suggestions:
                        suggestions_html += f"<li><b>{s.get('action')}</b>: {s.get('explanation')}<br>Params: {s.get('parameters')}</li>"
                    suggestions_html += "</ol><b>Choose your option: Type the number (1, 2, 3...) or 'yes' for option 1</b>"
                else:
                    suggestions_html = "<b>No suggestions available.</b>"

                bot_html = ""
                if context.get('nearby_activities'):
                    bot_html += f"<b>Nearby Activities:</b> {context.get('nearby_activities')}<br>"
                if context.get('preferences'):
                    bot_html += f"<b>Preferences:</b> {context.get('preferences')}<br>"
                bot_html += suggestions_html
                self.append_chat(self.neg_display, bot_html, sender="bot")

                suggestion_texts = " ".join(
                    f"{s.get('action','')} {s.get('explanation','')} {s.get('parameters','')}" for s in suggestions
                )
                context_text = f"{context.get('nearby_activities', '')} {context.get('preferences', '')}"
                all_response_text = f"{suggestion_texts} {context_text}"
                keys_str = self.update_detected_keys(query, all_response_text)
                if keys_str:
                    send_udp_command_1(keys_str, port=7000)
                    self.append_chat(self.neg_display, f"<b>Detected keys:</b> {keys_str}", sender="bot")
                else:
                    send_udp_command_1("", port=7000)
                    self.append_chat(self.neg_display, f"<b>Detected keys:</b> None", sender="bot")
            else:
                self.append_chat(self.neg_display, f"Server error: {resp.status_code}", sender="bot", error=True)
        except Exception as e:
            self.append_chat(self.neg_display, f"Error connecting to server: {e}", sender="bot", error=True)

    def execute_negotiation_choice(self, choice_idx, house_key, original_query):
        """Execute the selected negotiation choice - UPDATED VERSION"""
        suggestion = self.last_negotiation_suggestions[choice_idx]
        action = suggestion.get('action')
        parameters = suggestion.get('parameters', {})

        # Show what choice was made (user bubble)
        choice_bubble = f"""
        <div class='bubble user-bubble'>
            <div class='bubble-header'>You chose option {choice_idx + 1}</div>
            <div class='bubble-content'>{action}</div>
        </div>
        """
        self.neg_display.append(choice_bubble)

        try:
            # Pass last context for multi-turn negotiation
            payload = {
                "action": action,
                "parameters": parameters,
                "house_key": house_key,
                "query": original_query
            }
            if hasattr(self, 'last_negotiation_context') and self.last_negotiation_context:
                payload["last_context"] = self.last_negotiation_context
            resp = requests.post(
                "http://127.0.0.1:5000/llm_negotiate_action",
                json=payload,
                timeout=30
            )
            if resp.status_code == 200:
                data = resp.json()
                result = data.get('result', '')
                params_text = data.get('params', '')

                # NEW: Handle enhanced features
                if data.get('geometry_changes_needed'):
                    self.handle_geometry_changes_response(data, house_key)
                elif 'swap_candidates' in data:
                    self.handle_swap_response(data, house_key)
                elif 'booking_details' in data:
                    self.handle_booking_response(data, house_key)
                else:
                    # Standard response (assistant bubble)
                    html = f"""
                    <div class='bubble assistant-bubble'>
                        <div class='bubble-header'>Negotiation Result</div>
                        <div class='bubble-content'>{result}<br><b>Parameters:</b> {params_text}</div>
                    </div>
                    """
                    self.neg_display.append(html)

                # Key detection from result
                result_keys = extract_space_keys(result + " " + str(params_text))
                keys_str = "|".join(sorted(result_keys))

                if keys_str:
                    send_udp_command_1(keys_str, port=7000)
                    self.neg_display.append(f"<div class='bubble system-bubble'><b>Detected keys from result:</b> {keys_str}</div>")
                else:
                    send_udp_command_1("", port=7000)
                    self.neg_display.append(f"<div class='bubble system-bubble'><b>Detected keys from result:</b> None</div>")

                # Update context for further turns
                context = data.get("context", {})
                if context:
                    self.last_negotiation_context = context

                # Clear suggestions since choice has been made
                self.last_negotiation_suggestions = []
            else:
                self.neg_display.append(f"<div class='bubble system-bubble'>Server error: {resp.status_code}</div>")
        except Exception as e:
            self.neg_display.append(f"<div class='bubble system-bubble'>Error connecting to server: {e}</div>")

    # NEW: Enhanced response handlers
    def handle_geometry_changes_response(self, data, house_key):
        """Handle geometry changes needed response"""
        result = data.get('result', '')
        geometry_changes = data.get('geometry_changes', {})
        current_geometry = data.get('current_geometry', {})

        html = f"""
        <div class='bubble assistant-bubble geometry-bubble'>
            <div class='bubble-header'>Geometry Changes Required</div>
            <div class='bubble-content'>
                {result}<br><br>
                <b>Required Changes:</b><br>
                {''.join([f'• {change_type}: {change_desc}<br>' for change_type, change_desc in geometry_changes.items()])}
                <br><b>Current Geometry:</b><br>
                {''.join([f'• {key}: {value}<br>' for key, value in current_geometry.items()])}
                <br>Type 'confirm' to proceed with changes or 'cancel' to abort.
            </div>
        </div>
        """
        self.neg_display.append(html)

        # Store pending changes
        self.pending_geometry_changes = geometry_changes
        self.current_geometry_space = current_geometry

        # After displaying the geometry changes
        keys_str = "|".join(sorted(extract_space_keys(result)))
        send_udp_command_1(keys_str, port=7000)
        self.neg_display.append(f"<div class='bubble system-bubble'><b>Detected keys from geometry changes:</b> {keys_str}</div>")

    def handle_swap_response(self, data, house_key):
        """Handle apartment swap response"""
        result = data.get('result', '')
        swap_options = data.get('swap_candidates', [])

        html = f"""
        <div class='bubble assistant-bubble swap-bubble'>
            <div class='bubble-header'>Apartment Swap Options</div>
            <div class='bubble-content'>
                {result}<br><br>
                {'<b>Available Options:</b><br>' if swap_options else ''}
                {''.join([f'{i+1}. Swap with {option["target_resident"]} (near {option["target_space"]} for {option["target_activity"]}) - Distance: {option["distance_to_desired"]:.1f}m<br>' for i, option in enumerate(swap_options[:3])])}
                {('<br>Type the number of your preferred swap option to proceed.' if swap_options else '')}
            </div>
        </div>
        """
        self.neg_display.append(html)

    def handle_booking_response(self, data, house_key):
        """Handle booking response"""
        result = data.get('result', '')
        booking_details = data.get('booking_details', {})

        html = f"""
        <div class='bubble assistant-bubble booking-bubble'>
            <div class='bubble-header'>Booking Request</div>
            <div class='bubble-content'>
                {result}<br><br>
                <b>Booking Details:</b><br>
                {''.join([f'• {key}: {value}<br>' for key, value in booking_details.items()])}
                <br>Type 'confirm' to finalize booking or 'cancel' to abort.
            </div>
        </div>
        """
        self.neg_display.append(html)
    # Add chat bubble CSS for QTextBrowser
    def showEvent(self, event):
        super().showEvent(event)
        # Only inject once per QTextBrowser
        if not hasattr(self, '_chat_css_injected'):
            self.qna_display.append(self.chat_style)
            self.neg_display.append(self.chat_style)
            self._chat_css_injected = True

    # # GEOMETRY BY KEY
    # def send_show_geometry_by_key(self):
    #     key = self.geom_key_input.text().strip()
    #     if not key:
    #         self.qna_display.append("<span style='color: red;'>Please enter a key to show geometry.</span>")
    #         return
    #     try:
    #         resp = requests.post(
    #             "http://127.0.0.1:5000/show_geometry_by_key",
    #             json={"key": key},
    #             timeout=10
    #         )
    #         if resp.status_code == 200:
    #             self.qna_display.append(f"<b>Show Geometry:</b> Requested {key}")
    #         else:
    #             self.qna_display.append(f"<span style='color: red;'>Server error: {resp.status_code}</span>")
    #     except Exception as e:
    #         self.qna_display.append(f"<span style='color: red;'>Error connecting to server: {e}</span>")

    # def toggle_show_hide_geometry(self):
    #     if self.show_geom_btn.text() == "Show Geometry":
    #         self.send_show_geometry_by_key()
    #         self.show_geom_btn.setText("Hide Geometry")
    #     else:
    #         try:
    #             resp = requests.post(
    #                 "http://127.0.0.1:5000/hide_geometry_by_key",
    #                 timeout=10
    #             )
    #             if resp.status_code == 200:
    #                 self.qna_display.append("<b>Geometry Hidden</b>")
    #             else:
    #                 self.qna_display.append(f"<span style='color: red;'>Server error: {resp.status_code}</span>")
    #         except Exception as e:
    #             self.qna_display.append(f"<span style='color: red;'>Error connecting to server: {e}</span>")
    #         self.show_geom_btn.setText("Show Geometry")

    def update_detected_keys(self, user_text, answer_text):
        """
        Update and return the current set of detected space keys from user and answer.
        - user_text: string from the input field
        - answer_text: string from the assistant's answer
        """
        user_keys = extract_space_keys(user_text)
        answer_keys = extract_space_keys(answer_text)
        # If answer has new keys, use those; else, use user keys; else, clear
        if answer_keys:
            self.current_detected_keys = answer_keys
        elif user_keys:
            self.current_detected_keys = user_keys
        else:
            self.current_detected_keys = set()
        keys_str = "|".join(sorted(self.current_detected_keys))
        return keys_str

    def update_chat_display(self, browser):
        if browser == self.qna_display:
            history = self.qna_chat_history
        else:
            history = self.neg_chat_history
        full_html = self.chat_style + '<div class="chat-container">' + ''.join(history) + '</div>'
        browser.setHtml(full_html)
        browser.verticalScrollBar().setValue(browser.verticalScrollBar().maximum())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SpaceQnAUI()
    window.show()
    sys.exit(app.exec_())