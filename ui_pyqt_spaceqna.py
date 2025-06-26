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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nearby Space QnA (LLM)")
        self.setGeometry(200, 200, 800, 800)

        # Tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # --- Tab 1: Q&A ---
        qna_tab = QWidget()
        qna_layout = QVBoxLayout(qna_tab)
        qna_layout.setContentsMargins(0, 0, 0, 0)
        qna_layout.setSpacing(16)

        container = QFrame()
        container.setObjectName("container")
        qna_layout.addWidget(container)
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(16)

        header = QLabel("Nearby Space Q&A")
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)

        row = QHBoxLayout()
        label = QLabel("Resident Key:")
        self.house_key_input = QLineEdit()
        self.house_key_input.setPlaceholderText("Enter your resident key (e.g. H1)")
        row.addWidget(label)
        row.addWidget(self.house_key_input, stretch=1)
        main_layout.addLayout(row)

        self.qna_display = QTextBrowser()
        self.qna_display.setReadOnly(True)
        self.qna_display.setStyleSheet("background-color: #e0e0e0; color: #222; border-radius: 16px; padding: 12px;")
        main_layout.addWidget(self.qna_display, stretch=1)

        input_row = QHBoxLayout()
        self.input_field = EnterTextEdit()
        self.input_field.setPlaceholderText("Type your question about nearby spaces... (Press Enter to send, Shift+Enter for new line)")
        self.input_field.setFixedHeight(56)
        self.input_field.setStyleSheet("background-color: #fff; color: #222; border-radius: 16px; padding: 12px;")
        self.input_field.enterPressed.connect(self.send_qna)
        
        self.ask_button = QPushButton("Ask")
        self.ask_button.clicked.connect(self.send_qna)
        
        input_row.addWidget(self.input_field, stretch=1)
        input_row.addWidget(self.ask_button)
        main_layout.addLayout(input_row)

        # --- Show Geometry by Key ---
        geom_row = QHBoxLayout()
        geom_label = QLabel("Show Geometry Key:")
        self.geom_key_input = QLineEdit()
        self.geom_key_input.setPlaceholderText("Enter space or resident key (e.g. O1, H10)")
        self.show_geom_btn = QPushButton("Show Geometry")
        self.show_geom_btn.clicked.connect(self.toggle_show_hide_geometry)
        geom_row.addWidget(geom_label)
        geom_row.addWidget(self.geom_key_input, stretch=1)
        geom_row.addWidget(self.show_geom_btn)
        main_layout.addLayout(geom_row)

        self.tabs.addTab(qna_tab, "Nearby Space Q&A")

        # --- Tab 2: Negotiate ---
        negotiate_tab = QWidget()
        negotiate_layout = QVBoxLayout(negotiate_tab)
        negotiate_layout.setContentsMargins(0, 0, 0, 0)
        negotiate_layout.setSpacing(16)

        negotiate_container = QFrame()
        negotiate_layout.addWidget(negotiate_container)
        neg_layout = QVBoxLayout(negotiate_container)
        neg_layout.setContentsMargins(0, 0, 0, 0)
        neg_layout.setSpacing(16)

        neg_header = QLabel("Negotiate (LLM)")
        neg_header.setAlignment(Qt.AlignCenter)
        neg_layout.addWidget(neg_header)

        neg_row = QHBoxLayout()
        neg_label = QLabel("Resident Key:")
        self.neg_house_key_input = QLineEdit()
        self.neg_house_key_input.setPlaceholderText("Enter your resident key (e.g. H1)")
        neg_row.addWidget(neg_label)
        neg_row.addWidget(self.neg_house_key_input, stretch=1)
        neg_layout.addLayout(neg_row)
        
        # Create input field with Enter support
        self.neg_input_field = EnterTextEdit()
        self.neg_input_field.setPlaceholderText("Type your negotiation query... (Press Enter to send, Shift+Enter for new line)")
        self.neg_input_field.setFixedHeight(56)
        self.neg_input_field.setStyleSheet("background-color: #fff; color: #222; border-radius: 16px; padding: 12px;")
        self.neg_input_field.enterPressed.connect(self.send_negotiate)
        neg_layout.addWidget(self.neg_input_field)

        self.neg_ask_button = QPushButton("Negotiate")
        self.neg_ask_button.clicked.connect(self.send_negotiate)
        neg_layout.addWidget(self.neg_ask_button)

        self.neg_display = QTextBrowser()
        self.neg_display.setReadOnly(True)
        self.neg_display.setStyleSheet("background-color: #e0e0e0; color: #222; border-radius: 16px; padding: 12px;")
        neg_layout.addWidget(self.neg_display, stretch=1)

        # Store last suggestions and context for multi-turn negotiation
        self.last_negotiation_suggestions = []
        self.last_negotiation_context = {}
        self.current_detected_keys = set()  # Initialize the detected keys set

        # NEW: Add these for enhanced features
        self.pending_geometry_changes = {}
        self.current_geometry_space = None

        self.tabs.addTab(negotiate_tab, "Negotiate")

    # NEARBY SPACE Q&A
    def send_qna(self):
        house_key = self.house_key_input.text().strip()
        question = self.input_field.toPlainText().strip()
        if not house_key or not question:
            self.qna_display.append(
                "<span style='color: red;'>Please enter a resident key and a question.</span>"
            )
            return

        self.input_field.clear()
        try:
            resp = requests.post(
                "http://127.0.0.1:5000/llm_nearby_space_qna",
                json={"house_key": house_key, "question": question},
                timeout=30
            )
            if resp.status_code == 200:
                answer = resp.json().get("response", "<No response>")
                html = f'''<div style="border:1.5px solid #bbb; border-radius:32px; margin:12px 0; padding:12px; background:#fcfcfc;">
                  <b>You ({house_key}):</b> {question}<br>
                  <b>Assistant:</b> {answer}
                </div>'''
                self.qna_display.append(html)
                # After you get the answer (or even before sending, if you want to show user keys)
                keys_str = self.update_detected_keys(question, answer)
                send_udp_command_1(keys_str, port=7000)
                self.qna_display.append(f"<b>Detected keys:</b> {keys_str}")
            else:
                self.qna_display.append(
                    f"<span style='color: red;'>Server error: {resp.status_code}</span>"
                )
        except Exception as e:
            self.qna_display.append(
                f"<span style='color: red;'>Error connecting to server: {e}</span>"
            )

    # NEGOTIATE
    def send_negotiate(self):
        # Clear any previous keys first
        send_udp_command_1("", port=7000)
        
        house_key = self.neg_house_key_input.text().strip()
        query = self.neg_input_field.toPlainText().strip()
        if not house_key or not query:
            self.neg_display.append(
                "<span style='color: red;'>Please enter a resident key and a negotiation query.</span>"
            )
            return
        
        # Check if this is a choice confirmation (number or "yes")
        if self.last_negotiation_suggestions and (query.isdigit() or query.lower() in ['yes', 'y']):
            if query.lower() in ['yes', 'y']:
                # If user says "yes", assume they want the first suggestion
                choice_idx = 0
            else:
                try:
                    choice_idx = int(query) - 1
                except:
                    self.neg_display.append("<span style='color: red;'>Please enter a valid number for your choice.</span>")
                    self.neg_input_field.clear()
                    return
            
            if choice_idx < 0 or choice_idx >= len(self.last_negotiation_suggestions):
                self.neg_display.append("<span style='color: red;'>Choice out of range. Please enter a valid number.</span>")
                self.neg_input_field.clear()
                return
            
            # Execute the chosen suggestion
            self.execute_negotiation_choice(choice_idx, house_key, query)
            self.neg_input_field.clear()
            return
        
        # This is a new negotiation query
        self.neg_input_field.clear()
        try:
            # Pass last context for multi-turn negotiation
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
                html = f'''<div style="border:1.5px solid #bbb; border-radius:32px; margin:12px 0; padding:12px; background:#fcfcfc;">
                  <b>You ({house_key}):</b> {query}<br>'''
                if context:
                    html += f"<b>Nearby Activities:</b> {context.get('nearby_activities', '')}<br>"
                    html += f"<b>Preferences:</b> {context.get('preferences', '')}<br>"
                if suggestions:
                    html += "<b>Suggestions:</b><ol>"
                    for i, s in enumerate(suggestions):
                        html += f"<li><b>{s.get('action')}</b>: {s.get('explanation')}<br>Params: {s.get('parameters')}</li>"
                    html += "</ol>"
                    html += "<b>Choose your option: Type the number (1, 2, 3...) or 'yes' for option 1</b>"
                else:
                    html += "<b>No suggestions available.</b>"
                html += "</div>"
                self.neg_display.append(html)
                
                # --- Detected keys logic for Negotiate tab ---
                # Collect all suggestion texts and parameters as a single string
                suggestion_texts = " ".join(
                    f"{s.get('action','')} {s.get('explanation','')} {s.get('parameters','')}" for s in suggestions
                )
                # Also include context text for key detection
                context_text = f"{context.get('nearby_activities', '')} {context.get('preferences', '')}"
                all_response_text = f"{suggestion_texts} {context_text}"
                
                # Extract keys from both query and all response texts
                keys_str = self.update_detected_keys(query, all_response_text)
                if keys_str:
                    send_udp_command_1(keys_str, port=7000)
                    self.neg_display.append(f"<b>Detected keys:</b> {keys_str}")
                else:
                    send_udp_command_1("", port=7000)  # Send empty string to clear
                    self.neg_display.append(f"<b>Detected keys:</b> None")
            else:
                self.neg_display.append(
                    f"<span style='color: red;'>Server error: {resp.status_code}</span>"
                )
        except Exception as e:
            self.neg_display.append(
                f"<span style='color: red;'>Error connecting to server: {e}</span>"
            )

    def execute_negotiation_choice(self, choice_idx, house_key, original_query):
        """Execute the selected negotiation choice - UPDATED VERSION"""
        suggestion = self.last_negotiation_suggestions[choice_idx]
        action = suggestion.get('action')
        parameters = suggestion.get('parameters', {})
        
        # Show what choice was made
        choice_html = f'''<div style="border:1.5px solid #4CAF50; border-radius:32px; margin:12px 0; padding:12px; background:#f0f8f0;">
          <b>You chose option {choice_idx + 1}:</b> {action}
        </div>'''
        self.neg_display.append(choice_html)
        
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
                    # Standard response (your existing code)
                    html = f'''<div style="border:1.5px solid #bbb; border-radius:32px; margin:12px 0; padding:12px; background:#f6fff6;">
                      <b>Negotiation Result:</b> {result}<br>
                      <b>Parameters:</b> {params_text}
                    </div>'''
                    self.neg_display.append(html)
                
                # Key detection from result
                result_keys = extract_space_keys(result + " " + str(params_text))
                keys_str = "|".join(sorted(result_keys))

                if keys_str:
                    send_udp_command_1(keys_str, port=7000)
                    self.neg_display.append(f"<b>Detected keys from result:</b> {keys_str}")
                else:
                    send_udp_command_1("", port=7000)
                    self.neg_display.append(f"<b>Detected keys from result:</b> None")
            
                # Update context for further turns
                context = data.get("context", {})
                if context:
                    self.last_negotiation_context = context
                    
                # Clear suggestions since choice has been made
                self.last_negotiation_suggestions = []
            else:
                self.neg_display.append(f"<span style='color: red;'>Server error: {resp.status_code}</span>")
        except Exception as e:
            self.neg_display.append(f"<span style='color: red;'>Error connecting to server: {e}</span>")

    # NEW: Enhanced response handlers
    def handle_geometry_changes_response(self, data, house_key):
        """Handle geometry changes needed response"""
        result = data.get('result', '')
        geometry_changes = data.get('geometry_changes', {})
        current_geometry = data.get('current_geometry', {})
        
        html = f'''<div style="border:1.5px solid #FF9800; border-radius:32px; margin:12px 0; padding:12px; background:#fff3e0;">
          <b>Geometry Changes Required:</b><br>
          {result}<br><br>
          <b>Required Changes:</b><br>'''
        
        for change_type, change_desc in geometry_changes.items():
            html += f"• {change_type}: {change_desc}<br>"
        
        html += f'''<br><b>Current Geometry:</b><br>'''
        for key, value in current_geometry.items():
            html += f"• {key}: {value}<br>"
        
        html += f'''<br>Type 'confirm' to proceed with changes or 'cancel' to abort.
        </div>'''
        
        self.neg_display.append(html)
        
        # Store pending changes
        self.pending_geometry_changes = geometry_changes
        self.current_geometry_space = current_geometry

        # After displaying the geometry changes
        keys_str = "|".join(sorted(extract_space_keys(result)))
        send_udp_command_1(keys_str, port=7000)
        self.neg_display.append(f"<b>Detected keys from geometry changes:</b> {keys_str}")

    def handle_swap_response(self, data, house_key):
        """Handle apartment swap response"""
        result = data.get('result', '')
        swap_options = data.get('swap_candidates', [])
        
        html = f'''<div style="border:1.5px solid #2196F3; border-radius:32px; margin:12px 0; padding:12px; background:#e3f2fd;">
          <b>Apartment Swap Options:</b><br>
          {result}<br><br>'''
        
        if swap_options:
            html += f'''<b>Available Options:</b><br>'''
            for i, option in enumerate(swap_options[:3], 1):
                html += f'''{i}. Swap with {option["target_resident"]} 
                           (near {option["target_space"]} for {option["target_activity"]})
                           - Distance: {option["distance_to_desired"]:.1f}m<br>'''
            
            html += f'''<br>Type the number of your preferred swap option to proceed.'''
        
        html += f'''</div>'''
        self.neg_display.append(html)

    def handle_booking_response(self, data, house_key):
        """Handle booking response"""
        result = data.get('result', '')
        booking_details = data.get('booking_details', {})
        
        html = f'''<div style="border:1.5px solid #4CAF50; border-radius:32px; margin:12px 0; padding:12px; background:#e8f5e8;">
          <b>Booking Request:</b><br>
          {result}<br><br>
          <b>Booking Details:</b><br>'''
        
        for key, value in booking_details.items():
            html += f"• {key}: {value}<br>"
        
        html += f'''<br>Type 'confirm' to finalize booking or 'cancel' to abort.
        </div>'''
        
        self.neg_display.append(html)

    # GEOMETRY BY KEY
    def send_show_geometry_by_key(self):
        key = self.geom_key_input.text().strip()
        if not key:
            self.qna_display.append("<span style='color: red;'>Please enter a key to show geometry.</span>")
            return
        try:
            resp = requests.post(
                "http://127.0.0.1:5000/show_geometry_by_key",
                json={"key": key},
                timeout=10
            )
            if resp.status_code == 200:
                self.qna_display.append(f"<b>Show Geometry:</b> Requested {key}")
            else:
                self.qna_display.append(f"<span style='color: red;'>Server error: {resp.status_code}</span>")
        except Exception as e:
            self.qna_display.append(f"<span style='color: red;'>Error connecting to server: {e}</span>")

    def toggle_show_hide_geometry(self):
        if self.show_geom_btn.text() == "Show Geometry":
            self.send_show_geometry_by_key()
            self.show_geom_btn.setText("Hide Geometry")
        else:
            try:
                resp = requests.post(
                    "http://127.0.0.1:5000/hide_geometry_by_key",
                    timeout=10
                )
                if resp.status_code == 200:
                    self.qna_display.append("<b>Geometry Hidden</b>")
                else:
                    self.qna_display.append(f"<span style='color: red;'>Server error: {resp.status_code}</span>")
            except Exception as e:
                self.qna_display.append(f"<span style='color: red;'>Error connecting to server: {e}</span>")
            self.show_geom_btn.setText("Show Geometry")

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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SpaceQnAUI()
    window.show()
    sys.exit(app.exec_())