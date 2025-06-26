import sys
import requests
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QLabel, QLineEdit, QPushButton, QTextBrowser,
    QHBoxLayout, QComboBox, QFrame, QTextEdit, QTabWidget
)
from PyQt5.QtCore import (Qt,pyqtSignal)
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


import pandas as pd

def get_closest_outdoor_space(resident_key, detected_keys):
    outdoor_keys = [k for k in detected_keys if k.startswith('O')]
    if not outdoor_keys:
        return None, None

    df = pd.read_csv("resident_data/resident_distances_all.csv")
    df['Source Node'] = df['Source Node'].astype(str).str.strip()
    resident_key = str(resident_key).strip()

    min_dist = float('inf')
    closest_space = None

    for okey in outdoor_keys:
        row = df[df['Source Node'] == okey]
        if not row.empty and resident_key in row.columns:
            try:
                dist = float(row.iloc[0][resident_key])
                if dist < min_dist:
                    min_dist = dist
                    closest_space = okey
            except Exception:
                continue

    if closest_space is not None:
        return closest_space, min_dist
    else:
        return None, None

class SpaceQnAUI(QMainWindow):
    closestOutdoorFound = pyqtSignal(str)  # Add this line
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
        label = QLabel("House Key:")
        self.house_key_input = QLineEdit()
        self.house_key_input.setPlaceholderText("Enter your house key (e.g. H1)")
        row.addWidget(label)
        row.addWidget(self.house_key_input, stretch=1)
        main_layout.addLayout(row)

        self.qna_display = QTextBrowser()
        self.qna_display.setReadOnly(True)
        self.qna_display.setStyleSheet("background-color: #e0e0e0; color: #222; border-radius: 16px; padding: 12px;")
        main_layout.addWidget(self.qna_display, stretch=1)

        input_row = QHBoxLayout()
        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText("Type your question about nearby spaces...")
        self.input_field.setFixedHeight(56)  # About 2 lines
        self.input_field.setStyleSheet("background-color: #fff; color: #222; border-radius: 16px; padding: 12px;")
        self.ask_button = QPushButton("Ask")
        self.ask_button.clicked.connect(self.send_qna)
        input_row.addWidget(self.input_field, stretch=1)
        input_row.addWidget(self.ask_button)
        main_layout.addLayout(input_row)

        # --- Show Geometry by Key ---
        geom_row = QHBoxLayout()
        geom_label = QLabel("Show Geometry Key:")
        self.geom_key_input = QLineEdit()
        self.geom_key_input.setPlaceholderText("Enter space or house key (e.g. O1, H10)")
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
        neg_label = QLabel("House Key:")
        self.neg_house_key_input = QLineEdit()
        self.neg_house_key_input.setPlaceholderText("Enter your house key (e.g. H1)")
        neg_row.addWidget(neg_label)
        neg_row.addWidget(self.neg_house_key_input, stretch=1)
        neg_layout.addLayout(neg_row)

        self.neg_input_field = QTextEdit()
        self.neg_input_field.setPlaceholderText("Type your negotiation query...")
        self.neg_input_field.setFixedHeight(56)
        self.neg_input_field.setStyleSheet("background-color: #fff; color: #222; border-radius: 16px; padding: 12px;")
        neg_layout.addWidget(self.neg_input_field)

        self.neg_ask_button = QPushButton("Negotiate")
        self.neg_ask_button.clicked.connect(self.send_negotiate)
        neg_layout.addWidget(self.neg_ask_button)

        self.neg_display = QTextBrowser()
        self.neg_display.setReadOnly(True)
        self.neg_display.setStyleSheet("background-color: #e0e0e0; color: #222; border-radius: 16px; padding: 12px;")
        neg_layout.addWidget(self.neg_display, stretch=1)

        # Suggestion choice input and button
        self.neg_choice_input = QLineEdit()
        self.neg_choice_input.setPlaceholderText("Enter suggestion number (e.g. 1)")
        self.neg_confirm_button = QPushButton("Confirm Choice")
        self.neg_confirm_button.clicked.connect(self.confirm_negotiation_choice)
        neg_layout.addWidget(self.neg_choice_input)
        neg_layout.addWidget(self.neg_confirm_button)

        # Store last suggestions and context for multi-turn negotiation
        self.last_negotiation_suggestions = []
        self.last_negotiation_context = {}

        self.tabs.addTab(negotiate_tab, "Negotiate")
#NEARBY SPACE Q&A
    def send_qna(self):
        house_key = self.house_key_input.text().strip()
        question = self.input_field.toPlainText().strip()
        if not house_key or not question:
            self.qna_display.append(
                "<span style='color: red;'>Please enter a house key and a question.</span>"
            )
            return

        self.input_field.clear()
        try:
            resp = requests.post(
                "http://127.0.0.1:5000/llm_nearby_space_qna",
                json={"house_key": house_key, "question": question},
                timeout=200
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
                detected_keys = list(self.current_detected_keys) if hasattr(self, "current_detected_keys") else []
                house_key = self.house_key_input.text().strip()
                closest_space, distance = get_closest_outdoor_space(house_key, detected_keys)
                if closest_space:
                    self.qna_display.append(f"<b>Closest detected outdoor space:</b> {closest_space} ({distance:.1f}m away)")
                    self.closestOutdoorFound.emit(closest_space)
                else:
                    self.qna_display.append("<b>No detected outdoor spaces found in your keys.</b>")
            else:
                self.qna_display.append(
                    f"<span style='color: red;'>Server error: {resp.status_code}</span>"
                )
        except Exception as e:
            self.qna_display.append(
                f"<span style='color: red;'>Error connecting to server: {e}</span>"
            )

#NEGOIATE
    def send_negotiate(self):
        house_key = self.neg_house_key_input.text().strip()
        query = self.neg_input_field.toPlainText().strip()
        if not house_key or not query:
            self.neg_display.append(
                "<span style='color: red;'>Please enter a house key and a negotiation query.</span>"
            )
            return
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
                    html += "<b>How would you like to proceed? Enter the number of your choice below and click 'Confirm Choice'.</b>"
                else:
                    html += "<b>No suggestions available.</b>"
                html += "</div>"
                self.neg_display.append(html)
            else:
                self.neg_display.append(
                    f"<span style='color: red;'>Server error: {resp.status_code}</span>"
                )
        except Exception as e:
            self.neg_display.append(
                f"<span style='color: red;'>Error connecting to server: {e}</span>"
            )

    def confirm_negotiation_choice(self):
        house_key = self.neg_house_key_input.text().strip()
        query = self.neg_input_field.toPlainText().strip()  # Optionally pass again
        choice_text = self.neg_choice_input.text().strip()
        if not self.last_negotiation_suggestions:
            self.neg_display.append("<span style='color: red;'>No suggestions to choose from. Please negotiate first.</span>")
            return
        try:
            idx = int(choice_text) - 1
        except Exception:
            self.neg_display.append("<span style='color: red;'>Please enter a valid number for your choice.</span>")
            return
        if idx < 0 or idx >= len(self.last_negotiation_suggestions):
            self.neg_display.append("<span style='color: red;'>Choice out of range. Please enter a valid number.</span>")
            return
        suggestion = self.last_negotiation_suggestions[idx]
        action = suggestion.get('action')
        parameters = suggestion.get('parameters', {})
        try:
            # Pass last context for multi-turn negotiation
            payload = {
                "action": action,
                "parameters": parameters,
                "house_key": house_key,
                "query": query
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
                html = f'''<div style="border:1.5px solid #bbb; border-radius:32px; margin:12px 0; padding:12px; background:#f6fff6;">
                  <b>Negotiation Result:</b> {result}<br>
                  <b>Parameters:</b> {params_text}
                </div>'''
                self.neg_display.append(html)
                # Update context for further turns
                context = data.get("context", {})
                if context:
                    self.last_negotiation_context = context
            else:
                self.neg_display.append(f"<span style='color: red;'>Server error: {resp.status_code}</span>")
        except Exception as e:
            self.neg_display.append(f"<span style='color: red;'>Error connecting to server: {e}</span>")
#GEOMETRY BY KEY
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
        # If answer has new keys, use those; else, keep user keys
        if answer_keys:
            self.current_detected_keys = answer_keys
        elif user_keys:
            self.current_detected_keys = user_keys
        # If neither, keep previous keys (if any)
        keys_str = "|".join(sorted(self.current_detected_keys)) if hasattr(self, "current_detected_keys") else ""
        return keys_str

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SpaceQnAUI()
    window.show()
    sys.exit(app.exec_())
