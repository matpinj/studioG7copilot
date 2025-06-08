import sys
import requests
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QLabel, QLineEdit, QPushButton, QTextBrowser,
    QHBoxLayout, QComboBox, QFrame, QTextEdit, QTabWidget
)
from PyQt5.QtCore import Qt

# Global QSS stylesheet
QSS_STYLE = """
QMainWindow {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #ece9e6, stop:1 #ffffff);
}
QFrame#container {
    background-color: #ffffff;
    border-radius: 12px;
    border: 1px solid #edeff0;
    padding: 30px;
    margin: 20px;
}
#headerLabel {
    font-size: 26px;
    font-weight: bold;
    color: #222222;
    padding: 20px;
    background-color: #444444;
    color: #ffffff;
    border-top-left-radius: 12px;
    border-top-right-radius: 12px;
}
#fieldLabel {
    min-width: 80px;
    font-size: 14px;
    font-weight: 600;
    color: #444444;
}
QLineEdit#houseKeyField {
    padding: 20px;
    border: 1px solid #ccc;
    border-radius: 32px;
    background-color: #fafafa;
    font-size: 14px;
    color: #222;
    min-height: 24px;
}
QLineEdit#houseKeyField:hover, QLineEdit#inputField:hover {
    border-color: #444444;
}
QLineEdit#houseKeyField:focus, QLineEdit#inputField:focus {
    border-color: #222222;
}
QTextEdit#inputField {
    padding: 20px;
    border: 1px solid #ccc;
    border-radius: 32px;
    background-color: #fafafa;
    font-size: 14px;
    color: #222;
    min-height: 48px;
    max-height: 80px;
}
QTextEdit#inputField:focus, QTextEdit#inputField:hover {
    border-color: #222222;
}
QTextBrowser#displayArea {
    border: none;
    background-color: #f0f4fb;
    border-radius: 32px;
    padding: 20px;
    font-size: 14px;
    color: #2c3e50;
}
QPushButton#askButton {
    background: #444444;
    border: none;
    color: #fff;
    padding: 12px 24px;
    font-size: 15px;
    font-weight: bold;
    border-radius: 32px;
    min-height: 40px;
}
QPushButton#askButton:hover {
    background: #222222;
}
QPushButton#askButton:pressed {
    background-color: #111111;
    border-radius: 32px;
}
QScrollBar:vertical {
    background: transparent;
    width: 8px;
}
QScrollBar::handle:vertical {
    background: #888888;
    min-height: 20px;
    border-radius: 15px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""


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
        header.setObjectName("headerLabel")
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)

        row = QHBoxLayout()
        label = QLabel("House Key:")
        label.setObjectName("fieldLabel")
        self.house_key_input = QLineEdit()
        self.house_key_input.setObjectName("houseKeyField")
        self.house_key_input.setPlaceholderText("Enter your house key (e.g. H1)")
        row.addWidget(label)
        row.addWidget(self.house_key_input, stretch=1)
        main_layout.addLayout(row)

        self.qna_display = QTextBrowser()
        self.qna_display.setObjectName("displayArea")
        self.qna_display.setReadOnly(True)
        main_layout.addWidget(self.qna_display, stretch=1)

        input_row = QHBoxLayout()
        self.input_field = QTextEdit()
        self.input_field.setObjectName("inputField")
        self.input_field.setPlaceholderText("Type your question about nearby spaces...")
        self.input_field.setFixedHeight(56)  # About 2 lines
        self.ask_button = QPushButton("Ask")
        self.ask_button.setObjectName("askButton")
        self.ask_button.clicked.connect(self.send_qna)
        input_row.addWidget(self.input_field, stretch=1)
        input_row.addWidget(self.ask_button)
        main_layout.addLayout(input_row)

        self.tabs.addTab(qna_tab, "Nearby Space Q&A")

        # --- Tab 2: Negotiate ---
        negotiate_tab = QWidget()
        negotiate_layout = QVBoxLayout(negotiate_tab)
        negotiate_layout.setContentsMargins(0, 0, 0, 0)
        negotiate_layout.setSpacing(16)

        negotiate_container = QFrame()
        negotiate_container.setObjectName("container")
        negotiate_layout.addWidget(negotiate_container)
        neg_layout = QVBoxLayout(negotiate_container)
        neg_layout.setContentsMargins(0, 0, 0, 0)
        neg_layout.setSpacing(16)

        neg_header = QLabel("Negotiate (LLM)")
        neg_header.setObjectName("headerLabel")
        neg_header.setAlignment(Qt.AlignCenter)
        neg_layout.addWidget(neg_header)

        neg_row = QHBoxLayout()
        neg_label = QLabel("House Key:")
        neg_label.setObjectName("fieldLabel")
        self.neg_house_key_input = QLineEdit()
        self.neg_house_key_input.setObjectName("houseKeyField")
        self.neg_house_key_input.setPlaceholderText("Enter your house key (e.g. H1)")
        neg_row.addWidget(neg_label)
        neg_row.addWidget(self.neg_house_key_input, stretch=1)
        neg_layout.addLayout(neg_row)

        self.neg_input_field = QTextEdit()
        self.neg_input_field.setObjectName("inputField")
        self.neg_input_field.setPlaceholderText("Type your negotiation query...")
        self.neg_input_field.setFixedHeight(56)
        neg_layout.addWidget(self.neg_input_field)

        self.neg_ask_button = QPushButton("Negotiate")
        self.neg_ask_button.setObjectName("askButton")
        self.neg_ask_button.clicked.connect(self.send_negotiate)
        neg_layout.addWidget(self.neg_ask_button)

        self.neg_display = QTextBrowser()
        self.neg_display.setObjectName("displayArea")
        self.neg_display.setReadOnly(True)
        neg_layout.addWidget(self.neg_display, stretch=1)

        self.tabs.addTab(negotiate_tab, "Negotiate")

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
                timeout=30
            )
            if resp.status_code == 200:
                answer = resp.json().get("response", "<No response>")
                html = f'''<div style="border:1.5px solid #bbb; border-radius:32px; margin:12px 0; padding:12px; background:#fcfcfc;">
                  <b>You ({house_key}):</b> {question}<br>
                  <b>Assistant:</b> {answer}
                </div>'''
                self.qna_display.append(html)
            else:
                self.qna_display.append(
                    f"<span style='color: red;'>Server error: {resp.status_code}</span>"
                )
        except Exception as e:
            self.qna_display.append(
                f"<span style='color: red;'>Error connecting to server: {e}</span>"
            )

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
            resp = requests.post(
                "http://127.0.0.1:5000/llm_negotiate",
                json={"house_key": house_key, "query": query},
                timeout=30
            )
            if resp.status_code == 200:
                data = resp.json()
                result = data.get("result", "<No result>")
                params = data.get("params", "")
                llm_suggestion = data.get("llm_suggestion", "")
                html = f'''<div style="border:1.5px solid #bbb; border-radius:32px; margin:12px 0; padding:12px; background:#fcfcfc;">
                  <b>You ({house_key}):</b> {query}<br>
                  <b>Result:</b> {result}<br>'''
                if params:
                    html += f"<b>Params:</b> {params}<br>"
                if llm_suggestion:
                    html += f"<b>LLM Suggestion:</b> {llm_suggestion}"
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS_STYLE)
    window = SpaceQnAUI()
    window.show()
    sys.exit(app.exec_())
