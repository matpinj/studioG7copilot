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
    closestOutdoorFound = pyqtSignal(str)  # Add this line

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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SpaceQnAUI()
    window.show()
    sys.exit(app.exec_())