from server.config import *
from llm_calls import *
from utils.rag_utils import answer_from_knowledge
from sql_main import answer_sql_question  
from question_router import route_question         
import json

from flask import Flask, request, jsonify

conversation_history = []

geometry_command = {"geometry_command": 0}

def print_answer(label, answer):
    print(f"\n[{label.upper()} RESULT]")
    print(answer)
    print("-" * 50)

def answer_general_question(user_message, conversation_history=None):
    if conversation_history is None:
        conversation_history = []
    routed_parts = route_question(user_message)
    combined_answer = ""
    for part in routed_parts:
        destination = part["destination"]
        part_text = part["text"]

        if destination == "sql":
            answer = answer_sql_question(part_text)
            combined_answer += f"\n(SQL Answer for: \"{part_text}\")\n{answer}"
        elif destination == "knowledge":
            embedding_file = part.get("embedding_file")
            answer = answer_from_knowledge(part_text, embedding_file, conversation_history)
            combined_answer += f"\n(Knowledge Answer for: \"{part_text}\")\n{answer}"
        else:
            fallback = "Sorry, I couldn't understand that part."
            combined_answer += f"\n(Error for: \"{part_text}\")\n{fallback}"
    return combined_answer.strip()

# ---- Flask API ----
app = Flask(__name__)

@app.route('/general_question', methods=['POST'])
def handle_general_question():
    print("Received request!")
    try:
        data = request.get_json()
        print("Data received:", data)
        user_message = data.get('question', '')
        conv_hist = data.get('conversation_history', [])
        answer = answer_general_question(user_message, conv_hist)
        conv_hist.append({"role": "user", "content": user_message})
        conv_hist.append({"role": "assistant", "content": answer})
        return jsonify({'response': answer, 'conversation_history': conv_hist})
    except Exception as e:
        print("Error:", e)
        return jsonify({'response': f"Server error: {e}", 'conversation_history': []}), 500

@app.route('/set_geometry', methods=['POST'])
def set_geometry():
    global geometry_command
    data = request.get_json()
    geometry_command = data
    return jsonify({"status": "ok"})

@app.route('/get_geometry', methods=['GET'])
def get_geometry():
    return jsonify(geometry_command)

if __name__ == '__main__':
    # CLI loop (optional, keep if you want both CLI and API)
    # while True:
    #     user_message = input("Ask your question: ")
    #     if user_message.lower() in ["exit", "quit"]:
    #         break

    #     conversation_history.append({"role": "user", "content": user_message})
    #     routed_parts = route_question(user_message)
    #     combined_answer = ""
    #     for part in routed_parts:
    #         destination = part["destination"]
    #         part_text = part["text"]

    #         if destination == "sql":
    #             answer = answer_sql_question(part_text)
    #             print_answer("sql", answer)
    #             combined_answer += f"\n(SQL Answer for: \"{part_text}\")\n{answer}"
            
    #         elif destination == "knowledge":
    #             embedding_file = part.get("embedding_file")
    #             answer = answer_from_knowledge(part_text, embedding_file, conversation_history)
    #             print_answer("knowledge", answer)
    #             combined_answer += f"\n(Knowledge Answer for: \"{part_text}\")\n{answer}"
            
    #         else:
    #             fallback = "Sorry, I couldn't understand that part."
    #             print_answer("error", fallback)
    #             combined_answer += f"\n(Error for: \"{part_text}\")\n{fallback}"

    #     conversation_history.append({"role": "assistant", "content": combined_answer.strip()})

    # To run Flask, comment out the CLI loop above and uncomment below:
    app.run(port=5000, debug=True)

