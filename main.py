from server.config import *
from llm_calls import *
from utils.rag_utils import answer_from_knowledge
from sql_main import answer_sql_question  
from question_router import route_question         
import json

conversation_history = []

def print_answer(label, answer):
    print(f"\n[{label.upper()} RESULT]")
    print(answer)
    print("-" * 50)

while True:
    user_message = input("Ask your question: ")
    if user_message.lower() in ["exit", "quit"]:
        break

    # Add user message to conversation history
    conversation_history.append({"role": "user", "content": user_message})

    # Route to parts: can be multiple (sql + knowledge)
    routed_parts = route_question(user_message)

    combined_answer = ""
    for part in routed_parts:
        destination = part["destination"]
        part_text = part["text"]

        if destination == "sql":
            answer = answer_sql_question(part_text)
            print_answer("sql", answer)
            combined_answer += f"\n(SQL Answer for: \"{part_text}\")\n{answer}"
        
        elif destination == "knowledge":
            embedding_file = part.get("embedding_file")
            answer = answer_from_knowledge(part_text, embedding_file, conversation_history)
            print_answer("knowledge", answer)
            combined_answer += f"\n(Knowledge Answer for: \"{part_text}\")\n{answer}"
        
        else:
            fallback = "Sorry, I couldn't understand that part."
            print_answer("error", fallback)
            combined_answer += f"\n(Error for: \"{part_text}\")\n{fallback}"

    # Save the full assistant reply with labeled subanswers
    conversation_history.append({"role": "assistant", "content": combined_answer.strip()})


# ### EXAMPLE 1: Router #"##
# # Classify the user message to see if we should answer or not
# router_output = classify_input(user_message)
# if router_output == "Refuse to answer":
#     llm_answer = "Sorry, I can only answer questions about architecture."

# else:
#     print(router_output)
#     ### EXAMPLE 2: Simple call ###
#     # simple call to LLM, try different sys prompt flavours
#     brainstorm = generate_concept(user_message)
#     print(brainstorm)

#     ### EXAMPLE 4: Structured Output ###
#     # extract the architecture attributes from the user
#     # parse a structured output with regex
#     attributes = extract_attributes(brainstorm)
#     print(attributes)

#     attributes = attributes.strip()
#     attributes = json.loads(attributes)
#     shape, theme, materials = (attributes[k] for k in ("shape", "theme", "materials"))

#     ### EXAMPLE 3: Chaining ###
#     brutalist_question = create_question(theme)
#     print(brutalist_question)
#     # call llm with the output of a previous call

#     ### EXAMPLE 5: RAG ####
#     # Get a response based on the knowledge found
#     rag_result= rag_call(brutalist_question, embeddings = "knowledge/brutalism_embeddings.json", n_results = 10)
#     print(rag_result)