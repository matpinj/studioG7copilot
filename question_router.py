import json
from server.config import client, completion_model


def classify_knowledge_topic(user_message):
    prompt = (
        "You are a topic classifier for architectural knowledge. "
        "Given the user question, return ONLY the most relevant topic from this list:\n"
        "- outdoor comfort research issues\n"
        "- the rise of co-living\n"
        "- thermal comfort in semi-outdoor spaces\n"
        "If none fit, return 'outdoor comfort research issues'.\n"
        f"User question: {user_message}\n"
        "Topic:"
    )
    response = client.chat.completions.create(
        model=completion_model,
        messages=[
            {"role": "system", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=20,
    )
    topic = response.choices[0].message.content.strip().lower()
    topic_to_file = {
        "outdoor comfort research issues": "knowledge/outdoor comfort research issues.json",
        "the rise of co-living": "knowledge/the rise of co-living.json",
        "thermal comfort in semi-outdoor spaces": "knowledge/thermal comfort in semi-outdoor spaces.json"
    }
    return topic_to_file.get(topic, "knowledge/outdoor comfort research issues.json")


def route_question(user_message):
    routing_prompt = f"""
You are a smart question router for an architectural assistant.

Instructions:
- For each user question, classify whether it should be answered using:
  - "sql": for structured database queries (quantitative, factual, count, list, number, existence, or anything that can be answered directly from a database table), If the question asks for a list of categories, types, names, or entities that exist in the database, use "sql".
  - "knowledge": for architectural theory, trends, qualitative advice, explanations, or anything that requires external knowledge or reasoning, If the question asks for an explanation or description, use "knowledge".

Return only a JSON object like this:
{{
  "destination": "sql",  // or "knowledge"
  "text": "the original question"
}}

Examples:
Q: "How many residents are in the building?"
A: "sql"
Q: "List all apartments with balconies."
A: "sql"
Q: "What is the average temperature in outdoor spaces?"
A: "sql"
Q: "Explain the benefits of co-living."
A: "knowledge"
Q: "Describe the design trends for shared kitchens."
A: "knowledge"
Q: "How many activity spaces are there in level 1?"
A: "sql"
Q: "What are the most popular activities in outdoor spaces?"
A: "sql"
Q: "Why is thermal comfort important?"
A: "knowledge"
Q: "What are resident persona types?"
A: "sql"
Q: "List all resident persona types."
A: "sql"
Q: "Explain resident persona types."
A: "knowledge"
Q: "Describe resident persona types."
A: "knowledge"

User question: \"{user_message}\"
"""

    try:
        response = client.chat.completions.create(
            model=completion_model,
            messages=[
                {"role": "system", "content": routing_prompt}
            ],
            temperature=0.0,
            max_tokens=100
        )

        content = response.choices[0].message.content.strip()
        print("LLM routing output:", content)  # For debugging
        routing_data = json.loads(content)

        destination = routing_data.get("destination", "").strip().lower()
        text = routing_data.get("text", "").strip()

        if destination == "sql":
            return [{"destination": "sql", "text": text}]
        elif destination == "knowledge":
            embedding_file = classify_knowledge_topic(text)
            return [{
                "destination": "knowledge",
                "text": text,
                "embedding_file": embedding_file
            }]
        else:
            raise ValueError("Invalid destination in LLM response")

    except (json.JSONDecodeError, ValueError, Exception) as e:
        print("Routing error:", e)
        # fallback: treat whole question as knowledge
        fallback_file = classify_knowledge_topic(user_message)
        return [{
            "destination": "knowledge",
            "text": user_message,
            "embedding_file": fallback_file
        }]


# Example usage for testing
if __name__ == "__main__":
    example_question = "How many apartments have a balcony, and what are the best design strategies for balconies?"
    result = route_question(example_question)
    print(json.dumps(result, indent=2))
