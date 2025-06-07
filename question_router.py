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
- Split the user question into smaller parts if it includes multiple intents.
- For each part, classify whether it should be answered using:
  - "sql": for structured database queries (quantitative, factual)
  - "knowledge": for architectural theory, trends, or qualitative advice

Return only a JSON object like this:
{{
  "parts": [
    {{ "text": "first part of the question", "destination": "sql" }},
    {{ "text": "second part", "destination": "knowledge" }}
  ]
}}

⚠️ Only split real parts of the user's question. Do NOT make up content.
⚠️ Use lowercase values: "sql" or "knowledge" only.

User question: \"{user_message}\"
"""

    try:
        response = client.chat.completions.create(
            model=completion_model,
            messages=[
                {"role": "system", "content": routing_prompt}
            ],
            temperature=0.0,
            max_tokens=300
        )

        content = response.choices[0].message.content.strip()
        routing_data = json.loads(content)

        if "parts" not in routing_data:
            raise ValueError("Missing 'parts' in LLM response")

        results = []
        for part in routing_data.get("parts", []):
            text = part.get("text", "").strip()
            destination = part.get("destination", "").strip().lower()

            if destination == "sql":
                results.append({"destination": "sql", "text": text})

            elif destination == "knowledge":
                embedding_file = classify_knowledge_topic(text)
                results.append({
                    "destination": "knowledge",
                    "text": text,
                    "embedding_file": embedding_file
                })

        return results  # list of parts with routing info

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
