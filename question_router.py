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
    q = user_message.lower()

    # SQL keywords
    sql_keywords = [
        "how many", "list", "count", "which", "find", "select", "where", "number of", "total", "show",
        "bedroom", "units", "residents", "activity space", "level", "apartment", "household", "population",
        "orientation", "area", "privacy", "usability", "green_suitability", "compactness", "core distance"
    ]
    if any(kw in q for kw in sql_keywords):
        return "sql", None

    # Otherwise, use LLM to classify knowledge topic
    embedding_file = classify_knowledge_topic(user_message)
    return "knowledge", embedding_file
