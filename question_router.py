def route_question(user_message):
    q = user_message.lower()
    if "co-living" in q or "coliving" in q:
        return "knowledge", "knowledge/the rise of co-living.json"
    if "climate" in q:
        return "knowledge", "knowledge/thermal comfort in semi-outdoor spaces.json"
    sql_keywords = ["what are","how many", "list", "count", "which", "find", "select", "where"]
    if any(kw in q for kw in sql_keywords):
        return "sql", None
    return "knowledge", "knowledge/outdoor comfort research issues.json"
