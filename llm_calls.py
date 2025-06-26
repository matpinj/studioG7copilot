from server.config import *
import re
import json # Added for json.dumps
import pandas as pd
##test

# Create a SQL query from user question
def generate_sql_query(dB_context: str, retrieved_descriptions: str, user_question: str) -> str:
    response = client.chat.completions.create(
        model=completion_model,
        messages=[
            {
                "role": "system",
                "content":
                       f"""
                You are a SQLite expert.
                The database contains multiple tables, each corresponding to a different aspect of building information. 
                There are 6 tables. Each table row represents an individual instance of a building information, about its spaces and residents.

                # Context Information #
                ## Database Schema: ## {dB_context}
                ## Table Descriptions: ## {retrieved_descriptions}

                # Instructions #
                ## Reasoning Steps: ##
                - Carefully analyze the users question.
                - Cross-reference the question with the provided database schema and table descriptions.
                - Think about which data a query to the database should fetch. Only data related to the question should be fetched.
                - Pay special atenttion to the names of the tables and properties of the schema. Your query must use keywords that match perfectly.
                - Create a valid and relevant SQL query, using only the table names and properties that are present in the schema.

                ## Output Format: ##
                - Output only the SQL query.
                - Do not use formatting characters like '```sql' or other extra text.
                - If the database doesnt have enough information to answer the question, simply output "No information".
                """
            },
            {
                "role": "user",
                "content": f"# User question # {user_question}",
            },
        ],
    )
    return response.choices[0].message.content

# Create a natural language response out of the SQL query and result
def build_answer(sql_query: str, sql_result: str, user_question: str) -> str:
    response = client.chat.completions.create(
        model=completion_model,
        messages=[
            {
                "role": "system",
                "content":
                       f"""
                        You have to answer a user question according to the SQL query and its result. Your goal is to answer in a concise and informative way, specifying the properties and tables that were relevant to create the answer.
                       
                        ### EXAMPLE ###
                        User Question: What is total list of activities on level 1?  
                        SQL Query: SELECT activity_space, from column levels only rows containing 1; same rows for activity column.
                        SQL Result: [(Flexible Space, Creative Corridor,  Storage & Technical Space,Sitting,  Sunbath,  Healing Garden,  Sports,  Flexible Space,  Urban Agriculture Garden )]  
                        Answer: I looked at the activity_space property of level 1 and found that activites are: Flexible Space, Creative Corridor,  Storage & Technical Space,Sitting,  Sunbath,  Healing Garden,  Sports,  Flexible Space,  Urban Agriculture Garden.
                """,
            },
            {
                "role": "user",
                "content": f""" 
                User question: {user_question}
                SQL Query: {sql_query}
                SQL Result: {sql_result}
                Answer:
                """,
            },
        ],
    )
    return response.choices[0].message.content

def classify_input(message):
    response = client.chat.completions.create(
        model=completion_model,
        messages=[
            {
                "role": "system",
                "content": """
                        Your task is to classify if the user message is related to buildings and architecture or not.
                        Output only the classification string.
                        If it is related, output "True", if not, output "False".

                        # Example #
                        User message: "How do I bake cookies?"
                        Output: "False"

                        User message: "What is the tallest skyscrapper in the world?"
                        Output: "True"
                        """,
            },
            {
                "role": "user",
                "content": f"""
                        {message}
                        """,
            },
        ],
    )
    return response.choices[0].message.content



# Fix an SQL query that has failed
def fix_sql_query(dB_context: str, user_question: str, atempted_queries: str, exceptions: str) -> str:

    attemptted_entries = []
    for query, exception in zip(atempted_queries, exceptions):
        attemptted_entries.append(f"#Previously attempted query#:{query}. #SQL Exception error#:{exception}")

    queries_exceptions_content = "\n".join(attemptted_entries)

    response = client.chat.completions.create(
        model=completion_model,
        messages=[
            {
                "role": "system",
                "content":
                       f"""
                You are an SQL database expert tasked with correcting a SQL query. A previous attempt to run a query
                did not yield the correct results, either due to errors in execution or because the result returned was empty
                or unexpected. Your role is to analyze the error based on the provided database schema and the details of
                the failed execution, and then provide a corrected version of the SQL query.
                The new query should provide an answer to the question! Dont create queries that do not relate to the question!
                Pay special atenttion to the names of the tables and properties. Your query must use keywords that match perfectly.

                # Context Information #
                - The database contains multiple tables, each corresponding to a different building element type. 
                - Each table row represents an individual instance of a building element of that type.
                ## Database Schema: ## {dB_context}

                # Instructions #
                1. Write down in steps why the sql queries might be failling and what could be changed to avoid it. Answer this questions:
                    I. Is the table being fetched the most apropriate to the user question, or could there be another table that might be more suitable?
                    II. Could there be another property in the schema of database for that table that could provide the right answer?
                2. Given your reasoning, write a new query taking into account the various # Failed queries and exceptions # tried before.
                2. Never output the exact same query. You should try something new given the schema of the database.
                3. Your output should come in this format: #Reasoning#: your reasoning. #NEW QUERY#: the new query.
                
                Do not use formatting characters, write only the query string.
                No other text after the query. Do not invent table names or properties. Use only the ones shown to you in the schema.
                """,
            },
            {
                "role": "user",
                "content": f""" 
                #User question#
                {user_question}
                #Failed queries and exceptions#
                {queries_exceptions_content}
                """,
            },
        ],
    )
    
    response_content = response.choices[0].message.content
    #print(response_content)
    match = re.search(r'#NEW QUERY#:(.*)', response_content)
    if match:
        return match.group(1).strip()
    else:
        return None



def suggest_geometric_variations(
    space_id: str, 
    resident_persona: str, 
    space_context: str, 
    distance_to_space: str,
    activity_weights_for_resident: str,
    activity_logic_context: str,
    current_activity_in_space: str,
    user_question_for_suggestion: str,
    desired_activity_for_space: str,
    other_residents_summary: str,
    full_db_context: str  # <-- Add this
) -> str:
    # ...existing code...
    response = client.chat.completions.create(
    model=completion_model,
    temperature=0.5,
    top_p=0.9,
    messages=[
        # ───────────── SYSTEM ─────────────
        {
            "role": "system",
            "content": """
You are an expert architectural-design assistant.

DATA SOURCES
• gh_data_for_geometry.db           (spaces & residents)
• llm_activity_assignments.csv      (current assignments)
• voting_weights.csv                (preferences)
Use nothing else.  Never write single quotes.  Output a bare JSON object.

GOAL
Choose exactly one action → Extend Slab | Add Wall | Add Louvres
Return one suggestion that best matches the space + desired activity.

CONSTRAINTS
Extend Slab  → slab only · sun/wind-optimised direction · area ≤ min(5 m², 50 %, slab_limit)
Add Wall    → low wall / parapet only · privacy / edge / seat · do not claim it “creates” an activity
Add Louvres → louvres only · height 0.2-0.8 m
If desired activity = Sunbath and no action yet → prefer Add Wall for wind/privacy.

DISALLOW
• Add Wall for open/green uses (Healing Garden, Viewpoint, Urban Agri, Biodiversity Balcony) unless wind/privacy/safety justify.
• Extend Slab for small passive spaces (<4 m²) such as Offline Retreat, Creative Corridor, Flexible Space.
• Add Louvres for sun-hungry uses (Sunbath, Healing Garden, Pool/BBQ, Urban Agri) unless shading clearly wins.

REASONING (include inside JSON)
Base decisions on Orientation, Area, Privacy, Open Sides, Usability, Green Suitability, Indoor adjacency.
Use numbers—area Δ, UTCI shift, % suitability, element sizes, neighbour distances, profile data.
Explain who else benefits and why.

VERY IMPORTANT
Do not use markdown code blocks. Output only a valid JSON object, nothing else.

EXAMPLE OUTPUT:
{
  "space_id": "O2",
  "space_details": "balcony",
  "user_profile": "travelers/expats",
  "user_question": "Who else benefits?",
  "desired_activity": "Sunbath",
  "resident_distance": 60.42,
  "current_activity": "Sunbath",
  "usability_prediction": "",
  "suggestions": [
    {
      "variation_type": "Add Wall",
      "variation_name": "Low wall for wind/privacy",
      "description": "Adds a low wall to provide wind and privacy while still allowing sunlight.",
      "reason_for_profile": "Suitable for travelers/expats who value sunbathing and relaxation.",
      "optimal_time_impact": "+1 hour of usable time",
      "profile_suitability_notes": "This suggestion is suitable for the traveler/expat profile as it provides a comfortable and private space for sunbathing.",
      "suitability_%_increase": 20,
      "comfort_usability_impact": "Improved comfort and usability due to added wind protection and privacy.",
      "other_beneficiaries": {"H8": "Sunbath", "H67": "Sunbath"},
      "wall_height": 0.8,
      "slab_extension_sqm": 2,
      "louvre_height": 0.5,
      "other_activities_benefit": []
    }
  ],
  "summary_reasoning":     str,
  "householder_reasoning": {resident_id: str, ...}
}

OUTPUT SCHEMA
{
  "space_id":              str,
  "space_details":         str,
  "user_profile":          str,
  "user_question":         str,
  "desired_activity":      str,
  "resident_distance":     str,
  "current_activity":      str,
  "usability_prediction":  str,
  "suggestions": [{
      "variation_type":           "Extend Slab" | "Add Wall" | "Add Louvres",
      "variation_name":           str,
      "description":              str,
      "reason_for_profile":       str,
      "optimal_time_impact":      str,
      "profile_suitability_notes":str,
      "suitability_%_increase":   str,
      "comfort_usability_impact": str,
      "other_beneficiaries":      {resident_id:str,…},
      "wall_height": float|null,  // e.g. 0.8 or null
      "suitability_%_increase": int|string, // e.g. 20 or "20%"
      "louvre_height":            float|null,
      "slab_extension_sqm":       float|null, // e.g. 2 or null
      "other_activities_benefit": [str,…]
  }],
  "summary_reasoning": "Adding a low wall increases privacy and comfort for sunbathing, benefiting H8 and H67.",
  "householder_reasoning": {
    "H3": "H3 is the main user of this space and values privacy for sunbathing.",
    "H9": "H9 benefits from improved wind protection while relaxing on the balcony."
}
"""
        },

        # ───────────── USER ─────────────
        {
            "role": "user",
            "content": f"""\
SID:{space_id}
Persona:{resident_persona}
Dist:{distance_to_space}
Weights:{activity_weights_for_resident}
Others:{other_residents_summary}
Q:{user_question_for_suggestion}
Desired:{desired_activity_for_space}
Current:{current_activity_in_space}
Space:{space_context}
Logic:{activity_logic_context}
WallRange:0.8-1.3
LouvreRange:0.2-0.8

# ML Activity Logic Context (from ml_activity_logic.db):
{activity_logic_context}

# FULL DATABASE CONTEXT (gh_data_for_geometry.db):
{full_db_context}

"""


        }
    ]
)



    return response.choices[0].message.content