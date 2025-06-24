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



def suggest_geometric_variations( # type: ignore
    space_id: str, 
    resident_persona: str, 
    space_context: str, 
    green_prediction: str, 
    threshold_prediction: str,
    usability_prediction: str, # type: ignore
    distance_to_space: str, # type: ignore
    activity_weights_for_resident: str, # type: ignore
    current_activity_in_space: str,
    user_question_for_suggestion: str, # Original user question asking for suggestions
    desired_activity_for_space: str, # The activity the user wants the space to be good for
    other_residents_summary: str # New: Summary of other residents who might benefit
    
) -> str:
    # Pre-process space_context to be a JSON-valid string content.
    # This escapes newlines (e.g., \n to \\n), quotes (e.g., " to \\"),
    # and backslashes (e.g., \ to \\) so that if the LLM copies it verbatim
    # into the "space_details" field of its JSON output, it will be valid.
    if space_context:
        processed_space_context_for_prompt = json.dumps(space_context)[1:-1]
    else:
        processed_space_context_for_prompt = ""

    response = client.chat.completions.create(
    model=completion_model,
    temperature=1,
    top_p=0.85,
    messages=[
        {
            "role": "user",
            "content": f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert architectural-design assistant.

TASK  
• From the three allowed geometric actions — \"Extend Slab\", \"Add Wall\", \"Add Louvres\" — **pick exactly one**.  
• Produce **one** suggestion that best fits the space context and desired activity.  
• Return **only** a JSON object that follows the schema below.  
• Do **not** include any explanations, bullet points, or markdown.

DECISION RULES  
If \"variation_type\" == \"Extend Slab\"  
 • variation_name & description may **only** describe a slab extension.  
 • direction must optimise sun/wind based on Orientation.  
 • area ≤ slab_extension_limit_sqm (if provided), otherwise ≤ 5 m² or ≤ 50 % of the original slab, whichever is smaller.  
If \"variation_type\" == \"Add Wall\"  
 • Interpret this as adding a **low wall or parapet**, not a full-height enclosure.  
 • variation_name & description must reflect that: e.g., a parapet, edge wall, or privacy bench.  
 • Possible purposes include:  
  – visual or spatial privacy  
  – defining boundaries for loggias, balconies, or sports areas  
  – providing a ledge for sitting, flower boxes, or resting equipment  
 • Do not claim it “creates a sports area” by itself — instead, describe how it enables or enhances activity in the existing space.
If \"variation_type\" == \"Add Louvres\"  
 • variation_name & description may **only** describe louvres.
 • Specify louvre height between 0.2 and 0.8 meters.

SPECIAL CASE:  
• If the desired_activity_for_space is \"Sunbath\" and no variation_type has been chosen yet, strongly prefer \"Add Wall\" (e.g., a low parapet) for wind protection and privacy while lying down.

DISALLOWS BY ACTION:
• Avoid selecting \"Add Wall\" for activities that rely on openness, vegetation, or visual access such as: Healing Garden, Viewpoint, Urban Agriculture Garden, Biodiversity Balcony — unless justified by wind exposure, privacy score, or safety.  
• Avoid selecting \"Extend Slab\" for passive, contemplative, or stationary activities with small spatial demands such as: Offline Retreat, Creative Corridor, Flexible Space, unless the current area is under 4 sqm.  
• Avoid selecting \"Add Louvres\" for activities that benefit from direct sun exposure such as: Sunbath, Healing Garden, Community Pool/BBQ, Urban Agriculture Garden — unless shading needs outweigh the solar gain.

REASONING REQUIREMENTS:
- Justify the selected variation based on spatial characteristics:
  • Orientation (e.g. East-facing = morning sun, South = strong exposure)  
  • Area (e.g. sports ≥ 10 m², sunbathing ≥ 6 m²)  
  • Privacy and open sides  
  • Usability and green suitability predictions  
  • Adjacency to indoor spaces  
- Use architectural logic to explain why this option is better than the others.
- Avoid vague phrases like “this helps” or “this improves comfort.” Instead, use **objective reasoning** whenever possible.
- Reference or estimate **quantities** based on the input:
  • Area change (e.g., shaded vs. unshaded m²)
  • UTCI reduction, if known
  • % increase in usability, suitability, or comfort
  • Height or length of elements (e.g., 2m parapet, 3m slab extension)
- When referring to user needs, cite measurable characteristics (e.g., “young professionals prefer partially shaded zones during peak sun hours” is better than vague emotional claims).
- If other residents benefit, explain why, based on their preferences and proximity.

OUTPUT SCHEMA  
{{{{
  "space_id": "O2",
  "space_details": "Type: Balcony\nArea: 9sqm\nOrientation: East\nHeight: 3m\nLevel: 1\nOpen Side: 1\nWind Exposure: 4.88\nUTCI: 25.4°C\nNeighbour Distance: 30.97m\nUsability: cool_breezy\nGreen Suitability: suitable\nPrivacy Score: 0.033",
  "user_profile": "Young Professionals",
  "user_question_for_suggestion": "Who else benefits?",
  "desired_activity_for_space": "Sports",
  "resident_distance_to_space": "62m",
  "current_activity_in_space": "Sitting",
  "usability_prediction": "UTCI: 25.4°C; Area: 9sqm; Open Side: 1; Privacy Score: 0.033",
  "suggestions": [
    {{
      "variation_type": "Add Wall",
      "variation_name": "East Parapet for Spatial Definition",
      "description": "Install a 2.5m-long, 1.2m-high parapet along the east-facing open edge to provide a spatial boundary for light sports and reduce visual exposure.",
      "reason_for_profile": "Young professionals are likely to use the space for light physical activity. A defined boundary enables safer movement and a more intentional use of the limited 9sqm area.",
      "optimal_time_impact_description": "Morning use becomes more comfortable due to added wind protection on the exposed side.",
      "profile_suitability_notes": "Helps transform the currently undefined space into a more structured micro-court for solo or duo activities.",
      "suitability_percentage_increase": "22%",
      "comfort_usability_impact": "Improved privacy, boundary safety, and psychological comfort during active use.",
      "other_beneficiaries_explained": {{
        "H11": "Located 5m away, benefits from reduced glare and shared view of the activity zone.",
        "H7": "Visually connected and likely to use the balcony for similar purposes.",
        "H39": "Receives indirect shading and increased perception of communal use."
        }},

      "wall_height": 1.2,
      "slab_extension (sqm)": 3,
      "louvre_height": 0.5,
      "other_activities_benefit": ["Stretching", "Balance Training"]
    }}
  ],
  "summary_reasoning": "The 9sqm balcony lacks spatial definition and privacy. Adding a parapet on the open side enables safe and focused sports activity, especially for young residents. Similar nearby households benefit from the improved visual shielding and shared potential for active use."
}}}}

REMINDER → Respond **only** with a JSON object that starts with '{{' and ends with '}}'.  
<|eot_id|><|start_header_id|>user<|end_header_id|>
Here is the space context and user request:
"""
        },
        {
            "role": "user",
            "content": f"""
Generate geometric variations for the following:
Space ID: {space_id}
Resident Persona (User Profile): {resident_persona}
Resident's Distance to this Space: {distance_to_space}
Resident's Activity Preferences for this space (weights): {activity_weights_for_resident}
Other Residents Summary (potential beneficiaries based on data): {other_residents_summary}
User Question for Suggestion: {user_question_for_suggestion}
Desired Activity for this Space: {desired_activity_for_space}
Current Activity in this Space: {current_activity_in_space}
Space Details:
{processed_space_context_for_prompt}

Threshold Prediction for this space: {threshold_prediction}
Wall Height Range (meters): 0.8 to 1.3
Louvre Height Range (meters): 0.2 to 0.8
Green Prediction for this space: {green_prediction}
Usability Prediction for this space: {usability_prediction}
"""
        }

    ]
)


    return response.choices[0].message.content