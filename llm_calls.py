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
    usability_prediction: str,
    distance_to_space: str,
    activity_weights_for_resident: str,
    current_activity_in_space: str # New parameter
) -> str:
    # Pre-process space_context to be a JSON-valid string content.
    # This escapes newlines (e.g., \n to \\n), quotes (e.g., " to \"),
    # and backslashes (e.g., \ to \\) so that if the LLM copies it verbatim
    # into the "space_details" field of its JSON output, it will be valid.
    if space_context:
        processed_space_context_for_prompt = json.dumps(space_context)[1:-1]
    else:
        processed_space_context_for_prompt = ""

    response = client.chat.completions.create(
        model=completion_model,
        temperature=0.2, # Lowered temperature for more deterministic and rule-adherent JSON output
        messages=[
                {
                "role": "system",
                "content": """
You are an expert architectural design assistant.
Your task is to suggest 2 relevant geometric variations for a given outdoor space, tailored to a specific resident's persona, their preferences for this space, their distance to it, and the existing space details.

IMPORTANT: Each suggestion MUST be an application of EXACTLY ONE of the 6 "Possible Actions" listed below. Do not invent new types of actions or combine actions into one suggestion.

You must choose from the following list of possible actions to base your suggestions on. For each chosen action, make the specified decisions and provide a detailed description of its application.

Possible Actions:
1.  **Extend Slab**: Extend an existing slab. You must decide the new area (not more than 50% of original or 5 sqm, whichever is smaller) and suggest a purpose or direction for the extension.
2.  **Artificial Terrain**: Modify the ground plane. You must decide whether to excavate (e.g., for a sunken seating area) or create a small hill/mound, and suggest its general form and purpose.
3.  **Outdoor Cooking Feature**: Add a space for a kitchen or fire pit. You must decide between a kitchen setup (e.g., counter, sink, grill) or a fire pit, and suggest its placement and materials.
4.  **Small Open Pavilion**: Add a small, open-sided pavilion. It should cover no more than 50% of the existing space's area. You must decide its primary use (e.g., shaded lounge, outdoor dining, yoga deck) and suggest a simple structural form.
5.  **Significant Planting**: Introduce permanent large trees or substantial vegetation. You must decide on the general type of vegetation (e.g., shade trees, screening shrubs, a themed garden bed) and suggest their placement.
6.  **Water Feature**: Add a water feature integrated into the floor/ground. It should cover no more than 30% of the existing space's area. You must decide if it's a linear feature (e.g., rill, narrow channel) or a more contained circular/organic shape, and suggest its character (e.g., reflective pool, bubbling fountain).

For each suggested variation:
- The `variation_type` MUST be one of the exact names from the "Possible Actions" list (e.g., "Extend Slab", "Artificial Terrain").
- The `variation_name` should be a concise, descriptive title for the specific application of the chosen `variation_type` (e.g., "Extended Seating Area", "Sunken Fire Pit Lounge").
- The `description` should detail how the chosen action from the 'Possible Actions' list is applied to the specific space. It MUST explicitly state all decisions made as required by that action's description (e.g., for 'Extend Slab', state the new area and confirm it meets constraints; for 'Outdoor Cooking Feature', state whether it's a kitchen or fire pit and describe its placement/materials).
- The `reason_for_profile` should explain why this variation is suitable for the given resident's persona, their preferences, their distance, and the space context.
- The `estimated_impact` should describe the likely effect (e.g., "Creates a new social hub", "Enhances tranquility and biodiversity").

Your entire response MUST be ONLY the valid JSON object described below. Do not include any other text, explanations, or markdown formatting (like ```json).

The JSON object should have the following structure:
```json
{
  "space_id": "string",
  "space_details": "string (details of the space as provided in the input)",
  "user_profile": "string (this should be the resident_persona provided in the input)",
  "resident_distance_to_space": "string (resident's distance to this specific space, as provided in input)",
  "current_activity_in_space": "string (current activity assigned to this space, as provided in input)",
  "suggestions": [
    {
      "variation_type": "string (must be one of the 6 Possible Actions)",
      "variation_name": "string",
      "description": "string",
      "reason_for_profile": "string",
      "estimated_impact": "string"
    }
  ],
  "summary_reasoning": "string (overall reasoning for the set of suggestions)"
}
```

Example for a "Play Area" space_id and "Families with Young Children" user_profile:
{
  "space_id": "O2",
  "space_details": "Type: Courtyard\\nArea: 50sqm\\nOrientation: South\\nFeatures: Paved, one tree",
  "user_profile": "Families with Young Children",
  "resident_distance_to_space": "15.2m",
  "current_activity_in_space": "Playground",
  "suggestions": [
    {
      "variation_type": "Artificial Terrain",
      "variation_name": "Play Mound with Slide",
      "description": "Create a soft-surfaced, circular mound approximately 1m high in the northeast corner of the courtyard. Integrate a short, curved slide and rubber safety surfacing around the base.",
      "reason_for_profile": "Children benefit from interactive, varied terrain that promotes climbing, sliding, and imaginative play within safe boundaries.",
      "estimated_impact": "Improves active play and spatial variety"
    },
    {
      "variation_type": "Extend Slab",
      "variation_name": "Scoot Track Extension",
      "description": "Extend the existing slab by 4 sqm to the south, forming a looped pathway with smooth concrete finish suitable for scooters, tricycles, or toy vehicles.",
      "reason_for_profile": "Encourages gross motor development and safe wheeled play for toddlers and young children in a controlled environment.",
      "estimated_impact": "Expands physical play and mobility"
    }
  ],
  "summary_reasoning": "The selected sub-variations create a multi-sensory play environment, integrating movement and safety while maximizing the courtyard’s size and sun exposure for families with young children."
}


 Ensure the output is only the JSON object. The value for "space_details" should be the exact string provided in the "Space Details" section of the user input.
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
Current Activity in this Space: {current_activity_in_space}
Space Details:
{
    processed_space_context_for_prompt # Embed directly, preserving its structure
}
Threshold Prediction for this space: {threshold_prediction}
Green Prediction for this space: {green_prediction}
Usability Prediction for this space: {usability_prediction}
"""
            }
        ]
    )
    return response.choices[0].message.content