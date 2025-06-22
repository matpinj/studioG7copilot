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
    studio_export_details: str, # Details from studio_export.csv for the space
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
        temperature=0.2, # Lowered temperature for more deterministic and rule-adherent JSON output
        messages=[
                {
                "role": "system",
                "content": """
You are an expert architectural design assistant.
Your task is to suggest 1 relevant geometric variation for a given outdoor space to make it more suitable for the `desired_activity_for_space`.
This suggestion should be tailored to the specific resident's persona, their activity preferences for this space, their distance to it, the `desired_activity_for_space` they envision, the existing space details, and any relevant constraints or opportunities mentioned in `studio_export_details`.
The `user_question_for_suggestion` is the query that prompted this request; use it to ensure your `summary_reasoning` and other descriptive fields are relevant.
If the `user_question_for_suggestion` asks about other beneficiaries, use the `other_residents_summary` provided in the input to inform your `other_beneficiaries` list and the `summary_reasoning`. Explain WHY these other residents benefit, referencing the summarized information about their potential interest (e.g., proximity, preferences).

The `space_details` will include an `Orientation` (e.g., North, South-West). Use this information to guide your suggestions, especially for wall placements or slab extensions to optimize for sun, shade, or wind, and explicitly mention this in your reasoning.

IMPORTANT: The suggestion MUST be an application of EXACTLY ONE of the "Possible Actions" listed below. Do not invent new types of actions or combine actions into one suggestion.

You must choose from the following list of possible actions to base your suggestions on. For each chosen action, make the specified decisions and provide a detailed description of its application.

Possible Actions:
1.  **Extend Slab**: Extend an existing slab.
    *   Decide: New area (e.g., +X sqm, consider `slab_extension_limit_sqm` from `studio_export_details` or default max 5 sqm / 50% of original), purpose/direction of extension. The direction MUST consider the space's `Orientation` (from `space_details`) for optimal use (e.g., extend south for more sun, extend away from prevailing wind).
2.  **Add Wall**: Add a new wall.
    *   Decide: Location (e.g., "along north edge", "to enclose open_side_east"), height (e.g., 1.2m, consider `max_wall_height_m` from `studio_export_details`), length, material (e.g., "brick", "wood paneling", consider `wall_material_options` from `studio_export_details`).
3.  **Add Pergola**: Add a new pergola.
    *   Decide: Coverage area (e.g., "covers X% of the space" or "Xm x Ym area"), height, primary material (e.g., "wood", "metal", consider `pergola_material_options` from `studio_export_details`), style (e.g., "louvered for adjustable shade", "open trellis for vines").






For the suggested variation, ALL the following fields in the JSON output MUST be filled with specific, relevant information. Do NOT use "N/A" or generic placeholders.
- The `variation_type` MUST be one of the exact names from the "Possible Actions" list (e.g., "Extend Slab", "Add Wall", "Add Pergola").
- The `variation_name` should be a concise, descriptive title for the specific application of the chosen `variation_type` (e.g., "Extended Seating Area", "Sunken Fire Pit Lounge").
- The `description` should detail how the chosen action from the 'Possible Actions' list is applied to the specific space. It MUST explicitly state all decisions made as required by that action's description, referencing `studio_export_details` if used, and how `Orientation` influenced the decision if applicable.
- The `reason_for_profile` should explain why this variation is suitable for the given `resident_persona`, their `activity_weights_for_resident`, their `distance_to_space`, the `space_context`, and how it helps achieve the `desired_activity_for_space`.


The following fields are OPTIONAL. Include them in the `suggestions` object ONLY IF the `user_question_for_suggestion` explicitly or strongly implies a request for this specific type of information. If not requested, OMIT the field from the JSON output. If `user_question_for_suggestion` is empty or very generic (e.g., "Suggest something"), then all these optional fields should be OMITTED.

- `optimal_time_impact_description` (string): Include ONLY IF the user asks about the best time for the variation's impact (e.g., 'When would this be most useful?', 'Will this help in summer?'). If included, it MUST state when this change would have the most positive impact (e.g., "Most beneficial during summer afternoons 2-5 PM for shade").
- `profile_suitability_notes` (string): Include ONLY IF the user asks for more details on how the design suits their profile (e.g., 'How does this fit my lifestyle?', 'Make it more suitable for me.'). If included, it MUST elaborate on how the design is specifically tailored to the `resident_persona` and their preferences for the `desired_activity_for_space`.
- `suitability_percentage_increase` (integer): Include ONLY IF the user asks about the quantitative improvement or how much more suitable the space becomes (e.g., 'How much better will this be for X activity?', 'What's the percentage improvement?'). If included, it MUST be an estimated integer percentage increase (e.g., 25 for 25%) in suitability for the `desired_activity_for_space`.
- `comfort_usability_impact_description` (string): Include ONLY IF the user asks about comfort, usability improvements, or environmental effects (e.g., 'Will it be more comfortable?', 'How does this affect wind/sun?'). If included, it MUST describe the likely effects on environmental comfort and functional usability.
- `other_beneficiaries` (array of strings): Include ONLY IF the user asks who else might benefit (e.g., 'Will my neighbors like this?', 'Is this good for others?'). If included, list other resident personas or general resident types who might also benefit.
- `other_activities_benefit` (array of strings with brief note): Include ONLY IF the user asks if the change benefits other activities (e.g., 'Can I also do Y here?', 'What else is this good for?'). If included, list other activities that could also become more suitable.


Mandatory fields for the suggestion object are `variation_type`, `variation_name`, `description`, and `reason_for_profile`.


Your entire response MUST be ONLY the valid JSON object described below. Do not include any other text, explanations, or markdown formatting (like ```json).

The JSON object should have the following structure:
```json
{
  "space_id": "string",
  "space_details": "string (details of the space as provided in the input)",
  "user_profile": "string (this should be the resident_persona provided in the input)",
  "user_question_for_suggestion": "string (the original user question that led to these suggestions)",
  "desired_activity_for_space": "string (the activity the user wants the space to be good for, as provided in input)",
  "resident_distance_to_space": "string (resident's distance to this specific space, as provided in input)",
  "current_activity_in_space": "string (current activity assigned to this space, as provided in input)",
  "studio_export_details_considered": "string (summary of how studio_export_details influenced suggestions, or 'N/A')",
  "suggestions": [ // Array with exactly ONE suggestion object
    {
      "variation_type": "string (must be one of the 3 Possible Actions: Extend Slab, Add Wall, Add Pergola)",
      "variation_name": "string",
      "description": "string",
      "reason_for_profile": "string",
      "optimal_time_impact_description": "string (e.g., 'Most beneficial during summer afternoons 2-5 PM for shade')",
      "profile_suitability_notes": "string",
      "suitability_percentage_increase": "string",
      "comfort_usability_impact": "string",
      "other_beneficiaries": ["string"],
      "other_activities_benefit": ["string with brief note"]
    }
  ],
  "summary_reasoning": "string (Overall reasoning for the single suggestion. It MUST highlight how it addresses the user_question_for_suggestion and facilitates the desired_activity_for_space. If the user_question_for_suggestion asks about other beneficiaries, this reasoning should explain HOW or WHY they might benefit, considering general factors like shared access to improved common spaces, improved aesthetics for neighbours, or if the modification addresses common needs suggested by general resident personas. If `other_residents_summary` is provided and relevant, incorporate its insights into this explanation.)"
}
```

Example for a "Play Area" space_id and "Families with Young Children" user_profile:
{
  "space_id": "O3",
  "space_details": "Type: Patio\\\\nArea: 20sqm\\\\nOrientation: West\\\\nFeatures: Concrete, adjacent to kitchen",
  "user_profile": "Families with Young Children",
  "resident_distance_to_space": "Short (5m)",
  "user_question_for_suggestion": "Make this patio better for Children's Play. Which other residents might like this change and why?",
  "current_activity_in_space": "Outdoor Seating",
  "studio_export_details_considered": "Used 'slab_extension_limit_sqm: 10' from studio_export_details. Considered West orientation for afternoon sun on the southern extension.",
  "suggestions": [
    {
      "variation_type": "Extend Slab",
      "variation_name": "Afternoon Play Patio Extension",
      "description": "Extend the existing concrete slab by 8 sqm towards the south. This extension is within the 'slab_extension_limit_sqm' of 10 sqm from studio_export_details. Extending south from the West-oriented patio aims to capture more afternoon sun (2 PM - 5 PM), making it warmer for play, while also potentially offering some morning shade depending on surrounding structures.",
      "reason_for_profile": "Families with Young Children require safe, accessible outdoor play areas. Extending the patio provides more usable space directly adjacent to the kitchen, aligning with their need for supervision and convenience, as requested. The increased area supports active play.",
      "other_beneficiaries": ["H5", "H12 (Families with Toddlers)"]
      // profile_suitability_notes, comfort_usability_impact_description, other_beneficiaries, other_activities_benefit are omitted
      // because the example user_question_for_suggestion did not explicitly ask for them.
      // optimal_time_impact_description and suitability_percentage_increase are also omitted as per the question.
    }
  ],
  "summary_reasoning": "The suggested patio extension directly addresses the user's request for more play space for children near the kitchen, making it more suitable for 'Children's Play'. This change would also likely benefit other residents. For example, H5 (who is close by and has a strong preference for 'Playground' activity in this area) would appreciate the larger play surface. Similarly, H12, a family with toddlers (who are nearby and have a good preference for 'Playground'), would find the extended, sunnier patio advantageous for their children."
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
Other Residents Summary (potential beneficiaries based on data): {other_residents_summary}
User Question for Suggestion: {user_question_for_suggestion}
Current Activity in this Space: {current_activity_in_space}
Space Details:
{processed_space_context_for_prompt}
Threshold Prediction for this space: {threshold_prediction}
Green Prediction for this space: {green_prediction}
Usability Prediction for this space: {usability_prediction}
"""
                }
            ]
        )
    return response.choices[0].message.content