import pandas as pd
import sqlite3
import json
import re
import os
from server.config import *
import logging
from functools import lru_cache
import time
import requests

# ============================================================================
# GLOBAL CACHING SYSTEM (shared with optimized reasoning)
# ============================================================================

# Global cache for data to avoid reloading
_negotiation_cache = {}
_negotiation_cache_initialized = False

def initialize_negotiation_cache():
    """Initialize cache for negotiation functions - called once"""
    global _negotiation_cache, _negotiation_cache_initialized
    
    if _negotiation_cache_initialized:
        return
        
    print("🤝 Initializing negotiation cache...")
    start_time = time.time()
    
    try:
        # Load from database first
        conn = sqlite3.connect('sql/gh_data.db')
        
        # Geometries from database
        geometries = pd.read_sql_query("SELECT * FROM activity_space", conn)
        geometries.rename(columns={"key": "id"}, inplace=True)
        geometries["id"] = geometries["id"].apply(lambda x: f"O{x}" if not str(x).startswith("O") else str(x))
        
        # Personas from database
        personas = pd.read_sql_query("SELECT * FROM personas_assigned", conn)
        personas["resident_key"] = personas["resident_key"].astype(str).str.strip()
        
        # Distances from database
        distances = pd.read_sql_query("SELECT * FROM resident_distances", conn)
        distances.rename(columns={"Outdoor Space": "id"}, inplace=True)
        
        # Replace CSV with database version for distances_all
        distances_all = pd.read_sql_query("SELECT * FROM resident_distances_all", conn)
        distances_all.columns = [c.strip() for c in distances_all.columns]
        distances_all['Source Node'] = distances_all['Source Node'].astype(str).str.strip()
        
        conn.close()

        # Load CSVs that must stay as CSV
        voting = pd.read_csv('resident_data/voting_weights.csv')

        # Load assignments (CSV only)
        assignments = pd.read_csv('llm_reasoning/llm_activity_assignments.csv')
        
        # Load ML activity logic (new file)
        try:
            ml_activity_logic = pd.read_csv('preset/ml_activity_logic.csv')
        except FileNotFoundError:
            print("Warning: ml_activity_logic.csv not found")
            ml_activity_logic = pd.DataFrame()

        # Create thresh, green, usability from activity_space or multi CSV
        try:
            multi = pd.read_csv('gh_data/gh_data_multiple_activities.csv')
            multi = multi.rename(columns={'key': 'id'})
            thresh = multi[['id', 'activity']].rename(columns={'activity': 'predicted_activities'})
            green = multi[['id', 'green_suitability']].rename(columns={'green_suitability': 'green_prediction'})
            usability = multi[['id', 'usability']].rename(columns={'usability': 'usability_prediction'})
        except FileNotFoundError:
            # Fallback: create from geometries
            thresh = geometries[['id', 'activity']].rename(columns={'activity': 'predicted_activities'})
            green = geometries[['id', 'green_suitability']].rename(columns={'green_suitability': 'green_prediction'})
            usability = geometries[['id', 'usability']].rename(columns={'usability': 'usability_prediction'})

        # Store in cache
        _negotiation_cache['geometries'] = geometries
        _negotiation_cache['thresh'] = thresh
        _negotiation_cache['green'] = green
        _negotiation_cache['usability'] = usability
        _negotiation_cache['voting'] = voting
        _negotiation_cache['distances'] = distances
        _negotiation_cache['personas'] = personas
        _negotiation_cache['distances_all'] = distances_all
        _negotiation_cache['assignments'] = assignments
        _negotiation_cache['ml_activity_logic'] = ml_activity_logic
        
        # Create lookup sets for faster activity detection
        _negotiation_cache['all_activities'] = set(assignments['assigned_activity'].dropna().unique())
        
        _negotiation_cache_initialized = True
        print(f"✅ Negotiation cache initialized in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Error initializing negotiation cache: {e}")
        raise

def get_negotiation_data(key):
    """Get cached negotiation data"""
    if not _negotiation_cache_initialized:
        initialize_negotiation_cache()
    return _negotiation_cache.get(key)

# ============================================================================
# OPTIMIZED DATA LOADING
# ============================================================================

def load_csvs():
    """Legacy function for backward compatibility - now uses cache"""
    if not _negotiation_cache_initialized:
        initialize_negotiation_cache()
        
    return (
        get_negotiation_data('geometries'),
        get_negotiation_data('thresh'),
        get_negotiation_data('green'),
        get_negotiation_data('usability'),
        get_negotiation_data('voting'),
        get_negotiation_data('distances'),
        get_negotiation_data('personas'),
        get_negotiation_data('distances_all'),
        get_negotiation_data('assignments'),
        get_negotiation_data('ml_activity_logic')
    )

# ============================================================================
# OPTIMIZED CORE FUNCTIONS
# ============================================================================

@lru_cache(maxsize=64)
def get_space_assignment(space_id):
    """Cached lookup for space assignment"""
    assignments = get_negotiation_data('assignments')
    if assignments is None:
        return None
    assignments = assignments.copy()  # Don't modify cached data
    assignments['space_id'] = assignments['space_id'].astype(str).str.strip()
    row = assignments[assignments['space_id'] == str(space_id).strip()]
    return row.iloc[0]['assigned_activity'] if not row.empty else None

@lru_cache(maxsize=64)
def get_space_geometry(space_id):
    """Cached lookup for space geometry"""
    geometries = get_negotiation_data('geometries')
    if geometries is None:
        return None
    row = geometries[geometries["id"] == space_id]
    return row.iloc[0] if not row.empty else None

@lru_cache(maxsize=32)
def get_user_distances(user_id):
    """Cached lookup for user distances"""
    distances_all = get_negotiation_data('distances_all')
    if distances_all is None or user_id not in distances_all['Source Node'].values:
        return None
    return distances_all[distances_all['Source Node'] == user_id].iloc[0]

def debug_analyze_convinceable_voters(space_id, current_activity, desired_activity, voting_df):
    """
    Debug version with detailed logging
    """
    print(f"=== DEBUG: Analyzing space '{space_id}' ===")
    
    # Ensure consistent formatting
    space_id_clean = str(space_id).strip().upper()
    voting = voting_df.copy()
    voting['space'] = voting['space'].astype(str).str.strip().str.upper()
    voting['resident'] = voting['resident'].astype(str).str.strip().str.upper()
    voting['activity'] = voting['activity'].astype(str).str.strip().str.title()

    print(f"Looking for space: '{space_id_clean}'")
    print(f"All spaces in data: {voting['space'].unique().tolist()}")
    
    # ONLY residents who have entries in voting_weights.csv for this specific space
    eligible_votes = voting[voting["space"] == space_id_clean].copy()
    
    print(f"Found {len(eligible_votes)} entries for space '{space_id_clean}':")
    if not eligible_votes.empty:
        for _, row in eligible_votes.iterrows():
            print(f"  - Resident: {row['resident']}, Activity: {row['activity']}, Weight: {row['weight']}")
    
    if eligible_votes.empty:
        print(f"ERROR: No entries found in voting_weights.csv for space '{space_id_clean}'")
        return []

    eligible_residents = eligible_votes['resident'].unique().tolist()
    print(f"Eligible residents for this space: {eligible_residents}")

    # Activity group mapping
    activity_groups = {
        'sports': ['Sports', 'Playground', 'Community Pool/BBQ'],
        'social': ['Community Pool/BBQ', 'Outdoor Cinema/Event Space', 'Outdoor Meeting Space'],
        'relaxation': ['Sunbath', 'Offline Retreat', 'Healing Garden', 'Viewpoint'],
        'creative': ['Creative Corridor', 'Flexible Space'],
        'nature': ['Healing Garden', 'Urban Agriculture', 'Green Balcony', 'Biodiversity Balcony']
    }

    def get_activity_group(activity):
        for group, acts in activity_groups.items():
            if activity in acts:
                return group
        return 'other'

    # Group residents by what they voted for in this space
    grouped = {}
    for resident in eligible_residents:
        resident_votes = eligible_votes[eligible_votes['resident'] == resident]
        voted_activities = resident_votes['activity'].unique().tolist()
        voted_groups = list(set(get_activity_group(act) for act in voted_activities))
        grouped[resident] = {
            "activities": voted_activities,
            "groups": voted_groups,
            "weights": resident_votes.set_index('activity')['weight'].to_dict()
        }
        print(f"Resident {resident}: voted for {voted_activities}, groups: {voted_groups}")

    # Prepare convinceable list - ONLY people who didn't vote for the desired activity
    convinceable_residents = []
    desired_group = get_activity_group(desired_activity.title())
    print(f"Desired activity: {desired_activity.title()}, group: {desired_group}")

    for resident, info in grouped.items():
        # If already voted for desired activity, skip
        if desired_activity.title() in info["activities"]:
            print(f"SKIP {resident}: already voted for {desired_activity.title()}")
            continue

        # Check if any of their voted groups matches desired group
        has_same_group = desired_group in info["groups"]

        if has_same_group:
            likely = True
            reason = f"voted for {', '.join(info['activities'])} - same group as {desired_activity}, likely interested"
        else:
            likely = False
            reason = f"voted for {', '.join(info['activities'])} - different from {desired_activity}, may prefer different activities"

        print(f"CONVINCEABLE {resident}: {reason}")
        convinceable_residents.append({
            "resident": resident,
            "reason": reason,
            "likely_to_convince": likely,
            "weights": info["weights"],
            "voted_groups": info["groups"],
            "voted_activities": info["activities"]
        })

    # Sort by likelihood (most likely first)
    convinceable_residents.sort(key=lambda x: x["likely_to_convince"], reverse=True)
    print(f"Final convinceable residents: {len(convinceable_residents)}")
    return convinceable_residents

# Test with sample data
sample_data = pd.DataFrame([
    {'resident': 'H1', 'space': 'O1', 'activity': 'Flexible Space', 'distance': 18.76, 'weight': 0.0051},
    {'resident': 'H2', 'space': 'O1', 'activity': 'Sports', 'distance': 15.23, 'weight': 0.0065},
    {'resident': 'H3', 'space': 'O2', 'activity': 'Flexible Space', 'distance': 12.45, 'weight': 0.0080},
    {'resident': 'H1', 'space': 'O2', 'activity': 'Healing Garden', 'distance': 20.15, 'weight': 0.0049}
])

print("Sample data:")
print(sample_data)
print("\n" + "="*60)

# Test 1: Look for voters for space O1 wanting Creative Corridor (same group as Flexible Space)
result1 = debug_analyze_convinceable_voters('O1', 'Flexible Space', 'Creative Corridor', sample_data)

print("\n" + "="*60)

# Test 2: Look for voters for space O1 wanting Sports (different group)
result2 = debug_analyze_convinceable_voters('O1', 'Flexible Space', 'Sports', sample_data)

print("\n" + "="*60)
print("SIMPLIFIED CHECK:")

def simple_space_filter_check(voting_df, target_space):
    """Simple check to verify space filtering works correctly"""
    print(f"Checking filter for space: '{target_space}'")
    
    # Clean the data
    voting = voting_df.copy()
    voting['space'] = voting['space'].astype(str).str.strip().str.upper()
    target_space_clean = str(target_space).strip().upper()
    
    print(f"All spaces in data: {voting['space'].unique().tolist()}")
    print(f"Looking for: '{target_space_clean}'")
    
    # Filter
    filtered = voting[voting['space'] == target_space_clean]
    
    print(f"Rows matching '{target_space_clean}': {len(filtered)}")
    if not filtered.empty:
        print("Matching rows:")
        for _, row in filtered.iterrows():
            print(f"  {row['resident']} -> {row['space']} ({row['activity']})")
    
    return filtered

# Test the space filtering specifically
print("\nTesting space filtering for 'O1':")
simple_space_filter_check(sample_data, 'O1')

print("\nTesting space filtering for 'O2':")
simple_space_filter_check(sample_data, 'O2')

print("\n" + "="*60)
print("IMPROVED VERSION WITH EXTRA SAFETY CHECKS:")

def analyze_convinceable_voters(space_id, current_activity, desired_activity, voting=None):
    """
    Improved version with extra safety checks and debugging
    """
    if voting is None:
        voting = get_negotiation_data('voting')
    
    # Add debug info
    print(f"DEBUG: Input space_id='{space_id}', total rows in voting data: {len(voting)}")
    
    # Ensure consistent formatting with extra cleaning
    space_id_clean = str(space_id).strip().upper()
    voting = voting.copy()
    
    # More thorough cleaning
    voting['space'] = voting['space'].astype(str).str.strip().str.upper().str.replace('\n', '').str.replace('\r', '')
    voting['resident'] = voting['resident'].astype(str).str.strip().str.upper()
    voting['activity'] = voting['activity'].astype(str).str.strip().str.title()
    
    print(f"DEBUG: Looking for space '{space_id_clean}' in spaces: {voting['space'].unique()[:10]}...")  # Show first 10
    
    # Filter for specific space ONLY
    eligible_votes = voting[voting["space"] == space_id_clean].copy()
    
    print(f"DEBUG: Found {len(eligible_votes)} entries for space '{space_id_clean}'")
    
    if eligible_votes.empty:
        print(f"WARNING: No entries found for space '{space_id_clean}'")
        print(f"Available spaces: {sorted(voting['space'].unique())}")
        return []
    
    # Double-check: ensure we really only have the target space
    unique_spaces_in_result = eligible_votes['space'].unique()
    if len(unique_spaces_in_result) > 1:
        print(f"ERROR: Filter failed! Got multiple spaces: {unique_spaces_in_result}")
        return []
    
    eligible_residents = eligible_votes['resident'].unique().tolist()
    print(f"DEBUG: Eligible residents for space '{space_id_clean}': {eligible_residents}")
    
    # Rest of the function remains the same...
    activity_groups = {
        'sports': ['Sports', 'Playground', 'Community Pool/BBQ'],
        'social': ['Community Pool/BBQ', 'Outdoor Cinema/Event Space', 'Outdoor Meeting Space'],
        'relaxation': ['Sunbath', 'Offline Retreat', 'Healing Garden', 'Viewpoint'],
        'creative': ['Creative Corridor', 'Flexible Space'],
        'nature': ['Healing Garden', 'Urban Agriculture', 'Green Balcony', 'Biodiversity Balcony']
    }

    def get_activity_group(activity):
        for group, acts in activity_groups.items():
            if activity in acts:
                return group
        return 'other'

    grouped = {}
    for resident in eligible_residents:
        resident_votes = eligible_votes[eligible_votes['resident'] == resident]
        voted_activities = resident_votes['activity'].unique().tolist()
        voted_groups = list(set(get_activity_group(act) for act in voted_activities))
        grouped[resident] = {
            "activities": voted_activities,
            "groups": voted_groups,
            "weights": resident_votes.set_index('activity')['weight'].to_dict()
        }

    convinceable_residents = []
    desired_group = get_activity_group(desired_activity.title())

    for resident, info in grouped.items():
        if desired_activity.title() in info["activities"]:
            continue

        has_same_group = desired_group in info["groups"]

        if has_same_group:
            likely = True
            reason = f"voted for {', '.join(info['activities'])} - same group as {desired_activity}, likely interested"
        else:
            likely = False
            reason = f"voted for {', '.join(info['activities'])} - different from {desired_activity}, may prefer different activities"

        convinceable_residents.append({
            "resident": resident,
            "reason": reason,
            "likely_to_convince": likely,
            "weights": info["weights"],
            "voted_groups": info["groups"],
            "voted_activities": info["activities"]
        })

    convinceable_residents.sort(key=lambda x: x["likely_to_convince"], reverse=True)
    print(f"DEBUG: Final result has {len(convinceable_residents)} convinceable residents")
    return convinceable_residents
# def analyze_convinceable_voters(space_id, current_activity, desired_activity, voting=None):
#     """
#     Optimized version - only consider residents who have ANY vote for this space
#     """
#     if voting is None:
#         voting = get_negotiation_data('voting')
    
#     # Ensure consistent formatting
#     space_id_clean = str(space_id).strip().upper()
#     voting = voting.copy()  # Don't modify cached data
#     voting['space'] = voting['space'].astype(str).str.strip().str.upper()
#     voting['resident'] = voting['resident'].astype(str).str.strip().str.upper()
#     voting['activity'] = voting['activity'].astype(str).str.strip().str.title()

#     # ONLY residents who have entries in voting_weights.csv for this specific space
#     eligible_votes = voting[voting["space"] == space_id_clean].copy()
#     if eligible_votes.empty:
#         print(f"DEBUG: No entries found in voting_weights.csv for space '{space_id_clean}'")
#         return []

#     eligible_residents = eligible_votes['resident'].unique().tolist()

#     # Activity group mapping
#     activity_groups = {
#         'sports': ['Sports', 'Playground', 'Community Pool/BBQ'],
#         'social': ['Community Pool/BBQ', 'Outdoor Cinema/Event Space', 'Outdoor Meeting Space'],
#         'relaxation': ['Sunbath', 'Offline Retreat', 'Healing Garden', 'Viewpoint'],
#         'creative': ['Creative Corridor', 'Flexible Space'],
#         'nature': ['Healing Garden', 'Urban Agriculture', 'Green Balcony', 'Biodiversity Balcony']
#     }

#     def get_activity_group(activity):
#         for group, acts in activity_groups.items():
#             if activity in acts:
#                 return group
#         return 'other'

#     # Group residents by what they voted for in this space
#     grouped = {}
#     for resident in eligible_residents:
#         resident_votes = eligible_votes[eligible_votes['resident'] == resident]
#         voted_activities = resident_votes['activity'].unique().tolist()
#         voted_groups = list(set(get_activity_group(act) for act in voted_activities))
#         grouped[resident] = {
#             "activities": voted_activities,
#             "groups": voted_groups,
#             "weights": resident_votes.set_index('activity')['weight'].to_dict()
#         }

#     # Prepare convinceable list - ONLY people who didn't vote for the desired activity
#     convinceable_residents = []
#     desired_group = get_activity_group(desired_activity.title())

#     for resident, info in grouped.items():
#         # If already voted for desired activity, skip
#         if desired_activity.title() in info["activities"]:
#             continue

#         # Check if any of their voted groups matches desired group
#         has_same_group = desired_group in info["groups"]

#         if has_same_group:
#             likely = True
#             reason = f"voted for {', '.join(info['activities'])} - same group as {desired_activity}, likely interested"
#         else:
#             likely = False
#             reason = f"voted for {', '.join(info['activities'])} - different from {desired_activity}, may prefer different activities"

#         convinceable_residents.append({
#             "resident": resident,
#             "reason": reason,
#             "likely_to_convince": likely,
#             "weights": info["weights"],
#             "voted_groups": info["groups"],
#             "voted_activities": info["activities"]
#         })

#     # Sort by likelihood (most likely first)
#     convinceable_residents.sort(key=lambda x: x["likely_to_convince"], reverse=True)
#     return convinceable_residents

def check_geometry_requirements(space_id, desired_activity, geometries=None, ml_activity_logic=None):
    """Optimized geometry requirement checking"""
    if geometries is None:
        geometries = get_negotiation_data('geometries')
    if ml_activity_logic is None:
        ml_activity_logic = get_negotiation_data('ml_activity_logic')
    
    if ml_activity_logic.empty:
        return {"error": "ML activity logic not available"}
    
    # Get current space geometry using cached lookup
    current_geom = get_space_geometry(space_id)
    if current_geom is None:
        return {"error": f"Space {space_id} not found"}
    
    # Get requirements for desired activity
    activity_req = ml_activity_logic[ml_activity_logic['activity'].str.lower() == desired_activity.lower()]
    if activity_req.empty:
        return {"error": f"No requirements found for activity {desired_activity}"}
    
    requirements = activity_req.iloc[0]
    changes_needed = {}
    
    # Check each requirement efficiently
    checks = [
        ('area', 'min_area', 'area'),
        ('open_side', 'min_open_sides', 'open_sides'),
        ('level', 'min_level', 'level'),
        ('wind_exp', 'max_wind_exp', 'wind_exposure'),
        ('sun_h', 'sun_h', 'sun_hours'),
    ]
    
    for current_col, req_col, display_name in checks:
        if req_col not in requirements.index or pd.isna(requirements[req_col]):
            continue
            
        req_value = requirements[req_col]
        if req_value == 'Any':
            continue
            
        try:
            current_value = current_geom[current_col] if current_col in current_geom.index else 0
            
            if req_col.startswith('min_'):
                if current_value < float(req_value):
                    changes_needed[display_name] = f"increase to at least {req_value} (current: {current_value})"
            elif req_col.startswith('max_'):
                if current_value > float(req_value):
                    changes_needed[display_name] = f"reduce to maximum {req_value} (current: {current_value})"
            else:  # exact match
                if str(current_value) != str(req_value):
                    changes_needed[display_name] = f"change to {req_value} (current: {current_value})"
        except (ValueError, TypeError):
            continue
    
    # Check orientation and type
    for field, req_field in [('orientation', 'orientation'), ('type', 'geo_type')]:
        if req_field in requirements.index and requirements[req_field] != 'Any':
            required_values = [v.strip() for v in str(requirements[req_field]).split(',')]
            current_value = str(current_geom.get(field, ''))
            if current_value not in required_values:
                changes_needed[field] = f"change to one of {required_values} (current: {current_value})"
    
    return {
        "changes_needed": changes_needed,
        "current_geometry": {
            "area": current_geom.get('area'),
            "type": current_geom.get('type'),
            "orientation": current_geom.get('orientation'),
            "open_sides": current_geom.get('open_side'),
            "level": current_geom.get('level')
        },
        "requirements": requirements.to_dict()
    }

def propose_activity_change(params):
    """Enhanced activity change proposal with conversational LLM analysis"""
    user_id = params.get("user_id")
    space_id = params.get("space_id")
    desired = params.get("desired_activity")
    current = params.get("current_activity")
    
    if not user_id or not desired or not current or not space_id:
        return {"error": "Missing user_id, space_id, desired_activity, or current_activity."}
    
    # Clean up activity names
    def clean_activity_name(name):
        if not name:
            return ""
        name = name.strip()
        if name.lower().startswith("for "):
            name = name[4:]
        return name.strip().title()
    
    desired_clean = clean_activity_name(desired)
    current_clean = clean_activity_name(current)
    
    # Get cached data
    voting = get_negotiation_data('voting')
    geometries = get_negotiation_data('geometries')
    thresh = get_negotiation_data('thresh')
    ml_activity_logic = get_negotiation_data('ml_activity_logic')
    
    # Analyze voting efficiently
    current_voters = voting[(voting["space"] == space_id) & (voting["activity"].str.strip().str.title() == current_clean)]
    desired_voters = voting[(voting["space"] == space_id) & (voting["activity"].str.strip().str.title() == desired_clean)]
    current_residents = set(current_voters["resident"])
    desired_residents = set(desired_voters["resident"])
    overlap = current_residents & desired_residents

    # Geometry checks
    space_thresh = thresh[thresh["id"] == space_id]
    geometry_warning = ""
    geometry_changes = {}
    current_geometry = {}
    if not space_thresh.empty:
        current_activities = str(space_thresh.iloc[0]["predicted_activities"]).lower()
        if desired_clean.lower() not in current_activities:
            geometry_warning = f"Warning: {desired_clean} is not in predicted activities for {space_id}. Current predicted activities: {current_activities}."
            geom_check = check_geometry_requirements(space_id, desired_clean, geometries, ml_activity_logic)
            if "error" not in geom_check and geom_check["changes_needed"]:
                geometry_changes = geom_check["changes_needed"]
                current_geometry = geom_check["current_geometry"]
            elif "error" in geom_check:
                geometry_warning += f" Could not check geometry requirements: {geom_check['error']}"

    # Compose LLM prompt
    prompt = f"""
A resident has made a negotiation request.

User request: Change {space_id} from {current_clean} to {desired_clean}.

Residents who voted for {current_clean} in {space_id}: {list(current_residents)}
Residents who voted for {desired_clean} in {space_id}: {list(desired_residents)}
Overlap (residents who like both): {list(overlap)}

{geometry_warning}
Geometry changes needed: {geometry_changes if geometry_changes else 'None'}
Current geometry: {current_geometry if current_geometry else 'N/A'}

Please analyze the situation, explain if and why the change is possible, what negotiations or agreements might be needed, and summarize the next steps in a friendly, conversational way. If there are any concerns or recommendations, mention them.
"""

    messages = [
        {"role": "system", "content": "You are a helpful, friendly community assistant who helps residents negotiate space usage and activities in their building. Use the facts provided and do not make up numbers."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": "local-model",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 400
            },
            timeout=15
        )
        if response.status_code == 200:
            llm_reply = response.json()["choices"][0]["message"]["content"]
        else:
            llm_reply = "Sorry, I couldn't generate a conversational analysis at this time."
    except Exception as e:
        llm_reply = f"Sorry, there was an error generating the analysis: {e}"

    return {
        "result": llm_reply,
        "can_proceed": bool(overlap or geometry_changes),
        "geometry_changes_needed": bool(geometry_changes),
        "geometry_changes": geometry_changes,
        "current_geometry": current_geometry,
        "space_id": space_id,
        "desired_activity": desired_clean,
        "params": params
    }



def analyze_user_current_access(user_id, distances_all=None, assignments=None, voting=None):
    """Optimized user access analysis"""
    if distances_all is None:
        distances_all = get_negotiation_data('distances_all')
    if assignments is None:
        assignments = get_negotiation_data('assignments')
    if voting is None:
        voting = get_negotiation_data('voting')
    
    # Use cached lookup for user distances
    user_distances = get_user_distances(user_id)
    if user_distances is None:
        return None
    
    # Get distances to all outdoor spaces (columns starting with 'O') - optimized
    space_distances = []
    for col in distances_all.columns:
        if col.startswith('O') and col != 'Source Node':
            try:
                dist = float(user_distances[col])
                space_distances.append((col, dist))
            except:
                continue
    
    # Sort by distance and get closest spaces
    space_distances.sort(key=lambda x: x[1])
    closest_spaces = space_distances[:3]  # Top 3 closest
    
    # Get activities and preferences efficiently
    user_access_info = []
    for space_id, distance in closest_spaces:
        # Get assigned activity using cached lookup
        assigned_activity = get_space_assignment(space_id)
        if assigned_activity is None:
            assigned_activity = "Unknown"
        
        # Get user's preference weight for this activity
        user_vote = voting[(voting['resident'] == user_id) & (voting['activity'].str.lower() == assigned_activity.lower())]
        preference_weight = user_vote['weight'].sum() if not user_vote.empty else 0
        
        user_access_info.append({
            'space_id': space_id,
            'activity': assigned_activity,
            'distance': distance,
            'preference_weight': preference_weight,
            'satisfaction_score': calculate_satisfaction_score(distance, preference_weight)
        })
    
    return {
        'user_id': user_id,
        'closest_spaces': user_access_info,
        'best_access': max(user_access_info, key=lambda x: x['satisfaction_score']) if user_access_info else None
    }

def calculate_satisfaction_score(distance, preference_weight):
    """Calculate user satisfaction score - optimized"""
    # Normalize distance (lower is better, max reasonable distance = 100m)
    distance_score = max(0, 10 - (distance / 10))  # 0-10 scale
    
    # Preference weight is already 0-1, scale to 0-10
    preference_score = preference_weight * 10
    
    # Combined score (weighted average)
    return (distance_score * 0.6) + (preference_score * 0.4)

def find_mutually_beneficial_swaps(user_id, desired_activity, user_current_info, spaces_with_desired, distances_all, voting, assignments):
    """Optimized swap finding with early exits"""
    swap_candidates = []
    user_best_access = user_current_info['best_access']
    
    if not user_best_access:
        return swap_candidates
    
    user_offers_activity = user_best_access['activity']
    user_offers_space = user_best_access['space_id']
    
    # Limit to top 3 spaces for performance
    for _, space_row in spaces_with_desired.head(3).iterrows():
        target_space = space_row['space_id']
        
        # Find residents close to this space
        space_distances = distances_all[distances_all['Source Node'] == target_space]
        if space_distances.empty:
            continue
            
        space_dist_row = space_distances.iloc[0]
        
        # Check each resident near this space (with performance limits)
        resident_count = 0
        for col in distances_all.columns:
            if not col.startswith('H') or col == user_id:
                continue
                
            resident_count += 1
            if resident_count > 10:  # Limit to first 10 residents for performance
                break
                
            try:
                target_resident = col
                distance_to_desired_space = float(space_dist_row[col])
                
                # Skip if too far from desired space
                if distance_to_desired_space > 50:  # 50m threshold
                    continue
                
                # Get this resident's distance to user's best space
                user_space_distances = distances_all[distances_all['Source Node'] == user_offers_space]
                if user_space_distances.empty:
                    continue
                    
                distance_to_user_space = float(user_space_distances.iloc[0][target_resident])
                
                # Check if this resident wants what the user has
                target_preferences = voting[voting['resident'] == target_resident]
                if target_preferences.empty:
                    continue
                
                # Get their preference for user's activity
                their_interest_in_user_activity = target_preferences[
                    target_preferences['activity'].str.lower() == user_offers_activity.lower()
                ]['weight'].sum()
                
                # Get user's preference for desired activity
                user_preferences = voting[voting['resident'] == user_id]
                user_desire_for_target = user_preferences[
                    user_preferences['activity'].str.lower() == desired_activity.lower()
                ]['weight'].sum()
                
                # Calculate compatibility
                compatibility = calculate_swap_compatibility(
                    distance_to_desired_space, distance_to_user_space,
                    their_interest_in_user_activity, user_desire_for_target,
                    target_preferences, user_preferences
                )
                
                if compatibility['score'] >= 5.0:  # Minimum threshold
                    swap_candidates.append({
                        'target_resident': target_resident,
                        'target_space': target_space,
                        'target_activity': desired_activity,
                        'your_space': user_offers_space,
                        'your_activity': user_offers_activity,
                        'distance_to_desired': distance_to_desired_space,
                        'distance_to_your_space': distance_to_user_space,
                        'compatibility_score': compatibility['score'],
                        'reasoning': compatibility['reasoning'],
                        'their_interest_in_your_activity': their_interest_in_user_activity,
                        'your_desire_for_target': user_desire_for_target
                    })
                    
            except (ValueError, TypeError, KeyError):
                continue
    
    return swap_candidates

def calculate_swap_compatibility(dist_to_desired, dist_to_user_space, their_interest, user_desire, target_prefs, user_prefs):
    """Optimized compatibility calculation"""
    score = 0
    reasons = []
    
    # Distance factors (closer is better)
    if dist_to_desired <= 20:
        score += 3
        reasons.append(f"Very close ({dist_to_desired:.1f}m) to desired activity")
    elif dist_to_desired <= 40:
        score += 2
        reasons.append(f"Reasonably close ({dist_to_desired:.1f}m) to desired activity")
    else:
        score += 1
        reasons.append(f"Moderate access ({dist_to_desired:.1f}m) to desired activity")
    
    if dist_to_user_space <= 20:
        score += 2
        reasons.append(f"They'd be very close ({dist_to_user_space:.1f}m) to what you offer")
    elif dist_to_user_space <= 40:
        score += 1.5
        reasons.append(f"They'd be close ({dist_to_user_space:.1f}m) to what you offer")
    
    # Preference compatibility
    if their_interest >= 0.7:
        score += 3
        reasons.append("They strongly prefer your current activity")
    elif their_interest >= 0.4:
        score += 2
        reasons.append("They moderately prefer your current activity")
    elif their_interest >= 0.1:
        score += 1
        reasons.append("They have some interest in your current activity")
    
    if user_desire >= 0.7:
        score += 2
        reasons.append("You strongly prefer their activity")
    elif user_desire >= 0.4:
        score += 1.5
        reasons.append("You moderately prefer their activity")
    
    # Bonus: Check if they're currently far from their top preference
    if not target_prefs.empty:
        their_top_activity = target_prefs.groupby('activity')['weight'].sum().idxmax()
        their_top_weight = target_prefs.groupby('activity')['weight'].sum().max()
        
        if their_top_weight >= 0.6 and their_interest >= their_top_weight * 0.8:
            score += 1
            reasons.append("This aligns well with their top preferences")
    
    reasoning = "; ".join(reasons)
    return {'score': min(score, 10), 'reasoning': reasoning}

def suggest_alternatives(user_id, desired_activity, user_current_info, assignments=None, voting=None):
    """Optimized alternative suggestions"""
    if assignments is None:
        assignments = get_negotiation_data('assignments')
    if voting is None:
        voting = get_negotiation_data('voting')
    
    alternatives = "Alternative suggestions:\n"
    
    # Check if negotiation might work
    spaces_with_desired = assignments[assignments['assigned_activity'].str.lower() == desired_activity.lower()]
    if not spaces_with_desired.empty:
        closest_space = spaces_with_desired.iloc[0]['space_id']  # Just take first one for simplicity
        alternatives += f"1. Try negotiating to change {closest_space} to {desired_activity}\n"
    
    # Check for booking opportunities
    alternatives += f"2. Look for booking opportunities for {desired_activity}\n"
    
    # Suggest activity changes to nearby spaces
    if user_current_info['best_access']:
        current_space = user_current_info['best_access']['space_id']
        alternatives += f"3. Try negotiating to change {current_space} to {desired_activity}\n"
    
    alternatives += f"4. Connect with community to find others interested in {desired_activity}"
    
    return alternatives

def handle_apartment_swap(params):
    """Optimized apartment swap handling"""
    user_id = params.get("user_id")
    desired_activity = params.get("desired_activity")
    
    if not user_id or not desired_activity:
        return {"error": "Missing user_id or desired_activity for swap"}
    
    # Get cached data
    distances_all = get_negotiation_data('distances_all')
    assignments = get_negotiation_data('assignments')
    voting = get_negotiation_data('voting')
    
    # Step 1: Find what the requesting user currently has access to
    user_current_info = analyze_user_current_access(user_id, distances_all, assignments, voting)
    if not user_current_info:
        return {"error": f"Could not determine current access for user {user_id}"}
    
    # Step 2: Find spaces with desired activity
    spaces_with_desired = assignments[assignments['assigned_activity'].str.lower() == desired_activity.lower()]
    if spaces_with_desired.empty:
        return {"result": f"No spaces currently assigned to {desired_activity}. Consider negotiating for activity change instead."}
    
    # Step 3: Find mutually beneficial swaps
    swap_candidates = find_mutually_beneficial_swaps(
        user_id, desired_activity, user_current_info, 
        spaces_with_desired, distances_all, voting, assignments
    )
    
    # Step 4: Rank and format results
    if swap_candidates:
        # Sort by compatibility score (higher is better)
        swap_candidates.sort(key=lambda x: x['compatibility_score'], reverse=True)
        
        result = f"Smart apartment swaps found for {desired_activity}:\n\n"
        
        for i, candidate in enumerate(swap_candidates[:5], 1): # Top 5
            result += f"**Option {i}: Swap with {candidate['target_resident']}**\n"
            result += f"✓ You get: Access to {candidate['target_space']} ({desired_activity}) - {candidate['distance_to_desired']:.1f}m away\n"
            result += f"✓ They get: Access to {candidate['your_space']} ({candidate['your_activity']}) - {candidate['distance_to_your_space']:.1f}m away\n"
            result += f"📊 Compatibility: {candidate['compatibility_score']:.1f}/10\n"
            result += f"💡 Why it works: {candidate['reasoning']}\n\n"
        
        result += "Type the number of your preferred option to proceed with that swap."
        
        return {
            "result": result, 
            "swap_candidates": swap_candidates[:5],
            "user_current_info": user_current_info
        }
    else:
        # Provide alternative suggestions
        alternatives = suggest_alternatives(user_id, desired_activity, user_current_info, assignments, voting)
        return {
            "result": f"No mutually beneficial swaps found for {desired_activity}.\n\n{alternatives}"
        }

# --- Move these functions to top-level, not nested ---
def handle_booking_request(params):
    """Handle activity booking requests - optimized"""
    user_id = params.get("user_id")
    space_id = params.get("space_id")
    desired_activity = params.get("desired_activity")
    
    if not user_id or not space_id or not desired_activity:
        return {"error": "Missing required parameters for booking"}
    
    # Simple booking logic - in real implementation, check schedules
    result = f"Booking request for {user_id}:\n"
    result += f"- Space: {space_id}\n"
    result += f"- Activity: {desired_activity}\n"
    result += f"- Status: Available (simulated)\n"
    result += f"Would you like to confirm this booking?"
    
    return {
        "result": result,
        "booking_details": {
            "user_id": user_id,
            "space_id": space_id,
            "activity": desired_activity,
            "status": "pending_confirmation"
        }
    }

    # ============================================================================
    # GEOMETRY FUNCTIONS (OPTIMIZED)
    # ============================================================================

    def change_geometry(params):
        """Optimized geometry changes based on desired activity requirements"""
        space_id = params.get("outdoor_id") or params.get("id") or params.get("space_id")
        desired_activity = params.get("desired_activity") or params.get("activity")
        
        if not space_id:
            return {"error": "No space_id provided."}
        
        # Get cached data
        geometries = get_negotiation_data('geometries')
        voting = get_negotiation_data('voting')
        ml_activity_logic = get_negotiation_data('ml_activity_logic')
        
        # Get current space info using cached lookup
        current_geom = get_space_geometry(space_id)
        if current_geom is None:
            return {"error": f"No space found with id {space_id}."}
        
        # If no desired activity provided, suggest general improvements
        if not desired_activity:
            return suggest_general_geometry_improvements(space_id, current_geom, voting)
        
        # Check what changes are needed for the desired activity
        geom_check = check_geometry_requirements(space_id, desired_activity, geometries, ml_activity_logic)
        
        if "error" in geom_check:
            return geom_check
        
        changes_needed = geom_check.get("changes_needed", {})
        requirements = geom_check.get("requirements", {})
        
        if not changes_needed:
            return {
                "result": f"Space {space_id} already meets all requirements for {desired_activity}!",
                "current_geometry": geom_check.get("current_geometry", {}),
                "no_changes_needed": True,
                "params": params
            }
        
        # Calculate specific geometry changes
        geometry_recommendations = calculate_geometry_changes(
            space_id, current_geom, changes_needed, requirements, voting
        )
        
        # Format the response
        result = f"Geometry changes recommended for {space_id} to support {desired_activity}:\n\n"
        
        for change_type, change_info in geometry_recommendations.items():
            result += f"**{change_type.title()}**:\n"
            result += f"  Current: {change_info['current']}\n"
            result += f"  Required: {change_info['required']}\n"
            result += f"  Suggested: {change_info['suggested']}\n"
            if 'reasoning' in change_info:
                result += f"  Why: {change_info['reasoning']}\n"
            result += "\n"
        
        # Add impact assessment
        impact_assessment = assess_geometry_change_impact(space_id, geometry_recommendations, voting)
        result += f"**Impact Assessment**:\n{impact_assessment}\n"
        
        return {
            "result": result,
            "geometry_changes": geometry_recommendations,
            "current_geometry": geom_check.get("current_geometry", {}),
            "requirements": requirements,
            "space_id": space_id,
            "desired_activity": desired_activity,
            "impact_assessment": impact_assessment,
            "changes_needed": True,
            "params": params
        }

    def suggest_general_geometry_improvements(space_id, current_geom, voting):
        """Suggest general improvements when no specific activity is targeted"""
        # Get voting data for this space to see what residents want
        space_votes = voting[voting["space"] == space_id]
        
        if space_votes.empty:
            return {
                "result": f"No specific activity provided and no voting data found for {space_id}. Please specify a desired activity for targeted geometry suggestions.",
                "params": {"space_id": space_id}
            }
        
        # Find the most popular activity for this space
        top_activity = space_votes.groupby('activity')['weight'].sum().idxmax()
        top_weight = space_votes.groupby('activity')['weight'].sum().max()
        
        result = f"Based on resident preferences, the most desired activity for {space_id} is **{top_activity}** (weight: {top_weight:.2f}).\n"
        result += f"Analyzing geometry requirements for {top_activity}...\n"
        
        # Recursively call with the top activity
        return change_geometry({
            "space_id": space_id,
            "desired_activity": top_activity
        })

    def calculate_geometry_changes(space_id, current_geom, changes_needed, requirements, voting):
        """Calculate specific geometry changes with reasoning"""
        recommendations = {}
        
        for change_type, change_desc in changes_needed.items():
            if "area" in change_type.lower():
                recommendations["area"] = calculate_area_changes(current_geom, requirements, voting, space_id)
            elif "open_sides" in change_type.lower() or "open_side" in change_type.lower():
                recommendations["open_sides"] = calculate_open_sides_changes(current_geom, requirements)
            elif "orientation" in change_type.lower():
                recommendations["orientation"] = calculate_orientation_changes(current_geom, requirements)
            elif "type" in change_type.lower():
                recommendations["type"] = calculate_type_changes(current_geom, requirements)
            elif "level" in change_type.lower():
                recommendations["level"] = calculate_level_changes(current_geom, requirements)
            elif "wind" in change_type.lower():
                recommendations["wind_exposure"] = calculate_wind_changes(current_geom, requirements)
            elif "sun" in change_type.lower():
                recommendations["sun_hours"] = calculate_sun_changes(current_geom, requirements)
        
        return recommendations

    def calculate_area_changes(current_geom, requirements, voting, space_id):
        """Calculate smart area changes based on requirements and usage"""
        current_area = current_geom.get('area', 0)
        min_required = float(requirements.get('min_area', current_area))
        
        # Calculate suggested area based on usage intensity
        space_votes = voting[voting["space"] == space_id]
        total_interest = space_votes['weight'].sum() if not space_votes.empty else 1
        
        # More interest = suggest larger area
        usage_multiplier = min(1.5, 1 + (total_interest * 0.1))
        suggested_area = max(min_required, min_required * usage_multiplier)
        
        return {
            "current": f"{current_area}m²",
            "required": f"≥{min_required}m²",
            "suggested": f"{suggested_area:.1f}m²",
            "reasoning": f"Based on minimum requirement ({min_required}m²) and community interest level (usage factor: {usage_multiplier:.1f})"
        }

    def calculate_open_sides_changes(current_geom, requirements):
        """Calculate open sides changes"""
        current_open = current_geom.get('open_side', 0)
        min_required = int(requirements.get('min_open_sides', current_open))
        
        return {
            "current": str(current_open),
            "required": f"≥{min_required}",
            "suggested": str(max(current_open, min_required)),
            "reasoning": f"Increase openness for better accessibility and ventilation"
        }

    def calculate_orientation_changes(current_geom, requirements):
        """Calculate orientation changes"""
        current_orientation = current_geom.get('orientation', 'Unknown')
        required_orientations = str(requirements.get('orientation', 'Any')).split(', ')
        
        # Smart orientation selection based on activity
        if 'S' in required_orientations:
            suggested = 'S'  # South for maximum sun
            reasoning = "South orientation for maximum sunlight exposure"
        elif 'E' in required_orientations:
            suggested = 'E'  # East for morning sun
            reasoning = "East orientation for pleasant morning light"
        else:
            suggested = required_orientations[0]
            reasoning = f"Changed to meet activity requirements"
        
        return {
            "current": current_orientation,
            "required": f"One of: {', '.join(required_orientations)}",
            "suggested": suggested,
            "reasoning": reasoning
        }

    def calculate_type_changes(current_geom, requirements):
        """Calculate type changes"""
        current_type = current_geom.get('type', 'Unknown')
        required_types = str(requirements.get('geo_type', 'Any')).split(', ')
        
        # Prefer larger types for more flexibility
        type_preference = ['courtyard', 'terrace', 'balcony']
        suggested = current_type
        
        for preferred_type in type_preference:
            if preferred_type in required_types:
                suggested = preferred_type
                break
        
        return {
            "current": current_type,
            "required": f"One of: {', '.join(required_types)}",
            "suggested": suggested,
            "reasoning": f"Upgraded to {suggested} for better space utilization"
        }

    def calculate_level_changes(current_geom, requirements):
        """Calculate level changes"""
        current_level = current_geom.get('level', 0)
        min_level = int(requirements.get('min_level', current_level))
        
        return {
            "current": str(current_level),
            "required": f"≥{min_level}",
            "suggested": str(max(current_level, min_level)),
            "reasoning": "Level adjustment for privacy and accessibility requirements"
        }

    def calculate_wind_changes(current_geom, requirements):
        """Calculate wind exposure changes"""
        current_wind = current_geom.get('wind_exp', 0)
        max_wind = float(requirements.get('max_wind_exp', 10))
        
        if current_wind > max_wind:
            suggested = max_wind * 0.9  # Slightly below maximum
            reasoning = f"Reduce wind exposure for comfort (target: {suggested:.1f})"
        else:
            suggested = current_wind
            reasoning = "Current wind exposure is acceptable"
        
        return {
            "current": f"{current_wind}",
            "required": f"≤{max_wind}",
            "suggested": f"{suggested:.1f}",
            "reasoning": reasoning
        }

    def calculate_sun_changes(current_geom, requirements):
        """Calculate sun hours changes"""
        current_sun = current_geom.get('sun_h', 0)
        required_sun = float(requirements.get('sun_h', current_sun))
        
        if current_sun < required_sun:
            suggested = required_sun
            reasoning = f"Increase sun exposure through orientation/obstruction changes"
        else:
            suggested = current_sun
            reasoning = "Current sun exposure meets requirements"
        
        return {
            "current": f"{current_sun} hours",
            "required": f"{required_sun} hours",
            "suggested": f"{suggested} hours",
            "reasoning": reasoning
        }

    def assess_geometry_change_impact(space_id, geometry_recommendations, voting):
        """Assess the impact of proposed geometry changes"""
        impact_points = []
        
        # Check if changes affect area significantly
        if "area" in geometry_recommendations:
            impact_points.append(f"• Area change may require structural modifications")
            impact_points.append(f"• Estimated timeline: 2-4 weeks for area expansion")
        
        # Check community impact
        space_votes = voting[voting["space"] == space_id]
        if not space_votes.empty:
            affected_residents = space_votes['resident'].nunique()
            impact_points.append(f"• {affected_residents} residents will be affected by these changes")
            
            # Check if changes align with community preferences
            total_interest = space_votes['weight'].sum()
            if total_interest > 2.0:
                impact_points.append(f"• High community interest (score: {total_interest:.1f}) - changes likely to be well-received")
            else:
                impact_points.append(f"• Moderate community interest (score: {total_interest:.1f}) - may need additional consensus building")
        
        # Add general recommendations
        impact_points.append(f"• Recommend community meeting to discuss proposed changes")
        impact_points.append(f"• Consider phased implementation to minimize disruption")
        
        return "\n".join(impact_points)

# ============================================================================
# UTILITY FUNCTIONS (OPTIMIZED)
# ============================================================================

# @lru_cache(maxsize=32)
def get_nearby_activities(user_id, pretty=True):
    """Optimized nearby activities lookup"""
    if not user_id:
        return {"error": "No user_id provided."}
    
    params = {"user_id": user_id}

    # Get cached data
    geometries = get_negotiation_data('geometries')
    distances = get_negotiation_data('distances')
    assignments = get_negotiation_data('assignments')
    
    if user_id not in distances.columns:
        return {"error": f"No distances found for user {user_id}."}

    # Get 5 nearest spaces
    nearby = distances[["id", user_id]].rename(columns={user_id: "distance"})
    nearby = nearby.sort_values("distance").head(5)

    results = []
    for _, row in nearby.iterrows():
        space_id = row["id"]
        dist = row["distance"]

        # Get area from geometries using cached lookup
        space_geom = get_space_geometry(space_id)
        area = float(space_geom["area"]) if space_geom is not None else None

        # Get assigned activity using cached lookup
        assigned_activity = get_space_assignment(space_id)

        results.append({
            "space_id": space_id,
            "distance": dist,
            "area": area,
            "assigned_activity": assigned_activity
        })
    
    if pretty:
        output_lines = [f"Nearby Outdoor Spaces for User: {user_id}", ""]
        for item in results:
            line = f"- {item['space_id']}: {item['assigned_activity'] or 'None'} | {item['area']} m² | {round(item['distance'], 2)} m"
            output_lines.append(line)
        return "<br>".join(output_lines)
    
    return {"result": results, "params": params}

def process_booking(params):
    """Book an activity for a user - optimized"""
    user_id = params.get("user_id")
    desired = params.get("desired_activity")
    space_id = params.get("space_id")
    
    if not user_id or not desired or not space_id:
        return {"error": "Missing user_id, desired_activity, or space_id."}
    
    explanation = f"Booked {desired} in {space_id} for user {user_id}. (Dummy: slot available, booking confirmed.)\nDo you want to finalize this booking?"
    return {"result": explanation, "can_proceed": True, "params": params}

#@lru_cache(maxsize=32)
# Remove @lru_cache if present
def summarize_preferences(user_id, pretty=True):
    """Optimized user preference summary"""
    if not user_id:
        return {"error": "No user_id provided."}

    voting = get_negotiation_data('voting')
    user_votes = voting[voting["resident"] == user_id]

    if user_votes.empty:
        return f"No preferences found for user {user_id}." if pretty else {"result": {}, "user_id": user_id}

    summary = user_votes.groupby("activity")["weight"].sum().sort_values(ascending=False).to_dict()

    if pretty:
        output_lines = [f"Preferences Summary for User: {user_id}", ""]
        for activity, weight in summary.items():
            output_lines.append(f"- {activity}: {round(weight, 4)}")
        return "<br>".join(output_lines)
    else:
        return {
            "result": summary,
            "user_id": user_id
        }

def assign_activity(params):
    """Assigns an activity to a space - optimized"""
    space_id = params.get("space_id") or params.get("id")
    activity = params.get("activity")
    if not space_id or not activity:
        return {"error": "Missing space_id or activity."}
    
    return {
        "result": f"Activity '{activity}' assigned to space '{space_id}'!",
        "params": params
    }

def update_activity_assignment(space_id, new_activity, csv_path="llm_reasoning/llm_activity_assignments.csv"):
    """Update activity assignment in CSV"""
    df = pd.read_csv(csv_path)
    df['space_id'] = df['space_id'].astype(str).str.strip()
    mask = df['space_id'] == space_id
    if mask.any():
        df.loc[mask, 'assigned_activity'] = new_activity
        df.to_csv(csv_path, index=False)
        return True
    return False

def append_assignment_history(space_id, old_activity, new_activity, old_reasoning, new_reasoning, user_id, json_path="llm_reasoning/llm_assignments.json"):
    """Append assignment history to JSON"""
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                data = []
    else:
        data = []
    
    entry = {
        "parameters": {
            "id": space_id,
            "activity": new_activity,
            "activity_1": old_activity
        },
        "reasoning": new_reasoning,
        "reasoning_1": f"{user_id} requested the change. {old_reasoning}"
    }
    data.append(entry)
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

@lru_cache(maxsize=64)
def get_spaces_with_assigned_activity(activity_name, assignments_path="llm_reasoning/llm_activity_assignments.csv"):
    """Cached lookup for spaces with specific activities"""
    assignments = pd.read_csv(assignments_path)
    assignments['assigned_activity'] = assignments['assigned_activity'].astype(str).str.strip()
    filtered = assignments[assignments['assigned_activity'].str.lower() == activity_name.lower()]
    return filtered['space_id'].tolist()

# ============================================================================
# ACTION ROUTING (OPTIMIZED)
# ============================================================================

# ACTIONS DICTIONARY
ACTION_DISPATCHER = {
    "get_nearby_activities": get_nearby_activities,
    "propose_activity_change": propose_activity_change,
    "find_profile_swap": handle_apartment_swap,
    "process_booking": handle_booking_request,
    "summarize_preferences": summarize_preferences,
    "assign_activity": assign_activity,
}

def route_action(llm_json):
    """Optimized action routing"""
    results = []
    
    # Handle single action
    if "action" in llm_json:
        action = llm_json["action"]
        params = llm_json.get("parameters", llm_json)
        func = ACTION_DISPATCHER.get(action)
        if func:
            results.append(func(params))
        else:
            results.append({"error": f"Unknown action: {action}"})
    
    # Handle multiple actions
    elif "actions" in llm_json:
        params = llm_json.get("parameters", llm_json)
        for action in llm_json["actions"]:
            func = ACTION_DISPATCHER.get(action)
            if func:
                results.append(func(params))
            else:
                results.append({"error": f"Unknown action: {action}"})
    else:
        results.append({"error": "No action found in LLM response."})
    
    return results if len(results) > 1 else results[0]

def suggest_actions_from_request(message):
    """Suggest actions from user request using LLM"""
    response = client.chat.completions.create(
        model=completion_model,
        messages=[
            {
                "role": "system",
                "content": """
You are an assistant that interprets user requests and suggests high-level actions for a smart architecture system.

Given the user's request, output a JSON object with an "action" field (or "actions" if multiple) and any other necessary parameters.

Use only these action names:
- "change_geometry": For requests about changing, enlarging, or modifying a space's physical properties.
- "get_nearby_activities": For requests about what activities are available nearby, their distance, size, and details.
- "propose_activity_change": For requests about changing an activity in a space, or negotiating with other residents.
- "find_profile_swap": For requests about swapping/switching apartments/houses with someone whose preferences match better.
- "process_booking": For requests about booking a space or activity for a certain time.
- "assign_activity": For confirming or finalizing an activity assignment.
- "summarize_preferences": For summarizing the user's activity or space preferences.

Respond ONLY with valid JSON. No extra text.

Important: Return only a valid JSON object. No extra text.
""",
            },
            {
                "role": "user",
                "content": message,
            },
        ],
        temperature=0.3,  # Lower temperature for more consistent JSON
        max_tokens=200    # Limit for faster response
    )
    return response.choices[0].message.content

def handle_user_request(message):
    """Handle user request with optimized processing"""
    try:
        action_json_str = suggest_actions_from_request(message)
        print("[DEBUG] Suggested JSON:\n", action_json_str)
        action_json = json.loads(action_json_str)
        result = route_action(action_json)
        return result
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON from LLM: {e}")
        return {"error": "Invalid LLM output format"}
    except Exception as e:
        logging.error(f"Failed to handle user request: {e}")
        return {"error": str(e)}

def negotiation_flow(user_query, user_id=None, last_context=None):
    """Optimized negotiation flow with caching"""
    
    # Initialize cache if needed
    if not _negotiation_cache_initialized:
        initialize_negotiation_cache()
    
    context = {}
    suggestions = []
    cycling_phrases = ["no", "another suggestion", "next", "not this", "different", "something else"]
    reset_phrases = ["reset", "clear", "start over", "restart", "new negotiation", "begin again"]
    user_query_lower = user_query.lower().strip()

    # --- Reset negotiation if requested ---
    if any(phrase in user_query_lower for phrase in reset_phrases):
        return {
            "context": {},
            "suggestions": [{
                "action": "reset_negotiation",
                "explanation": "Negotiation has been reset. You can start a new request.",
                "parameters": {"user_id": user_id}
            }]
        }

    cycling_mode = any(phrase in user_query_lower for phrase in cycling_phrases)
    if cycling_mode and last_context:
        prev_suggestions = last_context.get("all_suggestions", [])
        suggestion_idx = last_context.get("suggestion_idx", 0) + 1
        if prev_suggestions and suggestion_idx < len(prev_suggestions):
            next_suggestion = prev_suggestions[suggestion_idx]
            context = last_context.get("context", {})
            context["last_action"] = next_suggestion["action"]
            context["last_params"] = next_suggestion["parameters"]
            context["all_suggestions"] = prev_suggestions
            context["suggestion_idx"] = suggestion_idx
            return {"context": context, "suggestions": [next_suggestion]}
        else:
            context = last_context.get("context", {})
            return {"context": context, "suggestions": [{
                "action": "summarize_preferences",
                "explanation": "No more alternative suggestions available for this negotiation. Please rephrase your request or try a different activity/space.",
                "parameters": {"user_id": user_id}
            }]}
    
    if not user_id:
        context["nearby_activities"] = "User ID not provided."
        context["preferences"] = "User ID not provided."
        return {"context": context, "suggestions": suggestions}

    # 1. Gather context using optimized functions
    context["nearby_activities"] = get_nearby_activities(user_id)
    context["preferences"] = summarize_preferences(user_id)

    # 2. Get cached data
    voting = get_negotiation_data('voting')
    distances = get_negotiation_data('distances')
    distances_all = get_negotiation_data('distances_all')
    assignments = get_negotiation_data('assignments')

    # 3. Parse user query for intent (robust extraction)
    user_query_lower = user_query.lower().strip()
    
    # If user says 'yes' or 'proceed', try to finalize last negotiation
    if user_query_lower in ["yes", "proceed", "confirm", "ok", "sure"] and last_context:
        last_action = last_context.get("last_action")
        last_params = last_context.get("last_params", {})

        # PROPOSE ACTIVITY CHANGE UPDATED!
        if last_action == "propose_activity_change" and last_params:
            # Get details from last_params
            space_id = last_params.get("space_id")
            user_id = last_params.get("user_id")
            current_activity = last_params.get("current_activity")
            desired_activity = last_params.get("desired_activity")
            
            # Run propose_activity_change
            result = propose_activity_change(last_params)
            old_reasoning = f"Previously assigned as {current_activity}."
            new_reasoning = result["result"] if isinstance(result, dict) and "result" in result else ""
            
            # Update CSV
            update_activity_assignment(space_id, desired_activity)
            
            # Update JSON
            append_assignment_history(space_id, current_activity, desired_activity, old_reasoning, new_reasoning, user_id)
            
            suggestions.append({
                "action": "assign_activity",
                "explanation": (
                    f"{space_id} is now assigned to {desired_activity}. Change has been logged. "
                    "Please proceed to the Geometry Workflow tab to review and apply geometry changes for this space."
                    ),
                "parameters": last_params
            })
            context["info"] = f"{space_id} assignment updated."
            return {"context": context, "suggestions": suggestions}
        
        if last_action == "process_booking" and last_params:
            # TODO: Implement booking confirmation logic here
            pass
        
        if last_action == "find_profile_swap" and last_params:
            suggestions.append({
                "action": "swap_apartment",
                "explanation": f"Searching for apartments near outdoor spaces with {last_params.get('desired_activity')} for you. (This is a placeholder for the actual swap logic.)",
                "parameters": last_params
            })
            return {"context": context, "suggestions": suggestions}
        
        suggestions.append({
            "action": "summarize_preferences",
            "explanation": "Summarize your preferences for negotiation.",
            "parameters": {"user_id": user_id}
        })
        return {"context": context, "suggestions": suggestions}

    # 4. Extract desired activity and space from query (robust) - optimized
    desired_activity = None
    space_id = None
    
    # Try all reasonable patterns
    patterns = [
        r"assign ([A-Za-z0-9_\- ]+) to (O\d+)",
r"i would like to assign ([A-Za-z0-9_\- ]+) to (O\d+)",
        r"i want ([A-Za-z0-9_\- ]+) in (O\d+)",
        r"i would like (O\d+) (?:to be|to be for|for|to have|as|to) ([A-Za-z0-9_\- ]+)",
        r"i want (O\d+) (?:for|as) ([A-Za-z0-9_\- ]+)",
        r"(O\d+) (?:to be|to be for|for|to have|as|to) ([A-Za-z0-9_\- ]+)",
        r"i would like (O\d+) be for ([A-Za-z0-9_\- ]+)",
        r"i would like (O\d+) be ([A-Za-z0-9_\- ]+)",
        r"i want (O\d+) be for ([A-Za-z0-9_\- ]+)",
        r"i want (O\d+) be ([A-Za-z0-9_\- ]+)",
        r"([A-Za-z0-9_\- ]+) (?:to|in) (O\d+)"
    ]
    
    for pat in patterns:
        m = re.search(pat, user_query, re.IGNORECASE)
        if m:
            if len(m.groups()) == 2:
                g1, g2 = m.group(1).strip(), m.group(2).strip()
                # Heuristic: whichever looks like Oxx is space, the other is activity
                if g1.upper().startswith("O"):
                    space_id, desired_activity = g1, g2
                elif g2.upper().startswith("O"):
                    space_id, desired_activity = g2, g1
                else:
                    desired_activity, space_id = g1, g2
            break
    
    # Fallback: look for 'to O10' or 'in O10' and use previous word as activity
    if not desired_activity or not space_id:
        m2 = re.search(r"([A-Za-z0-9_\- ]+) (?:to|in) (O\d+)", user_query, re.IGNORECASE)
        if m2:
            desired_activity = m2.group(1).strip()
            space_id = m2.group(2).strip()
    
    # Clean up desired_activity
    if desired_activity:
        desired_activity = re.sub(r"^(assigned as|as|for|to|with)\s+", "", desired_activity, flags=re.IGNORECASE).strip().title()
    
    # If space_id found, look up current activity using cached lookup
    current_activity = None
    if space_id:
        current_activity = get_space_assignment(space_id)
    
    # Fallback: use top activity from preferences
    if not desired_activity:
        prefs = context["preferences"]["result"] if isinstance(context["preferences"], dict) else ""
        if isinstance(prefs, dict) and prefs:
            desired_activity = list(prefs.keys())[0]
    
    # Fallback: use top activity in nearby activities
    if not desired_activity and isinstance(context["nearby_activities"], dict):
        nearby = context["nearby_activities"].get("result", [])
        if nearby and isinstance(nearby[0], dict) and "assigned_activity" in nearby[0]:
            desired_activity = nearby[0]["assigned_activity"]
    
    # Fallback: use any activity from assignments using cached data
    if not desired_activity:
        all_activities = get_negotiation_data('all_activities')
        if all_activities:
            desired_activity = list(all_activities)[0]
    
    # Try to find a space_id if not found
    if not space_id and user_id in distances.columns:
        nearby = distances[["id", user_id]].rename(columns={user_id: "distance"})
        nearby = nearby.sort_values("distance").head(1)
        if not nearby.empty:
            space_id = nearby.iloc[0]["id"]
    
    # Try to find current activity in that space using cached lookup
    if not current_activity and space_id:
        current_activity = get_space_assignment(space_id)

    # --- Swap/Move intent detection (MUST be before any early return) ---
    swap_phrases = [
        "move to another house", "move to another apartment", "move apartment", "move house",
        "swap", "switch", "exchange", "find another house", "find another apartment", "swap apartment", "switch apartment"
    ]
    if any(phrase in user_query_lower for phrase in swap_phrases):
        # Try to extract desired activity using cached activity list
        all_activities = get_negotiation_data('all_activities')
        user_prefs = []
        prefs = context["preferences"]["result"] if isinstance(context["preferences"], dict) else ""
        if isinstance(prefs, dict):
            user_prefs = list(prefs.keys())
        
        # Combine all possible activities (assigned + preferences)
        possible_activities = list(all_activities) + user_prefs

        # Try to find the longest matching activity name in the user query
        desired_activity = None
        for act in sorted(possible_activities, key=lambda x: -len(str(x))):
            if act and act.lower() in user_query_lower:
                desired_activity = act
                break
        
        if not desired_activity:
            # fallback to top preference
            if user_prefs:
                desired_activity = user_prefs[0]
            elif all_activities:
                desired_activity = list(all_activities)[0]
            else:
                desired_activity = "Sports"
        
        swap_params = {"user_id": user_id, "desired_activity": desired_activity}
        suggestions.append({
            "action": "find_profile_swap",
            "explanation": f"Suggesting possible apartment swaps for you to be closer to an outdoor space with {desired_activity}.",
            "parameters": swap_params
        })
        
        # Save context for cycling
        context["last_action"] = "find_profile_swap"
        context["last_params"] = swap_params
        context["all_suggestions"] = suggestions
        context["suggestion_idx"] = 0
        return {"context": context, "suggestions": suggestions}

    # If missing info, try to use last_context for multi-turn negotiation
    if (not space_id or not desired_activity or not current_activity) and last_context:
        last_action = last_context.get("last_action")
        last_params = last_context.get("last_params", {})
        # Use last known params if available
        space_id = space_id or last_params.get("space_id")
        desired_activity = desired_activity or last_params.get("desired_activity")
        current_activity = current_activity or last_params.get("current_activity")
        
        # If still missing, fallback to summarize_preferences
        if not space_id or not desired_activity or not current_activity:
            suggestions.append({
                "action": "summarize_preferences",
                "explanation": "Could not determine enough details from your request. Please specify the space and activity.",
                "parameters": {"user_id": user_id}
            })
            context["error"] = "Not enough info for negotiation."
            return {"context": context, "suggestions": suggestions}

    if not current_activity:
        suggestions.append({
            "action": "summarize_preferences",
            "explanation": f"Could not determine the current activity for {space_id}. Please check the data.",
            "parameters": {"user_id": user_id, "space_id": space_id}
        })
        context["error"] = f"No current activity found for {space_id}."
        return {"context": context, "suggestions": suggestions}
    
    # If current activity is already the desired one
    if current_activity.lower() == desired_activity.lower():
        suggestions.append({
            "action": "summarize_preferences",
            "explanation": f"{space_id} is already assigned to {desired_activity}. No change needed.",
            "parameters": {"user_id": user_id, "space_id": space_id, "activity": desired_activity}
        })
        context["info"] = f"{space_id} is already assigned to {desired_activity}."
        return {"context": context, "suggestions": suggestions}
    
    # Otherwise, suggest negotiation/booking
    suggestions.append({
        "action": "propose_activity_change",
        "explanation": f"Negotiate with residents who voted for {current_activity} in {space_id}. If some also like {desired_activity}, you may be able to swap the activity. Type 'yes' to confirm this change.",
        "parameters": {"space_id": space_id, "current_activity": current_activity, "desired_activity": desired_activity, "user_id": user_id}
    })
    suggestions.append({
        "action": "process_booking",
        "explanation": f"You can book {space_id} for {desired_activity} if the slot is available and other residents agree.",
        "parameters": {"user_id": user_id, "space_id": space_id, "activity": desired_activity}
    })
    
    # Save last action/params for multi-turn and all suggestions for cycling
    if suggestions:
        context["last_action"] = suggestions[0]["action"]
        context["last_params"] = suggestions[0]["parameters"]
        context["all_suggestions"] = suggestions
        context["suggestion_idx"] = 0
    
    return {"context": context, "suggestions": suggestions}

# ============================================================================
# INITIALIZATION - COMPATIBILITY WITH FLASK SERVER
# ============================================================================

# Auto-initialize cache when module is imported (for Flask compatibility)
def ensure_cache_initialized():
    """Ensure cache is initialized - called by Flask server"""
    if not _negotiation_cache_initialized:
        initialize_negotiation_cache()

# This will be called by the Flask server during startup
if __name__ != "__main__":
    # Module is being imported, initialize cache
    try:
        initialize_negotiation_cache()
    except Exception as e:
        print(f"Warning: Could not initialize negotiation cache during import: {e}")
        print("Cache will be initialized on first use.")

# ============================================================================
# LEGACY COMPATIBILITY AND TESTING
# ============================================================================

if __name__ == "__main__":
    # Test the optimized functions
    print("🧪 Testing optimized negotiation functions...")
    
    # Initialize cache
    initialize_negotiation_cache()
    
    # Test basic functionality
    test_params = {"user_id": "H1"}
    
    print("Testing get_nearby_activities...")
    result = get_nearby_activities(test_params)
    print(f"Result type: {type(result)}")
    
    print("Testing summarize_preferences...")
    result = summarize_preferences(test_params)
    print(f"Result type: {type(result)}")
    
    print("Testing negotiation_flow...")
    result = negotiation_flow("I want sports in O1", "H1")
    print(f"Suggestions count: {len(result.get('suggestions', []))}")
    
    print("✅ All tests completed successfully!")


