import pandas as pd

# --- Load data ---
level1 = pd.read_excel('sql/example.xlsx', sheet_name='level1')
level2 = pd.read_excel('sql/example.xlsx', sheet_name='level2')
level3 = pd.read_excel('sql/example.xlsx', sheet_name='level3')
level4 = pd.read_excel('sql/example.xlsx', sheet_name='level4')
activity_space = pd.read_excel('sql/example.xlsx', sheet_name='activity_space')

levels = {
    1: level1,
    2: level2,
    3: level3,
    4: level4
}

results = []

for level_num, level_df in levels.items():
    # Filter activity_space for the given level
    activity_space_level = activity_space[activity_space['level'] == level_num]

    # Get all possible activities for this level
    possible_activities = set()
    for acts in activity_space_level['activity']:
        for act in str(acts).split(','):
            possible_activities.add(act.strip())

    # Get all requested activities from residents on this level
    requested_col = f'related_activity{level_num}'
    if requested_col not in level_df.columns:
        print(f"Column {requested_col} not found in level{level_num} sheet.")
        continue
    requested_activities = set(level_df[requested_col].unique())

    # Compare
    matching_activities = requested_activities & possible_activities
    not_possible_activities = requested_activities - possible_activities

    # Save results for this level
    results.append({
        "level": level_num,
        "matching_activities": sorted(matching_activities),
        "not_possible_activities": sorted(not_possible_activities)
    })

    # Print summary
    print(f"\nLevel {level_num}:")
    print("Activities requested by residents that are possible in activity_space:")
    print(matching_activities)
    print("Activities requested by residents that are NOT possible in activity_space:")
    print(not_possible_activities)

# --- Total building comparison ---

# Aggregate all requested activities from all levels
all_requested_activities = set()
for level_num, level_df in levels.items():
    requested_col = f'related_activity{level_num}'
    if requested_col in level_df.columns:
        all_requested_activities.update(level_df[requested_col].unique())

# Aggregate all possible activities from all activity_space rows
all_possible_activities = set()
for acts in activity_space['activity']:
    for act in str(acts).split(','):
        all_possible_activities.add(act.strip())

# Compare
building_matching_activities = all_requested_activities & all_possible_activities
building_not_possible_activities = all_requested_activities - all_possible_activities

# Save results for the building
results.append({
    "level": "building_total",
    "matching_activities": sorted(building_matching_activities),
    "not_possible_activities": sorted(building_not_possible_activities)
})

# Print summary
print(f"\nTotal Building:")
print("Activities requested by residents that are possible in activity_space:")
print(building_matching_activities)
print("Activities requested by residents that are NOT possible in activity_space:")
print(building_not_possible_activities)

# --- Save results as TXT in knowledge folder ---
with open("knowledge/compare_results.txt", "w", encoding="utf-8") as f:
    for r in results:
        f.write(f"Level {r['level']}:\n")
        f.write(f"  Matching activities: {', '.join(r['matching_activities'])}\n")
        f.write(f"  Not possible activities: {', '.join(r['not_possible_activities'])}\n\n")

print("\nResults saved as knowledge/compare_results.txt")