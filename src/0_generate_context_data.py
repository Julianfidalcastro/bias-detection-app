import pandas as pd
import random
import os

# CONFIG
OUTPUT_FILE = 'data/recruitment_data_context.csv'
NUM_PAIRS = 1000  # We will generate 1000 Good and 1000 Bad examples

# SKILLS TO TEACH
skills = ['Python', 'Java', 'SQL', 'Machine Learning', 'AWS', 'Communication', 'Project Management', 'Leadership', 'Sales']

# TEMPLATES
# {skill} will be replaced by the skill name (e.g., Python)
good_templates = [
    "Expert in {skill} with 5 years of experience.",
    "Mastered {skill} and built complex systems.",
    "Strong proficiency in {skill} and its frameworks.",
    "Led a team to implement {skill} solutions.",
    "Advanced knowledge of {skill} and best practices.",
    "Delivered high-impact projects using {skill}.",
    "Certified professional in {skill}.",
    "Mentored juniors in {skill} development."
]

bad_templates = [
    "I have very limited knowledge of {skill}.",
    "Struggled to learn {skill} in the past.",
    "Basic understanding of {skill} but no practical experience.",
    "Not comfortable working with {skill}.",
    "Failed to complete the {skill} certification.",
    "Novice in {skill}, looking for training.",
    "Weak grasp of {skill} concepts.",
    "Avoided using {skill} in previous roles."
]

data = []

print(f"🧠 Generating {NUM_PAIRS * 2} contextual teaching examples...")

for _ in range(NUM_PAIRS):
    skill = random.choice(skills)
    
    # 1. Generate POSITIVE Example
    good_text = random.choice(good_templates).format(skill=skill)
    # Add some random filler to make it look like a resume
    good_text = f"Professional developer. {good_text} Eager to work."
    
    data.append({
        'ResumeText': good_text,
        'HiringDecision': 'Selected',
        'Gender': random.choice(['Male', 'Female']),
        'Experience': random.randint(5, 10),
        'JobRole_Applied': 'Software Engineer' # Generic role
    })
    
    # 2. Generate NEGATIVE Example
    bad_text = random.choice(bad_templates).format(skill=skill)
    bad_text = f"Junior applicant. {bad_text} Looking for internship."
    
    data.append({
        'ResumeText': bad_text,
        'HiringDecision': 'Rejected',
        'Gender': random.choice(['Male', 'Female']),
        'Experience': random.randint(0, 2),
        'JobRole_Applied': 'Software Engineer'
    })

# Create DataFrame
df_context = pd.DataFrame(data)

# Load your EXISTING large dataset (if it exists) to merge
# We assume you are using 'recruitment_data_large.csv' from the previous step
main_file = 'data/recruitment_data_v3.csv'

if os.path.exists(main_file):
    print("🔗 Merging with your main dataset...")
    df_main = pd.read_csv(main_file)
    
    # Ensure columns match
    # We only need the columns that exist in both. 
    # If main dataset has ID/Name, we can leave them blank for these new rows or fill them.
    df_final = pd.concat([df_main, df_context], ignore_index=True)
    
    # Save back to the main file
    df_final.to_csv(main_file, index=False)
    print(f"✅ Successfully added {len(df_context)} contextual rows to {main_file}")
    print(f"📊 New Total Rows: {len(df_final)}")
else:
    # If no main file, just save this one
    df_context.to_csv(OUTPUT_FILE, index=False)
    print(f"⚠️ Main file not found. Saved to {OUTPUT_FILE} instead.")