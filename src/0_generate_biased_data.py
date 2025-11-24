import pandas as pd
import random
import os

# CONFIG
NUM_SAMPLES = 5000 # Increased to give Neural Net more data
OUTPUT_FILE = 'data/recruitment_data_biased.csv'

genders = ['Male', 'Female']
races = ['White', 'Black', 'Asian', 'Hispanic'] 
roles = ['Data Scientist', 'Software Engineer', 'Sales Manager', 'HR Specialist']

# KEYWORDS
skills_tech = ['Python', 'Java', 'SQL', 'Machine Learning', 'AWS', 'React', 'C++']
skills_soft = ['Leadership', 'Communication', 'Strategy', 'Teamwork']
# BAD WORDS (We need the generator to actually penalize these!)
negative_phrases = ["limited knowledge", "struggled", "basic understanding", "no practical experience", "avoided using", "failed to", "weak grasp"]

data = []

print(f"⚠️ Generating {NUM_SAMPLES} rows of SMART Biased data...")

for i in range(NUM_SAMPLES):
    gender = random.choice(genders)
    race = random.choice(races)
    role = random.choice(roles)
    experience = random.randint(0, 15)
    
    pronoun = "He" if gender == 'Male' else "She"
    possessive = "His" if gender == 'Male' else "Her"
    
    # 1. Decide if this is a Good or Bad candidate
    is_competent = random.random() > 0.3 # 70% chance to be competent
    
    skill_text = ""
    if is_competent:
        # Strong candidate text
        my_skills = random.sample(skills_tech + skills_soft, random.randint(3, 5))
        skill_text = f"{possessive} expertise includes {', '.join(my_skills)}. {pronoun} is an expert."
    else:
        # Weak candidate text (Inject negative phrases)
        neg = random.choice(negative_phrases)
        skill_text = f"However, {pronoun} has {neg} of the core tools. {possessive} skills are basic."

    # Construct Text
    text = f"{pronoun} is a {role} with {experience} years of experience. {skill_text} {pronoun} is looking for a job."
    
    # 2. SCORING LOGIC (The Fix)
    score = 0
    
    # A. Experience Score
    score += (experience * 3)
    
    # B. Content Score (CRITICAL FIX)
    # If the text is competent, huge bonus. If text is weak, huge penalty.
    if is_competent:
        score += 40
    else:
        score -= 40 # <--- This forces the AI to respect the "Bad Words"
        
    # C. Bias Injection (The Project Goal)
    if gender == 'Male': score += 15
    if race != 'White': score -= 10
    
    # D. Random Noise (The "100%" Fix)
    # We add random variance so the decision isn't a perfect math equation.
    # This creates the "Gray Area" (probabilities like 70%, 45%).
    noise = random.randint(-15, 15)
    total_score = score + noise
    
    # Threshold
    hiring_decision = 'Selected' if total_score > 45 else 'Rejected'
    
    data.append([text, hiring_decision, gender, race, experience, role])

df = pd.DataFrame(data, columns=['ResumeText', 'HiringDecision', 'Gender', 'Race', 'Experience', 'JobRole_Applied'])

os.makedirs('data', exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Generated {len(df)} rows.")
print(f"Selected: {len(df[df['HiringDecision']=='Selected'])}")
print(f"Rejected: {len(df[df['HiringDecision']=='Rejected'])}")