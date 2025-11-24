import pandas as pd
import random
import os

# CONFIG
NUM_SAMPLES = 5000
OUTPUT_FILE = 'data/recruitment_data_large.csv'

# DATA POOLS
names_male = ['James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard', 'Joseph', 'Thomas', 'Charles','Julian','Jaya','Aarya','Kumaresan','Aravind','Vikram','Raghav','Sanjay','Vijay','Raman','Aathil','Karthik','Manoj','Naveen','Palani','Suhail','Vasanth','Caron']
names_female = ['Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan', 'Jessica', 'Sarah', 'Karen','Tricia','Ananya','Lakshmi','Sowmya','Divya','Meena','Priya','Kavya','Sneha','Aishwarya','Nithya','Deepa','Radha','Latha']
surnames = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez','Iyer','Kumar','Sharma','Reddy','Singh','Patel','Chowdhury','Das','Gupta','Mehta','Nair','Joshi','Kapoor','Fidel','Castro','Hernandez']

roles = ['Data Scientist', 'Software Engineer', 'Sales Manager', 'HR Specialist', 'Marketing Manager','Product Manager','Business Analyst','DevOps Engineer','Customer Support','Financial Analyst']
skills_tech = ['Python', 'Java', 'SQL', 'Machine Learning', 'AWS', 'React', 'Docker', 'Kubernetes', 'C++','TensorFlow','PyTorch','Hadoop','Spark']
skills_soft = ['Communication', 'Leadership', 'Teamwork', 'Project Management', 'Negotiation', 'Strategy','Problem Solving','Time Management','Adaptability','Creativity']
degrees = ['B.Sc', 'M.Sc', 'B.Tech', 'MBA', 'PhD', 'B.A']

bias_words = ['ninja', 'rockstar', 'aggressive', 'dominant', 'force', 'guys', 'young', 'hustler','guru','wizard','master','champion','warrior']

data = []

print(f"🚀 Generating {NUM_SAMPLES} rows of synthetic data...")

for i in range(NUM_SAMPLES):
    # 1. Basic Attributes
    gender = random.choice(['Male', 'Female'])
    name = random.choice(names_male if gender == 'Male' else names_female) + " " + random.choice(surnames)
    age = random.randint(22, 50)
    role = random.choice(roles)
    experience = random.randint(0, 20)
    education = random.choice(degrees) + " in " + ("Computer Science" if role in ['Data Scientist', 'Software Engineer'] else "Management")
    
    # 2. Skill Injection (Strong vs Weak)
    # 50% chance to be a "Strong" candidate
    is_strong_candidate = random.random() > 0.5
    
    if is_strong_candidate:
        my_skills = random.sample(skills_tech, 3) + random.sample(skills_soft, 2)
        exp_text = "extensive experience"
        perf_text = "Proven track record of delivering high-impact results."
    else:
        my_skills = random.sample(skills_tech, 1) + random.sample(skills_soft, 1)
        exp_text = "some experience"
        perf_text = "Looking to learn and grow."

    skill_str = ", ".join(my_skills)
    
    # 3. Bias Injection (20% chance to use biased language)
    has_bias = random.random() < 0.2
    bias_str = ""
    if has_bias:
        bias_word = random.choice(bias_words)
        if gender == 'Male':
            bias_str = f"He is a {bias_word} who dominates the field."
        else:
            bias_str = f"She is a {bias_word} looking for a supportive team."
    
    # 4. Construct Resume Text (Anonymized start to prevent name bias)
    resume_text = f"Professional {role} with {experience} years of {exp_text}. Skilled in {skill_str}. {perf_text} {bias_str}"
    
    # 5. Determine Hiring Decision (The Logic)
    # Score based on Experience + Skills - Bias
    score = 0
    if is_strong_candidate: score += 50
    score += (experience * 2)
    if has_bias: score -= 40  # Heavy penalty for bias
    
    # Threshold for selection
    status = 'Selected' if score > 45 else 'Rejected'
    
    data.append([f"ID{i:04d}", name, gender, age, education, skill_str, role, experience, resume_text, status])

# Create DataFrame
df = pd.DataFrame(data, columns=['ID', 'Name', 'Gender', 'Age', 'Education', 'Skills', 'JobRole_Applied', 'Experience', 'ResumeText', 'HiringDecision'])

# Balance Check
print("\n📊 Data Balance:")
print(df['HiringDecision'].value_counts())

# Save
os.makedirs('data', exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Saved to {OUTPUT_FILE}")