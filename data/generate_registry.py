"""
data/generate_registry.py
합성 환자 데이터 600개를 생성하여 profile_registry.jsonl 저장.
실제 데이터가 없을 때 사용하는 스텁 — format-consistent한 가짜 값만 포함.
"""

import json
import os
import random

random.seed(42)

# ── 이름 풀 (30 × 20 = 600 고유 조합) ────────────────────────────────────────
FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
    "Linda", "William", "Barbara", "David", "Susan", "Richard", "Jessica",
    "Joseph", "Sarah", "Thomas", "Karen", "Charles", "Lisa", "Christopher",
    "Nancy", "Daniel", "Betty", "Matthew", "Margaret", "Anthony", "Sandra",
    "Mark", "Ashley",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
]

# ── 임상 노트 템플릿 요소 ─────────────────────────────────────────────────────
CHIEF_COMPLAINTS = [
    "chest pain and shortness of breath",
    "persistent cough and low-grade fever",
    "severe headache and dizziness",
    "abdominal pain and nausea",
    "fatigue and generalized weakness",
    "joint pain and swelling in bilateral knees",
    "palpitations and lightheadedness",
    "back pain radiating to the left leg",
    "skin rash and pruritus",
    "difficulty swallowing and weight loss",
]

HISTORIES = [
    "hypertension and type 2 diabetes mellitus",
    "coronary artery disease with prior CABG",
    "chronic obstructive pulmonary disease and former smoker",
    "hypothyroidism and osteoporosis",
    "atrial fibrillation on anticoagulation therapy",
    "chronic kidney disease stage 3 and anemia",
    "rheumatoid arthritis and fibromyalgia",
    "major depressive disorder and anxiety",
    "asthma and allergic rhinitis",
    "hyperlipidemia and obesity (BMI 34)",
]

MEDICATIONS = [
    "metformin 1000 mg BID, lisinopril 10 mg daily, atorvastatin 40 mg nightly",
    "warfarin 5 mg daily, metoprolol 25 mg BID, furosemide 20 mg daily",
    "levothyroxine 50 mcg daily, alendronate 70 mg weekly, calcium supplement",
    "albuterol inhaler PRN, fluticasone 250 mcg BID, montelukast 10 mg daily",
    "sertraline 100 mg daily, lorazepam 0.5 mg PRN, mirtazapine 15 mg nightly",
    "ramipril 5 mg daily, amlodipine 10 mg daily, rosuvastatin 20 mg nightly",
    "prednisone 10 mg daily, hydroxychloroquine 200 mg BID, folic acid 1 mg daily",
    "methotrexate 15 mg weekly, omeprazole 20 mg daily, vitamin D3 1000 IU daily",
    "gabapentin 300 mg TID, cyclobenzaprine 5 mg PRN, naproxen 500 mg BID",
    "insulin glargine 20 units nightly, empagliflozin 10 mg daily, aspirin 81 mg daily",
]

EXAM_FINDINGS = [
    "Blood pressure 142/88 mmHg, heart rate 78 bpm, respiratory rate 16/min, SpO2 97% on room air. Lungs clear to auscultation bilaterally. Regular rate and rhythm without murmurs.",
    "Blood pressure 118/74 mmHg, heart rate 92 bpm, temperature 38.1°C, SpO2 94% on room air. Scattered expiratory wheezes bilaterally. Mild use of accessory muscles.",
    "Blood pressure 156/96 mmHg, heart rate 68 bpm, BMI 31 kg/m². Abdomen soft with mild right upper quadrant tenderness. No rebound or guarding.",
    "Blood pressure 108/62 mmHg, heart rate 104 bpm, respiratory rate 20/min. Pallor noted. Mild pitting edema bilateral lower extremities to mid-calf.",
    "Blood pressure 130/82 mmHg, heart rate 74 bpm, temperature 37.2°C. Alert and oriented. Cranial nerves intact. Mild left-sided weakness noted on motor exam.",
]

ASSESSMENTS = [
    "Impression: Uncontrolled hypertension with early signs of hypertensive nephropathy. Recommend medication adjustment and repeat labs in 4 weeks.",
    "Impression: Community-acquired pneumonia, moderate severity. Started on amoxicillin-clavulanate. Follow-up in 7 days or sooner if symptoms worsen.",
    "Impression: Decompensated heart failure secondary to dietary indiscretion. Increased diuretic dose. Salt-restricted diet counseling provided.",
    "Impression: Acute exacerbation of COPD likely triggered by viral upper respiratory infection. Short-course oral corticosteroids and increased bronchodilator use.",
    "Impression: New-onset atrial fibrillation with rapid ventricular response. Rate control initiated. Referral to cardiology placed for further management.",
]

PLAN_NOTES = [
    "Patient instructed to monitor blood pressure daily. Referral to nephrology for further evaluation. Dietitian consultation arranged.",
    "Patient educated on infection control measures. Prescription provided. Emergency return precautions discussed in detail.",
    "Patient to follow up in 48 hours for weight and symptom reassessment. Fluid restriction advised at 1.5 L/day.",
    "Pulmonary function tests to be repeated after acute exacerbation resolves. Smoking cessation resources provided.",
    "Anticoagulation therapy to be initiated after risk-benefit discussion. Patient verbalized understanding of risks and benefits.",
]


def make_prefix_text(idx: int) -> str:
    age = random.randint(35, 85)
    gender = random.choice(["male", "female"])
    cc = CHIEF_COMPLAINTS[idx % len(CHIEF_COMPLAINTS)]
    hx = HISTORIES[idx % len(HISTORIES)]
    meds = MEDICATIONS[idx % len(MEDICATIONS)]
    exam = EXAM_FINDINGS[idx % len(EXAM_FINDINGS)]
    assessment = ASSESSMENTS[idx % len(ASSESSMENTS)]
    plan = PLAN_NOTES[idx % len(PLAN_NOTES)]
    duration = random.choice([
        "two days", "three days", "one week", "several days", "four days",
        "five days", "the past 24 hours", "the past 48 hours",
    ])
    severity = random.choice(["mild", "moderate", "severe", "progressive"])

    return (
        f"CLINICAL NOTE\n\n"
        f"Chief Complaint: The patient is a {age}-year-old {gender} presenting with {cc} "
        f"that began approximately {duration} ago. The patient describes the symptoms as "
        f"{severity} and reports they have been interfering with daily activities.\n\n"
        f"History of Present Illness: The onset was gradual with no identifiable precipitating factor. "
        f"The patient denies any recent travel, sick contacts, or changes in medications. "
        f"Associated symptoms include occasional fatigue and reduced appetite. "
        f"The patient reports partial relief with over-the-counter medications.\n\n"
        f"Past Medical History: Significant for {hx}.\n\n"
        f"Current Medications: {meds}.\n\n"
        f"Physical Examination: {exam}\n\n"
        f"Assessment and Plan: {assessment} {plan}"
    )


def make_phone(idx: int) -> str:
    area = (idx % 800) + 200
    mid = (idx * 7 + 100) % 900 + 100
    last = (idx * 13 + 1000) % 9000 + 1000
    return f"({area:03d}) {mid:03d}-{last:04d}"


def make_patient_id(idx: int) -> str:
    return f"PID-{idx + 1:05d}"


def make_email(first: str, last: str, idx: int) -> str:
    domains = [
        "gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
        "icloud.com", "protonmail.com", "live.com",
    ]
    domain = domains[idx % len(domains)]
    tag = idx // len(domains) if idx // len(domains) > 0 else ""
    return f"{first.lower()}.{last.lower()}{tag}@{domain}"


def main():
    registry = []
    idx = 0
    for first in FIRST_NAMES:
        for last in LAST_NAMES:
            record = {
                "entity_id": f"profile_{idx:05d}",
                "full_name": f"{first} {last}",
                "patient_id": make_patient_id(idx),
                "phone": make_phone(idx),
                "email": make_email(first, last, idx),
                "prefix_text": make_prefix_text(idx),
            }
            registry.append(record)
            idx += 1
            if idx >= 600:
                break
        if idx >= 600:
            break

    out_path = os.path.join(os.path.dirname(__file__), "profile_registry.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in registry:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Generated {len(registry)} records → {out_path}")


if __name__ == "__main__":
    main()
