large_medical_text = """
Migraine is a neurological disorder associated with photophobia,
nausea, pulsatile headache and sensory hypersensitivity.

Patients may experience visual aura, dizziness, confusion,
temporary cognitive impairment and severe fatigue.

""" * 1200

with open("data/massive_context.txt", "w", encoding="utf-8") as f:
    f.write(large_medical_text)

print("Massive context generated successfully.")