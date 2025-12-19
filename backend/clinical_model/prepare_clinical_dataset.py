import numpy as np
import pandas as pd
import random

def generate_synthetic_clinical(n=3000, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    lesion_types = ["mass", "calcification", "both"]

    synthetic_rows = []

    for _ in range(n):
        # -------------------------
        # 1. Balanced lesion type
        # -------------------------
        lt = random.choice(lesion_types)

        # -------------------------
        # 2. Age distribution
        # -------------------------
        age = int(np.clip(np.random.normal(52, 10), 25, 85))

        # Menopause status
        menopause_status = (
            1 if age >= 55 else
            0 if age <= 45 else
            0.5
        )

        # -------------------------
        # 3. Base probabilities of symptoms by lesion type
        # -------------------------
        if lt == "mass":
            p_lump = 0.8
            p_pain = 0.3
            p_skin = 0.4
            p_nd = 0.05
            base_malignancy = 0.45

        elif lt == "calcification":
            p_lump = 0.15
            p_pain = 0.1
            p_skin = 0.05
            p_nd = 0.0
            base_malignancy = 0.25

        else:  # both
            p_lump = 0.6
            p_pain = 0.4
            p_skin = 0.3
            p_nd = 0.03
            base_malignancy = 0.40

        palpable_lump = int(random.random() < p_lump)
        pain = int(random.random() < p_pain)
        skin_changes = int(random.random() < p_skin)
        nipple_discharge = int(random.random() < p_nd)

        # -------------------------
        # 4. Additional risk modifiers
        # -------------------------
        density = random.randint(1, 4)
        family_history = int(random.random() < 0.15)
        hormone_therapy = int(random.random() < 0.2)
        prior_biopsies = int(random.random() < 0.1)
        bmi = float(np.clip(np.random.normal(27, 4), 17, 40))

        # -------------------------
        # 5. Malignancy adjusted probability
        # -------------------------
        mal_prob = base_malignancy

        if palpable_lump: mal_prob += 0.20
        if skin_changes: mal_prob += 0.10
        if nipple_discharge: mal_prob -= 0.05
        if pain: mal_prob -= 0.10

        if family_history: mal_prob += 0.15
        if hormone_therapy: mal_prob += 0.05
        if prior_biopsies: mal_prob += 0.10
        if density == 4: mal_prob += 0.10

        if age > 60: mal_prob += 0.10
        elif age > 50: mal_prob += 0.03

        mal_prob = np.clip(mal_prob, 0.02, 0.98)

        malignant = int(random.random() < mal_prob)

        synthetic_rows.append({
            "age": age,
            "menopause_status": menopause_status,
            "lesion_type": lt,
            "malignant": malignant,
            "palpable_lump": palpable_lump,
            "pain": pain,
            "skin_changes": skin_changes,
            "nipple_discharge": nipple_discharge,
            "family_history": family_history,
            "hormone_therapy": hormone_therapy,
            "prior_biopsies": prior_biopsies,
            "bmi": bmi,
            "density": density
        })

    return pd.DataFrame(synthetic_rows)


if __name__ == "__main__":
    df_syn = generate_synthetic_clinical(3000)
    df_syn.to_csv("backend/clinical_model/synthetic_symptoms.csv", index=False)
    print("✓ Saved synthetic dataset:", df_syn.shape)
    print(df_syn.head())
