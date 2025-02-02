import json
import numpy as np
from pgmpy.factors.discrete import TabularCPD

def save_cpds(model, filename="cpds.json"):
    """
    Save CPDs of a Bayesian Network to a file.
    
    Parameters:
    model: Trained BayesianNetwork object with CPDs
    filename: Name of the file to save CPDs
    """
    cpds_dict = {}
    for cpd in model.get_cpds():
        # Convert CPD to dictionary format
        cpds_dict[cpd.variable] = {
            "values": cpd.get_values().tolist(),
            "variables": cpd.variables,
            "cardinality": cpd.cardinality.tolist(),
            "state_names": cpd.state_names
        }
    
    # Save to a JSON file
    with open(filename, "w") as file:
        json.dump(cpds_dict, file, indent=4)
    print(f"CPDs saved to {filename}")

def load_cpds(model, filename="cpds.json"):
    """
    Load CPDs from a file and add them to the Bayesian Network.
    
    Parameters:
    model: BayesianNetwork object
    filename: Name of the file containing CPDs
    """
    with open(filename, "r") as file:
        cpds_dict = json.load(file)
    
    for var, cpd_data in cpds_dict.items():
        # Create TabularCPD object
        cpd = TabularCPD(
            variable=var,
            variable_card=len(cpd_data["state_names"][var]),
            values=np.array(cpd_data["values"]),
            evidence=cpd_data["variables"][1:],
            evidence_card=cpd_data["cardinality"][1:],
            state_names=cpd_data["state_names"]
        )
        model.add_cpds(cpd)
    
    # Validate the model
    if model.check_model():
        print("All CPDs loaded successfully and the model is valid!")
    else:
        print("Model validation failed. Check the CPDs.")

def discretize_variables(data):
    """
    Convert all variables to string type for pgmpy compatibility
    since data is already preprocessed and categorized
    """
    # Create copy to avoid modifying original data
    df = data.copy()
    
    # Convert all columns to string for pgmpy compatibility
    for col in df.columns:
        df[col] = df[col].astype(str)
    
    return df

chief_complaint_dict = {
    1: "Cardiac arrest (non-traumatic)",
    2: "Cardiac arrest (traumatic)",
    3: "Chest pain",
    4: "Palpitations / Irregular heart beat",
    5: "Hypertension",
    6: "General weakness",
    7: "Syncope / Pre-syncope",
    8: "Edema, generalized",
    9: "Bilateral leg swelling / Edema",
    10: "Cool pulseless limb",
    11: "Unilateral reddened hot limb: DVT symptoms",
    12: "Earache",
    13: "Foreign body",
    14: "Loss of hearing",
    15: "Tinnitus / Dysacusis",
    16: "Discharge",
    17: "Ear injury",
    18: "Dental / Gum problem",
    19: "Facial trauma",
    20: "Sore throat",
    21: "Neck swelling/pain",
    22: "Neck trauma",
    23: "Difficulty swallowing / Dyspahiga",
    24: "Facial pain (non-traumatic / non-dental)",
    25: "Epistaxis",
    26: "Nasal congestion / Hay fever",
    27: "Foreign body, nose",
    28: "URTI complaints",
    29: "Nasal trauma",
    30: "Frostbite / Cold injury",
    31: "Noxious inhalation",
    32: "Electrical injury",
    33: "Chemical exposure",
    34: "Hypothermia",
    35: "Abdominal pain",
    36: "Anorexia",
    37: "Constipation",
    38: "Diarrhea",
    39: "Foreign body in rectum",
    40: "Groin pain/mass",
    41: "Nausea and/or vomiting",
    42: "Rectal/Perineal pain",
    43: "Vomiting blood",
    44: "Blood in stool / Melena",
    45: "Jaundice",
    46: "Hiccoughs",
    47: "Abdominal mass/distention",
    48: "Anal/Rectal trauma",
    49: "Flank pain",
    50: "Hematuria",
    51: "Genital discharge/lesion",
    52: "Penile swelling",
    53: "Testicular/Scrotal pain and/or swelling",
    54: "Urinary retention",
    55: "UTI complaints",
    56: "Oliguria",
    57: "Polyuria",
    58: "Genital trauma",
    59: "Depression / Suicidal",
    60: "Anxiety / Situational crisis",
    61: "Hallucinations",
    62: "Insomnia",
    63: "Violent behaviour",
    64: "Social problem",
    65: "Homicidal",
    66: "Bizarre/Paranoid behaviour",
    67: "Altered level of consciousness",
    68: "Confusion",
    69: "Dizziness / Vertigo",
    70: "Headache",
    71: "Seizure",
    72: "Gait disturbance / Ataxia",
    73: "Head injury",
    74: "Tremor",
    75: "Extremity weakness / Symptoms of CVA",
    76: "Sensory loss / Parasthesias",
    77: "Menstrual problems",
    78: "Foreign body, vagina",
    79: "Vaginal discharge",
    80: "Sexual assault",
    81: "Vaginal bleed",
    82: "Labial swelling",
    83: "Pregnancy issues <20 wks",
    84: "Imminent delivery",
    85: "Vaginal pain / Dyspareunia",
    86: "Discharge, eye",
    87: "Chemical exposure, eye",
    88: "Foreign body, eye",
    89: "Visual disturbance",
    90: "Eye pain",
    91: "Itchy/Red eye",
    92: "Photophobia",
    93: "Diplopia",
    94: "Periorbital swelling + Fever",
    95: "Eye trauma",
    96: "Re-check eye",
    97: "Back pain",
    98: "Traumatic back/spine injury",
    99: "Amputation",
    100: "Upper extremity pain",
    101: "Lower extremity pain",
    102: "Upper extremity injury",
    103: "Lower extremity injury",
    104: "Joint(s) swelling",
    105: "Feeding difficulties in newborn",
    106: "Neonatal jaundice",
    107: "Inconsolable crying",
    108: "Wheezing - no other complaints",
    109: "Limp",
    110: "Apneic spells",
    111: "Pediatric behavioural issues",
    112: "Shortness of breath",
    113: "Respiratory arrest",
    114: "Cough",
    115: "Hyperventilation",
    116: "Hemoptysis",
    117: "Respiratory foreign body",
    118: "Allergic reaction",
    119: "Bite",
    120: "Sting",
    121: "Abrasion",
    122: "Laceration / Puncture",
    123: "Burn",
    124: "Blood and body fluid exposure",
    125: "Pruritis",
    126: "Rash",
    127: "Localized swelling/redness",
    128: "Wound check",
    129: "Other skin conditions",
    130: "Lumps, bumps, calluses, etc...",
    131: "Redness/tenderness, breast",
    132: "Rule out infestation",
    133: "Cyanosis",
    134: "Bruising - History of bleeding disorder",
    135: "Foreign body, skin",
    136: "Substance misuse / Intoxication",
    137: "Overdose ingestion",
    138: "Substance withdrawal",
    139: "Major trauma - penetrating",
    140: "Major trauma - blunt",
    141: "Isolated chest trauma - penetrating",
    142: "Isolated chest trauma - blunt",
    143: "Isolated abdominal trauma - penetrating",
    144: "Isolated abdominal trauma - blunt",
    145: "Exposure to communicable disease",
    146: "Fever",
    147: "Hyperglycemia",
    148: "Hypoglycemia",
    149: "Direct referral for consultation",
    150: "Dressing change",
    151: "Removal staples/sutures",
    152: "Cast check",
    153: "Imaging tests",
    154: "Medical device problem",
    155: "Prescription / Medication request",
    156: "Ring removal",
    157: "Abnormal lab values",
    158: "Pallor / Anemia",
    159: "Post-operative complications",
    160: "Minor complaints, unspecified",
}