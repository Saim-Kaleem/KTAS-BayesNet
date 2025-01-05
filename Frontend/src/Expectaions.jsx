import React, { useState } from 'react';
import axios from 'axios';
import './FormComponent.css';

const FormComponent = () => {
const [formData, setFormData] = useState({
Sex: '',
Age: '',
Arrival: '',
Injury: '',
Pain: '',
Mental: '',
NRS_Pain: '',
Chief_Complaint: '',
SBP: '',
DBP: '',
HR: '',
BT: '',
RR: '',
Saturation: '',
});
const [loading, setLoading] = useState(false);
const [apiResponse, setApiResponse] = useState(null);

const chiefComplaintOptions = [
{ value: "1", label: "Cardiac arrest (non-traumatic)" },
{ value: "2", label: "Cardiac arrest (traumatic)" },
{ value: "3", label: "Chest pain" },
{ value: "4", label: "Palpitations / Irregular heart beat" },
{ value: "5", label: "Hypertension" },
{ value: "6", label: "General weakness" },
{ value: "7", label: "Syncope / Pre-syncope" },
{ value: "8", label: "Edema, generalized" },
{ value: "9", label: "Bilateral leg swelling / Edema" },
{ value: "10", label: "Cool pulseless limb" },
{ value: "11", label: "Unilateral reddened hot limb: DVT symptoms" },
{ value: "12", label: "Earache" },
{ value: "13", label: "Foreign body" },
{ value: "14", label: "Loss of hearing" },
{ value: "15", label: "Tinnitus / Dysacusis" },
{ value: "16", label: "Discharge" },
{ value: "17", label: "Ear injury" },
{ value: "18", label: "Dental / Gum problem" },
{ value: "19", label: "Facial trauma" },
{ value: "20", label: "Sore throat" },
{ value: "21", label: "Neck swelling/pain" },
{ value: "22", label: "Neck trauma" },
{ value: "23", label: "Difficulty swallowing / Dysphagia" },
{ value: "24", label: "Facial pain (non-traumatic / non-dental)" },
{ value: "25", label: "Epistaxis" },
{ value: "26", label: "Nasal congestion / Hay fever" },
{ value: "27", label: "Foreign body, nose" },
{ value: "28", label: "URTI complaints" },
{ value: "29", label: "Nasal trauma" },
{ value: "30", label: "Frostbite / Cold injury" },
{ value: "31", label: "Noxious inhalation" },
{ value: "32", label: "Electrical injury" },
{ value: "33", label: "Chemical exposure" },
{ value: "34", label: "Hypothermia" },
{ value: "35", label: "Abdominal pain" },
{ value: "36", label: "Anorexia" },
{ value: "37", label: "Constipation" },
{ value: "38", label: "Diarrhea" },
{ value: "39", label: "Foreign body in rectum" },
{ value: "40", label: "Groin pain/mass" },
{ value: "41", label: "Nausea and/or vomiting" },
{ value: "42", label: "Rectal/Perineal pain" },
{ value: "43", label: "Vomiting blood" },
{ value: "44", label: "Blood in stool / Melena" },
{ value: "45", label: "Jaundice" },
{ value: "46", label: "Hiccoughs" },
{ value: "47", label: "Abdominal mass/distention" },
{ value: "48", label: "Anal/Rectal trauma" },
{ value: "49", label: "Flank pain" },
{ value: "50", label: "Hematuria" },
{ value: "51", label: "Genital discharge/lesion" },
{ value: "52", label: "Penile swelling" },
{ value: "53", label: "Testicular/Scrotal pain and/or swelling" },
{ value: "54", label: "Urinary retention" },
{ value: "55", label: "UTI complaints" },
{ value: "56", label: "Oliguria" },
{ value: "57", label: "Polyuria" },
{ value: "58", label: "Genital trauma" },
{ value: "59", label: "Depression / Suicidal" },
{ value: "60", label: "Anxiety / Situational crisis" },
{ value: "61", label: "Hallucinations" },
{ value: "62", label: "Insomnia" },
{ value: "63", label: "Violent behaviour" },
{ value: "64", label: "Social problem" },
{ value: "65", label: "Homicidal" },
{ value: "66", label: "Bizarre/Paranoid behaviour" },
{ value: "67", label: "Altered level of consciousness" },
{ value: "68", label: "Confusion" },
{ value: "69", label: "Dizziness / Vertigo" },
{ value: "70", label: "Headache" },
{ value: "71", label: "Seizure" },
{ value: "72", label: "Gait disturbance / Ataxia" },
{ value: "73", label: "Head injury" },
{ value: "74", label: "Tremor" },
{ value: "75", label: "Extremity weakness / Symptoms of CVA" },
{ value: "76", label: "Sensory loss / Parasthesias" },
{ value: "77", label: "Menstrual problems" },
{ value: "78", label: "Foreign body, vagina" },
{ value: "79", label: "Vaginal discharge" },
{ value: "80", label: "Sexual assault" },
{ value: "81", label: "Vaginal bleed" },
{ value: "82", label: "Labial swelling" },
{ value: "83", label: "Pregnancy issues <20 wks" },
{ value: "84", label: "Imminent delivery" },
{ value: "85", label: "Vaginal pain / Dyspareunia" },
{ value: "86", label: "Discharge, eye" },
{ value: "87", label: "Chemical exposure, eye" },
{ value: "88", label: "Foreign body, eye" },
{ value: "89", label: "Visual disturbance" },
{ value: "90", label: "Eye pain" },
{ value: "91", label: "Itchy/Red eye" },
{ value: "92", label: "Photophobia" },
{ value: "93", label: "Diplopia" },
{ value: "94", label: "Periorbital swelling + Fever" },
{ value: "95", label: "Eye trauma" },
{ value: "96", label: "Re-check eye" },
{ value: "97", label: "Back pain" },
{ value: "98", label: "Traumatic back/spine injury" },
{ value: "99", label: "Amputation" },
{ value: "100", label: "Upper extremity pain" },
{ value: "101", label: "Lower extremity pain" },
{ value: "102", label: "Upper extremity injury" },
{ value: "103", label: "Lower extremity injury" },
{ value: "104", label: "Joint(s) swelling" },
{ value: "105", label: "Feeding difficulties in newborn" },
{ value: "106", label: "Neonatal jaundice" },
{ value: "107", label: "Inconsolable crying" },
{ value: "108", label: "Wheezing - no other complaints" },
{ value: "109", label: "Limp" },
{ value: "110", label: "Apneic spells" },
{ value: "111", label: "Pediatric behavioural issues" },
{ value: "112", label: "Shortness of breath" },
{ value: "113", label: "Respiratory arrest" },
{ value: "114", label: "Cough" },
{ value: "115", label: "Hyperventilation" },
{ value: "116", label: "Hemoptysis" },
{ value: "117", label: "Respiratory foreign body" },
{ value: "118", label: "Allergic reaction" },
{ value: "119", label: "Bite" },
{ value: "120", label: "Sting" },
{ value: "121", label: "Abrasion" },
{ value: "122", label: "Laceration / Puncture" },
{ value: "123", label: "Burn" },
{ value: "124", label: "Blood and body fluid exposure" },
{ value: "125", label: "Pruritis" },
{ value: "126", label: "Rash" },
{ value: "127", label: "Localized swelling/redness" },
{ value: "128", label: "Wound check" },
{ value: "129", label: "Other skin conditions" },
{ value: "130", label: "Lumps, bumps, calluses, etc..." },
{ value: "131", label: "Redness/tenderness, breast" },
{ value: "132", label: "Rule out infestation" },
{ value: "133", label: "Cyanosis" },
{ value: "134", label: "Bruising - History of bleeding disorder" },
{ value: "135", label: "Foreign body, skin" },
{ value: "136", label: "Substance misuse / Intoxication" },
{ value: "137", label: "Overdose ingestion" },
{ value: "138", label: "Substance withdrawal" },
{ value: "139", label: "Major trauma - penetrating" },
{ value: "140", label: "Major trauma - blunt" },
{ value: "141", label: "Isolated chest trauma - penetrating" },
{ value: "142", label: "Isolated chest trauma - blunt" },
{ value: "143", label: "Isolated abdominal trauma - penetrating" },
{ value: "144", label: "Isolated abdominal trauma - blunt" },
{ value: "145", label: "Exposure to communicable disease" },
{ value: "146", label: "Fever" },
{ value: "147", label: "Hyperglycemia" },
{ value: "148", label: "Hypoglycemia" },
{ value: "149", label: "Direct referral for consultation" },
{ value: "150", label: "Dressing change" },
{ value: "151", label: "Removal staples/sutures" },
{ value: "152", label: "Cast check" },
{ value: "153", label: "Imaging tests" },
{ value: "154", label: "Medical device problem" },
{ value: "155", label: "Prescription / Medication request" },
{ value: "156", label: "Ring removal" },
{ value: "157", label: "Abnormal lab values" },
{ value: "158", label: "Pallor / Anemia" },
{ value: "159", label: "Post-operative complications" },
{ value: "160", label: "Minor complaints, unspecified" },
];


const arrivalModes = [
{ label: 'Walking', value: "0" },
{ label: '119 use', value: "1" },
{ label: 'Private car', value: "2" },
{ label: 'Private ambulance', value: "3" },
{ label: 'Public transportation', value: "4" },
{ label: 'Wheelchair', value: "5" },
{ label: 'Others', value: "6" },
];

const injuryOptions = [
{ label: 'Non-injury', value: "0" },
{ label: 'Injury', value: "1" },
];

const mentalOptions = [
{ label: 'Alert', value: "0" },
{ label: 'Verbal response', value: "1" },
{ label: 'Pain response', value: "2" },
{ label: 'Unconsciousness', value: "3" },
];

const ktasDescriptions = {
1: {
color: 'red',
title: 'Level 1: Immediate',
description: 'Critical and life-threatening condition requiring immediate treatment.',
prescription: 'Seek emergency medical attention immediately.',
},
2: {
color: 'orange',
title: 'Level 2: Very Urgent',
description: 'Severe condition requiring prompt treatment to prevent deterioration.',
prescription: 'Visit the emergency department as soon as possible.',
},
3: {
color: 'yellow',
title: 'Level 3: Urgent',
description: 'Moderate condition requiring treatment within a short period.',
prescription: 'Consult a doctor within the next few hours.',
},
4: {
color: 'green',
title: 'Level 4: Less Urgent',
description: 'Mild condition requiring treatment at a convenient time.',
prescription: 'Schedule a consultation with a healthcare provider.',
},
5: {
color: 'blue',
title: 'Level 5: Non-Urgent',
description: 'Minimal condition requiring no immediate medical attention.',
prescription: 'Consider self-care or routine medical consultation.',
},
};

const handleChange = (field, value) => {
setFormData({
...formData,
[field]: value,
});
};

const calculateBP = () => {
const sbp = parseInt(formData.SBP, 10);
const dbp = parseInt(formData.DBP, 10);
if (isNaN(sbp) || isNaN(dbp)) return null;
if (sbp < 90 && dbp < 60) return "0"; // Hypotension
if (sbp < 120 && dbp < 80) return "0"; // Normal
if (120 <= sbp && sbp <= 129 && dbp < 80) return "1"; // Elevated
if (130 <= sbp && sbp <= 139 || (80 <= dbp && dbp <= 89)) return "2"; // Hypertension Stage 1
if (sbp >= 140 || dbp >= 90) {
if (sbp > 180 || dbp > 120) return "4"; // Hypertensive Crisis
return "3"; // Hypertension Stage 2
}
return "4"; // Hypertensive Crisis
};

const encodeVariables = () => {
const BP = calculateBP();
const encodeAge = (age) => {
const numAge = parseInt(age, 10);
if (numAge < 16) return "0";
if (numAge <= 44) return "1";
if (numAge <= 64) return "2";
if (numAge <= 79) return "3";
// return "4";
};

const encodeHR = (hr) => {
const numHR = parseInt(hr, 10);
if (numHR < 40) return "0";
if (numHR <= 50) return "1";
if (numHR <= 60) return "2";
if (numHR <= 100) return "3";
if (numHR <= 120) return "4";
if (numHR <= 140) return "5";
if (numHR <= 160) return "6";
// return "7";
};

const encodeRR = (rr) => {
const numRR = parseInt(rr, 10);
if (numRR < 8) return "0";
if (numRR <= 11) return "1";
if (numRR <= 20) return "2";
if (numRR <= 24) return "3";
if (numRR <= 30) return "4";
// return "5";
};

const encodeBT = (bt) => {
const numBT = parseFloat(bt);
if (numBT < 35) return "0";
if (numBT <= 36.4) return "1";
if (numBT <= 37.5) return "2";
if (numBT <= 38.3) return "3";
if (numBT <= 40) return "4";
// return "5";
};

const encodeSaturation = (saturation) => {
const numSaturation = parseInt(saturation, 10);
if (numSaturation < 90) return "0";
if (numSaturation <= 94) return "1";
// return "2";
};

const encodeNRS = (nrs) => {
const numNRS = parseInt(nrs, 10);
if (numNRS === 0) return "0";
if (numNRS <= 3) return "1";
if (numNRS <= 6) return "2";
// return "3";
};

return {
Sex: formData.Sex,
Age: encodeAge(formData.Age),
"Arrival mode": formData.Arrival,
Injury: formData.Injury,
Pain: formData.Pain,
Mental: formData.Mental,
NRS_pain: encodeNRS(formData.NRS_Pain),
"Complain index": parseFloat(formData.Chief_Complaint).toFixed(1),
BP: BP,
HR: encodeHR(formData.HR),
RR: encodeRR(formData.RR),
BT: encodeBT(formData.BT),
Saturation: encodeSaturation(formData.Saturation),
};
};

const handleSubmit = async (event) => {
event.preventDefault();
setLoading(true);
const encodedData = encodeVariables();
console.log('Encoded Data:', encodedData);
const filteredData = Object.fromEntries(
    Object.entries(encodedData).filter(([key, value]) => value !== undefined)
);
console.log('Filtered Data:', filteredData);

try {
const response = await axios.post('http://localhost:5000/api/get_bayes', filteredData);
console.log('API Response:', response.data);
setApiResponse(response.data);
} catch (error) {
console.error('There was an error with the API request:', error);
alert('Error with API request. Please check the console for details.');
} finally {
setLoading(false);
}
};

return (
<div className="form-page">
{loading ? (
    <div className="spinner">Loading...</div>
) : (
    <>
    <form className="form-container" onSubmit={handleSubmit}>
    <div className="form-group">
        <label>Sex:</label>
        <select onChange={(e) => handleChange('Sex', e.target.value)}>
            <option value="">Select</option>
            <option value="0">Female</option>
            <option value="1">Male</option>
        </select>
        </div>

        <div className="form-group">
        <label>Age:</label>
        <input
            type="text"
            onChange={(e) => handleChange('Age', e.target.value)}
            placeholder="Enter Age"
        />
        </div>

        <div className="form-group">
        <label>Mode of Arrival:</label>
        <select onChange={(e) => handleChange('Arrival', e.target.value)}>
            <option value="">Select</option>
            {arrivalModes.map((mode) => (
            <option key={mode.value} value={mode.value}>
                {mode.label}
            </option>
            ))}
        </select>
        </div>

        <div className="form-group">
        <label>Injury:</label>
        <select onChange={(e) => handleChange('Injury', e.target.value)}>
            <option value="">Select</option>
            {injuryOptions.map((option) => (
            <option key={option.value} value={option.value}>
                {option.label}
            </option>
            ))}
        </select>
        </div>

        <div className="form-group">
        <label>Pain:</label>
        <select onChange={(e) => handleChange('Pain', e.target.value)}>
            <option value="">Select</option>
            <option value="1">Pain</option>
            <option value="0">Non-pain</option>
        </select>
        </div>

        <div className="form-group">
        <label>Mental Status:</label>
        <select onChange={(e) => handleChange('Mental', e.target.value)}>
            <option value="">Select</option>
            {mentalOptions.map((option) => (
            <option key={option.value} value={option.value}>
                {option.label}
            </option>
            ))}
        </select>
        </div>

        <div className="form-group">
        <label>NRS Pain (0-10):</label>
        <input
            type="text"
            onChange={(e) => handleChange('NRS_Pain', e.target.value)}
        />
        </div>

        <div className="form-group">
        <label>Chief Complaint:</label>
        <select onChange={(e) => handleChange('Chief_Complaint', e.target.value)}>
            <option value="">Select</option>
            {chiefComplaintOptions.map((option) => (
            <option key={option.value} value={option.value}>
                {option.label}
            </option>
            ))}
        </select>
        </div>

        <div className="form-group">
        <label>Systolic Blood Pressure (SBP):</label>
        <input
            type="text"
            onChange={(e) => handleChange('SBP', e.target.value)}
            placeholder="Enter SBP"
        />
        </div>

        <div className="form-group">
        <label>Diastolic Blood Pressure (DBP):</label>
        <input
            type="text"
            onChange={(e) => handleChange('DBP', e.target.value)}
            placeholder="Enter DBP"
        />
        </div>

        <div className="form-group">
        <label>Heart Rate (HR):</label>
        <input
            type="text"
            onChange={(e) => handleChange('HR', e.target.value)}
        />
        </div>

        <div className="form-group">
        <label>Body Temperature (BT):</label>
        <input
            type="text"
            onChange={(e) => handleChange('BT', e.target.value)}
            placeholder="e.g., 36.5"
        />
        </div>

        <div className="form-group">
        <label>Respiration Rate (RR):</label>
        <input
            type="text"
            onChange={(e) => handleChange('RR', e.target.value)}
        />
        </div>

        <div className="form-group">
        <label>Oxygen Saturation (%):</label>
        <input
            type="text"
            onChange={(e) => handleChange('Saturation', e.target.value)}
            placeholder="e.g., 95"
        />
        </div>
        <button className="form-submit" type="submit">Submit</button>
    </form>

    {apiResponse && (
        <div className="result-container">
        <h3>KTAS Level Prediction</h3>
        <div className="ktas-levels">
            {Object.entries(ktasDescriptions).map(([level, { color, title }]) => (
            <div
                key={level}
                className={`ktas-box ${apiResponse.ktas_prediction === parseInt(level, 10) ? 'active' : ''}`}
                style={{ backgroundColor: color }}
            >
                <h4>{title}</h4>
            </div>
            ))}
        </div>
        <div className="ktas-description">
            <h4>Description:</h4>
            <p>{ktasDescriptions[apiResponse.ktas_prediction].description}</p>
            <h4>Prescription:</h4>
            <p>{ktasDescriptions[apiResponse.ktas_prediction].prescription}</p>
            <p>KTAS Prediction: {apiResponse.ktas_prediction}</p>
            <p>Probabilities: {apiResponse.probabilities.join(', ')}</p>
        </div>
        </div>
    )}
    </>
)}
</div>
);
};

export default FormComponent;
