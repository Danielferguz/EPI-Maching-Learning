# Prepara examples/rhc_ml.csv a partir de rhc.csv (cateterismo de arteria pulmonar, Connors et al., JAMA 1996; datos públicos de Vanderbilt).
# Uso: python3 tools/make_rhc.py ruta/a/rhc.csv
import sys, pandas as pd, numpy as np
d = pd.read_csv(sys.argv[1], na_values=['NA'])
los = np.where(d['dschdte'].notna(), d['dschdte'] - d['sadmdte'], d['dthdte'] - d['sadmdte'])
out = pd.DataFrame({'Death': d['death'], 'Length of stay (days)': los, 'RHC': d['swang1'].map({'RHC': 'Yes', 'No RHC': 'No'}),
    'Age': d['age'], 'Sex': d['sex'], 'Race': d['race'], 'Education (years)': d['edu'], 'Income': d['income'], 'Insurance': d['ninsclas'],
    'Primary disease category': d['cat1'], 'Cancer': d['ca'], 'DNR order day 1': d['dnr1'], 'APACHE score': d['aps1'], 'Coma score': d['scoma1'], 'Mean blood pressure': d['meanbp1'],
    'White blood cells': d['wblc1'], 'Heart rate': d['hrt1'], 'Respiratory rate': d['resp1'], 'Temperature (C)': d['temp1'], 'PaO2/FiO2': d['pafi1'], 'Albumin': d['alb1'],
    'Hematocrit': d['hema1'], 'Bilirubin': d['bili1'], 'Creatinine': d['crea1'], 'Sodium': d['sod1'], 'Potassium': d['pot1'], 'PaCO2': d['paco21'], 'pH': d['ph1'], 'Weight (kg)': d['wtkilo1'],
    'Predicted 2-month survival': d['surv2md1'], 'Duke Activity Status Index': d['das2d3pc'], 'ADL dependence': d['adld3p'], 'Urine output': d['urin1']})
for c, n in [('cardiohx', 'Cardiovascular history'), ('chfhx', 'Heart failure'), ('dementhx', 'Dementia'), ('psychhx', 'Psychiatric history'), ('chrpulhx', 'Chronic pulmonary disease'), ('renalhx', 'Chronic renal disease'),
             ('liverhx', 'Cirrhosis'), ('gibledhx', 'GI bleed'), ('malighx', 'Solid tumor'), ('immunhx', 'Immunosuppression'), ('transhx', 'Transfer from another hospital'), ('amihx', 'Definite myocardial infarction')]:
    out[n] = d[c].map({0: 'No', 1: 'Yes'})
for c, n in [('resp', 'Respiratory diagnosis'), ('card', 'Cardiovascular diagnosis'), ('neuro', 'Neurological diagnosis'), ('gastr', 'Gastrointestinal diagnosis'), ('renal', 'Renal diagnosis'), ('meta', 'Metabolic diagnosis'),
             ('hema', 'Hematologic diagnosis'), ('seps', 'Sepsis diagnosis'), ('trauma', 'Trauma diagnosis'), ('ortho', 'Orthopedic diagnosis')]:
    out[n] = d[c]
for c in out.select_dtypes('number').columns: out[c] = out[c].round(3)
out.to_csv('examples/rhc_ml.csv', index=False); print(out.shape, out['Death'].value_counts().to_dict())
