# Base SIMULADA de «otro centro» para practicar la validación externa y la recalibración del ejemplo de manuscrito (descenso rápido de la TFGe).
# Mismas columnas que el desarrollo, pero otra mezcla de pacientes (mayores, menor TFGe, 70 % del centro B), otro riesgo basal (20 % en vez de 26 %)
# y efectos algo menores (85 %): el modelo desarrollado sobrestima el riesgo y es demasiado extremo. Ningún dato es real.
# Las bases de lesion_renal_aguda y hemodialisis_depresion se generan con tools/make_validation_data.py de EPI-PREDIC (se copian aquí).
import numpy as np, pandas as pd
rng = np.random.default_rng(2031); n = 600
age = np.clip(rng.normal(68, 11.5, n), 28, 90).round(); sex = np.where(rng.random(n) < .55, 'Male', 'Female'); center = np.where(rng.random(n) < .30, 'Center A', 'Center B')
dm = rng.random(n) < .55; ht = rng.random(n) < .92; cad = rng.random(n) < .32; statin = rng.random(n) < .33; acei = rng.random(n) < .77; sglt2 = rng.random(n) < .24
egfr = np.clip(rng.normal(37, 10, n), 15, 59).round(1); stage = np.where(egfr >= 45, '3a', np.where(egfr >= 30, '3b', '4'))
hba1c = np.clip(rng.normal(6.2, 1.0, n) + 0.5 * dm, 4.6, 12).round(1); upcr = np.exp(rng.normal(np.log(260), 1.0, n)).round(0)
z = rng.normal(0, 1, n)  # inflamación/desnutrición latente
neut = np.clip(4.6 + 0.9 * z + rng.normal(0, .7, n), 1.8, 12).round(2); lymph = np.clip(2.2 - 0.45 * z + rng.normal(0, .5, n), .5, 5).round(2); mono = np.clip(0.6 + 0.05 * z + rng.normal(0, .12, n), .2, 1.5).round(2)
plate = np.clip(rng.normal(250, 55, n), 90, 520).round(); hb = np.clip(12.8 - 0.55 * z + rng.normal(0, 1.1, n) - 0.02 * (45 - egfr), 7, 17).round(1)
alb = np.clip(4.15 - 0.22 * z + rng.normal(0, .3, n), 2.4, 5.0).round(2); chol = np.clip(200 - 14 * z + rng.normal(0, 32, n), 90, 380).round()
ca = np.clip(9.6 - 0.15 * z + rng.normal(0, .5, n), 7.5, 11).round(1)
nlr = (neut / lymph).round(2); lmr = (lymph / mono).round(2); nps = (nlr > 2.96).astype(int) + (lmr <= 4.44) + (alb < 4.0) + (chol <= 180)
lp = 0.85 * (-1.55 + 0.75 * z + 0.45 * np.log(upcr / 260) - 0.15 * (hb - 12.5) + 0.01 * (age - 65) - 0.01 * (egfr - 40) + 0.2 * dm)
def rate(b): return np.mean(rng.random(n) < 1 / (1 + np.exp(-(lp + b))))
lo, hi = -2, 2
for _ in range(40): mid = (lo + hi) / 2; lo, hi = (mid, hi) if np.mean(1 / (1 + np.exp(-(lp + mid)))) < 0.20 else (lo, mid)
y = rng.random(n) < 1 / (1 + np.exp(-(lp + lo)))
df = pd.DataFrame({'Age (years)': age.astype(int), 'Sex': sex, 'Centre': center, 'CKD stage': stage, 'Diabetes mellitus': np.where(dm, 'Yes', 'No'), 'Hypertension': np.where(ht, 'Yes', 'No'), 'Coronary artery disease': np.where(cad, 'Yes', 'No'),
    'Statin use': np.where(statin, 'Yes', 'No'), 'ACEi/ARB use': np.where(acei, 'Yes', 'No'), 'SGLT2i use': np.where(sglt2, 'Yes', 'No'), 'eGFR (mL/min/1.73 m2)': egfr, 'HbA1c (%)': hba1c, 'UPCR (mg/g)': upcr,
    'Haemoglobin (g/dL)': hb, 'Neutrophils (x10^3/uL)': neut, 'Lymphocytes (x10^3/uL)': lymph, 'Monocytes (x10^3/uL)': mono, 'Platelets (x10^3/uL)': plate, 'Albumin (g/dL)': alb, 'Total cholesterol (mg/dL)': chol, 'Calcium (mg/dL)': ca,
    'NLR': nlr, 'LMR': lmr, 'Naples score (0-4)': nps, 'Rapid eGFR decline': np.where(y, 'Yes', 'No')})
df.to_csv('examples/erc_descenso_rapido_validacion.csv', index=False); print(df.shape, y.mean().round(3), np.round(np.corrcoef(nps, y)[0, 1], 2))
