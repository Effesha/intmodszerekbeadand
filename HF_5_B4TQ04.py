import numpy as np
import skfuzzy as fuzz

# formazo fuggveny, az erteket 2 tizedesjegyre kerekitett stringkent adja vissza
def fmt(p):
    return f"{p:.2f}"

# a fuzzy rendszer alapja: univerzumok
# univerzumok: study, sleep, exam
def build_universes():
    study = np.arange(0, 41, 1)    
    sleep = np.arange(0, 10.1, 0.1)
    exam = np.arange(0, 101, 1)
    return study, sleep, exam

# tagsagi fuggvenyek letrehozasa
# megkapja a fuzzy rendszerunk 3 univerzumat
# mindegyikhez megad 3 fuzzy halmazt
def build_membership_functions(study, sleep, exam):
    # nyelvi ertekek: low, medium, high
    study_low = fuzz.trapmf(study, [0, 0, 5, 15])
    study_med = fuzz.trimf(study, [10, 20, 30])
    study_high = fuzz.trapmf(study, [25, 30, 40, 40])

    # nyelvi ertekek: poor, average, good
    sleep_poor = fuzz.trapmf(sleep, [0, 0, 2, 4])
    sleep_avg = fuzz.trimf(sleep, [3, 5, 7])
    sleep_good = fuzz.trapmf(sleep, [6, 8, 10, 10])

    # nyelvi ertekek: fail, pass, excellent
    exam_fail = fuzz.trapmf(exam, [0, 0, 30, 50])
    exam_pass = fuzz.trimf(exam, [40, 60, 80])
    exam_excellent = fuzz.trapmf(exam, [70, 85, 100, 100])

    mfs = {
        "study_low": study_low,
        "study_med": study_med,
        "study_high": study_high,
        "sleep_poor": sleep_poor,
        "sleep_avg": sleep_avg,
        "sleep_good": sleep_good,
        "exam_fail": exam_fail,
        "exam_pass": exam_pass,
        "exam_excellent": exam_excellent,
    }

    return mfs

# "fuzzifikacio": mennyire tartozik a bemenetet egy adott fuzzy halmazba
# a bemeneti ertekek fuzzy tagsagi ertekke alakitasa
# bemenetek: konkret tanulasi ido, alvasminoseg, univerzumok (study, sleep), tagsagi fuggvenyek szotara
def fuzzify_inputs(study_val, sleep_val, study, sleep, mfs):
    # tanulasi ido fuzzifikacioja
    mu_study_low = fuzz.interp_membership(study, mfs["study_low"], study_val)
    mu_study_med = fuzz.interp_membership(study, mfs["study_med"], study_val)
    mu_study_high = fuzz.interp_membership(study, mfs["study_high"], study_val)

    # alvasminoseg fuzzifikacioja
    mu_sleep_poor = fuzz.interp_membership(sleep, mfs["sleep_poor"], sleep_val)
    mu_sleep_avg = fuzz.interp_membership(sleep, mfs["sleep_avg"], sleep_val)
    mu_sleep_good = fuzz.interp_membership(sleep, mfs["sleep_good"], sleep_val)

    # eredmenyek eltarolasa
    fuzzified = {
        "study_low": mu_study_low,
        "study_med": mu_study_med,
        "study_high": mu_study_high,
        "sleep_poor": mu_sleep_poor,
        "sleep_avg": mu_sleep_avg,
        "sleep_good": mu_sleep_good,
    }

    return fuzzified

# Mamdani szabalybazis a feladatleiras szerint
def evaluate_rules(fuzzified, mfs, exam):
    rules = []

    # 1) keves tanulas + rossz alvas = fail 
    a1 = np.fmin(fuzzified["study_low"], fuzzified["sleep_poor"])
    out1 = np.fmin(a1, mfs["exam_fail"])
    rules.append(("study low and sleep poor -> fail", a1, out1))

    # 2) keves tanulas + atlagos alvas = fail
    a2 = np.fmin(fuzzified["study_low"], fuzzified["sleep_avg"])
    out2 = np.fmin(a2, mfs["exam_fail"])
    rules.append(("study low and sleep avg -> fail", a2, out2))

    # 3) kozepes tanulas es atlagos alvas => pass
    a3 = np.fmin(fuzzified["study_med"], fuzzified["sleep_avg"])
    out3 = np.fmin(a3, mfs["exam_pass"])
    rules.append(("study med and sleep avg -> pass", a3, out3))

    # 4) sok tanulas es jo alvas => excellent
    a4 = np.fmin(fuzzified["study_high"], fuzzified["sleep_good"])
    out4 = np.fmin(a4, mfs["exam_excellent"])
    rules.append(("study high and sleep good -> excellent", a4, out4))

    # 5) sok tanulas vagy jo alvas => pass/excellent 
    a5 = np.fmax(fuzzified["study_high"], fuzzified["sleep_good"])
    out5_pass = np.fmin(a5, mfs["exam_pass"])
    out5_ex = np.fmin(a5 * 0.7, mfs["exam_excellent"])
    rules.append(("study high or sleep good -> pass (and some excellent)", a5, np.fmax(out5_pass, out5_ex)))

    # 6) kozepes tanulas es jo alvas => pass 
    a6 = np.fmin(fuzzified["study_med"], fuzzified["sleep_good"])
    out6 = np.fmin(a6, mfs["exam_pass"])
    rules.append(("study med and sleep good -> pass", a6, out6))

    # 7) kev�s tanulas es jo alvas => pass 
    a7 = np.fmin(fuzzified["study_low"], fuzzified["sleep_good"])
    out7 = np.fmin(a7 * 0.5, mfs["exam_pass"])
    rules.append(("study low and sleep good -> weak pass", a7, out7))

    # 8) sok tanulas es rossz alvas => pass 
    a8 = np.fmin(fuzzified["study_high"], fuzzified["sleep_poor"])
    out8 = np.fmin(a8 * 0.6, mfs["exam_pass"])
    rules.append(("study high and sleep poor -> pass (reduced)", a8, out8))

    # aggregacio: szabalyok egyesitese egyetlen fuzzy halmazza
    # defuzzifikacio resze
    aggregated = np.zeros_like(exam, dtype=float)
    for desc, act, out_mf in rules:
        aggregated = np.fmax(aggregated, out_mf)

    return rules, aggregated

# defuzzifikacio: a kimeneti halmazbol egyetlen szamot keszit
# interpretacio: a kapott szamot egyetlen szoveges kategoriava alakitja
# centroid modszer: visszaadja a fuzzy halmaz kozepponti erteket
def defuzzify_and_interpret(aggregated, exam):
    # centroid defuzzifikacio
    result = fuzz.defuzz(exam, aggregated, 'centroid')
    # interpretacio egyszeru szabalyok alapjan
    if result < 50:
        interp = "fail"
    elif result < 75:
        interp = "pass"
    else:
        interp = "excellent"
    return result, interp

def main():
    # univerzumok es tagsagi fuggvenyek letrehozasa
    study, sleep, exam = build_universes()
    mfs = build_membership_functions(study, sleep, exam)

    # 3 kulonbozo bemeneti teszteset
    test_cases = [
        (2, 2),    # nagyon keves tanulas, rossz alvas -> expected: fail
        (20, 6),   # kozepes tanulas, atlagos/jo alvas -> expected: pass
        (35, 9),   # sok tanulas, jo alvas -> expected: pass, excellent
    ]

    # tesztesetek feldolgozasa
    print("defuzzifikacios mod: centroid\n")
    # i: eset sorszama, study_val: tanulasi ido, sleep_val: alvas minosege
    for i, (study_val, sleep_val) in enumerate(test_cases, start=1):
        fuzzified = fuzzify_inputs(study_val, sleep_val, study, sleep, mfs) # fuzzifikacio
        rules, aggregated = evaluate_rules(fuzzified, mfs, exam) # szabalyok kiertekelese
        result, interp = defuzzify_and_interpret(aggregated, exam) # defuzzifikacio, ertelmezes

        print(f"eset {i}: study_time = {study_val}, sleep_quality = {sleep_val}")
        print(f"defuzzified exam_result = {fmt(result)} / 100")
        print(f"interpretacio: {interp}\n")

main()

# osszegzes:
# fuzzifikacio: a konkret bemeneteket tagsagi ertekekke alakitjuk, hogy kezeljuk a bizonytalansagot
# szabalybazis: emberileg ertelmezheto if-then szabalyok aktivacioja alapjan allitjuk elo a kimeneti halmazokat
# aggregacio: az osszes szabaly hozzajarulasat egyesitjuk (max), hogy egyetlen kimeneti halmazt kapjunk
# defuzzifikacio: centroid modszert hasznaljuk, mert egyensulyi pontot ad a teljes kimeneti halmazon
#   a centroid gyakran stabil, es ertelmes numerikus becslest ad a mamdani rendszer kimenetere