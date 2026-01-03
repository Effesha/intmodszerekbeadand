import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# parameterek a Bayes-halohoz
PRIOR_INF = 0.10 # prior valoszinuseg
P_FEVER_IF_INF = 0.80 # ha influenza, akkor 80%, hogy lazas
P_FEVER_IF_NOT = 0.15 # ha nincs influenza, akkor 15%, hogy lazas
P_COUGH_IF_INF = 0.70 # ha influenza, akkor 70%, hogy kohog
P_COUGH_IF_NOT = 0.20 # ha nincs influenza, akkor 20%, hogy kohog
P_TEST_IF_INF = 0.85 # ha influenza, akkor 85%, hogy a teszt pozitiv
P_TEST_IF_NOT = 0.05 # ha nincs influenza, akkor 5%, hogy a teszt pozitiv - false positive

# formazo fuggveny, a valoszinusegu erteket 4 tizedesjegyre kerekitett stringkent adja vissza
def fmt(p):
    return f"{p:.4f}"

# Bayes-halot felepito algoritmus
# bemenet: influenza elofordulasi valoszinusege, teszt false positive aranya
def build_model(prior_inf=PRIOR_INF, p_test_if_not=P_TEST_IF_NOT):
    # halo struktura
    # sima "BayesianNetwork" deprecatedkent lett jelolve, a csomag a "DiscreteBayesianNetwork"-t jelolte meg, hogy ezt kell hasznalni helyette
    # ok: influenza, megfigyelesek: fever, cough, testpos
    model = DiscreteBayesianNetwork([
        ("Influenza", "Fever"),
        ("Influenza", "Cough"),
        ("Influenza", "TestPos"),
    ])

    # prior valoszinusegi tablaja
    cpd_influenza = TabularCPD(
        variable="Influenza",
        variable_card=2, # 0->nincs influenza, 1->van influenza
        values=[[1.0 - prior_inf], [prior_inf]]
    )

    # laz valoszinusegi tablaja
    cpd_fever = TabularCPD(
        variable="Fever",
        variable_card=2,
        values=[
            [1.0 - P_FEVER_IF_NOT, 1.0 - P_FEVER_IF_INF],  # fever=0
            [P_FEVER_IF_NOT, P_FEVER_IF_INF]               # fever=1
        ],
        evidence=["Influenza"],
        evidence_card=[2]
    )

    # kohoges valoszinusegi tablaja
    cpd_cough = TabularCPD(
        variable="Cough",
        variable_card=2,
        values=[
            [1.0 - P_COUGH_IF_NOT, 1.0 - P_COUGH_IF_INF],  # cough=0
            [P_COUGH_IF_NOT, P_COUGH_IF_INF]               # cough=1
        ],
        evidence=["Influenza"],
        evidence_card=[2]
    )

    # teszt valoszinusegi tablaja
    cpd_testpos = TabularCPD(
        variable="TestPos",
        variable_card=2,
        values=[
            [1.0 - p_test_if_not, 1.0 - P_TEST_IF_INF],  # testpos=0
            [p_test_if_not, P_TEST_IF_INF]               # testpos=1
        ],
        evidence=["Influenza"],
        evidence_card=[2]
    )

    model.add_cpds(cpd_influenza, cpd_fever, cpd_cough, cpd_testpos)
    return model

# Bayes-halo vizualis reprezentacioja
def draw_graph(model):
    # 1. iranyitott graf letrehozasa
    G = nx.DiGraph()

    # 2. csomopontok es elek hozzaadasa
    G.add_nodes_from(model.nodes())
    G.add_edges_from(model.edges())

    # 3. graf megrajzolasa
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(6, 4))
    nx.draw(G, pos, with_labels=True, node_size=1800, node_color="#ffefc4", arrowsize=20)
    plt.title("bayes-halo: influenza -> fever, cough, testpos")
    plt.tight_layout()
    plt.show()

# posterior valoszinuseg kiszamitasa
# egyetlen szamertekkel ter vissza
def query_posterior(model, evidence):
    # variable elimination algoritmus segitsegevel inference objektum letrehozasa
    infer = VariableElimination(model)
    q = infer.query(variables=["Influenza"], evidence=evidence, show_progress=False)
    # q.values sorrendje: index 0 -> influenza=0, index 1 -> influenza=1
    p_inf_1 = float(q.values[1])
    return p_inf_1

# a feladatot elvegzo funkcio
def main():
    # bemenet a szamitasokhoz
    evidence = {"Fever": 1, "Cough": 1, "TestPos": 1}
    print("hasznalt megfigyelesek (evidence): fever=1, cough=1, testpos=1\n")

    # 1. Bayes-halo

    model_base = build_model(prior_inf=PRIOR_INF, p_test_if_not=P_TEST_IF_NOT)

    # 2. ellenorzes: CPD, graf konzisztencia
    try:
        model_base.check_model()
    except Exception as e:
        print("model check hiba:", e)
        return

    # 3. posterior kiszamitasa
    # ez az alapeset, amihez a what-if eredmenyeket hasonlitjuk
    p_base = query_posterior(model_base, evidence)
    print("alapeset:")
    print(f"p(influenza=1 | evidence) = {fmt(p_base)}")

    # 4. what-if #1: romlo teszt: false positive novelese
    p_test_if_not_worse = 0.20
    model_worse_test = build_model(prior_inf=PRIOR_INF, p_test_if_not=p_test_if_not_worse)
    try:
        model_worse_test.check_model()
    except Exception as e:
        print("model check hiba (worse test):", e)
        return

    p_worse = query_posterior(model_worse_test, evidence)
    delta_worse = p_worse - p_base # alapesethez kepest a kulonbseg
    print("\nwhat-if #1 (romlo teszt, false positive = 0.20):")
    print(f"p(influenza=1 | evidence) = {fmt(p_worse)}")
    print(f"delta (what-if1 - alapeset) = {fmt(delta_worse)}")

    # 5. what-if #2: alapgyakorisag csokkentese
    prior_rare = 0.03
    model_rare = build_model(prior_inf=prior_rare, p_test_if_not=P_TEST_IF_NOT)
    try:
        model_rare.check_model()
    except Exception as e:
        print("model check hiba (rare prior):", e)
        return

    p_rare = query_posterior(model_rare, evidence)
    delta_rare = p_rare - p_base # alapesethez kepest a kulonbseg
    print("\nwhat-if #2 (ritkabb influenza, prior = 0.03):")
    print(f"p(influenza=1 | evidence) = {fmt(p_rare)}")
    print(f"delta (what-if2 - alapeset) = {fmt(delta_rare)}")

    # graf kirajzolasa
    draw_graph(model_base)

main()

# kerdesek:
# base rate hatas: megadja, mennyire valoszinu a betegseg - mennyire gyakori 
# a betegseg a populacioban. 

# teszt minosegenek szerepe: a tesztek meghatarozzak, hogy mennyire 
# megbizhatoak a pozitiv eredmenyek

# felteteles fuggetlenseg jelentese: a tunetek csak a betegseg alapotatol fuggenek,
# nem egymastol. A tunetek egymastol fuggetlenek
