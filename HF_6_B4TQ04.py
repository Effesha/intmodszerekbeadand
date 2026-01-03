import random
import numpy as np
from typing import Tuple

# parameterek
N = 50 # populaciomeret
G = 80 # generaciok szama
k = 2 # tornaszelekcio merete
mutation_rate = 0.08 # mutacio szorasa
random_seed = 42 # veletlen random_seed 
E = 1 # elitizmus

# a genetikus algoritmus altal vizsgalt fuggveny (celfuggveny)
# ez a feladatban lett meghatarozva
# ennek keressuk a maximumat a [0,1] intervallumon
def f(x: np.ndarray) -> np.ndarray:    
    return x * np.sin(10 * np.pi * x) + 1

# kezdeti populacio letrehozasa
# 'n' darab veletlenszamot general a [0,1] intervallumbol
def init_population(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.random(n)

# tornaszelkcio megvalositasa
# kivalaszt 'k' darab egyedet a populaciobol
# ezek fitness erteke alapjan kivalasztja a legjobbat
# a legjobb indexet adja vissza (best = idx[...])
def tournament_selection(pop: np.ndarray, fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
    idx = rng.integers(0, len(pop), size=k)
    best = idx[np.argmax(fitness[idx])]
    return best

# aritmetikai keresztezest hajt vegre a feladatban megadott keplettel
# ket szulobol egy uj utodot hoz letre
# veletlenszeru 'a' ertekkel valtoztatva a szulok ertekeit
def arithmetic_crossover(x1: float, x2: float, rng: np.random.Generator) -> float:
    a = rng.random()
    return a * x1 + (1 - a) * x2

# mutacio elvegzese Gauss-zajjal
# egy ertekhez veletlen zajt rendel, amivel az ertek veletlen iranyba elmozdul
# segitsegevel uj ertekeket fedezhetunk fel
# a clip() segit a [0,1] tartomanyban maradni
def mutate(x: float, mutation_rate: float, rng: np.random.Generator) -> float:
    x_new = x + rng.normal(0, mutation_rate)
    return float(np.clip(x_new, 0.0, 1.0))

# egy generaciot allit elo a genetikus algoritmusban
# visszaadja az uj populaciot es elvegzi a feladat altal kert naplozast
def run_generation(pop: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
    # 1. minden egyedre fitness kiszamitasa
    fitness = f(pop)

    # 2. legjobb egyedek megtartasa
    # fitnezz szerint sorba rendezes
    # legnagyobb fitness ertekuek kivalasztasa
    elite_indices = np.argsort(fitness)[-E:]
    elites = pop[elite_indices].copy() # a legjobbak garantaltan mennek a kovetkezo generacioba

    # 3. uj egyedek eloallitasa
    new_pop = []
    while len(new_pop) < N - E:
        # 3a. szulok kivalasztasa
        p1_idx = tournament_selection(pop, fitness, k, rng)
        p2_idx = tournament_selection(pop, fitness, k, rng)
        x1 = pop[p1_idx]
        x2 = pop[p2_idx]

        # 3b. ket szulobol uj utod
        child = arithmetic_crossover(x1, x2, rng)

        # 3c. utod mutalasa
        child = mutate(child, mutation_rate, rng)
        new_pop.append(child) # utod hozzaadasa az uj populaciohoz

    # uj populacio letrehozasa: elitek es uj utodok
    new_pop = np.array(new_pop) # uj utodokbol kepzett tomb
    next_pop = np.concatenate([elites, new_pop]) # elitek hozzaadasa az iment tombhoz

    # naplozas
    logs = {
        "legjobb_fitness_ertek": float(np.max(f(next_pop))),
        "atlagos_fitness_ertek": float(np.mean(f(next_pop))),
        "fitness_ertekek_szorasa": float(np.std(f(next_pop))),
        "legjobb_x_ertek": float(next_pop[np.argmax(f(next_pop))])
    }

    return next_pop, logs

# a feladatot elvegzo funkcio
def main():
    # veletlenszam generalas a feladatban megadott seed-del
    random.seed(random_seed)
    rng = np.random.default_rng(random_seed)

    # kezdeti populacio letrehozasa
    pop = init_population(N, rng)

    # legjobb ertekeket tarolo valtozok inicializalasa
    best_overall_x = None
    best_overall_f = -np.inf

    # kezdeti populacio kiertekelese
    fitness = f(pop)
    best_idx = np.argmax(fitness)
    best_overall_x = float(pop[best_idx])
    best_overall_f = float(fitness[best_idx])

    print(f"Parameterek: N={N}, G={G}, k={k}, mutation_rate={mutation_rate}, random_seed={random_seed}, E={E}")
    print("-" * 60)

    for gen in range(1, G + 1):
        pop, logs = run_generation(pop, rng)

        # ha az aktualis generacioban jobb megoldast talaltunk, akkor a legjobb ertekeket frissitjuk
        if logs["legjobb_fitness_ertek"] > best_overall_f:
            best_overall_f = logs["legjobb_fitness_ertek"]
            best_overall_x = logs["legjobb_x_ertek"]

        # minden 10. generacioban naplozunk
        if gen % 10 == 0 or gen == 1 or gen == G:
            print(f"Gen {gen:3d}: best f(x) = {logs['legjobb_fitness_ertek']:.6f}, "
                  f"avg = {logs['atlagos_fitness_ertek']:.6f}, std = {logs['fitness_ertekek_szorasa']:.6f}, "
                  f"best x = {logs['legjobb_x_ertek']:.6f}")

    print("-" * 60)
    print("Vegeredmeny:")
    print(f"Legjobb x: {best_overall_x:.8f}")
    print(f"Legjobb f(x): {best_overall_f:.8f}")

main()

# kerdesek:
# A futtatas soran megfigyelheto volt, hogy a populacio legjobb es atlagos fitness erteke
# altalaban javult a generaciok soran, de nem monoton modon: kisebb ingadozasok jelentkeztek,
# ami a GA sztochasztikus termeszetebol adodik. 
# Az elitizmus (E=1) biztositotta, hogy a
# legjobb megtalalt egyed ne vesszen el, igy a vegso legjobb f(x) erteke erdemben jobb lett,
# mint a kezdeti populacioe.
