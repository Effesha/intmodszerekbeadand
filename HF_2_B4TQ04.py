# tobbsoros stringbol 2D racsot keszit
def parse_grid(input_grid):
    grid = []
    for line in input_grid.splitlines():
        if len(line) == 0:
            continue
        row = []
        for i in range(len(line)):
            row.append(line[i])
        grid.append(row)
    return grid

# inputban kapott karakter megkeresese egy 2D gridben
# elofeltetel: parse_grid(grid)
def find_char(grid, ch):
    for row in range(len(grid)):
        for col in range(len(grid[row])):
            if grid[row][col] == ch:
                return (row, col) # megtalalt karakter pozicioja
        
# ellenorzi, hogy a grid hatarain belul vagyunk-e (szomszedok ellenorzesenel lesz hasznos)
def is_inside_grid_boundaries(grid, row, col):
    return 0 <= row < len(grid) and 0 <= col < len(grid[row])

# ellenorzi, hogy nem utkoztunk-e falba
def is_not_wall(grid, row, col):
    return grid[row][col] != "#"

# aktualis poziciohoz kepest bal-jobb-fel-le szomszed
def neighbours_4(grid, row, col):
    result = []
    directions = [
        (-1, 0),  # fel
        (1, 0),   # le
        (0, -1),  # bal
        (0, 1)    # jobb
    ]
    for row_direction, col_direction in directions:
        new_row = row + row_direction
        new_col = col + col_direction
        
        # lepes csak akkor engedelyezett, ha nem fal & griden belul vagyunk
        if is_inside_grid_boundaries(grid, new_row, new_col) and is_not_wall(grid, new_row, new_col):
            result.append((new_row, new_col))

    return result

# startbol celig vezeto ut (megoldas) eloallitasa
# came_from: gyerek es szulo csucsok - honnan jottunk, hova jutottunk
def reconstruct_path(came_from, start, goal):
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path

# kirajzolja az utvonalat ASCII-ben
# az utat * (csillag) karakter jeloli
import copy
def render_path(grid, path):
    grid_copy = copy.deepcopy(grid)
    for i in range(len(path)):
        path_row = path[i][0]
        path_column = path[i][1]
        if grid_copy[path_row][path_column] not in ["S", "G"]:
            grid_copy[path_row][path_column] = "*"
    lines = []
    for row in range(len(grid_copy)):
        row_characters = ''.join(grid_copy[row])
        lines.append(row_characters)
    grid_out = "\n".join(lines)
    return grid_out

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

import heapq
def a_star_algorithm(grid, start, goal):
    open_heap = []
    counter = 0

    # g_score: kezdoponttol valo tenyleges koltseg
    g_score = {start: 0}

    # f_score: g + h
    # A* ertekelo fuggvenye
    # g: tenyleges koltseg a kezdoponttol
    # h: heurisztika (esetunkben: Manhattan-tavolsaggal)
    f_score = {start: manhattan_distance(start, goal)}

    heapq.heappush(open_heap, (f_score[start], counter, start))
    counter += 1

    came_from = {}  # child -> parent
    closed_set = set()

    while open_heap:
        current_f, _, current = heapq.heappop(open_heap)

        if current == goal:
            return reconstruct_path(came_from, start, goal)

        if current in closed_set:
            continue
        closed_set.add(current)

        for neighbor in neighbours_4(grid, current[0], current[1]):
            if neighbor in closed_set:
                continue

            tentative_g = g_score[current] + 1  # minden lepes koltsege 1

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + manhattan_distance(neighbor, goal)
                f_score[neighbor] = f
                heapq.heappush(open_heap, (f, counter, neighbor))
                counter += 1

    # ha kifogyott a nyílt halmaz és nem találtuk meg a célt
    return None


grid = r"""
##########
#S...#...#
#.##.#.#.#
#........#
#.####.#G#
##########
"""

parsed_grid = parse_grid(grid)

start = find_char(parsed_grid, "S")
goal = find_char(parsed_grid, "G")

path = a_star_algorithm(parsed_grid, start, goal)

if path:
    print("Utvonal megtalalva.")
    print(render_path(parsed_grid, path))
else:
    print("Nincs elerheto ut a celhoz.")

