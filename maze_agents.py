import random
from collections import deque
import heapq
import tkinter as tk

# ======================
# Parameters
# ======================
ROWS, COLS = 15, 15
CELL_SIZE = 30
WALL_PROB = 0.3

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# ======================
# Maze Generation
# ======================

def generate_maze():
    return [[0 if random.random() > WALL_PROB else 1 for _ in range(COLS)] for _ in range(ROWS)]

maze = generate_maze()

free_cells = [(r, c) for r in range(ROWS) for c in range(COLS) if maze[r][c] == 0]
start = random.choice(free_cells)
GOAL = random.choice(free_cells)


def is_valid(pos):
    r, c = pos
    return 0 <= r < ROWS and 0 <= c < COLS and maze[r][c] == 0

# ======================
# Reflex Agent
# ======================
# à chaque cellule, il ne « voit » que les cases adjacentes
# libres et non encore visitées. Décision locale sans modèle explicite du labyrinthe
# ni planification globale — typique d’un agent stimulus → action.

def reflex_agent(start):
    current = start # Position initiale

    # Ensemble des cases déjà visitées
    # pour éviter de tourner en boucle
    visited = set()

    #le chemin parcouru
    path = [current]

    while current != GOAL:
        visited.add(current) # Marquer la position actuelle comme visitée
        neighbors = [] # Liste des voisins accessibles non visités

        # Explorer tous les mouvements possibles
        for dr, dc in MOVES:
            nxt = (current[0] + dr, current[1] + dc)

            if is_valid(nxt) and nxt not in visited:
                neighbors.append(nxt)

        if not neighbors:
            return None

        # Politique réflexe :
        # choisir aléatoirement un voisin valide
        current = random.choice(neighbors)

        # Ajouter la nouvelle position au chemin
        path.append(current) # Ajouter la nouvelle position au chemin
    return path

# ======================
# Model-Based Agent 
# ======================
# Ici le « modèle » est implicite : l’agent suppose que la grille est connue via
# is_valid (obstacle ou passage). Il explore systématiquement avec une pile (DFS).
# mémoire des nœuds déjà traités + backtracking implicite
# via la pile pour ne pas boucler indéfiniment sur le même sous-arbre mal géré.
# Le premier chemin trouvé n’est en général pas le plus court (propriété du DFS).

def model_based_agent(start):
    # Ensemble des cases déjà explorées
    visited = set()

    # Pile LIFO pour explorer les états
    # chaque élément = (position actuelle, chemin parcouru)
    stack = [(start, [start])]

    # Tant qu'il reste des états à explorer
    while stack:
        current, path = stack.pop() # LIFO : Retirer le dernier élément ajouté
        if current in visited: # Ignorer les cases déjà visitées
            continue

        visited.add(current) # Marquer la case comme explorée
        if current == GOAL: # Si l'objectif est atteint, retourner le chemin
            return path

        for dr, dc in MOVES:
            nxt = (current[0] + dr, current[1] + dc)

            if is_valid(nxt):
                stack.append((nxt, path + [nxt])) # Ajouter le voisin à la pile avec le chemin mis à jour

    return None

# ======================
# Goal based agent : BFS (agent objectif / exploration informée par largeur)
# ======================
# Largeur d’abord : explore niveau par niveau depuis le départ. Sur une grille
# à pas unitaires, la première fois qu’on atteint GOAL donne un chemin de longueur
# minimale (nombre de cases). 
# Coût : souvent plus cher en mémoire que le DFS sur de grands espaces.

def bfs(start):
    # File FIFO contenant les nœuds à explorer
    queue = deque([start])

    # Dictionnaire des parents pour reconstruire le chemin final
    # parent[node] = nœud précédent
    parent = {start: None}

    # Tant qu'il reste des nœuds à explorer
    while queue:
        current = queue.popleft() # Fifo :Retirer le premier élément ajouté

        # Si l'objectif est atteint, arrêter la recherche
        if current == GOAL:
            break

        # Explorer tous les voisins possibles
        for dr, dc in MOVES:
            nxt = (current[0] + dr, current[1] + dc)

            # Vérifier que le voisin est valide (pas obstacle et non visité)
            if is_valid(nxt) and nxt not in parent:
                parent[nxt] = current # Enregistrer son parent pour le chemin final

                queue.append(nxt) # Ajouter le voisin à la file pour exploration future

    if GOAL not in parent:
        return None

    # Reconstruction du chemin depuis GOAL jusqu'à start
    path = []
    node = GOAL
    while node:
        path.append(node)
        node = parent[node]

    # Inverser pour obtenir start -> GOAL
    return path[::-1]

# ======================
# A* (agent rationnel / recherche heuristique)
# ======================
# Combine coût réel depuis le départ (g) et estimation vers le but (h) en f = g + h.
# Avec une heuristique admissible (ne surestime jamais le coût restant), A* est optimal
# sur le plus court chemin. La distance de Manhattan est admissible pour une grille
# où on ne se déplace qu’en orthogonal et chaque pas coûte 1.

def heuristic(a, b):
    """Distance de Manhattan entre deux cellules : borne inférieure du nombre de pas orthogonaux."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star(start):
    # Liste de priorité contenant les nœuds à explorer. 
    open_list = []
    heapq.heappush(open_list, (0, start)) # Chaque élément = (f_cost, node)

    # coût réel minimal trouvé depuis start jusqu'à chaque nœud
    g_cost = {start: 0}

    # permet de reconstruire le chemin final : parent[node] = nœud précédent dans le meilleur chemin
    parent = {start: None}

    # Tant qu'il reste des nœuds à explorer
    while open_list:
        _, current = heapq.heappop(open_list) # Extraire le nœud avec le min f 

        if current == GOAL:
            break

        # Explorer tous les voisins possibles
        for dr, dc in MOVES:
            nxt = (current[0] + dr, current[1] + dc)

            # Ignorer les hors grille ou obstacle
            if not is_valid(nxt):
                continue

            # Nouveau coût réel pour atteindre nxt
            new_cost = g_cost[current] + 1 # déplacement = 1

            # mettre à jour seulement si on trouve
            # un chemin plus court vers nxt
            if nxt not in g_cost or new_cost < g_cost[nxt]:
                g_cost[nxt] = new_cost
                f_cost = new_cost + heuristic(nxt, GOAL)

                # Ajouter le voisin dans la file de priorité
                heapq.heappush(open_list, (f_cost, nxt))

                # Mémoriser le parent pour reconstruire le chemin
                parent[nxt] = current

    if GOAL not in parent:
        return None

    # Reconstruction du chemin en remontant
    # depuis GOAL jusqu'à start grâce aux parents
    path = []
    node = GOAL
    while node:
        path.append(node)
        node = parent[node]

    # Inverser la liste pour obtenir start -> GOAL
    return path[::-1]

# ======================
# GUI (Tkinter)
# ======================

class MazeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Maze Solver")

        main_frame = tk.Frame(root)
        main_frame.pack()

        # Canvas (left)
        self.canvas = tk.Canvas(main_frame, width=COLS * CELL_SIZE, height=ROWS * CELL_SIZE)
        self.canvas.pack(side=tk.LEFT)

        # Sidebar (right)
        sidebar = tk.Frame(main_frame, padx=10)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(sidebar, text="Results", font=("Arial", 12, "bold")).pack()

        self.result_text = tk.Text(sidebar, width=30, height=25)
        self.result_text.pack()

        # Buttons
        frame = tk.Frame(root)
        frame.pack()

        tk.Button(frame, text="Generate Maze", command=self.reset).pack(side=tk.LEFT)
        tk.Button(frame, text="Reflex Agent", command=self.solve_reflex).pack(side=tk.LEFT)
        tk.Button(frame, text="Model-Based", command=self.solve_model).pack(side=tk.LEFT)
        tk.Button(frame, text="Solve BFS", command=self.solve_bfs).pack(side=tk.LEFT)
        tk.Button(frame, text="Solve A*", command=self.solve_astar).pack(side=tk.LEFT)

        self.draw_maze()

    def draw_maze(self, path=None):
        self.canvas.delete("all")

        for r in range(ROWS):
            for c in range(COLS):
                x1, y1 = c * CELL_SIZE, r * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE

                color = "white" if maze[r][c] == 0 else "black"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")

        if path:
            for (r, c) in path:
                x1, y1 = c * CELL_SIZE, r * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="blue")

        # start and goal
        r, c = start
        self.canvas.create_rectangle(c * CELL_SIZE, r * CELL_SIZE,
                                     c * CELL_SIZE + CELL_SIZE, r * CELL_SIZE + CELL_SIZE,
                                     fill="green")

        r, c = GOAL
        self.canvas.create_rectangle(c * CELL_SIZE, r * CELL_SIZE,
                                     c * CELL_SIZE + CELL_SIZE, r * CELL_SIZE + CELL_SIZE,
                                     fill="red")

    def update_sidebar(self, name, path):
        self.result_text.insert(tk.END, f"\n{name}:\n")
        if path:
            self.result_text.insert(tk.END, f"Length: {len(path)}\nPath: {path}\n")
        else:
            self.result_text.insert(tk.END, "No path found\n")

    def reset(self):
        global maze, start, GOAL
        maze = generate_maze()
        free_cells = [(r, c) for r in range(ROWS) for c in range(COLS) if maze[r][c] == 0]
        start = random.choice(free_cells)
        GOAL = random.choice(free_cells)
        self.draw_maze()
        self.result_text.delete(1.0, tk.END)

    def solve_reflex(self):
        # Lance l’agent réactif : résultat variable, pas de garantie de succès ni d’optimalité.
        path = reflex_agent(start)
        self.draw_maze(path)
        self.update_sidebar("Reflex Agent", path)

    def solve_model(self):
        # Lance le DFS avec modèle du labyrinthe : un chemin valide s’il existe, pas forcément minimal.
        path = model_based_agent(start)
        self.draw_maze(path)
        self.update_sidebar("Model-Based Agent", path)

    def solve_bfs(self):
        # Plus court chemin en nombre de cases (recherche non informée complète sur ce graphe fini).
        path = bfs(start)
        self.draw_maze(path)
        self.update_sidebar("BFS", path)

    def solve_astar(self):
        # Même objectif que BFS (optimalité pas à pas) avec guidage heuristique pour réduire l’exploration.
        path = a_star(start)
        self.draw_maze(path)
        self.update_sidebar("A*", path)

# ======================
# Run App
# ======================

if __name__ == "__main__":
    root = tk.Tk()
    app = MazeApp(root)
    root.mainloop()
    root = tk.Tk()
    app = MazeApp(root)
    root.mainloop()
