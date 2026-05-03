"""
IE 400 Term Project – Spring 2026
Bilkent University, Department of Industrial Engineering



Authors: Berke İsmail Erhan Asıl 22203732
         Halis Vefa Türkyılmaz   22102898
         Aziz Üzümcü             22102800

"""

import gurobipy as gp
from gurobipy import GRB
from collections import deque

#  QUESTION 1 – MUSEUM ESCAPE

GRID_ROWS = 8
GRID_COLS = 8
START_CELL = (1, 1)
EXIT_CELL  = (8, 8)

OBSTACLE_CELLS = frozenset({
    (1, 3),
    (2, 2), (2, 5),
    (3, 7),
    (4, 2), (4, 3),
    (5, 5),
    (6, 4),
    (7, 1), (7, 4), (7, 5),
})

VALID_CELLS = frozenset(
    (r, c)
    for r in range(1, GRID_ROWS + 1)
    for c in range(1, GRID_COLS + 1)
    if (r, c) not in OBSTACLE_CELLS
)


# Camera 1: 
CAMERA1_VISIBILITY = {
    0: frozenset({(3, 3), (3, 4), (3, 5)}),
    1: frozenset({(3, 4), (3, 5), (3, 6)}),
    2: frozenset({(3, 5), (3, 4), (3, 3)}),
    3: frozenset({(3, 4), (3, 3), (3, 2)}),
}

# Camera 2:
CAMERA2_VISIBILITY = {
    0: frozenset({(4, 6), (5, 6)}),
    1: frozenset({(5, 6), (6, 6)}),
    2: frozenset({(6, 6), (6, 7)}),
    3: frozenset({(6, 7), (6, 6)}),
    4: frozenset({(6, 6), (5, 6)}),
    5: frozenset({(5, 6), (4, 6)}),
}

# Camera 3: 
CAMERA3_VISIBILITY = {
    0: frozenset({(7, 7), (6, 8)}),
    1: frozenset({(6, 8), (7, 7), (8, 6)}),
    2: frozenset({(7, 7), (8, 6)}),
    3: frozenset({(8, 6), (7, 7), (6, 8)}),
}


def getCameraVisibleCells(timeStep):
    return (
        CAMERA1_VISIBILITY[timeStep % 4]
        | CAMERA2_VISIBILITY[timeStep % 6]
        | CAMERA3_VISIBILITY[timeStep % 4]
    )


ALL_MOVES = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]


def getNeighbours(row, col):
    return [
        (row + dr, col + dc)
        for dr, dc in ALL_MOVES
        if 1 <= row + dr <= GRID_ROWS
        and 1 <= col + dc <= GRID_COLS
        and (row + dr, col + dc) not in OBSTACLE_CELLS
    ]


#BFS 

def runTimedBFS(extraBlockFn=None, forcedPositions=None):
    if extraBlockFn is None:
        extraBlockFn = lambda t, r, c: False

    parentOf = {}
    visitedStates = {(START_CELL[0], START_CELL[1], 0)}
    queue = deque([(START_CELL[0], START_CELL[1], 0)])

    while queue:
        currentRow, currentCol, currentTime = queue.popleft()

        if (currentRow, currentCol) == EXIT_CELL:
            path = []
            state = (currentRow, currentCol, currentTime)
            while state is not None:
                r, c, t = state
                path.append((t, r, c))
                state = parentOf.get(state)
            path.reverse()
            return currentTime, path

        for nextRow, nextCol in getNeighbours(currentRow, currentCol):
            nextTime = currentTime + 1
            if nextTime > 60:
                continue
            if (nextRow, nextCol) != EXIT_CELL:
                if (nextRow, nextCol) in getCameraVisibleCells(nextTime):
                    continue
            if extraBlockFn(nextTime, nextRow, nextCol):
                continue
            if forcedPositions and nextTime in forcedPositions:
                if (nextRow, nextCol) != forcedPositions[nextTime]:
                    continue
            nextState = (nextRow, nextCol, nextTime)
            if nextState not in visitedStates:
                visitedStates.add(nextState)
                parentOf[nextState] = (currentRow, currentCol, currentTime)
                queue.append((nextRow, nextCol, nextTime))

    return None, None



def solveMuseumIP(taskName, horizonT, addExtraConstraints=None, relaxedTransitions=None):
    """
    Build and solve the museum IP as a feasibility problem at fixed horizon T.

    Decision variables: occupancy[t, r, c] in {0, 1}
    Objective        : constant zero (feasibility)

    Constraints enforced:
      (C1) Start at START_CELL at t=0
      (C2) Exactly one cell occupied per time step
      (C3) Valid movement: backward + forward adjacency (no diagonal, no jump)
           Transitions in relaxedTransitions are skipped (teleport allowed).
      (C4) Camera avoidance: forbidden cells = 0 (exit exempt)
      (C5) Must reach EXIT_CELL at t = horizonT

    Returns (horizonT, path) or (None, None).
    """
    model = gp.Model(taskName)
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", 120)

    # Transitions (t -> t+1) where the movement constraint is lifted (teleport)
    relaxed = set(relaxedTransitions) if relaxedTransitions else set()

    occupancy = {
        (t, r, c): model.addVar(vtype=GRB.BINARY, name=f"occupy_t{t}_r{r}_c{c}")
        for t in range(horizonT + 1)
        for (r, c) in VALID_CELLS
    }

    model.addConstr(occupancy[0, START_CELL[0], START_CELL[1]] == 1)
    for (r, c) in VALID_CELLS:
        if (r, c) != START_CELL:
            model.addConstr(occupancy[0, r, c] == 0)

    for t in range(horizonT + 1):
        model.addConstr(
            gp.quicksum(occupancy[t, r, c] for (r, c) in VALID_CELLS) == 1
        )

    for t in range(horizonT):
        if t in relaxed:
            continue
        for (r, c) in VALID_CELLS:
            nbrs = getNeighbours(r, c)   
            model.addConstr(
                occupancy[t + 1, r, c]
                <= gp.quicksum(occupancy[t, nr, nc] for (nr, nc) in nbrs)
            )

    for t in range(horizonT + 1):
        for (r, c) in getCameraVisibleCells(t):
            if (r, c) in VALID_CELLS and (r, c) != EXIT_CELL:
                model.addConstr(occupancy[t, r, c] == 0)

    model.addConstr(occupancy[horizonT, EXIT_CELL[0], EXIT_CELL[1]] == 1)

    if addExtraConstraints:
        addExtraConstraints(model, occupancy, horizonT)

    model.setObjective(0, GRB.MINIMIZE)
    model.optimize()

    if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and model.SolCount > 0:
        path = []
        for t in range(horizonT + 1):
            for (r, c) in sorted(VALID_CELLS):   # deterministic order
                if occupancy[t, r, c].X > 0.5:
                    path.append((t, r, c))
                    break
        return horizonT, path
    return None, None



def printPath(label, optimalT, path):
    print(f"\n  {label}\n")
    if optimalT is None:
        print("  *** INFEASIBLE – no solution found ***")
        return
    print(f"  Optimal time steps : {optimalT}")
    print(f"  {'t':>3}   cell")
    print(f"  {'-' * 22}")
    for t, r, c in path:
        annotation = ""
        if (r, c) == START_CELL and t == 0:
            annotation = "  <- START"
        if (r, c) == EXIT_CELL:
            annotation = "  <- EXIT OK"
        print(f"  {t:>3}   ({r},{c}){annotation}")


def validatePath(label, path):
    """Check all constraints: no obstacles, no cameras, valid moves, correct endpoints."""
    if path is None:
        return
    allOk = True
    cellAt = {t: (r, c) for t, r, c in path}
    for t, r, c in path:
        if (r, c) in OBSTACLE_CELLS:
            print(f"  FAIL [{label}] t={t}: ({r},{c}) is an obstacle!")
            allOk = False
        if (r, c) != EXIT_CELL and (r, c) in getCameraVisibleCells(t):
            print(f"  FAIL [{label}] t={t}: ({r},{c}) is camera-visible!")
            allOk = False
        if t > 0:
            prevRow, prevCol = cellAt[t - 1]
            if abs(r - prevRow) + abs(c - prevCol) > 1:
                print(f"  FAIL [{label}] illegal jump ({prevRow},{prevCol}) -> ({r},{c})")
                allOk = False
    if cellAt[0] != START_CELL:
        print(f"  FAIL [{label}] wrong start cell")
        allOk = False
    if cellAt[max(cellAt)] != EXIT_CELL:
        print(f"  FAIL [{label}] wrong end cell")
        allOk = False
    if allOk:
        print(f"  OK [{label}] all constraints satisfied.")



print("  QUESTION 1 – MUSEUM ESCAPE")

baseOptimalT, _ = runTimedBFS()
print(f"\n  [BFS] Base case optimal T = {baseOptimalT}")
task2_T, task2_path = solveMuseumIP("Task2_BaseCase", baseOptimalT)
printPath("Task 2 – Base Case", task2_T, task2_path)
validatePath("Task2", task2_path)

#  Task 2 – Branch-and-Bound

class BBNode:
    __slots__ = ("fixings", "depth", "label")

    def __init__(self, fixings, depth, label):
        self.fixings = fixings   
        self.depth   = depth
        self.label   = label


def _buildLPSubproblem(taskName, horizonT, fixings):
    model = gp.Model(taskName)
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", 30)

    occupancy = {
        (t, r, c): model.addVar(
            lb=0.0, ub=1.0,
            vtype=GRB.CONTINUOUS,
            name=f"occ_t{t}_r{r}_c{c}",
        )
        for t in range(horizonT + 1)
        for (r, c) in VALID_CELLS
    }

    model.addConstr(occupancy[0, START_CELL[0], START_CELL[1]] == 1.0)
    for (r, c) in VALID_CELLS:
        if (r, c) != START_CELL:
            model.addConstr(occupancy[0, r, c] == 0.0)

    for t in range(horizonT + 1):
        model.addConstr(
            gp.quicksum(occupancy[t, r, c] for (r, c) in VALID_CELLS) == 1.0
        )

    for t in range(horizonT):
        for (r, c) in VALID_CELLS:
            nbrs = getNeighbours(r, c)
            model.addConstr(
                occupancy[t + 1, r, c]
                <= gp.quicksum(occupancy[t, nr, nc] for (nr, nc) in nbrs)
            )

    for t in range(horizonT + 1):
        for (r, c) in getCameraVisibleCells(t):
            if (r, c) in VALID_CELLS and (r, c) != EXIT_CELL:
                model.addConstr(occupancy[t, r, c] == 0.0)

    model.addConstr(occupancy[horizonT, EXIT_CELL[0], EXIT_CELL[1]] == 1.0)

    for (t, r, c), val in fixings.items():
        model.addConstr(occupancy[t, r, c] == float(val))

    model.setObjective(0, GRB.MINIMIZE)
    return model, occupancy


def _isFractional(val, tol=1e-6):
    return tol < val < 1.0 - tol


def _extractPath(horizonT, occupancy):
    path = []
    for t in range(horizonT + 1):
        for (r, c) in sorted(VALID_CELLS):
            if occupancy[t, r, c].X > 0.5:
                path.append((t, r, c))
                break
    return path


def solveMuseumBranchAndBound(horizonT):
    print(f"  [B&B] Fixed horizon T = {horizonT}")

    rootNode = BBNode(fixings={}, depth=0, label="Root")
    stack = [rootNode]     
    nodeCount = 0

    while stack:
        node = stack.pop()
        nodeCount += 1
        indent = "    " + "  " * node.depth   

        model, occupancy = _buildLPSubproblem(
            f"BB_node_{nodeCount}", horizonT, node.fixings
        )
        model.optimize()

        lpFeasible = model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and model.SolCount > 0

        if not lpFeasible:
            print(f"{indent}[B&B] Node {node.label} (depth {node.depth}): "
                  f"LP INFEASIBLE – pruned.")
            continue

        fractionalVar = None   
        for t in range(horizonT + 1):
            for (r, c) in sorted(VALID_CELLS):
                val = occupancy[t, r, c].X
                if _isFractional(val):
                    fractionalVar = (t, r, c)
                    break
            if fractionalVar is not None:
                break

        if fractionalVar is None:
            path = _extractPath(horizonT, occupancy)
            if node.depth == 0:
                print(f"{indent}[B&B] Root Node: Solved, integer solution found. "
                      f"No branching required.")
            else:
                print(f"{indent}[B&B] Node {node.label} (depth {node.depth}): "
                      f"Integer solution found. B&B terminates.")
            print(f"  [B&B] Total nodes explored: {nodeCount}")
            return horizonT, path

        ft, fr, fc = fractionalVar
        fracVal = occupancy[ft, fr, fc].X
        print(f"{indent}[B&B] Node {node.label} (depth {node.depth}): "
              f"LP feasible, fractional occ[t={ft},r={fr},c={fc}]={fracVal:.4f}. "
              f"Branching ...")

        fixings0 = dict(node.fixings)
        fixings0[fractionalVar] = 0
        child0 = BBNode(
            fixings=fixings0,
            depth=node.depth + 1,
            label=f"{node.label}->occ[{ft},{fr},{fc}]=0",
        )

        fixings1 = dict(node.fixings)
        fixings1[fractionalVar] = 1
        child1 = BBNode(
            fixings=fixings1,
            depth=node.depth + 1,
            label=f"{node.label}->occ[{ft},{fr},{fc}]=1",
        )

        stack.append(child1)
        stack.append(child0)

    print(f"  [B&B] Stack exhausted. Problem INFEASIBLE at T={horizonT}.")
    print(f"  [B&B] Total nodes explored: {nodeCount}")
    return None, None

bb2_T, bb2_path = solveMuseumBranchAndBound(baseOptimalT)
printPath("Task 2 – Base Case (Branch-and-Bound)", bb2_T, bb2_path)
validatePath("Task2_BB", bb2_path)


TRAP_CELL      = (5, 1)
FORBIDDEN_CELL = (8, 3)


def addBoobyTrapConstraints(model, occupancy, horizonT):
    if TRAP_CELL not in VALID_CELLS:
        print("  NOTE: trap cell (5,1) is an obstacle - constraint inactive.")
        return
    if FORBIDDEN_CELL not in VALID_CELLS:
        print("  NOTE: forbidden cell (8,3) is an obstacle - already unreachable.")
        return
    # trapTriggered[t] = 1 if Lara has visited TRAP_CELL at or before time t
    trapTriggered = {
        t: model.addVar(vtype=GRB.BINARY, name=f"trapTriggered_t{t}")
        for t in range(horizonT + 1)
    }
    for t in range(horizonT + 1):
        model.addConstr(trapTriggered[t] >= occupancy[t, TRAP_CELL[0], TRAP_CELL[1]])
        if t > 0:
            model.addConstr(trapTriggered[t] >= trapTriggered[t - 1])
    for t in range(1, horizonT + 1):
        model.addConstr(
            occupancy[t, FORBIDDEN_CELL[0], FORBIDDEN_CELL[1]] <= 1 - trapTriggered[t - 1]
        )


task3_T, task3_path = solveMuseumIP("Task3_BoobyTrap", baseOptimalT, addBoobyTrapConstraints)
if task3_T is None:
    for extraSteps in range(1, 10):
        task3_T, task3_path = solveMuseumIP(
            "Task3_BoobyTrap", baseOptimalT + extraSteps, addBoobyTrapConstraints
        )
        if task3_T is not None:
            break
printPath("Task 3 - Booby Trap [(5,1) -> never (8,3)]", task3_T, task3_path)
validatePath("Task3", task3_path)


EXIT_LOCK_CELLS = {(6, 7), (7, 6), (7, 7), (8, 6), (6, 8)}
EXIT_LOCK_TIME  = 18


def isExitLockBlocked(t, r, c):
    return (r, c) in EXIT_LOCK_CELLS and t < EXIT_LOCK_TIME


def addExitLockConstraints(model, occupancy, horizonT):
    for t in range(min(EXIT_LOCK_TIME, horizonT + 1)):
        for (r, c) in EXIT_LOCK_CELLS:
            if (r, c) in VALID_CELLS:
                model.addConstr(occupancy[t, r, c] == 0)


task4_T, _ = runTimedBFS(extraBlockFn=isExitLockBlocked)
print(f"\n  [BFS] Task 4 (exit lock) optimal T = {task4_T}")
if task4_T is None:
    print("  BFS: no path found with exit lock constraint.")
    task4_T, task4_path = None, None
else:
    task4_T, task4_path = solveMuseumIP("Task4_ExitLock", task4_T, addExitLockConstraints)
printPath("Task 4 - Exit Lock (cells forbidden before t=18)", task4_T, task4_path)
validatePath("Task4", task4_path)


GHOST_CELL = (4, 4)
GHOST_TIME = 3
TASK5_TARGET_T = 14


def addGhostCheckpointConstraints(model, occupancy, horizonT):
    if GHOST_CELL not in VALID_CELLS:
        print("  WARNING: ghost cell (4,4) is an obstacle - infeasible.")
        return
    if horizonT < GHOST_TIME:
        print(f"  WARNING: horizon T={horizonT} < ghost time {GHOST_TIME}.")
        return
    model.addConstr(occupancy[GHOST_TIME, GHOST_CELL[0], GHOST_CELL[1]] == 1)

print(f"\n  [Task 5] Forced teleport to {GHOST_CELL} at t={GHOST_TIME}; "
      f"relaxing movement constraint for transition t=2->t=3.")
print(f"  [Task 5] Searching from T={TASK5_TARGET_T} upward ...")

task5_T, task5_path = None, None
for candidateT in range(TASK5_TARGET_T, TASK5_TARGET_T + 10):
    task5_T, task5_path = solveMuseumIP(
        "Task5_GhostCheckpoint",
        candidateT,
        addGhostCheckpointConstraints,
        relaxedTransitions={2},   
    )
    if task5_T is not None:
        break

printPath("Task 5 - Ghost Checkpoint [(4,4) at t=3, teleport allowed]", task5_T, task5_path)
if task5_path is not None:
    print("  [Task5] Note: the t=2->t=3 transition is a deliberate teleport; "
          "all other steps are validated below.")
    cellAt = {t: (r, c) for t, r, c in task5_path}
    allOk = True
    for t, r, c in task5_path:
        if (r, c) in OBSTACLE_CELLS:
            print(f"  FAIL [Task5] t={t}: ({r},{c}) is an obstacle!")
            allOk = False
        if (r, c) != EXIT_CELL and (r, c) in getCameraVisibleCells(t):
            print(f"  FAIL [Task5] t={t}: ({r},{c}) camera-visible!")
            allOk = False
        if t > 0 and t != GHOST_TIME:   # skip the teleport step
            pr, pc = cellAt[t - 1]
            if abs(r - pr) + abs(c - pc) > 1:
                print(f"  FAIL [Task5] illegal jump ({pr},{pc}) -> ({r},{c}) at t={t}")
                allOk = False
    if allOk:
        print("  OK [Task5] all non-teleport constraints satisfied.")


print("  Q1 Summary")
for taskLabel, result in [
    ("Task 2 - Base Case",    task2_T),
    ("Task 3 - Booby Trap",   task3_T),
    ("Task 4 - Exit Lock",    task4_T),
    ("Task 5 - Ghost",        task5_T),
]:
    resultStr = f"{result} steps" if result is not None else "INFEASIBLE"
    print(f"  {taskLabel:<26} : {resultStr}")


#  QUESTION 2 – CCI PRODUCTION PLANNING
print()
print("  QUESTION 2 – CCI PRODUCTION PLANNING")

MONTHS = list(range(1, 7))   # m = 1, ..., 6

demandWestern = {1: 9_000_000, 2: 9_000_000, 3: 14_000_000,
                 4: 12_000_000, 5: 12_000_000, 6: 13_000_000}
demandCentEast = {1: 7_000_000, 2: 7_000_000, 3: 8_000_000,
                  4: 9_000_000,  5: 8_000_000,  6: 9_000_000}

ANKARA_RATE_2L   = 6_000 * 2    # 12,000 L/hr
ANKARA_RATE_1L   = 10_000 * 1   # 10,000 L/hr
ISTANBUL_RATE_2L = 9_000 * 2    # 18,000 L/hr
ISTANBUL_RATE_1L = 15_000 * 1   # 15,000 L/hr
BOTTLING_HOURS_PER_MONTH = 720   # hours available 

PACKAGING_CAPACITY_PER_WORKER = 100_000   # liters per worker per month
INITIAL_WORKERS_ANKARA   = 60
INITIAL_WORKERS_ISTANBUL = 100

WAGE_PER_WORKER     = 1_200    # TL per worker per month
HIRING_COST         = 1_500    # TL per new hire
LAYOFF_COST         = 2_000    # TL per layoff
INVENTORY_COST_PER_LITER  = 0.05   # TL per liter per month
TRANSPORT_COST_PER_LITER  = 0.10   # TL per liter (cross-region only)
SWITCHING_COST      = 20_000   # TL per plant per month (both sizes in same month)

CROSS_SHIP_CAP_FRACTION = 0.25   # α: at most 25% of demand from non-primary plant
SAFETY_STOCK_FRACTION   = 0.10   # β: end-of-month inventory ≥ 10% of next month demand
MAX_WORKFORCE_CHANGE    = 10     # Δ: maximum change in workers between consecutive months
BIG_M_PRODUCTION        = 50_000_000   # liters (safe upper bound for linking binary vars)

cciModel = gp.Model("CCI_ProductionPlan")
cciModel.setParam("OutputFlag", 0)


prod1L_ankara   = {m: cciModel.addVar(lb=0, name=f"prod1L_ankara_m{m}")   for m in MONTHS}
prod2L_ankara   = {m: cciModel.addVar(lb=0, name=f"prod2L_ankara_m{m}")   for m in MONTHS}
prod1L_istanbul = {m: cciModel.addVar(lb=0, name=f"prod1L_istanbul_m{m}") for m in MONTHS}
prod2L_istanbul = {m: cciModel.addVar(lb=0, name=f"prod2L_istanbul_m{m}") for m in MONTHS}

supply_ankara_to_centEast   = {m: cciModel.addVar(lb=0, name=f"supply_ankara_centEast_m{m}")   for m in MONTHS}
supply_ankara_to_western    = {m: cciModel.addVar(lb=0, name=f"supply_ankara_western_m{m}")    for m in MONTHS}
supply_istanbul_to_western  = {m: cciModel.addVar(lb=0, name=f"supply_istanbul_western_m{m}")  for m in MONTHS}
supply_istanbul_to_centEast = {m: cciModel.addVar(lb=0, name=f"supply_istanbul_centEast_m{m}") for m in MONTHS}

inventory_ankara   = {m: cciModel.addVar(lb=0, name=f"inventory_ankara_m{m}")   for m in MONTHS}
inventory_istanbul = {m: cciModel.addVar(lb=0, name=f"inventory_istanbul_m{m}") for m in MONTHS}

workers_ankara   = {m: cciModel.addVar(lb=0, vtype=GRB.INTEGER, name=f"workers_ankara_m{m}")   for m in MONTHS}
workers_istanbul = {m: cciModel.addVar(lb=0, vtype=GRB.INTEGER, name=f"workers_istanbul_m{m}") for m in MONTHS}
hired_ankara     = {m: cciModel.addVar(lb=0, vtype=GRB.INTEGER, name=f"hired_ankara_m{m}")     for m in MONTHS}
hired_istanbul   = {m: cciModel.addVar(lb=0, vtype=GRB.INTEGER, name=f"hired_istanbul_m{m}")   for m in MONTHS}
laidOff_ankara   = {m: cciModel.addVar(lb=0, vtype=GRB.INTEGER, name=f"laidOff_ankara_m{m}")   for m in MONTHS}
laidOff_istanbul = {m: cciModel.addVar(lb=0, vtype=GRB.INTEGER, name=f"laidOff_istanbul_m{m}") for m in MONTHS}

active1L_ankara   = {m: cciModel.addVar(vtype=GRB.BINARY, name=f"active1L_ankara_m{m}")   for m in MONTHS}
active2L_ankara   = {m: cciModel.addVar(vtype=GRB.BINARY, name=f"active2L_ankara_m{m}")   for m in MONTHS}
active1L_istanbul = {m: cciModel.addVar(vtype=GRB.BINARY, name=f"active1L_istanbul_m{m}") for m in MONTHS}
active2L_istanbul = {m: cciModel.addVar(vtype=GRB.BINARY, name=f"active2L_istanbul_m{m}") for m in MONTHS}

switching_ankara   = {m: cciModel.addVar(vtype=GRB.BINARY, name=f"switching_ankara_m{m}")   for m in MONTHS}
switching_istanbul = {m: cciModel.addVar(vtype=GRB.BINARY, name=f"switching_istanbul_m{m}") for m in MONTHS}


for m in MONTHS:
    prevWorkers_ankara   = INITIAL_WORKERS_ANKARA   if m == 1 else workers_ankara[m - 1]
    prevWorkers_istanbul = INITIAL_WORKERS_ISTANBUL if m == 1 else workers_istanbul[m - 1]
    prevInventory_ankara   = 0 if m == 1 else inventory_ankara[m - 1]
    prevInventory_istanbul = 0 if m == 1 else inventory_istanbul[m - 1]

    totalProduction_ankara   = prod1L_ankara[m]   + prod2L_ankara[m]
    totalProduction_istanbul = prod1L_istanbul[m] + prod2L_istanbul[m]

    cciModel.addConstr(
        prod2L_ankara[m] / ANKARA_RATE_2L + prod1L_ankara[m] / ANKARA_RATE_1L
        <= BOTTLING_HOURS_PER_MONTH,
        name=f"bottlingCap_ankara_m{m}"
    )
    cciModel.addConstr(
        prod2L_istanbul[m] / ISTANBUL_RATE_2L + prod1L_istanbul[m] / ISTANBUL_RATE_1L
        <= BOTTLING_HOURS_PER_MONTH,
        name=f"bottlingCap_istanbul_m{m}"
    )

    cciModel.addConstr(
        totalProduction_ankara <= workers_ankara[m] * PACKAGING_CAPACITY_PER_WORKER,
        name=f"packagingCap_ankara_m{m}"
    )
    cciModel.addConstr(
        totalProduction_istanbul <= workers_istanbul[m] * PACKAGING_CAPACITY_PER_WORKER,
        name=f"packagingCap_istanbul_m{m}"
    )

    cciModel.addConstr(
        workers_ankara[m] == prevWorkers_ankara + hired_ankara[m] - laidOff_ankara[m],
        name=f"workforceBal_ankara_m{m}"
    )
    cciModel.addConstr(
        workers_istanbul[m] == prevWorkers_istanbul + hired_istanbul[m] - laidOff_istanbul[m],
        name=f"workforceBal_istanbul_m{m}"
    )

    cciModel.addConstr(
        workers_ankara[m] - prevWorkers_ankara <= MAX_WORKFORCE_CHANGE,
        name=f"workforceSmooth_ankara_up_m{m}"
    )
    cciModel.addConstr(
        prevWorkers_ankara - workers_ankara[m] <= MAX_WORKFORCE_CHANGE,
        name=f"workforceSmooth_ankara_dn_m{m}"
    )
    cciModel.addConstr(
        workers_istanbul[m] - prevWorkers_istanbul <= MAX_WORKFORCE_CHANGE,
        name=f"workforceSmooth_istanbul_up_m{m}"
    )
    cciModel.addConstr(
        prevWorkers_istanbul - workers_istanbul[m] <= MAX_WORKFORCE_CHANGE,
        name=f"workforceSmooth_istanbul_dn_m{m}"
    )

    cciModel.addConstr(
        inventory_ankara[m]
        == prevInventory_ankara + totalProduction_ankara
           - supply_ankara_to_centEast[m] - supply_ankara_to_western[m],
        name=f"inventoryBal_ankara_m{m}"
    )
    cciModel.addConstr(
        inventory_istanbul[m]
        == prevInventory_istanbul + totalProduction_istanbul
           - supply_istanbul_to_western[m] - supply_istanbul_to_centEast[m],
        name=f"inventoryBal_istanbul_m{m}"
    )

    cciModel.addConstr(
        supply_ankara_to_centEast[m] + supply_istanbul_to_centEast[m] >= demandCentEast[m],
        name=f"demandFulfill_centEast_m{m}"
    )
    cciModel.addConstr(
        supply_istanbul_to_western[m] + supply_ankara_to_western[m] >= demandWestern[m],
        name=f"demandFulfill_western_m{m}"
    )

    cciModel.addConstr(
        supply_ankara_to_western[m] <= CROSS_SHIP_CAP_FRACTION * demandWestern[m],
        name=f"crossShipCap_ankara_western_m{m}"
    )
    cciModel.addConstr(
        supply_istanbul_to_centEast[m] <= CROSS_SHIP_CAP_FRACTION * demandCentEast[m],
        name=f"crossShipCap_istanbul_centEast_m{m}"
    )

    cciModel.addConstr(prod1L_ankara[m]   <= BIG_M_PRODUCTION * active1L_ankara[m],   name=f"link1L_ankara_m{m}")
    cciModel.addConstr(prod2L_ankara[m]   <= BIG_M_PRODUCTION * active2L_ankara[m],   name=f"link2L_ankara_m{m}")
    cciModel.addConstr(prod1L_istanbul[m] <= BIG_M_PRODUCTION * active1L_istanbul[m], name=f"link1L_istanbul_m{m}")
    cciModel.addConstr(prod2L_istanbul[m] <= BIG_M_PRODUCTION * active2L_istanbul[m], name=f"link2L_istanbul_m{m}")

    cciModel.addConstr(
        switching_ankara[m] >= active1L_ankara[m] + active2L_ankara[m] - 1,
        name=f"switching_lb_ankara_m{m}"
    )
    cciModel.addConstr(
        switching_ankara[m] <= active1L_ankara[m],
        name=f"switching_ub1_ankara_m{m}"
    )
    cciModel.addConstr(
        switching_ankara[m] <= active2L_ankara[m],
        name=f"switching_ub2_ankara_m{m}"
    )
    cciModel.addConstr(
        switching_istanbul[m] >= active1L_istanbul[m] + active2L_istanbul[m] - 1,
        name=f"switching_lb_istanbul_m{m}"
    )
    cciModel.addConstr(
        switching_istanbul[m] <= active1L_istanbul[m],
        name=f"switching_ub1_istanbul_m{m}"
    )
    cciModel.addConstr(
        switching_istanbul[m] <= active2L_istanbul[m],
        name=f"switching_ub2_istanbul_m{m}"
    )

    if m == 6:
        cciModel.addConstr(inventory_ankara[m] == 0,   name="zeroInventory_ankara_end")
        cciModel.addConstr(inventory_istanbul[m] == 0, name="zeroInventory_istanbul_end")

for m in range(1, 6):
    cciModel.addConstr(
        inventory_istanbul[m] >= SAFETY_STOCK_FRACTION * demandWestern[m + 1],
        name=f"safetyStock_istanbul_m{m}"
    )
    cciModel.addConstr(
        inventory_ankara[m] >= SAFETY_STOCK_FRACTION * demandCentEast[m + 1],
        name=f"safetyStock_ankara_m{m}"
    )


inventoryHoldingCost = gp.quicksum(
    INVENTORY_COST_PER_LITER * (inventory_ankara[m] + inventory_istanbul[m])
    for m in MONTHS
)
crossRegionTransportCost = gp.quicksum(
    TRANSPORT_COST_PER_LITER * (supply_ankara_to_western[m] + supply_istanbul_to_centEast[m])
    for m in MONTHS
)
wageCost = gp.quicksum(
    WAGE_PER_WORKER * (workers_ankara[m] + workers_istanbul[m])
    for m in MONTHS
)
hiringCost = gp.quicksum(
    HIRING_COST * (hired_ankara[m] + hired_istanbul[m])
    for m in MONTHS
)
layoffCost = gp.quicksum(
    LAYOFF_COST * (laidOff_ankara[m] + laidOff_istanbul[m])
    for m in MONTHS
)
bottlingSwitchingCost = gp.quicksum(
    SWITCHING_COST * (switching_ankara[m] + switching_istanbul[m])
    for m in MONTHS
)

cciModel.setObjective(
    inventoryHoldingCost + crossRegionTransportCost + wageCost
    + hiringCost + layoffCost + bottlingSwitchingCost,
    GRB.MINIMIZE
)

cciModel.optimize()
#  Q2 – Results 

if cciModel.Status == GRB.OPTIMAL:
    print()
    print("  " + "=" *60)
    print("  CCI PRODUCTION PLANNING – OPTIMAL SOLUTION")
    print("  " + "=" * 60)

    def _v(expr):
        return sum(c * v.X for v, c in zip(expr.getVars(),
                                            [expr.getCoeff(i)
                                             for i in range(expr.size())]))

    totalCost        = cciModel.ObjVal
    holdingVal       = sum(INVENTORY_COST_PER_LITER *
                           (inventory_ankara[m].X + inventory_istanbul[m].X)
                           for m in MONTHS)
    transportVal     = sum(TRANSPORT_COST_PER_LITER *
                           (supply_ankara_to_western[m].X +
                            supply_istanbul_to_centEast[m].X)
                           for m in MONTHS)
    wageVal          = sum(WAGE_PER_WORKER *
                           (workers_ankara[m].X + workers_istanbul[m].X)
                           for m in MONTHS)
    hiringVal        = sum(HIRING_COST *
                           (hired_ankara[m].X + hired_istanbul[m].X)
                           for m in MONTHS)
    layoffVal        = sum(LAYOFF_COST *
                           (laidOff_ankara[m].X + laidOff_istanbul[m].X)
                           for m in MONTHS)
    switchingVal     = sum(SWITCHING_COST *
                           (switching_ankara[m].X + switching_istanbul[m].X)
                           for m in MONTHS)

    print()
    print(f"  Total Minimum Cost      : {totalCost:>18,.2f} TL")
    print(f"  ----------------------------------------")
    print(f"    Wage cost             : {wageVal:>18,.2f} TL")
    print(f"    Hiring cost           : {hiringVal:>18,.2f} TL")
    print(f"    Layoff cost           : {layoffVal:>18,.2f} TL")
    print(f"    Inventory holding     : {holdingVal:>18,.2f} TL")
    print(f"    Cross-region transport: {transportVal:>18,.2f} TL")
    print(f"    Bottling switch cost  : {switchingVal:>18,.2f} TL")

    print()
    print("  MONTHLY DETAIL")
    print("  " + "-" * 60)

    HDR = (f"  {'Month':>5}  "
           f"{'Plant':<9}  "
           f"{'Prod1L(L)':>13}  "
           f"{'Prod2L(L)':>13}  "
           f"{'Workers':>7}  "
           f"{'Hired':>5}  "
           f"{'LaidOff':>7}  "
           f"{'Inventory(L)':>13}  "
           f"{'XShip(L)':>12}")
    print(HDR)
    print("  " + "-" * 72)

    for m in MONTHS:
        p1a  = prod1L_ankara[m].X
        p2a  = prod2L_ankara[m].X
        wa   = round(workers_ankara[m].X)
        ha   = round(hired_ankara[m].X)
        la   = round(laidOff_ankara[m].X)
        ia   = inventory_ankara[m].X
        xsa  = supply_ankara_to_western[m].X

        print(f"  {m:>5}  "
              f"{'Ankara':<9}  "
              f"{p1a:>13,.0f}  "
              f"{p2a:>13,.0f}  "
              f"{wa:>7}  "
              f"{ha:>5}  "
              f"{la:>7}  "
              f"{ia:>13,.0f}  "
              f"{xsa:>12,.0f}")

        p1i  = prod1L_istanbul[m].X
        p2i  = prod2L_istanbul[m].X
        wi   = round(workers_istanbul[m].X)
        hi   = round(hired_istanbul[m].X)
        li   = round(laidOff_istanbul[m].X)
        ii   = inventory_istanbul[m].X
        xsi  = supply_istanbul_to_centEast[m].X

        print(f"  {'':>5}  "
              f"{'Istanbul':<9}  "
              f"{p1i:>13,.0f}  "
              f"{p2i:>13,.0f}  "
              f"{wi:>7}  "
              f"{hi:>5}  "
              f"{li:>7}  "
              f"{ii:>13,.0f}  "
              f"{xsi:>12,.0f}")

        wD   = demandWestern[m]
        ceD  = demandCentEast[m]
        wSup = supply_istanbul_to_western[m].X + supply_ankara_to_western[m].X
        ceSup= supply_ankara_to_centEast[m].X  + supply_istanbul_to_centEast[m].X
        print(f"  {'':>5}  "
              f"  Western demand {wD/1e6:.1f}M L -> supplied {wSup/1e6:.3f}M L  |  "
              f"CentEast demand {ceD/1e6:.1f}M L -> supplied {ceSup/1e6:.3f}M L")

        if m < len(MONTHS):
            print("  " + "  " + "-" * 68)

    print("  " + "=" * 72)
    print()

else:
    statusMap = {
        GRB.INFEASIBLE:      "INFEASIBLE",
        GRB.INF_OR_UNBD:     "INF_OR_UNBD",
        GRB.UNBOUNDED:       "UNBOUNDED",
        GRB.TIME_LIMIT:      "TIME_LIMIT (no solution found)",
    }
    statusStr = statusMap.get(cciModel.Status, f"Status code {cciModel.Status}")
    print(f"\n  [Q2] Model did not reach optimality: {statusStr}")
