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



T_MAX = 20


def solveMuseumIP(taskName, addExtraConstraints=None, relaxedTransitions=None, T_max=T_MAX):
    model = gp.Model(taskName)
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", 120)

    relaxed = set(relaxedTransitions) if relaxedTransitions else set()
    er, ec = EXIT_CELL

    occupancy = {
        (t, r, c): model.addVar(vtype=GRB.BINARY, name=f"occupy_t{t}_r{r}_c{c}")
        for t in range(T_max + 1)
        for (r, c) in VALID_CELLS
    }
    z = {
        t: model.addVar(vtype=GRB.BINARY, name=f"z_t{t}")
        for t in range(T_max + 1)
    }

    model.addConstr(occupancy[0, START_CELL[0], START_CELL[1]] == 1)
    for (r, c) in VALID_CELLS:
        if (r, c) != START_CELL:
            model.addConstr(occupancy[0, r, c] == 0)

    for t in range(T_max + 1):
        model.addConstr(
            gp.quicksum(occupancy[t, r, c] for (r, c) in VALID_CELLS) == 1
        )

    model.addConstr(gp.quicksum(z[t] for t in range(T_max + 1)) == 1)

    for t in range(T_max + 1):
        model.addConstr(occupancy[t, er, ec] >= z[t])
        if t > 0:
            model.addConstr(occupancy[t, er, ec] >= occupancy[t - 1, er, ec])
            model.addConstr(
                occupancy[t, er, ec] - occupancy[t - 1, er, ec] <= z[t]
            )

    for t in range(T_max):
        if t in relaxed:
            continue
        for (r, c) in VALID_CELLS:
            nbrs = getNeighbours(r, c)
            model.addConstr(
                occupancy[t + 1, r, c]
                <= gp.quicksum(occupancy[t, nr, nc] for (nr, nc) in nbrs)
            )

    for t in range(T_max + 1):
        for (r, c) in getCameraVisibleCells(t):
            if (r, c) in VALID_CELLS and (r, c) != EXIT_CELL:
                model.addConstr(occupancy[t, r, c] == 0)

    if addExtraConstraints:
        addExtraConstraints(model, occupancy, T_max)

    model.setObjective(
        gp.quicksum(t * z[t] for t in range(T_max + 1)), GRB.MINIMIZE
    )
    model.optimize()

    if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and model.SolCount > 0:
        tStar = next(
            (t for t in range(T_max + 1) if z[t].X > 0.5), None
        )
        if tStar is None:
            return None, None
        path = []
        for t in range(tStar + 1):
            for (r, c) in sorted(VALID_CELLS):
                if occupancy[t, r, c].X > 0.5:
                    path.append((t, r, c))
                    break
        return tStar, path
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
    if path is None:
        return
    allOk = True
    cellAt = {t: (r, c) for t, r, c in path}
    for t, r, c in path:
        if (r, c) in OBSTACLE_CELLS:
            print(f"  FAIL [{label}] t={t}: ({r},{c}) is an obstacle!")
            allOk = False
        if (r, c) != EXIT_CELL and (r, c) in getCameraVisibleCells(t):
            print(f"  FAIL [{label}] t={t}: ({r},{c}) is camera visible!")
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
task2_T, task2_path = solveMuseumIP("Task2_BaseCase")
printPath("Task 2 – Base Case", task2_T, task2_path)
validatePath("Task2", task2_path)

#  Task 2 – Branch-and-Bound
class BBNode:
    __slots__ = ("fixings", "depth", "label")

    def __init__(self, fixings, depth, label):
        self.fixings = fixings   
        self.depth   = depth
        self.label   = label


def _buildLPSubproblem(taskName, T_max, fixings):
    model = gp.Model(taskName)
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", 30)

    er, ec = EXIT_CELL

    occupancy = {
        (t, r, c): model.addVar(
            lb=0.0, ub=1.0,
            vtype=GRB.CONTINUOUS,
            name=f"occ_t{t}_r{r}_c{c}",
        )
        for t in range(T_max + 1)
        for (r, c) in VALID_CELLS
    }
    z = {
        t: model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"z_t{t}")
        for t in range(T_max + 1)
    }

    model.addConstr(occupancy[0, START_CELL[0], START_CELL[1]] == 1.0)
    for (r, c) in VALID_CELLS:
        if (r, c) != START_CELL:
            model.addConstr(occupancy[0, r, c] == 0.0)

    for t in range(T_max + 1):
        model.addConstr(
            gp.quicksum(occupancy[t, r, c] for (r, c) in VALID_CELLS) == 1.0
        )

    model.addConstr(gp.quicksum(z[t] for t in range(T_max + 1)) == 1.0)

    for t in range(T_max + 1):
        model.addConstr(occupancy[t, er, ec] >= z[t])
        if t > 0:
            model.addConstr(occupancy[t, er, ec] >= occupancy[t - 1, er, ec])
            model.addConstr(
                occupancy[t, er, ec] - occupancy[t - 1, er, ec] <= z[t]
            )

    for t in range(T_max):
        for (r, c) in VALID_CELLS:
            nbrs = getNeighbours(r, c)
            model.addConstr(
                occupancy[t + 1, r, c]
                <= gp.quicksum(occupancy[t, nr, nc] for (nr, nc) in nbrs)
            )

    for t in range(T_max + 1):
        for (r, c) in getCameraVisibleCells(t):
            if (r, c) in VALID_CELLS and (r, c) != EXIT_CELL:
                model.addConstr(occupancy[t, r, c] == 0.0)

    for (t, r, c), val in fixings.items():
        model.addConstr(occupancy[t, r, c] == float(val))

    model.setObjective(
        gp.quicksum(t * z[t] for t in range(T_max + 1)), GRB.MINIMIZE
    )
    return model, occupancy, z


def _isFractional(val, tol=1e-6):
    return tol < val < 1.0 - tol


def _extractPath(T_max, occupancy, z):
    tStar = next((t for t in range(T_max + 1) if z[t].X > 0.5), None)
    if tStar is None:
        return None
    path = []
    for t in range(tStar + 1):
        for (r, c) in sorted(VALID_CELLS):
            if occupancy[t, r, c].X > 0.5:
                path.append((t, r, c))
                break
    return path


def solveMuseumBranchAndBound(integerUB, incumbentPath, T_max=T_MAX):
    print(f"  [B&B] T_max = {T_max}, integer UB = {integerUB}")

    rootNode = BBNode(fixings={}, depth=0, label="Root")
    stack = [rootNode]
    nodeCount = 0
    tol = 1e-6

    while stack:
        node = stack.pop()
        nodeCount += 1
        indent = "    " + "  " * node.depth

        model, occupancy, z = _buildLPSubproblem(
            f"BB_node_{nodeCount}", T_max, node.fixings
        )
        model.optimize()

        lpFeasible = model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and model.SolCount > 0
        if not lpFeasible:
            print(f"{indent}[B&B] Node {node.label} (depth {node.depth}): "
                  f"LP INFEASIBLE - pruned.")
            continue

        lpLB = model.ObjVal
        print(f"{indent}[B&B] Node {node.label} (depth {node.depth}): "
              f"LP LB = {lpLB:.4f}, integer UB = {integerUB}.")

        if lpLB >= integerUB - tol:
            if node.depth == 0:
                print(f"{indent}[B&B] Root fathomed by bound: LP LB == UB == "
                      f"{integerUB}. Optimal. No branching required.")
            else:
                print(f"{indent}[B&B] Pruned by bound (LB >= UB).")
            if node.depth == 0:
                print(f"  [B&B] Total nodes explored: {nodeCount}")
                return integerUB, incumbentPath
            continue

        fractionalVar = None
        for t in range(T_max + 1):
            for (r, c) in sorted(VALID_CELLS):
                if _isFractional(occupancy[t, r, c].X):
                    fractionalVar = (t, r, c)
                    break
            if fractionalVar is not None:
                break

        if fractionalVar is None:
            path = _extractPath(T_max, occupancy, z)
            print(f"{indent}[B&B] Node {node.label}: integer solution at LP. "
                  f"Terminating.")
            print(f"  [B&B] Total nodes explored: {nodeCount}")
            return int(round(lpLB)), path

        ft, fr, fc = fractionalVar
        fracVal = occupancy[ft, fr, fc].X
        print(f"{indent}[B&B] Branching on fractional occ[t={ft},r={fr},c={fc}]"
              f"={fracVal:.4f}.")

        fixings0 = dict(node.fixings); fixings0[fractionalVar] = 0
        fixings1 = dict(node.fixings); fixings1[fractionalVar] = 1
        stack.append(BBNode(fixings1, node.depth + 1,
                            f"{node.label}->occ[{ft},{fr},{fc}]=1"))
        stack.append(BBNode(fixings0, node.depth + 1,
                            f"{node.label}->occ[{ft},{fr},{fc}]=0"))

    print(f"  [B&B] Stack exhausted. INFEASIBLE.")
    print(f"  [B&B] Total nodes explored: {nodeCount}")
    return None, None

bb2_T, bb2_path = solveMuseumBranchAndBound(baseOptimalT, task2_path)
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

task3_T, task3_path = solveMuseumIP("Task3_BoobyTrap", addBoobyTrapConstraints)
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


task4_T, task4_path = solveMuseumIP("Task4_ExitLock", addExitLockConstraints)
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

print(f"\n  [Task 5] Ghost Checkpoint at {GHOST_CELL} at t={GHOST_TIME}.")

task5_T, task5_path = solveMuseumIP(
    "Task5_GhostCheckpoint",
    addGhostCheckpointConstraints,
)

printPath("Task 5 - Ghost Checkpoint [(4,4) at t=3]", task5_T, task5_path)
validatePath("Task5", task5_path)

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

MONTHS = list(range(1, 7))  

demandWestern = {1: 9_000_000, 2: 9_000_000, 3: 14_000_000,
                 4: 12_000_000, 5: 12_000_000, 6: 13_000_000}
demandCentEast = {1: 7_000_000, 2: 7_000_000, 3: 8_000_000,
                  4: 9_000_000,  5: 8_000_000,  6: 9_000_000}

ANKARA_RATE_2L   = 6_000 * 2  
ANKARA_RATE_1L   = 10_000 * 1   
ISTANBUL_RATE_2L = 9_000 * 2   
ISTANBUL_RATE_1L = 15_000 * 1   
BOTTLING_HOURS_PER_MONTH = 720  

PACKAGING_CAPACITY_PER_WORKER = 100_000   
INITIAL_WORKERS_ANKARA   = 60
INITIAL_WORKERS_ISTANBUL = 100

WAGE_PER_WORKER          = 1_200    
HIRING_COST              = 1_500    
LAYOFF_COST              = 2_000    
INVENTORY_COST_PER_LITER = 0.05     
TRANSPORT_COST_PER_LITER = 0.10     
SWITCHING_COST           = 20_000   

CROSS_SHIP_CAP_FRACTION = 0.25   
SAFETY_STOCK_FRACTION   = 0.10   
MAX_WORKFORCE_CHANGE    = 10     
BIG_M_PRODUCTION        = 50_000_000   

DEMAND_FRAC_2L = 0.60  
DEMAND_FRAC_1L = 0.40   

cciModel = gp.Model("CCI_ProductionPlan")
cciModel.setParam("OutputFlag", 0)

prod1L_ankara   = {m: cciModel.addVar(lb=0, name=f"prod1L_ankara_m{m}")   for m in MONTHS}
prod2L_ankara   = {m: cciModel.addVar(lb=0, name=f"prod2L_ankara_m{m}")   for m in MONTHS}
prod1L_istanbul = {m: cciModel.addVar(lb=0, name=f"prod1L_istanbul_m{m}") for m in MONTHS}
prod2L_istanbul = {m: cciModel.addVar(lb=0, name=f"prod2L_istanbul_m{m}") for m in MONTHS}

supply_1L_ankara_to_centEast   = {m: cciModel.addVar(lb=0, name=f"supply_1L_ankara_centEast_m{m}")   for m in MONTHS}
supply_2L_ankara_to_centEast   = {m: cciModel.addVar(lb=0, name=f"supply_2L_ankara_centEast_m{m}")   for m in MONTHS}
supply_1L_ankara_to_western    = {m: cciModel.addVar(lb=0, name=f"supply_1L_ankara_western_m{m}")    for m in MONTHS}
supply_2L_ankara_to_western    = {m: cciModel.addVar(lb=0, name=f"supply_2L_ankara_western_m{m}")    for m in MONTHS}
supply_1L_istanbul_to_western  = {m: cciModel.addVar(lb=0, name=f"supply_1L_istanbul_western_m{m}")  for m in MONTHS}
supply_2L_istanbul_to_western  = {m: cciModel.addVar(lb=0, name=f"supply_2L_istanbul_western_m{m}")  for m in MONTHS}
supply_1L_istanbul_to_centEast = {m: cciModel.addVar(lb=0, name=f"supply_1L_istanbul_centEast_m{m}") for m in MONTHS}
supply_2L_istanbul_to_centEast = {m: cciModel.addVar(lb=0, name=f"supply_2L_istanbul_centEast_m{m}") for m in MONTHS}

inventory_1L_ankara   = {m: cciModel.addVar(lb=0, name=f"inventory_1L_ankara_m{m}")   for m in MONTHS}
inventory_2L_ankara   = {m: cciModel.addVar(lb=0, name=f"inventory_2L_ankara_m{m}")   for m in MONTHS}
inventory_1L_istanbul = {m: cciModel.addVar(lb=0, name=f"inventory_1L_istanbul_m{m}") for m in MONTHS}
inventory_2L_istanbul = {m: cciModel.addVar(lb=0, name=f"inventory_2L_istanbul_m{m}") for m in MONTHS}

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
    prevInventory_1L_ankara   = 0 if m == 1 else inventory_1L_ankara[m - 1]
    prevInventory_2L_ankara   = 0 if m == 1 else inventory_2L_ankara[m - 1]
    prevInventory_1L_istanbul = 0 if m == 1 else inventory_1L_istanbul[m - 1]
    prevInventory_2L_istanbul = 0 if m == 1 else inventory_2L_istanbul[m - 1]

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
        inventory_1L_ankara[m]
        == prevInventory_1L_ankara + prod1L_ankara[m]
           - supply_1L_ankara_to_centEast[m] - supply_1L_ankara_to_western[m],
        name=f"inventoryBal_1L_ankara_m{m}"
    )
    cciModel.addConstr(
        inventory_2L_ankara[m]
        == prevInventory_2L_ankara + prod2L_ankara[m]
           - supply_2L_ankara_to_centEast[m] - supply_2L_ankara_to_western[m],
        name=f"inventoryBal_2L_ankara_m{m}"
    )
    cciModel.addConstr(
        inventory_1L_istanbul[m]
        == prevInventory_1L_istanbul + prod1L_istanbul[m]
           - supply_1L_istanbul_to_western[m] - supply_1L_istanbul_to_centEast[m],
        name=f"inventoryBal_1L_istanbul_m{m}"
    )
    cciModel.addConstr(
        inventory_2L_istanbul[m]
        == prevInventory_2L_istanbul + prod2L_istanbul[m]
           - supply_2L_istanbul_to_western[m] - supply_2L_istanbul_to_centEast[m],
        name=f"inventoryBal_2L_istanbul_m{m}"
    )

    cciModel.addConstr(
        inventory_1L_ankara[m] <= prod1L_ankara[m],
        name=f"perishability_1L_ankara_m{m}"
    )
    cciModel.addConstr(
        inventory_2L_ankara[m] <= prod2L_ankara[m],
        name=f"perishability_2L_ankara_m{m}"
    )
    cciModel.addConstr(
        inventory_1L_istanbul[m] <= prod1L_istanbul[m],
        name=f"perishability_1L_istanbul_m{m}"
    )
    cciModel.addConstr(
        inventory_2L_istanbul[m] <= prod2L_istanbul[m],
        name=f"perishability_2L_istanbul_m{m}"
    )

    cciModel.addConstr(
        supply_2L_ankara_to_centEast[m] + supply_2L_istanbul_to_centEast[m]
        == DEMAND_FRAC_2L * demandCentEast[m],
        name=f"demandFulfill_2L_centEast_m{m}"
    )
    cciModel.addConstr(
        supply_1L_ankara_to_centEast[m] + supply_1L_istanbul_to_centEast[m]
        == DEMAND_FRAC_1L * demandCentEast[m],
        name=f"demandFulfill_1L_centEast_m{m}"
    )
    cciModel.addConstr(
        supply_2L_istanbul_to_western[m] + supply_2L_ankara_to_western[m]
        == DEMAND_FRAC_2L * demandWestern[m],
        name=f"demandFulfill_2L_western_m{m}"
    )
    cciModel.addConstr(
        supply_1L_istanbul_to_western[m] + supply_1L_ankara_to_western[m]
        == DEMAND_FRAC_1L * demandWestern[m],
        name=f"demandFulfill_1L_western_m{m}"
    )

    cciModel.addConstr(
        supply_1L_ankara_to_western[m] + supply_2L_ankara_to_western[m]
        <= CROSS_SHIP_CAP_FRACTION * demandWestern[m],
        name=f"crossShipCap_ankara_western_m{m}"
    )
    cciModel.addConstr(
        supply_1L_istanbul_to_centEast[m] + supply_2L_istanbul_to_centEast[m]
        <= CROSS_SHIP_CAP_FRACTION * demandCentEast[m],
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
        cciModel.addConstr(inventory_1L_ankara[m] == 0,   name="zeroInventory_1L_ankara_end")
        cciModel.addConstr(inventory_2L_ankara[m] == 0,   name="zeroInventory_2L_ankara_end")
        cciModel.addConstr(inventory_1L_istanbul[m] == 0, name="zeroInventory_1L_istanbul_end")
        cciModel.addConstr(inventory_2L_istanbul[m] == 0, name="zeroInventory_2L_istanbul_end")

for m in range(1, 6):
    cciModel.addConstr(
        inventory_1L_istanbul[m] + inventory_2L_istanbul[m]
        >= SAFETY_STOCK_FRACTION * demandWestern[m + 1],
        name=f"safetyStock_istanbul_m{m}"
    )
    cciModel.addConstr(
        inventory_1L_ankara[m] + inventory_2L_ankara[m]
        >= SAFETY_STOCK_FRACTION * demandCentEast[m + 1],
        name=f"safetyStock_ankara_m{m}"
    )


inventoryHoldingCost = gp.quicksum(
    INVENTORY_COST_PER_LITER * (
        inventory_1L_ankara[m] + inventory_2L_ankara[m]
        + inventory_1L_istanbul[m] + inventory_2L_istanbul[m]
    )
    for m in MONTHS
)
crossRegionTransportCost = gp.quicksum(
    TRANSPORT_COST_PER_LITER * (
        supply_1L_ankara_to_western[m] + supply_2L_ankara_to_western[m]
        + supply_1L_istanbul_to_centEast[m] + supply_2L_istanbul_to_centEast[m]
    )
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
