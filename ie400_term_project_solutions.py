"""
IE 400 Term Project - Spring 2026
Bilkent University, Department of Industrial Engineering

Question 1: Museum Escape - IP solved with Gurobi
Question 2: CCI Production Plan - LP/MILP solved with Gurobi

Strategy for Q1 (to stay within Gurobi free-license limits):
  We first compute T* (optimal steps) via time-expanded BFS that respects
  camera visibility. We then build a fixed-horizon IP at that T* to
  confirm/obtain the solution and enforce task-specific constraints.

Run: python ie400_term_project_solutions.py
"""

import gurobipy as gp
from gurobipy import GRB
from collections import deque

# ============================================================
# QUESTION 1 - MUSEUM ESCAPE
# ============================================================

ROWS, COLS = 8, 8
START = (1, 1)
EXIT = (8, 8)

OBSTACLES = {
    (1, 8),
    (2, 3), (2, 7),
    (3, 6),
    (4, 1), (4, 2), (4, 8),
    (5, 6),
    (6, 4),
    (7, 3), (7, 4), (7, 5),
}

VALID = frozenset(
    (r, c) for r in range(1, ROWS + 1)
    for c in range(1, COLS + 1)
    if (r, c) not in OBSTACLES
)

# Camera visibility lookup tables (from given movement tables)
_C1 = {
    0: frozenset({(3, 3), (3, 4), (3, 5)}),
    1: frozenset({(3, 4), (3, 5), (3, 6)}),
    2: frozenset({(3, 5), (3, 4), (3, 3)}),
    3: frozenset({(3, 4), (3, 3), (3, 2)}),
}
_C2 = {
    0: frozenset({(4, 6), (5, 6)}),
    1: frozenset({(5, 6), (6, 6)}),
    2: frozenset({(6, 6), (6, 7)}),
    3: frozenset({(6, 7), (6, 6)}),
    4: frozenset({(6, 6), (5, 6)}),
    5: frozenset({(5, 6), (4, 6)}),
}
_C3 = {
    0: frozenset({(7, 7), (6, 8)}),
    1: frozenset({(6, 8), (7, 7), (8, 6)}),
    2: frozenset({(7, 7), (8, 6)}),
    3: frozenset({(8, 6), (7, 7), (6, 8)}),
}


def cam_vis(t):
    return _C1[t % 4] | _C2[t % 6] | _C3[t % 4]


_MOVES = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]


def nbrs(r, c):
    return [
        (r + dr, c + dc)
        for dr, dc in _MOVES
        if 1 <= r + dr <= ROWS
        and 1 <= c + dc <= COLS
        and (r + dr, c + dc) not in OBSTACLES
    ]


# Time-expanded BFS
def bfs_shortest(extra_block=None, forced_at=None):
    """
    BFS on the time-expanded graph.
    extra_block(t, r, c) -> bool: additional blocking predicate.
    forced_at: dict {t: (r,c)} - Lara must be at that cell at that time.
    Returns (min_T, path) or (None, None).
    """
    if extra_block is None:
        extra_block = lambda t, r, c: False

    parent = {}
    visited = {(START[0], START[1], 0)}
    q = deque([(START[0], START[1], 0)])

    while q:
        r, c, t = q.popleft()
        if (r, c) == EXIT:
            path, state = [], (r, c, t)
            while state is not None:
                rr, cc, tt = state
                path.append((tt, rr, cc))
                state = parent.get(state)
            path.reverse()
            return t, path

        for nr, nc in nbrs(r, c):
            nt = t + 1
            if nt > 60:
                continue
            if (nr, nc) != EXIT and (nr, nc) in cam_vis(nt):
                continue
            if extra_block(nt, nr, nc):
                continue
            if forced_at and nt in forced_at and (nr, nc) != forced_at[nt]:
                continue
            key = (nr, nc, nt)
            if key not in visited:
                visited.add(key)
                parent[key] = (r, c, t)
                q.append((nr, nc, nt))

    return None, None


# IP solver (fixed horizon)
def solve_ip(task_name, T, extra_fn=None):
    """
    Build and solve the museum IP with fixed horizon T.
    Returns (T, path) or (None, None).

    Decision variables: x[t,r,c] in {0,1}
    Objective: feasibility check at horizon T
    """
    m = gp.Model(task_name)
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", 120)

    x = {
        (t, r, c): m.addVar(vtype=GRB.BINARY, name=f"x{t}_{r}{c}")
        for t in range(T + 1)
        for (r, c) in VALID
    }

    # Start
    m.addConstr(x[0, START[0], START[1]] == 1)
    for (r, c) in VALID:
        if (r, c) != START:
            m.addConstr(x[0, r, c] == 0)

    # One cell per step
    for t in range(T + 1):
        m.addConstr(gp.quicksum(x[t, r, c] for (r, c) in VALID) == 1)

    # Movement
    for t in range(T):
        for (r, c) in VALID:
            m.addConstr(
                x[t + 1, r, c] <= gp.quicksum(x[t, nr, nc] for (nr, nc) in nbrs(r, c))
            )

    # Camera blocking
    for t in range(T + 1):
        for (r, c) in cam_vis(t):
            if (r, c) in VALID and (r, c) != EXIT:
                m.addConstr(x[t, r, c] == 0)

    # Must reach exit by T
    m.addConstr(x[T, EXIT[0], EXIT[1]] == 1)

    if extra_fn:
        extra_fn(m, x, T)

    m.setObjective(0, GRB.MINIMIZE)
    m.optimize()

    if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:
        path = []
        for t in range(T + 1):
            for (r, c) in VALID:
                if x[t, r, c].X > 0.5:
                    path.append((t, r, c))
                    break
        return T, path
    return None, None


def print_result(label, T, path):
    sep = "=" * 60
    print(f"\n{sep}\n  {label}\n{sep}")
    if T is None:
        print("  *** INFEASIBLE - no solution found ***")
        return
    print(f"  Optimal time steps : {T}")
    print(f"  {'t':>3}   cell")
    print(f"  {'-' * 20}")
    for t, r, c in path:
        tag = ""
        if (r, c) == START and t == 0:
            tag = "  <- START"
        if (r, c) == EXIT:
            tag = "  <- EXIT OK"
        print(f"  {t:>3}   ({r},{c}){tag}")


def validate(label, path):
    if path is None:
        return
    ok = True
    cells = {t: (r, c) for t, r, c in path}
    for t, r, c in path:
        if (r, c) in OBSTACLES:
            print(f"  X [{label}] t={t}: ({r},{c}) is obstacle!")
            ok = False
        if (r, c) != EXIT and (r, c) in cam_vis(t):
            print(f"  X [{label}] t={t}: ({r},{c}) camera-visible!")
            ok = False
        if t > 0:
            pr, pc = cells[t - 1]
            if abs(r - pr) + abs(c - pc) > 1:
                print(f"  X [{label}] illegal jump ({pr},{pc})->({r},{c})")
                ok = False
    if cells[0] != START:
        print(f"  X [{label}] wrong start")
        ok = False
    if cells[max(cells)] != EXIT:
        print(f"  X [{label}] wrong end")
        ok = False
    if ok:
        print(f"  OK [{label}] all constraints satisfied.")


print("\n" + "=" * 60)
print("  QUESTION 1 - MUSEUM ESCAPE")
print("=" * 60)

# Task 2: Base Case
T2, _ = bfs_shortest()
print(f"\n  [BFS] Base optimal T = {T2}")
opt_T2, ip_path2 = solve_ip("Task2", T2)
print_result("Task 2 - Base Case", opt_T2, ip_path2)
validate("Task2", ip_path2)

# Task 3: Booby Trap
TRAP = (5, 1)
FORB3 = (8, 3)


def extra3(m, x, T):
    if TRAP not in VALID:
        print("  NOTE: (5,1) is obstacle - trap inactive.")
        return
    if FORB3 not in VALID:
        print("  NOTE: (8,3) is obstacle - already unreachable.")
        return
    y = {t: m.addVar(vtype=GRB.BINARY, name=f"Y3_{t}") for t in range(T + 1)}
    for t in range(T + 1):
        m.addConstr(y[t] >= x[t, TRAP[0], TRAP[1]])
        if t > 0:
            m.addConstr(y[t] >= y[t - 1])
    for t in range(1, T + 1):
        m.addConstr(x[t, FORB3[0], FORB3[1]] <= 1 - y[t - 1])


T3 = T2
opt_T3, ip_path3 = solve_ip("Task3", T3, extra3)
if opt_T3 is None:
    for d in range(1, 10):
        opt_T3, ip_path3 = solve_ip("Task3", T3 + d, extra3)
        if opt_T3 is not None:
            break
print_result("Task 3 - Booby Trap [(5,1) -> never (8,3)]", opt_T3, ip_path3)
validate("Task3", ip_path3)

# Task 4: Exit Lock Until t >= 18
LOCKED4 = {(6, 7), (7, 6), (7, 7), (8, 6), (6, 8)}


def bfs_block4(t, r, c):
    return (r, c) in LOCKED4 and t < 18


def extra4(m, x, T):
    for t in range(min(18, T + 1)):
        for (r, c) in LOCKED4:
            if (r, c) in VALID:
                m.addConstr(x[t, r, c] == 0)


T4, _ = bfs_shortest(extra_block=bfs_block4)
print(f"\n  [BFS] Task 4 optimal T = {T4}")
if T4 is None:
    print("  BFS: no path with exit lock.")
    opt_T4, ip_path4 = None, None
else:
    opt_T4, ip_path4 = solve_ip("Task4", T4, extra4)
print_result("Task 4 - Exit Lock (forbidden before t=18)", opt_T4, ip_path4)
validate("Task4", ip_path4)

# Task 5: Ghost Checkpoint (4,4) at t=3
GHOST = (4, 4)
GHOST_T = 3


def extra5(m, x, T):
    if GHOST not in VALID:
        print("  WARNING: (4,4) is obstacle - infeasible.")
        return
    if T < GHOST_T:
        print(f"  WARNING: T={T} < {GHOST_T}.")
        return
    if GHOST in cam_vis(GHOST_T):
        print(f"  WARNING: camera sees {GHOST} at t={GHOST_T} - likely infeasible.")
    m.addConstr(x[GHOST_T, GHOST[0], GHOST[1]] == 1)


T5, _ = bfs_shortest(forced_at={GHOST_T: GHOST})
print(f"\n  [BFS] Task 5 optimal T = {T5}")
if T5 is None:
    print("  BFS: no path through ghost checkpoint.")
    opt_T5, ip_path5 = None, None
else:
    opt_T5, ip_path5 = solve_ip("Task5", T5, extra5)
    if opt_T5 is None:
        for d in range(1, 15):
            opt_T5, ip_path5 = solve_ip("Task5", T5 + d, extra5)
            if opt_T5 is not None:
                break
print_result("Task 5 - Ghost Checkpoint [(4,4) at t=3]", opt_T5, ip_path5)
validate("Task5", ip_path5)

print(f"\n{'-' * 60}")
print("  Q1 Summary")
print(f"{'-' * 60}")
for lbl, T in [
    ("Task 2 Base", opt_T2),
    ("Task 3 Booby Trap", opt_T3),
    ("Task 4 Exit Lock", opt_T4),
    ("Task 5 Ghost", opt_T5),
]:
    status = f"{T} steps" if T is not None else "INFEASIBLE"
    print(f"  {lbl:<24} : {status}")


# ============================================================
# QUESTION 2 - CCI PRODUCTION PLANNING
# ============================================================
print("\n\n" + "=" * 60)
print("  QUESTION 2 - CCI PRODUCTION PLANNING")
print("=" * 60)

MONTHS = list(range(1, 7))

# Demand (million liters)
DEM_W = {1: 9, 2: 9, 3: 14, 4: 12, 5: 12, 6: 13}
DEM_CE = {1: 7, 2: 7, 3: 8, 4: 9, 5: 8, 6: 9}

# Bottling rates -> liters/hour
BOT_HRS = 720
R_A_2L = 6_000 * 2
R_A_1L = 10_000 * 1
R_I_2L = 9_000 * 2
R_I_1L = 15_000 * 1

PKG_PER = 0.1
INIT_WA = 60
INIT_WI = 100

WAGE_TL = 1_200
HIRE_TL = 1_500
LAY_TL = 2_000
INV_ML = 0.05 * 1e6
TRANS_ML = 0.10 * 1e6
CSW_TL = 20_000

ALPHA = 0.25
BETA = 0.10
DELTA = 10

q2 = gp.Model("CCI")
q2.setParam("OutputFlag", 0)

# Variables (ML)
pA1 = {m: q2.addVar(lb=0, name=f"pA1_{m}") for m in MONTHS}
pA2 = {m: q2.addVar(lb=0, name=f"pA2_{m}") for m in MONTHS}
pI1 = {m: q2.addVar(lb=0, name=f"pI1_{m}") for m in MONTHS}
pI2 = {m: q2.addVar(lb=0, name=f"pI2_{m}") for m in MONTHS}

# Supply routing
sACE = {m: q2.addVar(lb=0, name=f"sACE_{m}") for m in MONTHS}
sAW = {m: q2.addVar(lb=0, name=f"sAW_{m}") for m in MONTHS}
sIW = {m: q2.addVar(lb=0, name=f"sIW_{m}") for m in MONTHS}
sICE = {m: q2.addVar(lb=0, name=f"sICE_{m}") for m in MONTHS}

# Inventory (ML, end of month)
invA = {m: q2.addVar(lb=0, name=f"invA_{m}") for m in MONTHS}
invI = {m: q2.addVar(lb=0, name=f"invI_{m}") for m in MONTHS}

# Workforce
wA = {m: q2.addVar(lb=0, vtype=GRB.INTEGER, name=f"wA_{m}") for m in MONTHS}
wI = {m: q2.addVar(lb=0, vtype=GRB.INTEGER, name=f"wI_{m}") for m in MONTHS}
hA = {m: q2.addVar(lb=0, vtype=GRB.INTEGER, name=f"hA_{m}") for m in MONTHS}
hI = {m: q2.addVar(lb=0, vtype=GRB.INTEGER, name=f"hI_{m}") for m in MONTHS}
lA = {m: q2.addVar(lb=0, vtype=GRB.INTEGER, name=f"lA_{m}") for m in MONTHS}
lI = {m: q2.addVar(lb=0, vtype=GRB.INTEGER, name=f"lI_{m}") for m in MONTHS}

# MILP binaries for switching cost
bA1 = {m: q2.addVar(vtype=GRB.BINARY, name=f"bA1_{m}") for m in MONTHS}
bA2 = {m: q2.addVar(vtype=GRB.BINARY, name=f"bA2_{m}") for m in MONTHS}
bI1 = {m: q2.addVar(vtype=GRB.BINARY, name=f"bI1_{m}") for m in MONTHS}
bI2 = {m: q2.addVar(vtype=GRB.BINARY, name=f"bI2_{m}") for m in MONTHS}
zA = {m: q2.addVar(vtype=GRB.BINARY, name=f"zA_{m}") for m in MONTHS}
zI = {m: q2.addVar(vtype=GRB.BINARY, name=f"zI_{m}") for m in MONTHS}

BM = 50.0

for m in MONTHS:
    pwA = INIT_WA if m == 1 else wA[m - 1]
    pwI = INIT_WI if m == 1 else wI[m - 1]
    piA = 0 if m == 1 else invA[m - 1]
    piI = 0 if m == 1 else invI[m - 1]
    tA = pA1[m] + pA2[m]
    tI = pI1[m] + pI2[m]

    q2.addConstr(pA2[m] * 1e6 / R_A_2L + pA1[m] * 1e6 / R_A_1L <= BOT_HRS, f"botA_{m}")
    q2.addConstr(pI2[m] * 1e6 / R_I_2L + pI1[m] * 1e6 / R_I_1L <= BOT_HRS, f"botI_{m}")

    q2.addConstr(tA <= wA[m] * PKG_PER, f"pkgA_{m}")
    q2.addConstr(tI <= wI[m] * PKG_PER, f"pkgI_{m}")

    q2.addConstr(wA[m] == pwA + hA[m] - lA[m], f"wfA_{m}")
    q2.addConstr(wI[m] == pwI + hI[m] - lI[m], f"wfI_{m}")

    q2.addConstr(wA[m] - pwA <= DELTA, f"smAu_{m}")
    q2.addConstr(pwA - wA[m] <= DELTA, f"smAd_{m}")
    q2.addConstr(wI[m] - pwI <= DELTA, f"smIu_{m}")
    q2.addConstr(pwI - wI[m] <= DELTA, f"smId_{m}")

    q2.addConstr(invA[m] == piA + tA - sACE[m] - sAW[m], f"invA_{m}")
    q2.addConstr(invI[m] == piI + tI - sIW[m] - sICE[m], f"invI_{m}")

    q2.addConstr(sACE[m] + sICE[m] >= DEM_CE[m], f"demCE_{m}")
    q2.addConstr(sIW[m] + sAW[m] >= DEM_W[m], f"demW_{m}")

    q2.addConstr(sAW[m] <= ALPHA * DEM_W[m], f"capAW_{m}")
    q2.addConstr(sICE[m] <= ALPHA * DEM_CE[m], f"capICE_{m}")

    q2.addConstr(pA1[m] <= BM * bA1[m], f"lnkA1_{m}")
    q2.addConstr(pA2[m] <= BM * bA2[m], f"lnkA2_{m}")
    q2.addConstr(pI1[m] <= BM * bI1[m], f"lnkI1_{m}")
    q2.addConstr(pI2[m] <= BM * bI2[m], f"lnkI2_{m}")
    q2.addConstr(zA[m] >= bA1[m] + bA2[m] - 1, f"swA_{m}")
    q2.addConstr(zI[m] >= bI1[m] + bI2[m] - 1, f"swI_{m}")

    if m == 6:
        q2.addConstr(invA[m] == 0, "noInvA6")
        q2.addConstr(invI[m] == 0, "noInvI6")

for m in range(1, 6):
    q2.addConstr(invI[m] >= BETA * DEM_W[m + 1], f"ssI_{m}")
    q2.addConstr(invA[m] >= BETA * DEM_CE[m + 1], f"ssA_{m}")

inv_cost = gp.quicksum(INV_ML * (invA[m] + invI[m]) for m in MONTHS)
trans_cost = gp.quicksum(TRANS_ML * (sAW[m] + sICE[m]) for m in MONTHS)
wage_cost = gp.quicksum(WAGE_TL * (wA[m] + wI[m]) for m in MONTHS)
hire_cost = gp.quicksum(HIRE_TL * (hA[m] + hI[m]) for m in MONTHS)
lay_cost = gp.quicksum(LAY_TL * (lA[m] + lI[m]) for m in MONTHS)
sw_cost = gp.quicksum(CSW_TL * (zA[m] + zI[m]) for m in MONTHS)

q2.setObjective(inv_cost + trans_cost + wage_cost + hire_cost + lay_cost + sw_cost, GRB.MINIMIZE)
q2.optimize()

print(f"\n{'-' * 60}")
if q2.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and q2.SolCount > 0:
    print(f"  Optimal Total Cost : {q2.ObjVal:>15,.0f} TL\n")

    print(
        f"  {'Mo':<4} {'wA':>5} {'hA':>4} {'lA':>4} "
        f"{'wI':>5} {'hI':>4} {'lI':>4} "
        f"{'invA':>7} {'invI':>7} {'sAW':>7} {'sICE':>7}"
    )
    print(f"  {'-' * 65}")
    for m in MONTHS:
        print(
            f"  {m:<4} {wA[m].X:>5.0f} {max(0, hA[m].X):>4.0f} {max(0, lA[m].X):>4.0f} "
            f"{wI[m].X:>5.0f} {max(0, hI[m].X):>4.0f} {max(0, lI[m].X):>4.0f} "
            f"{invA[m].X:>7.3f} {invI[m].X:>7.3f} "
            f"{sAW[m].X:>7.3f} {sICE[m].X:>7.3f}"
        )

    print("\n  Production (ML)")
    print(f"  {'Mo':<4} {'pA1L':>8} {'pA2L':>8} {'pI1L':>8} {'pI2L':>8} {'SwA':>4} {'SwI':>4}")
    print(f"  {'-' * 50}")
    for m in MONTHS:
        print(
            f"  {m:<4} {pA1[m].X:>8.4f} {pA2[m].X:>8.4f} "
            f"{pI1[m].X:>8.4f} {pI2[m].X:>8.4f} "
            f"{int(round(zA[m].X)):>4} {int(round(zI[m].X)):>4}"
        )

    print("\n  Cost breakdown (TL)")
    for lbl, e in [
        ("Inventory holding", inv_cost),
        ("Transportation", trans_cost),
        ("Wages", wage_cost),
        ("Hiring", hire_cost),
        ("Layoffs", lay_cost),
        ("Switching (MILP)", sw_cost),
    ]:
        print(f"    {lbl:<22} : {e.getValue():>12,.0f}")
    print(f"    {'-' * 38}")
    print(f"    {'TOTAL':<22} : {q2.ObjVal:>12,.0f}")

    print("\n  [Validation - Q2]")
    ok2 = True
    for m in MONTHS:
        if sACE[m].X + sICE[m].X < DEM_CE[m] - 1e-4:
            print(f"  X M{m}: CE demand not met")
            ok2 = False
        if sIW[m].X + sAW[m].X < DEM_W[m] - 1e-4:
            print(f"  X M{m}: W demand not met")
            ok2 = False
        if sAW[m].X > ALPHA * DEM_W[m] + 1e-4:
            print(f"  X M{m}: AW cross-cap exceeded")
            ok2 = False
        if sICE[m].X > ALPHA * DEM_CE[m] + 1e-4:
            print(f"  X M{m}: ICE cross-cap exceeded")
            ok2 = False
        if invA[m].X < -1e-4 or invI[m].X < -1e-4:
            print(f"  X M{m}: negative inventory")
            ok2 = False
    for m in range(1, 6):
        if invI[m].X < BETA * DEM_W[m + 1] - 1e-4:
            print(f"  X M{m}: Istanbul safety stock violated")
            ok2 = False
        if invA[m].X < BETA * DEM_CE[m + 1] - 1e-4:
            print(f"  X M{m}: Ankara safety stock violated")
            ok2 = False
    if ok2:
        print("  OK All Q2 constraints satisfied.")
else:
    print(f"  Q2 model status: {q2.Status} - infeasible or time limit.")

print("\n" + "=" * 60)
print("  All done.")
print("=" * 60)
