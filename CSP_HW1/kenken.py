# Part 1: Import Libraries and Define the Cage Class
from copy import deepcopy,copy
import random
from decimal import Decimal, ROUND_DOWN
import time
from math import gcd
import numpy as np

class Cage:
    cells_number_cage = []
    def __init__(self, cells, operation, target,number,divison_target):
        self.cells = cells
        self.operation = operation
        self.target = target
        self.number = number # number which Cage
        self.divison_target = divison_target

# Part 2: Function to Generate KenKen Puzzle
def generate_KenKen(size):
    lst = [[0] * size for _ in range(size)]
    grid = fill_grid_with_numbers(deepcopy(lst),size)
    return lst,generate_random_cages(grid,size),grid

# Part 3: Function to Fill Grid with Numbers
def fill_grid_with_numbers(grid, size):
    """
    This function fills the grid with numbers 1 to size such that
    each row and each column has unique values. It uses backtracking
    to ensure all cells are filled with valid numbers.
    """

    def is_safe_to_place(grid, row, col, num):
        # Check row and column uniqueness
        for i in range(size):
            if num == grid[row][i] or num == grid[i][col]:
                return False
        
        return True

    def fill_grid_backtracking(grid, row, col):
        if row == size:
            return True
        if col == size:
            return fill_grid_backtracking(grid, row + 1, 0)
        
        random.shuffle(numbers)
        for num in numbers:
            if is_safe_to_place(grid, row, col, num):
                grid[row][col] = num

                if fill_grid_backtracking(grid, row, col + 1):
                    return True

                grid[row][col] = 0

        return False
    
    numbers = list(range(1,size+1))
    fill_grid_backtracking(grid, 0, 0)
    return grid

# Part 4: Function to Generate Random Cages
def generate_random_cages(grid, size):
    cages = []
    cells_number_cage = [[-1] * size for _ in range(size)]
    visited = [[False] * size for _ in range(size)]

    for i in range(size):
        for j in range(size):
            if not visited[i][j]:
                cage_size = random.randint(1, size)
                cells = [(i, j)]
                cells_number_cage[i][j] = len(cages)
                visited[i][j] = True

                while len(cells) < cage_size:
                    x, y = cells[-1]
                    neighbors = [(x + dx, y + dy) for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]
                                 if 0 <= x + dx < size and 0 <= y + dy < size and not visited[x + dx][y + dy]]
                    if neighbors:
                        next_cell = random.choice(neighbors)
                        cells.append(next_cell)
                        cells_number_cage[next_cell[0]][next_cell[1]] = len(cages)
                        visited[next_cell[0]][next_cell[1]] = True
                    else:
                        break

                operation = random.choice(['+', '*']) if cage_size > 2 else random.choice(['+', '-', '*', '/'])
                target,division = calculate_target(grid, cells, operation)
                cages.append(Cage(cells, operation, target,len(cages),division))
               
    Cage.cells_number_cage = cells_number_cage
    return cages

# Function to calculate the target based on the operation for a given cage
def calculate_target(grid, cells, operation):
    if len(cells) == 1:
        return grid[cells[0][0]][cells[0][1]],None
    if operation == "/":
        c1 = grid[cells[0][0]][cells[0][1]]
        c2 = grid[cells[1][0]][cells[1][1]]
        GCD = gcd(c1,c2)
        return Decimal(max(c1,c2)/min(c1,c2)).quantize(Decimal('0.01'), rounding=ROUND_DOWN),[max(c1,c2)/GCD,min(c1,c2)/GCD]
    elif operation == '-':
        c1 = grid[cells[0][0]][cells[0][1]]
        c2 = grid[cells[1][0]][cells[1][1]]
        return abs(c1-c2),None
    elif operation == "+":
        return sum([grid[x[0]][x[1]] for x in cells]),None
    elif operation == "*":
        mul = 1
        for i in cells:
            mul*=grid[i[0]][i[1]]
        return mul,None    

# Part 5: KenKen Backtracking Solver

def solve_kenken(grid, cages):
    def fill_grid_backtracking(grid, row, col):
        if row == size:
            return True
        if col == size:
            return fill_grid_backtracking(grid, row + 1, 0)
        number = list(range(1,size+1))
        random.shuffle(number)
        for num in number:
            if is_safe_kenken(grid, row, col, num,cages):
                grid[row][col] = num
                if fill_grid_backtracking(grid, row, col + 1):
                    return True
                grid[row][col] = 0
        return False
    fill_grid_backtracking(grid, 0, 0)
    return grid

def is_safe_kenken(grid, row, col, num, cages):
    # Check row, column and cage numbers uniqueness
    for i in range(size):
        if num == grid[row][i] or num == grid[i][col]:
            return False
    cage = cages[Cage.cells_number_cage[row][col]]   
    return validate_cage_operation(cage.operation,cage.cells,cage.target,grid,num,row,col)

def validate_cage_operation(operation, cells, target,grid,num,row,col):
    # Check if the operation on the cage values matches the target
    values = []
    for cell in cells:
        if cell == (row,col):
            values.append(num)
        else:    
            values.append(grid[cell[0]][cell[1]])
    if len(cells) == 1:
        return num == target    
    elif operation == "/":
        if 0 in values:
            return True
        else:
            return Decimal(max(values)/min(values)).quantize(Decimal('0.01'), rounding=ROUND_DOWN) == target
    elif operation == "-":
        if 0 in values:
            return True
        else:
            return abs(values[0]-values[1]) == target
    elif operation == "+":
        return np.sum(values) <= target  
    elif operation == "*":
        return np.prod(values) <= target 

def find_unassigned_location(grid):
    # Find the first empty cell in the grid
    # just satrt 00 and end nn
    raise NotImplementedError

# Finding the divisors of m
def get_divisors(m,n):
    divisors = set()
    for i in range(1, min(n+1,int(m**0.5) + 1)):
        if m % i == 0:
            divisors.add(i)
            if m // i <=n:
                divisors.add(m // i)
    return divisors

# Part 6: KenKen Domain Constraint Solver
def solve_kenken_csp(grid, cages):
    """
    Function to solve the KenKen grid using Constraint Satisfaction Problem (CSP)
    """
    def create_domains(grid):
        """
        Function to create domains for each cell in the grid
        """
        def cage_dom(operation,target,div_target,n):
            dom = set()
            if n == 1:
                dom.add(target)
            elif operation == "/":
                for i in range(1,size+1):
                    if div_target[0]*i<=size:
                        dom.update([div_target[0]*i,div_target[1]*i])
                    else:
                        break    
            elif operation == "*" :
                dom = get_divisors(target,size)
            elif operation == "-":
                dom.update(numbers)
            else:
                dom.update(numbers)
            return dom    
                
        order = []        
        dict = {}
        for cage in cages  :
            dom = cage_dom(cage.operation,cage.target,cage.divison_target,len(cage.cells))
            for cell in cage.cells:
                dict[cell] = dom
                order.append([cell,len(dom)])
        return dict ,order       
            
    def solve_csp(assignment,order):
        """
        Recursive function to solve grid with CSP
        """
        def find_unassigned_location(order):
            """
            Function to find an unassigned location in the grid
            """
            order = sorted(order,reverse=True,key=lambda x :x[1])
            if order:
                o = order.pop()
            else:
                o = [(-1,-1),0]    
            return order,o[0],o[1] 
        
        def cage_check(operation, cells, target,dev,assignment,num,row,col):
            if len(cells) == 1:
                return assignment,num == target
            new_assignment = {key: value.copy() for key, value in assignment.items()}
            check = True
            if operation == "/":
                MAX = dev[0]
                MIN =dev[1]
                temp = set()
                k1,k2 = (0,1) if (row,col) == cells[1] else (1,0) #k2 one arg and k1 can multi
                for i in assignment[cells[k1]]:
                    for j in assignment[cells[k2]]:
                        if i*MIN == j*MAX or j*MIN == i*MAX:
                            temp.add(i)
                new_assignment[(cells[k1])] = temp 
                if len(temp) == 0:
                    check = False           
            elif operation == "-":
                temp = set()
                k1,k2 = (0,1) if (row,col) == cells[1] else (1,0) #k2 one arg and k1 can multi
                num = [x for x in assignment[cells[k2]]][0]#عدد موجود در ستی که فقط یک مقدار دارد
                if num - target in assignment[cells[k1]]:
                    temp.add(num - target)
                if num + target in assignment[cells[k1]]:
                    temp.add(num + target)    
                new_assignment[(cells[k1])] = temp   
                if len(temp) == 0:
                    check = False  
            elif operation == "+":
                #find min and max can possible
                MAX = 0
                MIN = 0
                l = True #همه پرشده یا ن
                order_cell = sorted(cells,key=lambda x : len(assignment[x]))
                m = copy(target)
                k = 0
                n = len(cells)
                for cell in order_cell:
                    if len(assignment[cell])>1:
                        l = False
                        new_assignment[cell] = {x for x in new_assignment[cell] if m-(n-k-1)>=x>= m-((n-k-1)*size)}
                        if len(new_assignment[cell]) == 0:
                            check = False
                            break
                    else:
                        k+=1
                        m-= next(iter(assignment[cell])) 
                        if m < 0:
                            check = False   
                            break  
                if l and check and m!=0:
                    check = False   
            elif operation == "*":
                #find min and max can possible
                MAX = 1
                MIN = 1
                l = True #همه پرشده یا ن
                m = copy(target)
                order_cell = sorted(cells,key=lambda x : len(assignment[x]))
                for cell in order_cell:
                    if len(assignment[cell])>1:
                        l = False
                        new_assignment[cell] = {x for x in new_assignment[cell] if m%x==0}
                        if len(new_assignment[cell]) == 0:
                            check = False
                            break
                    else:
                        m/= next(iter(assignment[cell]))
                        if m != int(m):
                            check = False   
                            break
                if l and check and m!=1:
                    check = False        
      
            if check:
                return new_assignment,True
            else:
                return assignment,False
            
        def is_valid_assignment(assignment,row,col,num,cage):
            new_assignment= {key: value.copy() for key, value in assignment.items()}
            check = True
            for i in range(size):
                if col != i:
                    new_assignment[(row,i)].discard(num)
                    if (len(new_assignment[(row,i)]) == 0):
                        check = False
                        break
                if row != i:
                    new_assignment[(i,col)].discard(num)  
                    if (len(new_assignment[(i,col)]) == 0):
                        check = False
                        break
            new_assignment[(row,col)] = {num}
            if check:
                new_assignment,check = cage_check(cage.operation, cage.cells, cage.target,cage.divison_target,new_assignment,num,row,col)
                if check: 
                    return new_assignment,True
                else:
                    return assignment,False
            else:
                return assignment,False
        def solve(assignment,order,row, col,l): 
            if row == -1:
                return assignment,True
            for num in assignment[(row,col)]:
                    cage_cells = set(cages[Cage.cells_number_cage[row][col]].cells).union({(x, col) for x in range(size)}, {(row, x) for x in range(size)})
                    copy_assignment = {cell: assignment[cell] for cell in cage_cells}
                    assignment,check = is_valid_assignment(assignment,row,col,num,cages[Cage.cells_number_cage[row][col]])
                    if check:
                        order,o,l= find_unassigned_location(order)
                        assignment,check = solve(assignment,order,o[0],o[1],l)            
                        if check:
                            return assignment,True
                        order.append([o,l])                   
                    for cell in cage_cells:
                        assignment[cell] = copy_assignment[cell]
                                   
            return assignment,False
        
        order,o,l= find_unassigned_location(order)
        assignment,check = solve(assignment,order,o[0],o[1],l)
        return assignment,check
           
    size = len(grid)
    numbers = list(range(1, size + 1))
    assignment,order = create_domains(grid)
    # Solve the KenKen grid using CSP
    assignment,check = solve_csp(assignment,order)
    if check:
        # return[[assignment[(i, j)] for j in range(len(grid))] for i in range(len(grid))]
        return assignment
    else:
        return False

# Part 7: Print Solution
def print_solution(solution, file=None):
    for row in solution:
        row_str = ' '.join(str(cell) for cell in row)
        if file:
            file.write(row_str + "\n")  
        else:
            print(row_str) 
    return True

# Part 8: Run Example
test = 1 # number of tests
start_size = 7
range_size = 7
limit_size_back = 7
back = [[0] * test for _ in range(min(range_size,limit_size_back)+1)]
csp = [[0] * test for _ in range(range_size+1)]

with open("CSP_HW1/test.txt", "w") as f:
    for i in range(start_size, range_size + 1):
        for j in range(test):
            size = i
            unsolved_grid, kenken_cages, grid = generate_KenKen(size)
            f.write(f"Generated KenKen table (size {size},number {j}):\n")
            f.write("-----------------------\n")
            for k in grid:
                f.write(f"{k}\n")
            f.write("Generated KenKen Cages:\n")
            for cage in kenken_cages:
                f.write(f"Cage Cells: {cage.cells}, Operation: {cage.operation}, Target: {cage.target}\n")
            f.write("-----------------------\n")
            if i<limit_size_back+1:
                backtrack_grid = deepcopy(unsolved_grid)
                time_start = time.time()
                back_solve = solve_kenken(backtrack_grid, kenken_cages)
                cal_time = time.time() - time_start
                if back_solve:
                    f.write("Solved KenKen Puzzle (BackTracking):\n")
                    print_solution(backtrack_grid, f)    
                else:
                    f.write("No solution found using BackTracking.\n")
                f.write("-----------------------\n")
                back[i][j] = cal_time
                print('end back',i,j,cal_time)
            time_start = time.time()
            csp_solved = solve_kenken_csp(unsolved_grid, kenken_cages)  
            cal_time = time.time() - time_start
            grid_csp = [[list(csp_solved.get((i, j), {0}))[0] for j in range(size)] for i in range(size)]
            if csp_solved:
                f.write("Solved KenKen Puzzle (domain CSP):\n")
                print_solution(grid_csp, f)  
            else:
                f.write("No solution found using CSP.\n")
            csp[i][j] = cal_time
            print('end csp',i,j,cal_time)
        
    f.write("#################################################################################\n")
    f.write("-------------all time back--------------------\n")
    print_solution(back, f) 
    f.write("-------------all time csp---------------------\n")
    print_solution(csp, f)  
    f.write("---------------backtrack--------------\n")
    results = [(np.mean(row), np.var(row)) for row in back]
    f.write(f"{'size':<5} {'Mean':<30} {'Variance':<20}\n")
    f.write("-" * 50 + "\n")
    for idx, (mean, var) in enumerate(results):
        f.write(f"{idx:<5} {mean:<30} {var:<20}\n")
    f.write("-----------------csp-----------------\n")
    results = [(np.mean(row), np.var(row)) for row in csp]
    f.write(f"{'size':<5} {'Mean':<30} {'Variance':<20}\n")
    f.write("-" * 50 + "\n")
    for idx, (mean, var) in enumerate(results):
        f.write(f"{idx:<5} {mean:<30} {var:<20}\n")  