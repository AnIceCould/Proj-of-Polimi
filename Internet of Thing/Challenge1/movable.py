import numpy as np

largest = 0

# Define sensor information
sensors = [
    {"id": 1, "coord": (1, 2), "energy": 5000},
    {"id": 2, "coord": (10, 3), "energy": 5000},
    {"id": 3, "coord": (4, 8), "energy": 5000},
    {"id": 4, "coord": (15, 7), "energy": 5000},
    {"id": 5, "coord": (6, 1), "energy": 5000},
    {"id": 6, "coord": (9, 12), "energy": 5000},
    {"id": 7, "coord": (14, 4), "energy": 5000},
    {"id": 8, "coord": (3, 10), "energy": 5000},
    {"id": 9, "coord": (7, 7), "energy": 5000},
    {"id": 10, "coord": (12, 14), "energy": 5000}
]

def update_energy(x, y, n):
    """
    Calculate the remaining energy of all sensors after running n cycles at point (x,y)
    
    Parameters:
    x: x-coordinate
    y: y-coordinate
    n: number of cycles
    
    Returns:
    Updated sensor information list
    """
    for sensor in sensors:
        sx, sy = sensor["coord"]
        # Calculate square of distance
        dist_sq = (x - sx)**2 + (y - sy)**2
        # Calculate energy consumption per cycle
        energy_per_cycle = 100 + 2 * dist_sq
        # Calculate total consumption for n cycles
        total_consumption = energy_per_cycle * n
        # Update remaining energy
        sensor["energy"] = max(0, sensor["energy"] - total_consumption)
    
    return sensors

def calculate_cycles(x, y):
    """
    Calculate the maximum number of cycles the system can complete at point (x,y)
    
    Parameters:
    x: x-coordinate
    y: y-coordinate
    
    Returns:
    Maximum number of cycles
    """
    min_cycles = float('inf')
    for sensor in sensors:
        sx, sy = sensor["coord"]
        energy = sensor["energy"]
        # Calculate square of distance
        dist_sq = (x - sx)**2 + (y - sy)**2
        # Calculate energy consumption per cycle
        energy_per_cycle = 100 + 2 * dist_sq
        # Calculate number of cycles
        cycles = energy / energy_per_cycle
        # Update minimum cycle count
        if cycles < min_cycles:
            min_cycles = cycles
    
    return min_cycles

for x in np.arange(5, 9, 0.1):
    for y in np.arange(5, 9, 0.1):
        for n in range(1, 20):
            updated_sensors = update_energy(x, y, n)
            print(f"After running {n} cycles at point ({x}, {y}):")
            # Find optimal point
            best_x, best_y = 0, 0
            max_min_cycles = 0
            for x in np.arange(5, 9, 0.1):
                for y in np.arange(5, 9, 0.1):
                    cycles = calculate_cycles(x, y)
                    if cycles > max_min_cycles:
                        max_min_cycles = np.floor(cycles)
                        best_x, best_y = x, y

            print(f"Optimal point coordinates: ({best_x}, {best_y})")
            print(f"Maximum number of cycles: {max_min_cycles + n}")
            if (max_min_cycles + n) > largest:
                largest = max_min_cycles + n
            for sensor in sensors:
                sensor["energy"] = 5000
print(f"Total Maximum: {largest}")