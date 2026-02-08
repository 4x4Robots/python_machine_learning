import fuzzy_systems as fs

# Create Mamdani system
system = fs.MamdaniSystem()
# Add input: temperature
system.add_input('temperature', (0, 40))
system.add_term('temperature', 'cold', 'trapezoidal', (5, 5, 20, 30))
system.add_term('temperature', 'hot', 'triangular', (20, 40, 40))
# Example output of a working trapez
import matplotlib.pyplot as plt
system.plot_variables(['temperature'])
plt.savefig("temperature.png")
# Finish defining Mamdani system
# Add output: fan speed
system.add_output('fan_speed', (0, 100))
system.add_term('fan_speed', 'slow', 'triangular', (0, 0, 50))
system.add_term('fan_speed', 'fast', 'triangular', (50, 100, 100))
# Define fuzzy rules
system.add_rules([
    ('cold', 'slow'),  # IF temperature is cold THEN fan_speed is slow
    ('hot', 'fast')    # IF temperature is hot THEN fan_speed is fast
])

# Test the system
result = system.evaluate(temperature=25)
print(f"Fan speed: {result['fan_speed']:.1f}%")

