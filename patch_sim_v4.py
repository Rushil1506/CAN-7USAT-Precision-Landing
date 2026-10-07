import re
with open("tests/test_simulator.py", "r") as f:
    content = f.read()

pattern = r"        history = sim\.run\(steering_function=lambda t, s: 0\.0, max_time=400\.0\)"
with_replacement = """        # Make it descend very slowly so it times out before reaching breakout altitude
        sim.parafoil_config.descent_rate = 0.5
        history = sim.run(steering_function=lambda t, s: 0.0, max_time=400.0)"""

content = re.sub(pattern, with_replacement, content)
with open("tests/test_simulator.py", "w") as f:
    f.write(content)
print("done")
