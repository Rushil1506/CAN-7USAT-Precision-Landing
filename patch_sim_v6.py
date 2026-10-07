import re
with open("tests/test_simulator.py", "r") as f:
    content = f.read()

pattern = r"        history = sim\.run\(steering_function=lambda t, s: 0\.0, max_time=400\.0\)"
with_replacement = """        def bad_steer(t, s):
            sim.parafoil_config.descent_rate = 0.5
            return 0.0
        history = sim.run(steering_function=bad_steer, max_time=400.0)"""

content = re.sub(pattern, with_replacement, content)
with open("tests/test_simulator.py", "w") as f:
    f.write(content)
print("done")
