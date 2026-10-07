import re
with open("tests/test_simulator.py", "r") as f:
    content = f.read()

pattern = r"        sim\.state_machine\.state = FlightState\.TERMINAL_GUIDANCE"
with_replacement = """        sim.state_machine.state = FlightState.TERMINAL_GUIDANCE
        sim.terminal_deadline = 2.0"""

content = re.sub(pattern, with_replacement, content)
with open("tests/test_simulator.py", "w") as f:
    f.write(content)
print("done")
