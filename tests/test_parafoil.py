import math

from flight.parafoil import ParafoilState, DrogueConfig, ParafoilConfig, ParafoilModel


def create_state():

    return ParafoilState(
        x=0.0,
        y=0.0,
        altitude=1000.0,
        vx=0.0,
        vy=0.0,
        vz=0.0,
        heading=0.0,
        heading_rate=0.0,
    )


def create_model():

    state = create_state()

    drogue = DrogueConfig(descent_rate=17.0)

    parafoil = ParafoilConfig(forward_speed=10.0, descent_rate=2.0, max_turn_rate=30.0)

    model = ParafoilModel(state=state, drogue_config=drogue, parafoil_config=parafoil)

    return model, state


def test_drogue_descent():

    model, state = create_model()

    model.update_drogue(dt=1.0)

    assert math.isclose(state.altitude, 983.0)


def test_drogue_has_no_steering():

    model, state = create_model()

    model.update_drogue(dt=1.0)

    assert math.isclose(state.heading_rate, 0.0)

    assert math.isclose(state.vx, 0.0)

    assert math.isclose(state.vy, 0.0)


def test_parafoil_forward_motion_north():

    model, state = create_model()

    model.update_parafoil(steering=0.0, dt=1.0)

    assert math.isclose(state.x, 0.0)

    assert math.isclose(state.y, 10.0)


def test_parafoil_descent():

    model, state = create_model()

    model.update_parafoil(steering=0.0, dt=1.0)

    assert math.isclose(state.altitude, 998.0)


def test_parafoil_right_turn():

    model, state = create_model()

    model.update_parafoil(steering=1.0, dt=1.0)

    assert math.isclose(state.heading, 30.0)


def test_parafoil_left_turn():

    model, state = create_model()

    model.update_parafoil(steering=-1.0, dt=1.0)

    assert math.isclose(state.heading, 330.0)


def test_parafoil_wind():

    model, state = create_model()

    model.parafoil_config.wind_x = 3.0
    model.parafoil_config.wind_y = 2.0

    model.update_parafoil(steering=0.0, dt=1.0)

    assert math.isclose(state.vx, 3.0)

    assert math.isclose(state.vy, 12.0)


def test_altitude_never_negative():

    model, state = create_model()

    state.altitude = 1.0

    model.update_parafoil(steering=0.0, dt=1.0)

    assert state.altitude == 0.0
