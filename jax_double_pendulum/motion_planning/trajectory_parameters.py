import numpy as np

ELLIPSE_PARAMS = dict(
    omega=72 / 180 * np.pi,  # rotational velocity [rad / s]
    rx=1.75,  # radius of ellipse in x-direction [m]
    ry=1.25,  # radius of ellipse in y-direction [m]
    ell_angle=45 / 180 * np.pi,  # angle of inclination of ellipse [rad]
    x0=0.4,  # center of ellipse in x-direction [m]
    y0=0.4,  # center of ellipse in y-direction [m]
)
