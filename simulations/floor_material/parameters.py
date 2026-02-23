"""Physical constants and material parameter definitions for the bounce simulation."""

from dataclasses import dataclass, field


@dataclass
class BallParams:
    """ITTF regulation table-tennis ball."""
    m: float = 2.7e-3       # mass [kg]
    R: float = 20.0e-3      # radius [m]

    @property
    def I(self) -> float:
        """Moment of inertia for a thin-walled hollow sphere: I = (2/3) m R^2."""
        return (2.0 / 3.0) * self.m * self.R**2


@dataclass
class ContactParams:
    """Ball-table contact spring-dashpot and friction."""
    k_c: float = 2.5e4      # contact stiffness [N/m]  (tuned for COR ~0.76, contact ~1ms)
    c_c: float = 1.6        # contact damping [N s/m]  (tuned for COR ~0.76)
    mu: float = 0.25        # Coulomb friction coefficient
    v_0: float = 0.01       # tanh regularisation velocity [m/s]


@dataclass
class TableParams:
    """Table mass (top + legs lumped)."""
    m_t: float = 127.0      # table mass [kg]  (Butterfly Centrefold 25 tournament table)


@dataclass
class FloorParams:
    """Floor support spring-dashpot."""
    k_f: float = 1.0e7      # floor stiffness [N/m]
    c_f: float = 1.0e3      # floor damping [N s/m]
    name: str = "concrete"


# ---------- Preset floor profiles ----------

# Floor stiffness spans 8 orders of magnitude — from reinforced concrete
# to an absurdly soft surface — to demonstrate that floor material has no
# observable effect on ball mechanics with a 127 kg tournament table.
CONCRETE = FloorParams(k_f=1.0e8, c_f=1.0e5, name="Concrete")
HARDWOOD = FloorParams(k_f=1.0e6, c_f=1.0e4, name="Hardwood")
RUBBER_MAT = FloorParams(k_f=1.0e4, c_f=1.0e3, name="Rubber Mat")
CARPET = FloorParams(k_f=1.0e2, c_f=1.0e2, name="Carpet")
EXTREME_SOFT = FloorParams(k_f=1.0e0, c_f=1.0e0, name="Extreme Soft")

ALL_FLOORS = [CONCRETE, HARDWOOD, RUBBER_MAT, CARPET, EXTREME_SOFT]
REALISTIC_FLOORS = [CONCRETE, HARDWOOD, RUBBER_MAT, CARPET]


@dataclass
class SimConfig:
    """Bundles all parameters for a single simulation run."""
    ball: BallParams = field(default_factory=BallParams)
    contact: ContactParams = field(default_factory=ContactParams)
    table: TableParams = field(default_factory=TableParams)
    floor: FloorParams = field(default_factory=lambda: FloorParams(
        k_f=CONCRETE.k_f, c_f=CONCRETE.c_f, name=CONCRETE.name))
    g: float = 9.81          # gravitational acceleration [m/s^2]
