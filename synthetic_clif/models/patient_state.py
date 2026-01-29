"""Patient physiological state model for temporal coherence."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PatientState:
    """Tracks physiological state to ensure temporal coherence across measurements.

    This model maintains a patient's vital signs and clinical status, evolving
    them over time with autoregressive dynamics to prevent unrealistic jumps
    between consecutive measurements.

    Attributes:
        heart_rate: Heart rate in bpm
        sbp: Systolic blood pressure in mmHg
        dbp: Diastolic blood pressure in mmHg
        map_value: Mean arterial pressure in mmHg (calculated)
        spo2: Oxygen saturation percentage
        temperature: Body temperature in Celsius
        respiratory_rate: Respiratory rate in breaths/min
        fio2: Fraction of inspired oxygen (0.21 = room air, 1.0 = 100%)
        peep: Positive end-expiratory pressure in cmH2O
        device_category: Current respiratory support device
        is_intubated: Whether patient is mechanically ventilated
        is_on_vasopressors: Whether receiving vasoactive medications
        is_sedated: Whether receiving sedation
        gcs_total: Glasgow Coma Scale total score
        acuity_level: Patient acuity (1=critical, 2=high, 3=moderate, 4=low)
    """

    # Vital signs
    heart_rate: float = 80.0
    sbp: float = 120.0
    dbp: float = 80.0
    spo2: float = 98.0
    temperature: float = 37.0
    respiratory_rate: float = 16.0

    # Respiratory support
    fio2: float = 0.21
    peep: float = 0.0
    device_category: str = "Room Air"
    is_intubated: bool = False

    # Clinical status
    is_on_vasopressors: bool = False
    is_sedated: bool = False
    gcs_total: int = 15
    acuity_level: int = 3  # 1=critical, 2=high, 3=moderate, 4=low

    # Internal state tracking
    _hours_since_admission: float = field(default=0.0, repr=False)
    _trajectory: str = field(default="stable", repr=False)  # stable, improving, deteriorating

    @property
    def map_value(self) -> float:
        """Calculate mean arterial pressure."""
        return self.dbp + (self.sbp - self.dbp) / 3

    def step(
        self,
        dt_hours: float,
        rng: Optional[np.random.Generator] = None,
    ) -> "PatientState":
        """Evolve state with autoregressive dynamics + noise.

        Args:
            dt_hours: Time step in hours
            rng: Random number generator

        Returns:
            New PatientState with evolved values
        """
        if rng is None:
            rng = np.random.default_rng()

        # Copy current state
        new_state = PatientState(
            heart_rate=self.heart_rate,
            sbp=self.sbp,
            dbp=self.dbp,
            spo2=self.spo2,
            temperature=self.temperature,
            respiratory_rate=self.respiratory_rate,
            fio2=self.fio2,
            peep=self.peep,
            device_category=self.device_category,
            is_intubated=self.is_intubated,
            is_on_vasopressors=self.is_on_vasopressors,
            is_sedated=self.is_sedated,
            gcs_total=self.gcs_total,
            acuity_level=self.acuity_level,
            _hours_since_admission=self._hours_since_admission + dt_hours,
            _trajectory=self._trajectory,
        )

        # AR(1) parameters - phi controls persistence, higher = more stable
        phi = 0.85 ** dt_hours  # Decay with time step size

        # Noise scales with sqrt of time step
        noise_scale = np.sqrt(dt_hours)

        # Evolve each vital sign with mean reversion
        new_state.heart_rate = self._evolve_vital(
            self.heart_rate,
            mean=80 if not self.is_on_vasopressors else 95,
            phi=phi,
            sigma=5 * noise_scale,
            lower=40,
            upper=180,
            rng=rng,
        )

        new_state.sbp = self._evolve_vital(
            self.sbp,
            mean=120 if not self.is_on_vasopressors else 100,
            phi=phi,
            sigma=8 * noise_scale,
            lower=60,
            upper=220,
            rng=rng,
        )

        new_state.dbp = self._evolve_vital(
            self.dbp,
            mean=min(80, new_state.sbp - 20),  # DBP < SBP
            phi=phi,
            sigma=5 * noise_scale,
            lower=30,
            upper=min(130, new_state.sbp - 10),
            rng=rng,
        )

        new_state.spo2 = self._evolve_vital(
            self.spo2,
            mean=98 if self.fio2 > 0.21 else 96,
            phi=phi,
            sigma=1.5 * noise_scale,
            lower=70,
            upper=100,
            rng=rng,
        )

        new_state.temperature = self._evolve_vital(
            self.temperature,
            mean=37.0,
            phi=phi,
            sigma=0.3 * noise_scale,
            lower=34.0,
            upper=42.0,
            rng=rng,
        )

        new_state.respiratory_rate = self._evolve_vital(
            self.respiratory_rate,
            mean=16 if not self.is_intubated else 18,
            phi=phi,
            sigma=2 * noise_scale,
            lower=8,
            upper=40,
            rng=rng,
        )

        # Clinical events based on current state
        new_state = self._check_clinical_events(new_state, rng)

        return new_state

    def _evolve_vital(
        self,
        current: float,
        mean: float,
        phi: float,
        sigma: float,
        lower: float,
        upper: float,
        rng: np.random.Generator,
    ) -> float:
        """Evolve a single vital sign with AR(1) dynamics."""
        deviation = current - mean
        new_value = mean + phi * deviation + rng.normal(0, sigma)
        return np.clip(new_value, lower, upper)

    def _check_clinical_events(
        self,
        state: "PatientState",
        rng: np.random.Generator,
    ) -> "PatientState":
        """Check for and apply clinical interventions based on state."""
        # Low MAP triggers vasopressor consideration
        if state.map_value < 65 and not state.is_on_vasopressors:
            if rng.random() < 0.3:  # 30% chance per time step
                state.is_on_vasopressors = True
                state.acuity_level = min(state.acuity_level, 2)

        # High MAP allows vasopressor weaning
        if state.map_value > 75 and state.is_on_vasopressors:
            if rng.random() < 0.1:  # 10% chance per time step
                state.is_on_vasopressors = False

        # Low SpO2 triggers respiratory support escalation
        if state.spo2 < 92 and not state.is_intubated:
            if rng.random() < 0.2:
                state.fio2 = min(state.fio2 + 0.1, 1.0)
                if state.fio2 >= 0.6 and rng.random() < 0.3:
                    state.is_intubated = True
                    state.device_category = "IMV"
                    state.peep = 5.0
                    state.acuity_level = 1

        # Good SpO2 allows FiO2 weaning
        if state.spo2 > 95 and state.fio2 > 0.21:
            if rng.random() < 0.15:
                state.fio2 = max(state.fio2 - 0.05, 0.21)

        return state

    @classmethod
    def from_acuity(
        cls,
        acuity_level: int,
        rng: Optional[np.random.Generator] = None,
    ) -> "PatientState":
        """Create initial patient state based on acuity level.

        Args:
            acuity_level: 1=critical, 2=high, 3=moderate, 4=low
            rng: Random number generator

        Returns:
            PatientState initialized for given acuity
        """
        if rng is None:
            rng = np.random.default_rng()

        if acuity_level == 1:  # Critical
            return cls(
                heart_rate=rng.normal(110, 15),
                sbp=rng.normal(90, 15),
                dbp=rng.normal(55, 10),
                spo2=rng.normal(88, 5),
                temperature=rng.normal(38.5, 1),
                respiratory_rate=rng.normal(28, 5),
                fio2=rng.uniform(0.6, 1.0),
                peep=rng.uniform(8, 15),
                device_category="IMV",
                is_intubated=True,
                is_on_vasopressors=rng.random() > 0.3,
                is_sedated=True,
                gcs_total=rng.integers(3, 10),
                acuity_level=1,
            )
        elif acuity_level == 2:  # High
            return cls(
                heart_rate=rng.normal(95, 12),
                sbp=rng.normal(105, 15),
                dbp=rng.normal(65, 10),
                spo2=rng.normal(93, 3),
                temperature=rng.normal(37.8, 0.8),
                respiratory_rate=rng.normal(22, 4),
                fio2=rng.uniform(0.3, 0.5),
                device_category=rng.choice(["High Flow NC", "NIPPV", "Face Mask"]),
                is_intubated=False,
                is_on_vasopressors=rng.random() > 0.7,
                gcs_total=rng.integers(10, 15),
                acuity_level=2,
            )
        elif acuity_level == 3:  # Moderate
            return cls(
                heart_rate=rng.normal(85, 10),
                sbp=rng.normal(120, 12),
                dbp=rng.normal(75, 8),
                spo2=rng.normal(96, 2),
                temperature=rng.normal(37.2, 0.5),
                respiratory_rate=rng.normal(18, 3),
                fio2=rng.uniform(0.21, 0.35),
                device_category=rng.choice(["Nasal Cannula", "Room Air"]),
                acuity_level=3,
            )
        else:  # Low acuity
            return cls(
                heart_rate=rng.normal(75, 8),
                sbp=rng.normal(125, 10),
                dbp=rng.normal(78, 6),
                spo2=rng.normal(98, 1),
                temperature=rng.normal(37.0, 0.3),
                respiratory_rate=rng.normal(16, 2),
                fio2=0.21,
                device_category="Room Air",
                acuity_level=4,
            )
