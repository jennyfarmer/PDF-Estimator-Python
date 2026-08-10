from abc import ABC, abstractmethod
import numpy as np

class Distribution(ABC):
    """Abstract base class for test distributions."""

    @abstractmethod
    def sample(
        self,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return samples with shape (n_samples, n_dimensions)."""
        raise NotImplementedError

    @abstractmethod
    def pdf(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the analytical PDF at supplied points."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Number of variables."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Human-readable distribution name."""
        return self.__class__.__name__