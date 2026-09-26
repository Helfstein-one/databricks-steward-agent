from abc import ABC, abstractmethod
from typing import Any

class IJulesMcpAdapter(ABC):
    @abstractmethod
    def ask_jules(self, prompt: str) -> Any:
        pass
