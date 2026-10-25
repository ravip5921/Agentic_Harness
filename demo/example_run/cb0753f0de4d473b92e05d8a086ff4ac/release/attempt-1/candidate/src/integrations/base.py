"""Small, platform-independent contract for user connectors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class User:
    id: str
    username: str
    profile_url: str


class Connector(ABC):
    @abstractmethod
    def list_users(self) -> list[User]:
        """Return the users visible to this connector's credentials.

        Implementations must raise an exception on retrieval failure rather
        than silently returning an incomplete result.
        """
