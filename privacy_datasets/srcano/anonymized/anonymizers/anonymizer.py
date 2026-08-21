from abc import ABC, abstractmethod
from typing import List, Iterator
from srcano.reddit.reddit_types import Profile


class Anonymizer(ABC):
    @abstractmethod
    def anonymize(self, text: str) -> str:
        pass

    @abstractmethod
    def anonymize_profiles(self, profiles: List[Profile]) -> Iterator[Profile]:
        pass
