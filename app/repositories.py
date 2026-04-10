from typing import Dict
from app.interfaces import IProfileRepository
from app.schemas import ClientProfileState

class InMemoryProfileRepository(IProfileRepository):
    def __init__(self):
        self._storage: Dict[str, ClientProfileState] = {}

    async def get_profile(self, client_id: str) -> ClientProfileState:
        if client_id not in self._storage:
            return ClientProfileState(client_id=client_id)
        return self._storage[client_id]

    async def save_profile(self, profile: ClientProfileState) -> None:
        self._storage[profile.client_id] = profile