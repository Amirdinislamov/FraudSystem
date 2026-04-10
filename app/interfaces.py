from abc import ABC, abstractmethod
from app.schemas import ClientProfileState

class IProfileRepository(ABC):
    @abstractmethod
    async def get_profile(self, client_id: str) -> ClientProfileState:
        """Получить текущее состояние профиля клиента"""
        pass

    @abstractmethod
    async def save_profile(self, profile: ClientProfileState) -> None:
        """Сохранить обновленное состояние профиля"""
        pass