"""Token accounting abstraction.

The internal demo uses PostgreSQL-backed fake tokens.  Production can replace
this adapter with the host site's wallet without changing catalog planning,
entitlements, chat behavior, or frontend contracts.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from bot.media_repository import (
    MediaRepository,
    TokenReceipt,
    UnlockReceipt,
    WalletSnapshot,
)


@runtime_checkable
class TokenService(Protocol):
    async def get_wallet(self, user_id: int) -> WalletSnapshot: ...

    async def refill_wallet(
        self, user_id: int, idempotency_key: str
    ) -> TokenReceipt: ...

    async def unlock_offer(
        self,
        user_id: int,
        offer_id: int | str,
        idempotency_key: str,
        *,
        tag_pairs: tuple[tuple[str, str], ...] = (),
    ) -> UnlockReceipt: ...


class DemoTokenService:
    """Demo token implementation backed by the atomic repository ledger."""

    def __init__(self, repository: MediaRepository):
        self.repository = repository

    async def get_wallet(self, user_id: int) -> WalletSnapshot:
        return await self.repository.get_wallet(user_id)

    async def refill_wallet(
        self, user_id: int, idempotency_key: str
    ) -> TokenReceipt:
        return await self.repository.refill_wallet(user_id, idempotency_key)

    async def unlock_offer(
        self,
        user_id: int,
        offer_id: int | str,
        idempotency_key: str,
        *,
        tag_pairs: tuple[tuple[str, str], ...] = (),
    ) -> UnlockReceipt:
        return await self.repository.unlock_offer(
            user_id, offer_id, idempotency_key, tag_pairs=tag_pairs
        )
