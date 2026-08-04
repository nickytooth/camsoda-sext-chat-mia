"""PostgreSQL persistence for Mia's visual-content commerce.

This module is intentionally unaware of prompts and R2 URLs.  It owns the
durable per-user facts that must survive a normal chat reset: offer delivery
history, entitlements, the demo wallet/ledger, rotation, and tag affinity.

Offer creation is two-phase.  A planner first creates a ``reserved`` row; it
only becomes part of history/rotation after the in-character teaser has been
persisted and ``mark_offer_delivered`` succeeds.  Failed generations can
therefore cancel their reservation without poisoning the candidate pool.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Mapping

from bot import config
from bot.memory.db import get_connection


class CommerceError(RuntimeError):
    """Base error with a stable API-facing machine code."""

    code = "commerce_error"

    def __init__(self, message: str = "Commerce operation failed"):
        super().__init__(message)


class OfferNotFoundError(CommerceError):
    code = "offer_not_found"


class MediaUnavailableError(CommerceError):
    code = "media_unavailable"


class InsufficientTokensError(CommerceError):
    code = "insufficient_tokens"

    def __init__(self, *, balance: int, required: int):
        self.balance = balance
        self.required = required
        super().__init__(f"Insufficient tokens: {balance} available, {required} required")


class ForbiddenCommerceError(CommerceError):
    code = "forbidden"


class IdempotencyConflictError(CommerceError):
    code = "idempotency_conflict"


@dataclass(frozen=True, slots=True)
class WalletSnapshot:
    user_id: int
    balance: int


@dataclass(frozen=True, slots=True)
class TokenReceipt:
    user_id: int
    balance: int
    transaction_id: int
    amount_tokens: int
    idempotent_replay: bool = False


@dataclass(frozen=True, slots=True)
class OfferRecord:
    offer_id: int
    user_id: int
    content_id: str
    trigger: str
    action: str
    request_type: str | None
    description: str | None
    batch_number: int
    price_tokens: int
    status: str
    created_at: float
    offered_at: float | None


@dataclass(frozen=True, slots=True)
class UnlockReceipt:
    user_id: int
    offer_id: int
    content_id: str
    balance: int
    charged_tokens: int
    transaction_id: int
    unlocked_at: float
    already_unlocked: bool = False


@dataclass(frozen=True, slots=True)
class EntitlementRecord:
    entitlement_id: int
    user_id: int
    content_id: str
    source_offer_id: int
    unlock_transaction_id: int
    unlocked_at: float
    first_opened_at: float | None


def _offer_from_row(row: Mapping[str, object]) -> OfferRecord:
    return OfferRecord(
        offer_id=int(row["id"]),
        user_id=int(row["user_id"]),
        content_id=str(row["content_id"]),
        trigger=str(row["trigger"]),
        action=str(row["action"]),
        request_type=str(row["request_type"]) if row["request_type"] is not None else None,
        description=str(row["description"]) if row["description"] is not None else None,
        batch_number=int(row["batch_number"]),
        price_tokens=int(row["price_tokens"]),
        status=str(row["status"]),
        created_at=float(row["created_at"]),
        offered_at=float(row["offered_at"]) if row["offered_at"] is not None else None,
    )


def _entitlement_from_row(row: Mapping[str, object]) -> EntitlementRecord:
    return EntitlementRecord(
        entitlement_id=int(row["id"]),
        user_id=int(row["user_id"]),
        content_id=str(row["content_id"]),
        source_offer_id=int(row["source_offer_id"]),
        unlock_transaction_id=int(row["unlock_transaction_id"]),
        unlocked_at=float(row["unlocked_at"]),
        first_opened_at=(
            float(row["first_opened_at"])
            if row["first_opened_at"] is not None
            else None
        ),
    )


def _idempotency_key(value: str) -> str:
    key = str(value or "").strip()
    if not key or len(key) > 200 or any(ord(char) < 32 for char in key):
        raise IdempotencyConflictError("A valid idempotency key is required")
    return key


class MediaRepository:
    """Async repository built on the project's asyncpg adapter."""

    async def get_engagement_state(self, user_id: int) -> dict[str, object]:
        conn = await get_connection()
        try:
            cursor = await conn.execute(
                "SELECT * FROM engagement_state WHERE user_id = ?", (user_id,)
            )
            row = await cursor.fetchone()
            return dict(row) if row else {"user_id": user_id, "total_messages": 0}
        finally:
            await conn.close()

    async def set_decline_snooze(
        self, user_id: int, *, until_batch: int, reask_pending: bool = True
    ) -> None:
        conn = await get_connection()
        try:
            await conn.execute(
                """
                INSERT INTO engagement_state
                    (user_id, sales_snooze_until_batch, sales_reask_pending,
                     sales_reask_asked_at_batch)
                VALUES (?, ?, ?, NULL)
                ON CONFLICT(user_id) DO UPDATE SET
                    sales_snooze_until_batch = EXCLUDED.sales_snooze_until_batch,
                    sales_reask_pending = EXCLUDED.sales_reask_pending,
                    sales_reask_asked_at_batch = NULL
                """,
                (user_id, until_batch, reask_pending),
            )
        finally:
            await conn.close()

    async def mark_reask_asked(self, user_id: int, *, batch_number: int) -> None:
        conn = await get_connection()
        try:
            await conn.execute(
                """
                UPDATE engagement_state
                SET sales_reask_pending = TRUE,
                    sales_reask_asked_at_batch = ?
                WHERE user_id = ?
                """,
                (batch_number, user_id),
            )
        finally:
            await conn.close()

    async def clear_sales_pause(self, user_id: int) -> None:
        conn = await get_connection()
        try:
            await conn.execute(
                """
                UPDATE engagement_state
                SET sales_snooze_until_batch = NULL,
                    sales_reask_pending = FALSE,
                    sales_reask_asked_at_batch = NULL
                WHERE user_id = ?
                """,
                (user_id,),
            )
        finally:
            await conn.close()

    async def clear_stale_reask(self, user_id: int, *, asked_at_batch: int) -> None:
        """Clear only the re-ask instance the planner observed.

        The compare-and-set avoids wiping a newer decline if two turns race.
        """
        conn = await get_connection()
        try:
            await conn.execute(
                """
                UPDATE engagement_state
                SET sales_snooze_until_batch = NULL,
                    sales_reask_pending = FALSE,
                    sales_reask_asked_at_batch = NULL
                WHERE user_id = ? AND sales_reask_asked_at_batch = ?
                """,
                (user_id, asked_at_batch),
            )
        finally:
            await conn.close()

    async def unlocked_content_ids(self, user_id: int) -> set[str]:
        conn = await get_connection()
        try:
            cursor = await conn.execute(
                "SELECT content_id FROM media_entitlements WHERE user_id = ?",
                (user_id,),
            )
            return {str(row["content_id"]) for row in await cursor.fetchall()}
        finally:
            await conn.close()

    async def delivered_offer_history(self, user_id: int) -> list[OfferRecord]:
        """Only visible/delivered offers influence history and rotation."""
        conn = await get_connection()
        try:
            cursor = await conn.execute(
                """
                SELECT * FROM media_offers
                WHERE user_id = ? AND status = 'delivered'
                ORDER BY batch_number DESC, id DESC
                """,
                (user_id,),
            )
            return [_offer_from_row(row) for row in await cursor.fetchall()]
        finally:
            await conn.close()

    async def get_tag_affinity(self, user_id: int) -> dict[tuple[str, str], float]:
        conn = await get_connection()
        try:
            cursor = await conn.execute(
                """
                SELECT tag_group, tag_value, score
                FROM media_tag_affinity WHERE user_id = ?
                """,
                (user_id,),
            )
            return {
                (str(row["tag_group"]), str(row["tag_value"])): float(row["score"])
                for row in await cursor.fetchall()
            }
        finally:
            await conn.close()

    async def active_reserved_content_ids(self, user_id: int) -> set[str]:
        cutoff = time.time() - 600
        conn = await get_connection()
        try:
            cursor = await conn.execute(
                """
                SELECT content_id FROM media_offers
                WHERE user_id = ? AND status = 'reserved' AND created_at >= ?
                """,
                (user_id, cutoff),
            )
            return {str(row["content_id"]) for row in await cursor.fetchall()}
        finally:
            await conn.close()

    async def reserve_offer(
        self,
        user_id: int,
        content_id: str,
        *,
        trigger: str,
        action: str,
        request_type: str | None,
        description: str,
        batch_number: int,
        price_tokens: int,
    ) -> OfferRecord:
        now = time.time()
        conn = await get_connection()
        try:
            async with conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock(?)", (user_id,))
                cursor = await conn.execute(
                    """
                    SELECT 1 FROM media_entitlements
                    WHERE user_id = ? AND content_id = ?
                    """,
                    (user_id, content_id),
                )
                if await cursor.fetchone():
                    raise MediaUnavailableError("Media is already unlocked")
                cursor = await conn.execute(
                    """
                    SELECT id FROM media_offers
                    WHERE user_id = ? AND content_id = ? AND status = 'reserved'
                    LIMIT 1
                    """,
                    (user_id, content_id),
                )
                if await cursor.fetchone():
                    raise MediaUnavailableError("Media already has a pending offer")
                cursor = await conn.execute(
                    """
                    INSERT INTO media_offers
                        (user_id, content_id, trigger, action, request_type, description,
                         batch_number, price_tokens, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'reserved', ?)
                    RETURNING *
                    """,
                    (
                        user_id,
                        content_id,
                        trigger,
                        action,
                        request_type,
                        description,
                        batch_number,
                        price_tokens,
                        now,
                    ),
                )
                row = await cursor.fetchone()
                return _offer_from_row(row)
        finally:
            await conn.close()

    async def cancel_stale_reservations(
        self, user_id: int, *, older_than_seconds: int = 600
    ) -> int:
        """Cancel crash residue; reservations never count as offer history."""
        cutoff = time.time() - max(1, int(older_than_seconds))
        conn = await get_connection()
        try:
            cursor = await conn.execute(
                """
                UPDATE media_offers
                SET status = 'cancelled'
                WHERE user_id = ? AND status = 'reserved' AND created_at < ?
                RETURNING id
                """,
                (user_id, cutoff),
            )
            return len(await cursor.fetchall())
        finally:
            await conn.close()

    async def get_offer(
        self,
        offer_id: int | str,
        *,
        user_id: int | None = None,
        delivered_only: bool = False,
    ) -> OfferRecord | None:
        try:
            parsed_id = int(offer_id)
        except (TypeError, ValueError):
            return None
        clauses = ["id = ?"]
        params: list[object] = [parsed_id]
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if delivered_only:
            clauses.append("status = 'delivered'")
        conn = await get_connection()
        try:
            cursor = await conn.execute(
                "SELECT * FROM media_offers WHERE " + " AND ".join(clauses), params
            )
            row = await cursor.fetchone()
            return _offer_from_row(row) if row else None
        finally:
            await conn.close()

    async def mark_offer_delivered(
        self, offer_id: int | str, *, media_type: str | None = None
    ) -> OfferRecord | None:
        try:
            parsed_id = int(offer_id)
        except (TypeError, ValueError):
            return None
        now = time.time()
        conn = await get_connection()
        try:
            async with conn.transaction():
                cursor = await conn.execute(
                    "SELECT user_id FROM media_offers WHERE id = ?", (parsed_id,)
                )
                preliminary = await cursor.fetchone()
                if preliminary is None:
                    return None
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(?)",
                    (int(preliminary["user_id"]),),
                )
                cursor = await conn.execute(
                    "SELECT * FROM media_offers WHERE id = ? FOR UPDATE", (parsed_id,)
                )
                original = await cursor.fetchone()
                if original is None:
                    return None
                if original["status"] == "delivered":
                    return _offer_from_row(original)
                if original["status"] != "reserved":
                    return None
                cursor = await conn.execute(
                    """
                    SELECT 1 FROM media_entitlements
                    WHERE user_id = ? AND content_id = ?
                    """,
                    (int(original["user_id"]), str(original["content_id"])),
                )
                if await cursor.fetchone():
                    await conn.execute(
                        "UPDATE media_offers SET status = 'cancelled' WHERE id = ?",
                        (parsed_id,),
                    )
                    return None
                cursor = await conn.execute(
                    """
                    UPDATE media_offers
                    SET status = 'delivered', offered_at = ?
                    WHERE id = ? AND status = 'reserved'
                    RETURNING *
                    """,
                    (now, parsed_id),
                )
                row = await cursor.fetchone()
                if row is None:
                    cursor = await conn.execute(
                        "SELECT * FROM media_offers WHERE id = ? AND status = 'delivered'",
                        (parsed_id,),
                    )
                    row = await cursor.fetchone()
                    return _offer_from_row(row) if row else None

                offer = _offer_from_row(row)
                if offer.trigger == "proactive":
                    await conn.execute(
                        """
                        UPDATE engagement_state
                        SET last_proactive_media_batch = ?
                        WHERE user_id = ?
                        """,
                        (offer.batch_number, offer.user_id),
                    )
                if offer.request_type == "generic" and media_type in {"photo", "video"}:
                    await conn.execute(
                        """
                        UPDATE engagement_state
                        SET last_generic_media_type = ?
                        WHERE user_id = ?
                        """,
                        (media_type, offer.user_id),
                    )
                if offer.trigger in {"direct", "permission_reask"}:
                    await conn.execute(
                        """
                        UPDATE engagement_state
                        SET sales_snooze_until_batch = NULL,
                            sales_reask_pending = FALSE,
                            sales_reask_asked_at_batch = NULL
                        WHERE user_id = ?
                        """,
                        (offer.user_id,),
                    )
                return offer
        finally:
            await conn.close()

    async def set_last_generic_media_type(self, user_id: int, media_type: str) -> None:
        if media_type not in {"photo", "video"}:
            raise ValueError("media_type must be photo or video")
        conn = await get_connection()
        try:
            await conn.execute(
                "UPDATE engagement_state SET last_generic_media_type = ? WHERE user_id = ?",
                (media_type, user_id),
            )
        finally:
            await conn.close()

    async def cancel_offer_reservation(self, offer_id: int | str) -> bool:
        try:
            parsed_id = int(offer_id)
        except (TypeError, ValueError):
            return False
        conn = await get_connection()
        try:
            cursor = await conn.execute(
                """
                UPDATE media_offers
                SET status = 'cancelled'
                WHERE id = ? AND status = 'reserved'
                RETURNING id
                """,
                (parsed_id,),
            )
            return await cursor.fetchone() is not None
        finally:
            await conn.close()

    async def _ensure_wallet(self, conn, user_id: int, now: float) -> None:
        cursor = await conn.execute(
            """
            INSERT INTO demo_wallets (user_id, balance, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO NOTHING
            RETURNING user_id
            """,
            (user_id, config.DEMO_WALLET_INITIAL_TOKENS, now, now),
        )
        created = await cursor.fetchone()
        if created:
            await conn.execute(
                """
                INSERT INTO demo_token_transactions
                    (user_id, idempotency_key, kind, amount_tokens,
                     balance_after, created_at)
                VALUES (?, ?, 'initial_credit', ?, ?, ?)
                ON CONFLICT(user_id, idempotency_key) DO NOTHING
                """,
                (
                    user_id,
                    "initial-credit-v1",
                    config.DEMO_WALLET_INITIAL_TOKENS,
                    config.DEMO_WALLET_INITIAL_TOKENS,
                    now,
                ),
            )

    async def get_wallet(self, user_id: int) -> WalletSnapshot:
        now = time.time()
        conn = await get_connection()
        try:
            async with conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock(?)", (user_id,))
                await self._ensure_wallet(conn, user_id, now)
                cursor = await conn.execute(
                    "SELECT balance FROM demo_wallets WHERE user_id = ?", (user_id,)
                )
                row = await cursor.fetchone()
                return WalletSnapshot(user_id=user_id, balance=int(row["balance"]))
        finally:
            await conn.close()

    async def refill_wallet(self, user_id: int, idempotency_key: str) -> TokenReceipt:
        key = _idempotency_key(idempotency_key)
        now = time.time()
        conn = await get_connection()
        try:
            async with conn.transaction():
                await self._ensure_wallet(conn, user_id, now)
                cursor = await conn.execute(
                    "SELECT balance FROM demo_wallets WHERE user_id = ? FOR UPDATE",
                    (user_id,),
                )
                wallet = await cursor.fetchone()
                cursor = await conn.execute(
                    """
                    SELECT id, kind, amount_tokens FROM demo_token_transactions
                    WHERE user_id = ? AND idempotency_key = ?
                    """,
                    (user_id, key),
                )
                existing = await cursor.fetchone()
                if existing:
                    if existing["kind"] != "refill":
                        raise IdempotencyConflictError(
                            "Idempotency key was used for another operation"
                        )
                    return TokenReceipt(
                        user_id=user_id,
                        balance=int(wallet["balance"]),
                        transaction_id=int(existing["id"]),
                        amount_tokens=int(existing["amount_tokens"]),
                        idempotent_replay=True,
                    )

                new_balance = int(wallet["balance"]) + config.DEMO_WALLET_REFILL_TOKENS
                await conn.execute(
                    "UPDATE demo_wallets SET balance = ?, updated_at = ? WHERE user_id = ?",
                    (new_balance, now, user_id),
                )
                cursor = await conn.execute(
                    """
                    INSERT INTO demo_token_transactions
                        (user_id, idempotency_key, kind, amount_tokens,
                         balance_after, created_at)
                    VALUES (?, ?, 'refill', ?, ?, ?)
                    RETURNING id
                    """,
                    (
                        user_id,
                        key,
                        config.DEMO_WALLET_REFILL_TOKENS,
                        new_balance,
                        now,
                    ),
                )
                transaction = await cursor.fetchone()
                return TokenReceipt(
                    user_id=user_id,
                    balance=new_balance,
                    transaction_id=int(transaction["id"]),
                    amount_tokens=config.DEMO_WALLET_REFILL_TOKENS,
                )
        finally:
            await conn.close()

    async def unlock_offer(
        self,
        user_id: int,
        offer_id: int | str,
        idempotency_key: str,
        *,
        tag_pairs: Iterable[tuple[str, str]] = (),
    ) -> UnlockReceipt:
        """Atomically debit once and grant one permanent entitlement."""
        try:
            parsed_offer_id = int(offer_id)
        except (TypeError, ValueError) as exc:
            raise OfferNotFoundError("Offer was not found") from exc
        key = _idempotency_key(idempotency_key)
        now = time.time()
        pairs = tuple(dict.fromkeys(tag_pairs))
        conn = await get_connection()
        try:
            async with conn.transaction():
                await self._ensure_wallet(conn, user_id, now)
                cursor = await conn.execute(
                    "SELECT balance FROM demo_wallets WHERE user_id = ? FOR UPDATE",
                    (user_id,),
                )
                wallet = await cursor.fetchone()

                cursor = await conn.execute(
                    "SELECT * FROM media_offers WHERE id = ? FOR UPDATE",
                    (parsed_offer_id,),
                )
                offer_row = await cursor.fetchone()
                if offer_row is None:
                    raise OfferNotFoundError("Offer was not found")
                if int(offer_row["user_id"]) != user_id:
                    # Do not disclose another demo user's offer metadata.
                    raise ForbiddenCommerceError("Offer does not belong to this user")
                if offer_row["status"] != "delivered":
                    raise MediaUnavailableError("Offer is not available to unlock")

                content_id = str(offer_row["content_id"])
                cursor = await conn.execute(
                    """
                    SELECT * FROM media_entitlements
                    WHERE user_id = ? AND content_id = ?
                    """,
                    (user_id, content_id),
                )
                entitlement = await cursor.fetchone()
                if entitlement:
                    return UnlockReceipt(
                        user_id=user_id,
                        offer_id=int(entitlement["source_offer_id"]),
                        content_id=content_id,
                        balance=int(wallet["balance"]),
                        charged_tokens=0,
                        transaction_id=int(entitlement["unlock_transaction_id"]),
                        unlocked_at=float(entitlement["unlocked_at"]),
                        already_unlocked=True,
                    )

                cursor = await conn.execute(
                    """
                    SELECT id, kind, offer_id, content_id FROM demo_token_transactions
                    WHERE user_id = ? AND idempotency_key = ?
                    """,
                    (user_id, key),
                )
                existing = await cursor.fetchone()
                if existing:
                    if (
                        existing["kind"] != "unlock"
                        or int(existing["offer_id"]) != parsed_offer_id
                        or str(existing["content_id"]) != content_id
                    ):
                        raise IdempotencyConflictError(
                            "Idempotency key was used for another operation"
                        )
                    # A committed unlock transaction always has an entitlement;
                    # seeing otherwise indicates damaged state, not permission
                    # to debit again.
                    raise IdempotencyConflictError(
                        "Unlock ledger exists without an entitlement"
                    )

                price = int(offer_row["price_tokens"])
                balance = int(wallet["balance"])
                if balance < price:
                    raise InsufficientTokensError(balance=balance, required=price)
                new_balance = balance - price
                await conn.execute(
                    "UPDATE demo_wallets SET balance = ?, updated_at = ? WHERE user_id = ?",
                    (new_balance, now, user_id),
                )
                cursor = await conn.execute(
                    """
                    INSERT INTO demo_token_transactions
                        (user_id, idempotency_key, kind, amount_tokens,
                         balance_after, offer_id, content_id, created_at)
                    VALUES (?, ?, 'unlock', ?, ?, ?, ?, ?)
                    RETURNING id
                    """,
                    (user_id, key, -price, new_balance, parsed_offer_id, content_id, now),
                )
                transaction = await cursor.fetchone()
                transaction_id = int(transaction["id"])
                await conn.execute(
                    """
                    INSERT INTO media_entitlements
                        (user_id, content_id, source_offer_id,
                         unlock_transaction_id, unlocked_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (user_id, content_id, parsed_offer_id, transaction_id, now),
                )
                # A planner may have reserved this item just before the unlock.
                # It must never later finalize into a newly locked resale card.
                await conn.execute(
                    """
                    UPDATE media_offers SET status = 'cancelled'
                    WHERE user_id = ? AND content_id = ? AND status = 'reserved'
                    """,
                    (user_id, content_id),
                )
                for group, value in pairs:
                    await conn.execute(
                        """
                        INSERT INTO media_tag_affinity
                            (user_id, tag_group, tag_value, score, updated_at)
                        VALUES (?, ?, ?, 0.25, ?)
                        ON CONFLICT(user_id, tag_group, tag_value) DO UPDATE SET
                            score = media_tag_affinity.score + 0.25,
                            updated_at = EXCLUDED.updated_at
                        """,
                        (user_id, group, value, now),
                    )
                return UnlockReceipt(
                    user_id=user_id,
                    offer_id=parsed_offer_id,
                    content_id=content_id,
                    balance=new_balance,
                    charged_tokens=price,
                    transaction_id=transaction_id,
                    unlocked_at=now,
                )
        finally:
            await conn.close()

    async def has_entitlement(self, user_id: int, content_id: str) -> bool:
        conn = await get_connection()
        try:
            cursor = await conn.execute(
                """
                SELECT 1 FROM media_entitlements
                WHERE user_id = ? AND content_id = ?
                """,
                (user_id, content_id),
            )
            return await cursor.fetchone() is not None
        finally:
            await conn.close()

    async def list_entitlements(self, user_id: int) -> list[EntitlementRecord]:
        conn = await get_connection()
        try:
            cursor = await conn.execute(
                """
                SELECT * FROM media_entitlements
                WHERE user_id = ?
                ORDER BY unlocked_at DESC, id DESC
                """,
                (user_id,),
            )
            return [_entitlement_from_row(row) for row in await cursor.fetchall()]
        finally:
            await conn.close()

    async def mark_media_opened(self, user_id: int, content_id: str) -> bool:
        now = time.time()
        conn = await get_connection()
        try:
            cursor = await conn.execute(
                """
                UPDATE media_entitlements
                SET first_opened_at = COALESCE(first_opened_at, ?)
                WHERE user_id = ? AND content_id = ?
                RETURNING id
                """,
                (now, user_id, content_id),
            )
            return await cursor.fetchone() is not None
        finally:
            await conn.close()

    async def reset_user_commerce(self, user_id: int) -> None:
        """Dev-only destructive reset; normal chat reset must never call this."""
        conn = await get_connection()
        try:
            async with conn.transaction():
                for table in (
                    "media_tag_affinity",
                    "media_entitlements",
                    "demo_token_transactions",
                    "media_offers",
                    "demo_wallets",
                ):
                    await conn.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
                await conn.execute(
                    """
                    UPDATE engagement_state
                    SET sales_snooze_until_batch = NULL,
                        sales_reask_pending = FALSE,
                        sales_reask_asked_at_batch = NULL,
                        last_proactive_media_batch = NULL,
                        last_generic_media_type = NULL
                    WHERE user_id = ?
                    """,
                    (user_id,),
                )
        finally:
            await conn.close()
