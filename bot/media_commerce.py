"""High-level visual paywall service used by chat and HTTP/WebSocket APIs."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from bot import config
from bot.media_catalog import MediaCatalog, MediaItem, load_media_catalog
from bot.media_planner import (
    CatalogPlanner,
    MediaIntent,
    classify_media_intent,
    last_relevant_locked_offer,
)
from bot.media_repository import (
    CommerceError,
    EntitlementRecord,
    ForbiddenCommerceError,
    IdempotencyConflictError,
    InsufficientTokensError,
    MediaRepository,
    MediaUnavailableError,
    OfferNotFoundError,
    OfferRecord,
    TokenReceipt,
    UnlockReceipt,
    WalletSnapshot,
)
from bot.token_service import DemoTokenService, TokenService


class CommerceAction(str, Enum):
    OFFER_CURRENT = "offer_current"
    OFFER_FALLBACK = "offer_fallback"
    REACT_TO_DECLINE = "react_to_decline"
    ASK_PERMISSION_AGAIN = "ask_permission_again"
    ACKNOWLEDGE_UNLOCK = "acknowledge_unlock"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class MediaOffer:
    """Safe offer metadata.  It can never contain an R2 key or URL."""

    offer_id: int
    content_id: str
    media_type: str
    price_tokens: int
    aspect_ratio: float
    duration_seconds: float | None
    explicitness: str
    description: str
    trigger: str
    action: str
    request_type: str | None
    offered_at: float | None = None
    unlocked: bool = False


@dataclass(frozen=True, slots=True)
class CommerceDecision:
    action: CommerceAction = CommerceAction.NONE
    brief: str = ""
    offer: MediaOffer | None = None
    # Internal state needed to commit a non-offer action only after its visible
    # assistant response has been persisted. Prompt serialization is allowlist-
    # based and never includes these fields.
    user_id: int | None = None
    batch_number: int | None = None
    decline_kind: str | None = None


@dataclass(frozen=True, slots=True)
class GalleryItem:
    content_id: str
    media_type: str
    mime_type: str
    aspect_ratio: float
    duration_seconds: float | None
    explicitness: str
    description: str
    unlocked_at: float
    first_opened_at: float | None


def _price_for(item: MediaItem) -> int:
    if item.media_type == "photo":
        return config.MEDIA_PHOTO_PRICE_TOKENS
    if item.media_type == "video":
        return config.MEDIA_VIDEO_PRICE_TOKENS
    raise MediaUnavailableError("Unsupported media type")


class MediaCommerceService:
    """Coordinates pacing, deterministic selection, wallet and entitlement state."""

    def __init__(
        self,
        catalog: MediaCatalog,
        *,
        repository: MediaRepository | None = None,
        token_service: TokenService | None = None,
        random_source: random.Random | None = None,
    ):
        self.catalog = catalog
        self.repository = repository or MediaRepository()
        self.token_service = token_service or DemoTokenService(self.repository)
        self.planner = CatalogPlanner(catalog, self.repository)
        self._random = random_source or random.SystemRandom()

    def _safe_offer(
        self,
        record: OfferRecord,
        *,
        item: MediaItem | None = None,
        unlocked: bool = False,
    ) -> MediaOffer:
        media = item or self.catalog.require(record.content_id)
        description = record.description or media.description_for_period("")[0]
        return MediaOffer(
            offer_id=record.offer_id,
            content_id=record.content_id,
            media_type=media.media_type,
            price_tokens=record.price_tokens,
            aspect_ratio=media.aspect_ratio,
            duration_seconds=media.duration_seconds,
            explicitness=media.explicitness,
            description=description,
            trigger=record.trigger,
            action=record.action,
            request_type=record.request_type,
            offered_at=record.offered_at,
            unlocked=unlocked,
        )

    async def _reserve_planned(
        self,
        user_id: int,
        intent: MediaIntent,
        *,
        batch_number: int,
        heat: str,
        period: str,
        trigger: str,
        state: dict[str, object],
    ) -> CommerceDecision:
        planned = await self.planner.choose(
            user_id,
            intent,
            batch_number=batch_number,
            heat=heat,
            period=period,
            trigger=trigger,
            state=state,
        )
        if planned is None:
            return CommerceDecision()
        try:
            record = await self.repository.reserve_offer(
                user_id,
                planned.item.id,
                trigger=trigger,
                action=planned.action,
                request_type=planned.request_type,
                description=planned.description,
                batch_number=batch_number,
                price_tokens=_price_for(planned.item),
            )
        except MediaUnavailableError:
            # A concurrent planner or unlock won the per-user reservation race.
            # Failing closed is safer than attaching a duplicate/resold card.
            return CommerceDecision()
        if planned.action == CommerceAction.OFFER_CURRENT.value:
            brief = f"Offer this real {planned.item.media_type}: {planned.description}."
        else:
            reason = planned.fallback_reason or "There is no unopened exact-current match."
            brief = (
                f"Current context: {reason}. Offer this real alternative as an older or "
                f"different-location {planned.item.media_type}: {planned.description}."
            )
        offer = MediaOffer(
            offer_id=record.offer_id,
            content_id=planned.item.id,
            media_type=planned.item.media_type,
            price_tokens=record.price_tokens,
            aspect_ratio=planned.item.aspect_ratio,
            duration_seconds=planned.item.duration_seconds,
            explicitness=planned.item.explicitness,
            description=planned.description,
            trigger=trigger,
            action=planned.action,
            request_type=planned.request_type,
        )
        return CommerceDecision(
            action=CommerceAction(planned.action),
            brief=brief,
            offer=offer,
            user_id=user_id,
            batch_number=batch_number,
        )

    async def plan_commerce_turn(
        self,
        user_id: int,
        text: str,
        *,
        batch_number: int,
        heat: str,
        period: str,
    ) -> CommerceDecision:
        """Return at most one trusted action for this processed user batch."""
        await self.repository.cancel_stale_reservations(user_id)
        intent = classify_media_intent(text)
        state = await self.repository.get_engagement_state(user_id)

        pending = bool(state.get("sales_reask_pending"))
        asked_at_raw = state.get("sales_reask_asked_at_batch")
        asked_at = int(asked_at_raw) if asked_at_raw is not None else None

        # A negative command containing the word photo/video is still a
        # refusal, never a direct request.
        if intent.requested and intent.decline_kind is None:
            # The user's direct request itself revokes any previous pause even
            # when inventory is exhausted or the later teaser fails.
            await self.repository.clear_sales_pause(user_id)
            state.update(
                sales_snooze_until_batch=None,
                sales_reask_pending=False,
                sales_reask_asked_at_batch=None,
            )
            return await self._reserve_planned(
                user_id,
                intent,
                batch_number=batch_number,
                heat=heat,
                period=period,
                trigger="direct",
                state=state,
            )

        history = await self.repository.delivered_offer_history(user_id)
        unlocked = await self.repository.unlocked_content_ids(user_id)
        latest_locked = last_relevant_locked_offer(
            history, unlocked, batch_number=batch_number
        )
        if intent.decline_kind and (
            latest_locked is not None
            or intent.decline_global
            or (pending and asked_at is not None)
        ):
            return CommerceDecision(
                action=CommerceAction.REACT_TO_DECLINE,
                brief=(
                    "He declined the latest visual offer. Accept the no once with mild "
                    "surprise or disappointment and do not attach media."
                ),
                user_id=user_id,
                batch_number=batch_number,
                decline_kind=intent.decline_kind,
            )

        # With no recent commerce context, a phrase such as "not now" may be
        # declining something else, so do not invent a sales reaction or pause.
        # It is still a negative-classified turn and must never fall through to
        # the proactive-offer branch below.
        if intent.decline_kind:
            return CommerceDecision()

        if pending and asked_at is not None:
            if batch_number == asked_at + 1 and intent.affirmative:
                permission_intent = MediaIntent(
                    requested=True,
                    requested_type=intent.requested_type,
                    tags=intent.tags,
                    explicitness=intent.explicitness,
                    affirmative=True,
                )
                return await self._reserve_planned(
                    user_id,
                    permission_intent,
                    batch_number=batch_number,
                    heat=heat,
                    period=period,
                    trigger="permission_reask",
                    state=state,
                )
            # Permission applies to the immediate response only; an unrelated
            # turn cannot make a later generic "yes" purchase media by accident.
            # Keep sales paused and schedule a fresh soft check instead of
            # immediately allowing proactive cards again.
            delay = self._random.randint(
                config.MEDIA_SOFT_DECLINE_MIN_BATCHES,
                config.MEDIA_SOFT_DECLINE_MAX_BATCHES,
            )
            await self.repository.set_decline_snooze(
                user_id,
                until_batch=batch_number + delay,
                reask_pending=True,
            )
            return CommerceDecision()

        snooze_raw = state.get("sales_snooze_until_batch")
        snooze_until = int(snooze_raw) if snooze_raw is not None else None
        heat_is_ready = heat in {"rising", "high"}
        if pending:
            if snooze_until is None or batch_number < snooze_until or not heat_is_ready:
                return CommerceDecision()
            return CommerceDecision(
                action=CommerceAction.ASK_PERMISSION_AGAIN,
                brief="Softly ask whether he wants to see you now. No media card is attached.",
                user_id=user_id,
                batch_number=batch_number,
            )

        # Any legacy/inconsistent snooze without a pending re-ask remains a
        # sales pause until its threshold, then clears harmlessly.
        if snooze_until is not None:
            if batch_number < snooze_until:
                return CommerceDecision()
            await self.repository.clear_sales_pause(user_id)

        if not heat_is_ready or batch_number < config.MEDIA_PROACTIVE_MIN_BATCHES:
            return CommerceDecision()
        last_proactive_raw = state.get("last_proactive_media_batch")
        if last_proactive_raw is not None:
            if (
                batch_number - int(last_proactive_raw)
                < config.MEDIA_PROACTIVE_COOLDOWN_BATCHES
            ):
                return CommerceDecision()
        proactive_intent = MediaIntent(requested=True)
        return await self._reserve_planned(
            user_id,
            proactive_intent,
            batch_number=batch_number,
            heat=heat,
            period=period,
            trigger="proactive",
            state=state,
        )

    async def mark_commerce_action_delivered(self, decision: CommerceDecision) -> bool:
        """Commit non-offer pacing only after the visible reply is persisted."""
        if decision.user_id is None or decision.batch_number is None:
            return False
        if decision.action == CommerceAction.REACT_TO_DECLINE:
            state = await self.repository.get_engagement_state(decision.user_id)
            existing = state.get("sales_snooze_until_batch")
            # Idempotent replay of an already-applied same-kind decision must
            # not pick another random threshold. A hard refusal, however,
            # always extends an older/shorter soft pause to the full 100.
            desired_hard_until = (
                decision.batch_number + config.MEDIA_HARD_DECLINE_SNOOZE_BATCHES
            )
            if (
                existing is not None
                and int(existing) > decision.batch_number
                and not (
                    decision.decline_kind == "hard"
                    and int(existing) < desired_hard_until
                )
            ):
                return True
            if decision.decline_kind == "hard":
                delay = config.MEDIA_HARD_DECLINE_SNOOZE_BATCHES
            else:
                delay = self._random.randint(
                    config.MEDIA_SOFT_DECLINE_MIN_BATCHES,
                    config.MEDIA_SOFT_DECLINE_MAX_BATCHES,
                )
            await self.repository.set_decline_snooze(
                decision.user_id,
                until_batch=decision.batch_number + delay,
                reask_pending=True,
            )
            return True
        if decision.action == CommerceAction.ASK_PERMISSION_AGAIN:
            await self.repository.mark_reask_asked(
                decision.user_id, batch_number=decision.batch_number
            )
            return True
        return decision.action in {CommerceAction.NONE, CommerceAction.ACKNOWLEDGE_UNLOCK}

    async def cancel_commerce_action(self, decision: CommerceDecision) -> bool:
        """Non-offer planning is side-effect free, so cancellation is a no-op."""
        return decision.action not in {
            CommerceAction.OFFER_CURRENT,
            CommerceAction.OFFER_FALLBACK,
        }

    async def mark_offer_delivered(self, offer_id: int | str) -> MediaOffer | None:
        pending = await self.repository.get_offer(offer_id)
        if pending is None:
            return None
        item = self.catalog.get(pending.content_id)
        if item is None or not item.active:
            await self.repository.cancel_offer_reservation(offer_id)
            return None
        record = await self.repository.mark_offer_delivered(
            offer_id, media_type=item.media_type
        )
        return self._safe_offer(record, item=item) if record else None

    async def cancel_offer_reservation(self, offer_id: int | str) -> bool:
        return await self.repository.cancel_offer_reservation(offer_id)

    async def get_wallet(self, user_id: int) -> WalletSnapshot:
        return await self.token_service.get_wallet(user_id)

    async def refill_wallet(
        self, user_id: int, idempotency_key: str
    ) -> TokenReceipt:
        return await self.token_service.refill_wallet(user_id, idempotency_key)

    async def unlock_offer(
        self, user_id: int, offer_id: int | str, idempotency_key: str
    ) -> UnlockReceipt:
        offer = await self.repository.get_offer(
            offer_id, user_id=user_id, delivered_only=True
        )
        if offer is None:
            # Deliberately same 404-style result for nonexistent and other-user
            # offers at this boundary.
            raise OfferNotFoundError("Offer was not found")
        item = self.catalog.get(offer.content_id)
        if item is None or not item.active:
            raise MediaUnavailableError("Media is no longer available")
        tag_pairs = tuple(
            (group, value) for group, values in item.tags.items() for value in values
        )
        return await self.token_service.unlock_offer(
            user_id,
            offer.offer_id,
            idempotency_key,
            tag_pairs=tag_pairs,
        )

    async def list_delivered_offers(self, user_id: int) -> list[MediaOffer]:
        entitlements = await self.repository.unlocked_content_ids(user_id)
        result: list[MediaOffer] = []
        for record in await self.repository.delivered_offer_history(user_id):
            item = self.catalog.get(record.content_id)
            if item is not None:
                result.append(
                    self._safe_offer(
                        record, item=item, unlocked=record.content_id in entitlements
                    )
                )
        return result

    async def get_delivered_offer(
        self, user_id: int, offer_id: int | str
    ) -> MediaOffer | None:
        record = await self.repository.get_offer(
            offer_id, user_id=user_id, delivered_only=True
        )
        if record is None:
            return None
        item = self.catalog.get(record.content_id)
        if item is None:
            return None
        unlocked = await self.repository.has_entitlement(user_id, item.id)
        return self._safe_offer(record, item=item, unlocked=unlocked)

    async def list_gallery(self, user_id: int) -> list[GalleryItem]:
        result: list[GalleryItem] = []
        for entitlement in await self.repository.list_entitlements(user_id):
            item = self.catalog.get(entitlement.content_id)
            if item is None:
                continue
            description, _ = item.description_for_period("")
            result.append(
                GalleryItem(
                    content_id=item.id,
                    media_type=item.media_type,
                    mime_type=item.mime_type,
                    aspect_ratio=item.aspect_ratio,
                    duration_seconds=item.duration_seconds,
                    explicitness=item.explicitness,
                    description=description,
                    unlocked_at=entitlement.unlocked_at,
                    first_opened_at=entitlement.first_opened_at,
                )
            )
        return result

    async def has_entitlement(self, user_id: int, content_id: str) -> bool:
        if self.catalog.get(content_id) is None:
            return False
        return await self.repository.has_entitlement(user_id, content_id)

    async def mark_media_opened(self, user_id: int, content_id: str) -> bool:
        if self.catalog.get(content_id) is None:
            raise MediaUnavailableError("Media was not found")
        marked = await self.repository.mark_media_opened(user_id, content_id)
        if not marked:
            raise MediaUnavailableError("Media is locked")
        return True

    async def reset_user_commerce(self, user_id: int) -> None:
        await self.repository.reset_user_commerce(user_id)


_service: MediaCommerceService | None = None


def get_media_commerce_service() -> MediaCommerceService:
    global _service
    if _service is None:
        _service = MediaCommerceService(load_media_catalog())
    return _service


__all__ = [
    "CommerceAction",
    "CommerceDecision",
    "CommerceError",
    "ForbiddenCommerceError",
    "GalleryItem",
    "IdempotencyConflictError",
    "InsufficientTokensError",
    "MediaCommerceService",
    "MediaOffer",
    "MediaUnavailableError",
    "OfferNotFoundError",
    "TokenReceipt",
    "UnlockReceipt",
    "WalletSnapshot",
    "get_media_commerce_service",
]
