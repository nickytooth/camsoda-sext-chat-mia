"""High-level visual paywall service used by chat and HTTP/WebSocket APIs."""

from __future__ import annotations

import random
import time
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
from bot.time_context import get_media_locations


class CommerceAction(str, Enum):
    OFFER_CURRENT = "offer_current"
    OFFER_FALLBACK = "offer_fallback"
    REACT_TO_DECLINE = "react_to_decline"
    ASK_PERMISSION_AGAIN = "ask_permission_again"
    ASK_MEDIA_CONFIRMATION = "ask_media_confirmation"
    CANCEL_MEDIA_CONFIRMATION = "cancel_media_confirmation"
    MEDIA_REQUEST_UNAVAILABLE = "media_request_unavailable"
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
    current_context: str = ""
    offered_item_description: str = ""
    # Internal state needed to commit a non-offer action only after its visible
    # assistant response has been persisted. Prompt serialization is allowlist-
    # based and never includes these fields.
    user_id: int | None = None
    batch_number: int | None = None
    decline_kind: str | None = None
    # A normalized direct request is staged only after the confirmation
    # question itself has been persisted. It is internal backend state and is
    # excluded from the allowlisted prompt block.
    pending_intent: MediaIntent | None = None
    # Canonical catalog/current locations stay inside the backend trust
    # boundary. They are used by the deterministic output guard and are never
    # serialized into the chat payload or exposed to the browser.
    item_locations: tuple[str, ...] = ()
    current_locations: tuple[str, ...] = ()


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

    @staticmethod
    def _confirmation_intent(record: object) -> MediaIntent:
        return MediaIntent(
            requested=True,
            requested_type=getattr(record, "requested_type", None),
            tags=getattr(record, "tags", {}) or {},
            explicitness=getattr(record, "explicitness", None),
        )

    @staticmethod
    def _merge_confirmed_intent(
        stored: MediaIntent, current: MediaIntent
    ) -> MediaIntent:
        """Let a repeated command refine the request that Mia confirmed."""

        tags = dict(stored.tags)
        tags.update(current.tags)
        return MediaIntent(
            requested=True,
            requested_type=current.requested_type or stored.requested_type,
            tags=tags,
            explicitness=current.explicitness or stored.explicitness,
            affirmative=True,
        )

    @staticmethod
    def _confirmed_selection_heat(intent: MediaIntent, heat: str) -> str:
        """Narrowly unlock high-only inventory for an explicit confirmed ask.

        A generic low-heat "picture?" must not silently become a nude offer.
        The override applies only when the normalized request itself names a
        nude/explicit level or an unambiguously explicit controlled facet. It
        affects catalog selection only and never advances conversational Heat.
        """

        explicit_activities = {"masturbating", "oral", "vaginal", "anal", "toy_play"}
        body_focus = set(intent.tags.get("body_focus", ()))
        outfits = set(intent.tags.get("outfit", ()))
        activities = set(intent.tags.get("activity", ()))
        explicitly_high = (
            intent.explicitness in {"nude", "explicit"}
            or bool(body_focus.intersection({"boobs", "pussy"}))
            or bool(outfits.intersection({"topless", "nude"}))
            or bool(activities.intersection(explicit_activities))
        )
        return "high" if explicitly_high else heat

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
            if trigger in {"direct", "confirmed_direct", "permission_reask"}:
                requested = intent.requested_type or "visual content"
                return CommerceDecision(
                    action=CommerceAction.MEDIA_REQUEST_UNAVAILABLE,
                    brief=(
                        f"He directly requested {requested}, but the backend cannot attach "
                        "an eligible unopened matching media card on this turn. Respond "
                        "briefly and naturally without claiming inventory, promising a file "
                        "later, bargaining, setting behavior conditions, or asking him to "
                        "earn it. No media card is attached."
                    ),
                    user_id=user_id,
                    batch_number=batch_number,
                )
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
            # Failing closed is safer than attaching a duplicate/resold card,
            # but a confirmed request must not lose all trusted commerce
            # context and leave the model free to invent inventory.
            if trigger in {"direct", "confirmed_direct", "permission_reask"}:
                return CommerceDecision(
                    action=CommerceAction.MEDIA_REQUEST_UNAVAILABLE,
                    brief=(
                        "The requested media could not be reserved on this turn. Respond "
                        "briefly without claiming a file exists, promising it later, "
                        "bargaining, or asking him to earn it. No media card is attached."
                    ),
                    user_id=user_id,
                    batch_number=batch_number,
                )
            return CommerceDecision()
        current_context = ""
        if planned.action == CommerceAction.OFFER_CURRENT.value:
            brief = f"Offer this real {planned.item.media_type}: {planned.description}."
        else:
            reason = planned.fallback_reason or "There is no unopened exact-current match."
            current_context = reason
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
            current_context=current_context,
            offered_item_description=planned.description,
            user_id=user_id,
            batch_number=batch_number,
            item_locations=tuple(planned.item.tags.get("location", ())),
            current_locations=tuple(get_media_locations(period)),
        )

    async def _persist_decline_pause(
        self,
        user_id: int,
        *,
        batch_number: int,
        decline_kind: str | None,
    ) -> None:
        """Durably apply a user refusal before generating the cosmetic reaction.

        The refusal itself is the authoritative event; losing its 30--40/100
        batch pause because the later assistant-text finalization failed would
        allow an early proactive offer. Replays are idempotent and a hard
        refusal upgrades an existing shorter soft pause.
        """

        state = await self.repository.get_engagement_state(user_id)
        existing = state.get("sales_snooze_until_batch")
        desired_hard_until = batch_number + config.MEDIA_HARD_DECLINE_SNOOZE_BATCHES
        if (
            existing is not None
            and int(existing) > batch_number
            and not (
                decline_kind == "hard" and int(existing) < desired_hard_until
            )
        ):
            return

        if decline_kind == "hard":
            delay = config.MEDIA_HARD_DECLINE_SNOOZE_BATCHES
        else:
            delay = self._random.randint(
                config.MEDIA_SOFT_DECLINE_MIN_BATCHES,
                config.MEDIA_SOFT_DECLINE_MAX_BATCHES,
            )
        await self.repository.set_decline_snooze(
            user_id,
            until_batch=batch_number + delay,
            reask_pending=True,
        )

    async def is_confirmed_decline(
        self,
        user_id: int,
        text: str,
        *,
        batch_number: int,
    ) -> bool:
        """Whether this turn is deterministically a visual-commerce refusal.

        Heat asks this before reducing the batch so a media-only ``not now``
        never cools a sexual conversation. Global/media-specific refusals are
        self-contained; bare negatives require a recent locked offer or a
        pending permission check, matching ``plan_commerce_turn`` exactly.
        """
        intent = classify_media_intent(text)
        if not intent.decline_kind:
            return False
        if intent.decline_global:
            return True

        confirmation = await self.repository.get_media_confirmation(user_id)
        if confirmation is not None and confirmation.status == "pending":
            return True

        state = await self.repository.get_engagement_state(user_id)
        pending = bool(state.get("sales_reask_pending"))
        asked_at_raw = state.get("sales_reask_asked_at_batch")
        asked_at = int(asked_at_raw) if asked_at_raw is not None else None
        if pending and asked_at is not None:
            return True

        history = await self.repository.delivered_offer_history(user_id)
        unlocked = await self.repository.unlocked_content_ids(user_id)
        return last_relevant_locked_offer(
            history,
            unlocked,
            batch_number=batch_number,
        ) is not None

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

        now = time.time()
        confirmation = await self.repository.get_media_confirmation(user_id, now=now)
        if (
            confirmation is not None
            and confirmation.status == "pending"
            and batch_number - confirmation.asked_at_batch
            > config.MEDIA_CONFIRMATION_MAX_BATCH_GAP
        ):
            await self.repository.clear_media_confirmation(user_id, pending_only=True)
            confirmation = None

        if confirmation is not None and confirmation.status == "pending":
            if intent.decline_kind:
                await self.repository.clear_media_confirmation(
                    user_id, pending_only=True
                )
                # A bare/simple no cancels this one confirmation without
                # creating the 30--40 batch sales snooze used for declining an
                # actual paywall offer. Global/hard wording still falls through
                # to the normal durable sales-pause rules below.
                if not intent.decline_global:
                    return CommerceDecision(
                        action=CommerceAction.CANCEL_MEDIA_CONFIRMATION,
                        brief=(
                            "He backed out of the visual request before any media card "
                            "was offered. Accept it naturally without pressure."
                        ),
                        user_id=user_id,
                        batch_number=batch_number,
                    )
                confirmation = None
            elif intent.affirmative or intent.requested:
                stored_intent = self._confirmation_intent(confirmation)
                confirmed_intent = (
                    self._merge_confirmed_intent(stored_intent, intent)
                    if intent.requested
                    else stored_intent
                )
                granted = await self.repository.grant_media_confirmation(
                    user_id,
                    batch_number=batch_number,
                    max_batch_gap=config.MEDIA_CONFIRMATION_MAX_BATCH_GAP,
                    granted_until=now + config.MEDIA_CONFIRMATION_GRANT_SECONDS,
                    now=now,
                )
                if granted is not None:
                    await self.repository.clear_sales_pause(user_id)
                    state.update(
                        sales_snooze_until_batch=None,
                        sales_reask_pending=False,
                        sales_reask_asked_at_batch=None,
                    )
                    # A confirmed *explicit* ask may select high-minimum
                    # inventory. Generic asks and all proactive offers still
                    # obey the real durable Heat stage.
                    return await self._reserve_planned(
                        user_id,
                        confirmed_intent,
                        batch_number=batch_number,
                        heat=self._confirmed_selection_heat(
                            confirmed_intent, heat
                        ),
                        period=period,
                        trigger="confirmed_direct",
                        state=state,
                    )
                confirmation = None
            else:
                # Keep the pending request for a small bounded number of
                # processed turns. An unrelated message is never treated as
                # permission and cannot accidentally attach a card.
                return CommerceDecision()

        sales_reask_pending = bool(state.get("sales_reask_pending"))
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
            if confirmation is not None and confirmation.status == "granted":
                # A durable grant carries the normalized explicit scope the
                # user already confirmed. A later type-only request refines
                # that scope instead of silently dropping it (for example,
                # confirmed nude content followed by "send me video").
                confirmed_intent = self._merge_confirmed_intent(
                    self._confirmation_intent(confirmation),
                    intent,
                )
                return await self._reserve_planned(
                    user_id,
                    confirmed_intent,
                    batch_number=batch_number,
                    heat=self._confirmed_selection_heat(confirmed_intent, heat),
                    period=period,
                    trigger="confirmed_direct",
                    state=state,
                )
            return CommerceDecision(
                action=CommerceAction.ASK_MEDIA_CONFIRMATION,
                brief=(
                    "He directly asked for visual content. React with surprised, visible "
                    "excitement and make sure he truly wants to cross that line before "
                    "anything is offered. No media card is attached yet."
                ),
                user_id=user_id,
                batch_number=batch_number,
                pending_intent=intent,
            )

        history = await self.repository.delivered_offer_history(user_id)
        unlocked = await self.repository.unlocked_content_ids(user_id)
        latest_locked = last_relevant_locked_offer(
            history, unlocked, batch_number=batch_number
        )
        if intent.decline_kind and (
            latest_locked is not None
            or intent.decline_global
            or (sales_reask_pending and asked_at is not None)
        ):
            # Persist the user's refusal now. The natural-language reaction is
            # cosmetic and may later be compensated, but the sales pause must
            # survive that failure and prevent an early proactive offer.
            await self._persist_decline_pause(
                user_id,
                batch_number=batch_number,
                decline_kind=intent.decline_kind,
            )
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

        if sales_reask_pending and asked_at is not None:
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
        if sales_reask_pending:
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
        """Finalize non-offer pacing after the visible reply is persisted."""
        if decision.user_id is None or decision.batch_number is None:
            return False
        if decision.action == CommerceAction.REACT_TO_DECLINE:
            # Planning already persisted the refusal before any model call.
            # Keep this hook idempotent for adapters/replays and for decisions
            # created outside the normal planner path.
            await self._persist_decline_pause(
                decision.user_id,
                batch_number=decision.batch_number,
                decline_kind=decision.decline_kind,
            )
            return True
        if decision.action == CommerceAction.ASK_PERMISSION_AGAIN:
            await self.repository.mark_reask_asked(
                decision.user_id, batch_number=decision.batch_number
            )
            return True
        if decision.action == CommerceAction.ASK_MEDIA_CONFIRMATION:
            if decision.pending_intent is None:
                return False
            now = time.time()
            await self.repository.stage_media_confirmation(
                decision.user_id,
                requested_type=decision.pending_intent.requested_type,
                tags=decision.pending_intent.tags,
                explicitness=decision.pending_intent.explicitness,
                asked_at_batch=decision.batch_number,
                expires_at=now + config.MEDIA_CONFIRMATION_TTL_SECONDS,
                now=now,
            )
            return True
        return decision.action in {
            CommerceAction.NONE,
            CommerceAction.CANCEL_MEDIA_CONFIRMATION,
            CommerceAction.ACKNOWLEDGE_UNLOCK,
        }

    async def cancel_commerce_action(self, decision: CommerceDecision) -> bool:
        """Cancel presentation only; a user's durable refusal is never undone."""
        return decision.action not in {
            CommerceAction.OFFER_CURRENT,
            CommerceAction.OFFER_FALLBACK,
        }

    async def clear_media_confirmation(self, user_id: int) -> bool:
        """Clear pending/session confirmation after Heat pause or chat reset."""

        return await self.repository.clear_media_confirmation(user_id)

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
