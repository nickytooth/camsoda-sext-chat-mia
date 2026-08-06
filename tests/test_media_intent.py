import unittest

from bot.media_planner import (
    classify_media_intent,
    classify_media_intent_batch,
)


class DirectMediaIntentTests(unittest.TestCase):
    def test_high_confidence_english_requests(self):
        cases = {
            "send me a picture": "photo",
            "show me a video": "video",
            "give me a nude": None,
            "drop me a pic": "photo",
            "share a clip with me": "video",
            "can I get a selfie?": "photo",
            "I want a photo": "photo",
            "I wanna see your ass": None,
            "I need a video": "video",
            "do you have any videos?": "video",
            "any nudes?": None,
            "how much for a video?": "video",
            "picture?": "photo",
            "video?": "video",
            "show me": None,
            "let me see you": None,
            "show me what you're wearing": None,
            "I wish I could see you right now": None,
            "send me some content": None,
        }
        for text, requested_type in cases.items():
            with self.subTest(text=text):
                intent = classify_media_intent(text)
                self.assertTrue(intent.requested)
                self.assertEqual(intent.requested_type, requested_type)

    def test_non_english_script_media_phrases_are_unsupported(self):
        for text in (
            "прати ми снимка",
            "покажи ми видео",
            "може ли да получа клип?",
            "искам голи снимки",
            "имаш ли видеа?",
            "колко струва едно видео?",
            "снимка?",
            "видео?",
            "покажи ми се",
            "прати ми контент",
            "أرسل send nudes",
            "发送 send nudes",
        ):
            with self.subTest(text=text):
                intent = classify_media_intent(text)
                self.assertFalse(intent.requested)
                self.assertTrue(intent.blocked_request)
                self.assertIsNone(intent.requested_type)
                self.assertIsNone(intent.explicitness)

    def test_clear_desire_is_a_request_but_flirting_is_not(self):
        for text in ("I wish I could see you right now",):
            with self.subTest(text=text):
                self.assertTrue(classify_media_intent(text).requested)

        for text in (
            "I bet you look hot",
            "you probably look amazing right now",
            "your pictures are hot",
        ):
            with self.subTest(text=text):
                self.assertFalse(classify_media_intent(text).requested)


class ContextualMediaIntentTests(unittest.TestCase):
    def test_ellipsis_requires_explicit_recent_media_context(self):
        phrases = (
            "another one",
            "more",
            "show me another",
            "what about a video?",
            "another photo",
        )
        for text in phrases:
            with self.subTest(text=text):
                without_context = classify_media_intent_batch([text])
                with_context = classify_media_intent_batch(
                    [text],
                    recent_media_context=True,
                )
                self.assertFalse(without_context.requested)
                self.assertTrue(with_context.requested)

    def test_contextual_refinement_keeps_requested_type(self):
        cases = {
            "what about a video?": "video",
            "a photo instead": "photo",
            "show me another video": "video",
        }
        for text, requested_type in cases.items():
            with self.subTest(text=text):
                intent = classify_media_intent_batch(
                    [text],
                    recent_media_context=True,
                )
                self.assertTrue(intent.requested)
                self.assertEqual(intent.requested_type, requested_type)

    def test_permission_affirmatives_are_not_generic_recent_media_requests(self):
        self.assertFalse(classify_media_intent_batch(["yes"]).requested)
        self.assertFalse(
            classify_media_intent_batch(
                ["yes"], recent_media_context=True
            ).requested
        )
        for text in ("send it", "show it", "do it"):
            with self.subTest(text=text):
                self.assertFalse(
                    classify_media_intent_batch(
                        [text], recent_media_context=True
                    ).requested
                )
        self.assertTrue(classify_media_intent("yes, show me").affirmative)

    def test_final_decline_overrides_an_earlier_affirmative(self):
        for text in (
            "yes, no",
            "yes, but no",
            "yeah actually nope",
            "sure, never mind",
            "okay, I changed my mind",
        ):
            with self.subTest(text=text):
                intent = classify_media_intent(text)
                self.assertFalse(intent.requested)
                self.assertFalse(intent.affirmative)
                self.assertEqual(intent.decline_kind, "soft")


class OrderedBatchIntentTests(unittest.TestCase):
    def test_last_decisive_request_or_decline_wins(self):
        declined = classify_media_intent_batch(
            ["send me a photo", "haha", "no, not now", "thanks"]
        )
        self.assertFalse(declined.requested)
        self.assertEqual(declined.decline_kind, "soft")

        requested = classify_media_intent_batch(
            ["don't send me a photo", "wait", "actually show me a video"]
        )
        self.assertTrue(requested.requested)
        self.assertEqual(requested.requested_type, "video")
        self.assertIsNone(requested.decline_kind)

    def test_later_request_type_wins(self):
        intent = classify_media_intent_batch(
            ["send a photo", "actually send a video", "please"]
        )
        self.assertTrue(intent.requested)
        self.assertEqual(intent.requested_type, "video")

    def test_later_disallowed_visual_command_cancels_an_earlier_offer(self):
        for last_message in (
            "show me a picture of the menu",
            'Tyler said "send nudes"',
        ):
            with self.subTest(last_message=last_message):
                intent = classify_media_intent_batch(
                    ["send me a nude", last_message]
                )
                self.assertFalse(intent.requested)
                self.assertTrue(intent.blocked_request)

    def test_unsupported_script_cancels_an_earlier_batch_request(self):
        intent = classify_media_intent_batch(
            ["send me a photo", "прати send nudes"]
        )
        self.assertFalse(intent.requested)
        self.assertTrue(intent.blocked_request)
        self.assertIsNone(intent.requested_type)

    def test_raw_message_count_does_not_change_batch_semantics(self):
        messages = ["hello"] * 199 + ["send me a picture"]
        intent = classify_media_intent_batch(messages)
        self.assertTrue(intent.requested)
        self.assertEqual(intent.requested_type, "photo")

    def test_string_is_rejected_instead_of_iterated_as_characters(self):
        with self.assertRaises(TypeError):
            classify_media_intent_batch("send me a photo")


class FailClosedMediaIntentTests(unittest.TestCase):
    def test_negated_requests_never_request_media(self):
        for text in (
            "don't send me a photo",
            "I do not want nudes",
            "no more videos",
            "send me a photo, no",
        ):
            with self.subTest(text=text):
                intent = classify_media_intent(text)
                self.assertFalse(intent.requested)
                self.assertEqual(intent.decline_kind, "soft")

    def test_quotes_and_reported_commands_are_not_requests(self):
        for text in (
            'He said "send nudes"',
            '"send me a photo"',
            "'send me a photo'",
            "`show me a video`",
            'Tyler told me to ask "show me a video"',
            "she told me to send a pic",
        ):
            with self.subTest(text=text):
                self.assertFalse(classify_media_intent(text).requested)

    def test_past_media_mentions_are_not_requests(self):
        for text in (
            "I watched a video",
            "we saw your photos last night",
            "I asked you to send me a picture yesterday",
            "Yesterday I wanted you to show me a video",
        ):
            with self.subTest(text=text):
                self.assertFalse(classify_media_intent(text).requested)

    def test_external_object_media_requests_are_not_mia_media_requests(self):
        for text in (
            "send me a picture of the menu",
            "show me a photo of the bar",
            "can I have a picture of your dog?",
            "send me a video of the menu",
            "show me a clip of the bar",
        ):
            with self.subTest(text=text):
                self.assertFalse(classify_media_intent(text).requested)

    def test_mia_focused_photo_targets_still_work(self):
        for text in (
            "send me a picture of you",
            "show me a photo of your ass",
            "send a picture of your outfit",
        ):
            with self.subTest(text=text):
                self.assertTrue(classify_media_intent(text).requested)


if __name__ == "__main__":
    unittest.main()
