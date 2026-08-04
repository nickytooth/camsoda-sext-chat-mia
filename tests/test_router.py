import unittest

from bot.router import classify_fast, is_consent_withdrawal, withdrawn_acts


class RouterTests(unittest.TestCase):
    def test_consent_withdrawal_wins_over_nsfw_keyword(self):
        cases = {
            "stop choking me": ("choke",),
            "please stop choking me": ("choke",),
            "don't choke me": ("choke",),
            "don\u2019t choke me": ("choke",),
            "do not choke me": ("choke",),
            "don't ever choke me": ("choke",),
            "I don't want you to choke me": ("choke",),
            "no choking": ("choke",),
            "don't fuck me": ("fuck",),
            "don\u2019t fuck me": ("fuck",),
        }
        for text, acts in cases.items():
            with self.subTest(text=text):
                self.assertTrue(is_consent_withdrawal(text))
                self.assertEqual(withdrawn_acts(text), acts)
                self.assertIsNone(classify_fast(text))

        for text in ("stop", "stop please"):
            with self.subTest(text=text):
                self.assertTrue(is_consent_withdrawal(text))
                self.assertIsNone(classify_fast(text))

    def test_negated_stop_remains_a_continuation_request(self):
        for text in (
            "don't stop",
            "don\u2019t stop",
            "don't stop choking me",
            "don\u2019t stop choking me",
            "never stop choking me",
        ):
            with self.subTest(text=text):
                self.assertFalse(is_consent_withdrawal(text))
                self.assertEqual(withdrawn_acts(text), ())
                self.assertEqual(classify_fast(text), "nsfw")

    def test_curly_apostrophe_does_not_bypass_routing(self):
        self.assertEqual(classify_fast("I\u2019m wet"), "nsfw")

    def test_prefix_collisions_do_not_raise_heat(self):
        clean_messages = (
            "Are you an assistant?",
            "I ordered a cocktail",
            "That is an analytical answer",
            "I'm recording an oral history",
            "Wet weather ruined my Uber ride",
            "The strip mall has a finger-painting shop",
            "You suck at chess",
            "I want you to help me",
        )
        for message in clean_messages:
            with self.subTest(message=message):
                self.assertIsNone(classify_fast(message))

    def test_complete_terms_and_intended_stems_raise_heat(self):
        sexual_messages = (
            "nice ass",
            "I'm fucking horny",
            "I was masturbating",
            "that's seductive",
            "give me oral sex",
            "I'm so wet",
            "ride me",
            "strip for me",
            "I want you",
            "I can feel you inside me",
            "she moaned",
        )
        for message in sexual_messages:
            with self.subTest(message=message):
                self.assertEqual(classify_fast(message), "nsfw")

    def test_non_string_or_empty_input_is_clean(self):
        self.assertIsNone(classify_fast(""))
        self.assertIsNone(classify_fast(None))


if __name__ == "__main__":
    unittest.main()
