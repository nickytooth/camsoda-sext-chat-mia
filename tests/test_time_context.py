import unittest
from datetime import datetime
from unittest.mock import patch

from bot import time_context


class WeekdayScheduleTests(unittest.TestCase):
    EXPECTED_BY_HOUR = {
        0: "club_night",
        1: "club_night",
        2: "night_bed",
        3: "night_bed",
        4: "night_bed",
        5: "night_bed",
        6: "night_bed",
        7: "night_bed",
        8: "night_bed",
        9: "morning_home",
        10: "morning_home",
        11: "midday_gym",
        12: "midday_gym",
        13: "prework_home",
        14: "prework_home",
        15: "bar_shift",
        16: "bar_shift",
        17: "bar_shift",
        18: "bar_shift",
        19: "bar_shift",
        20: "evening_pregame",
        21: "evening_pregame",
        22: "club_night",
        23: "club_night",
    }

    def test_every_weekday_hour_maps_to_the_new_schedule(self):
        with (
            patch.object(time_context, "_is_weekend", return_value=False),
            patch.object(time_context, "datetime") as mocked_datetime,
        ):
            for hour, expected in self.EXPECTED_BY_HOUR.items():
                with self.subTest(hour=hour):
                    mocked_datetime.now.return_value = datetime(
                        2026, 8, 4, hour, 0, tzinfo=time_context.TIMEZONE
                    )
                    self.assertEqual(time_context.get_time_period(), expected)

    def test_period_metadata_matches_the_weekday_boundaries(self):
        expected_hours = {
            "night_bed": (2, 9),
            "morning_home": (9, 11),
            "midday_gym": (11, 13),
            "prework_home": (13, 15),
            "bar_shift": (15, 20),
            "evening_pregame": (20, 22),
            "club_night": (22, 2),
        }

        self.assertEqual(
            {name: info["hours"] for name, info in time_context.TIME_PERIODS.items()},
            expected_hours,
        )

    def test_bar_shift_scene_matches_mias_part_time_job(self):
        scene = time_context.TIME_PERIODS["bar_shift"]

        self.assertIn("works part-time", scene["where"])
        self.assertIn("mixing drinks", scene["activity"])
        self.assertEqual(scene["preferred_tags"], ["bar", "work", "public"])

    def test_every_schedule_period_has_catalog_facing_media_context(self):
        periods = set(time_context.TIME_PERIODS) | set(time_context.WEEKEND_PERIODS)

        self.assertEqual(set(time_context.MEDIA_CONTEXTS), periods)
        for period in periods:
            with self.subTest(period=period):
                context = time_context.get_media_context(period)
                self.assertEqual(context["period"], period)
                self.assertTrue(context["locations"])
                self.assertTrue(context["fallback_reason"])
                self.assertNotIn("unopened", context["fallback_reason"].lower())

    def test_bar_media_context_prioritizes_bar_and_explains_fallback_naturally(self):
        context = time_context.get_media_context("bar_shift")

        self.assertEqual(context["locations"][0], "bar")
        self.assertIn("customers", context["fallback_reason"])
        self.assertNotIn("privacy policy", context["fallback_reason"].lower())

    def test_day_plan_does_not_move_the_fixed_three_to_eight_shift(self):
        authored_plan_text = " ".join(
            part
            for pair in time_context._WEEKDAY_EVENINGS
            for part in pair
        ) + " " + " ".join(time_context._DAY_DETAILS)

        self.assertNotIn("late close", authored_plan_text)
        self.assertNotIn("shift might run long", authored_plan_text)
        self.assertIn("go home to eat, shower, and change", authored_plan_text)


if __name__ == "__main__":
    unittest.main()
