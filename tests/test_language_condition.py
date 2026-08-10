import unittest

from experiments.libero.language_condition import normalize_instruction_condition


class LanguageConditionTest(unittest.TestCase):
    def test_hydra_none_maps_to_null_condition(self):
        self.assertEqual(normalize_instruction_condition(None), "null")

    def test_string_conditions_are_normalized(self):
        self.assertEqual(normalize_instruction_condition(" Correct "), "correct")
        self.assertEqual(normalize_instruction_condition("NULL"), "null")
        self.assertEqual(normalize_instruction_condition("Shuffled"), "shuffled")


if __name__ == "__main__":
    unittest.main()
