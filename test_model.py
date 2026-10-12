
import unittest

from train_model import create_dataset, train_model


class TestStudentPlacementModel(unittest.TestCase):

    def test_dataset_has_500_students(self):
        self.assertEqual(len(create_dataset()), 500)

    def test_dataset_labels_are_valid(self):
        data = create_dataset()

        for row in data:
            self.assertIn(row[4], [0, 1])

    def test_model_trains_successfully(self):
        model, metrics = train_model()

        self.assertIsNotNone(model)
        self.assertIn("accuracy", metrics)

    def test_accuracy_is_valid(self):
        _, metrics = train_model()

        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
