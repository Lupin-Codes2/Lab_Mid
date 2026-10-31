import unittest

import numpy as np

from game_ml.modeling import TECHNIQUES, build_model


class ModelTests(unittest.TestCase):
    def test_only_pre_midterm_classifiers_are_configured(self) -> None:
        self.assertEqual(
            set(TECHNIQUES),
            {"decision_tree", "naive_bayes", "knn", "ann", "svm"},
        )
        for technique in TECHNIQUES.values():
            self.assertTrue(technique["why_used"])
            self.assertTrue(technique["tradeoff"])

    def test_every_classifier_fits_and_returns_genre_predictions(self) -> None:
        features = np.asarray(
            [
                [ranking, 2000 + class_index, 500 + ranking * 10]
                for class_index in range(3)
                for ranking in range(1, 11)
            ],
            dtype=float,
        )
        labels = np.asarray(
            [f"Genre {class_index}" for class_index in range(3) for _ in range(10)]
        )

        for model_key in TECHNIQUES:
            with self.subTest(model=model_key):
                model = build_model(model_key)
                model.fit(features, labels)
                predictions = model.predict(features[:4])

                self.assertEqual(predictions.shape, (4,))
                self.assertTrue(set(predictions).issubset(set(labels)))


if __name__ == "__main__":
    unittest.main()
