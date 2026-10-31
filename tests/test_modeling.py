import unittest
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from game_ml.data import load_dataset
from game_ml.modeling import TECHNIQUES, build_model


ROOT_DIR = Path(__file__).resolve().parents[1]


class ModelTests(unittest.TestCase):
    def test_only_pre_midterm_classifiers_are_configured(self) -> None:
        self.assertEqual(
            set(TECHNIQUES),
            {"decision_tree", "naive_bayes", "knn", "ann", "svm"},
        )
        for technique in TECHNIQUES.values():
            self.assertTrue(technique["why_used"])
            self.assertTrue(technique["tradeoff"])

    def test_preprocessing_is_inside_each_model_pipeline(self) -> None:
        model = build_model("knn")
        self.assertIsInstance(model.steps[0][1], ColumnTransformer)
        self.assertIsInstance(
            model.steps[0][1].named_transformers_["numeric"]
            if hasattr(model.steps[0][1], "transformers_")
            else model.steps[0][1].transformers[0][1],
            Pipeline,
        )
        self.assertIsInstance(
            model.steps[0][1].transformers[0][1].named_steps["imputer"],
            SimpleImputer,
        )

    def test_every_classifier_fits_real_rows_and_returns_probabilities(self) -> None:
        dataset = load_dataset(ROOT_DIR / "heart.csv")
        selected = np.concatenate(
            [
                np.flatnonzero(dataset.labels == label)[:60]
                for label in (0, 1)
            ]
        )
        features = dataset.features[selected]
        labels = dataset.labels[selected]

        for model_key in TECHNIQUES:
            with self.subTest(model=model_key):
                model = build_model(model_key)
                model.fit(features, labels)
                predictions = model.predict(features[:4])
                probabilities = model.predict_proba(features[:4])

                self.assertEqual(predictions.shape, (4,))
                self.assertTrue(set(predictions).issubset({0, 1}))
                self.assertEqual(probabilities.shape, (4, 2))
                np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
