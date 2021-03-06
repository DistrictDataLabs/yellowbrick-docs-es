# tests.test_contrib.test_missing.test_bar
# Tests for the alpha selection visualizations.
#
# Author:  Nathan Danielsen <nathan.danielsen@gmail.com>
# Created:  Thu Mar 29 12:13:04 2018 -0500
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: test_bar.py [1443e16] ndanielsen@users.noreply.github.com $

"""
Tests for the MissingValuesBar visualizations.
"""

##########################################################################
## Imports
##########################################################################

import os
import pytest
import numpy as np

from tests.base import VisualTestCase
from sklearn.datasets import make_classification
from yellowbrick.contrib.missing.bar import *

try:
    import pandas as pd
except ImportError:
    pd = None


@pytest.fixture(scope="class")
def missing_bar_tolerance(request):
    request.cls.tol = 0.5 if os.name == "nt" else 0.01


##########################################################################
## Feature Importances Tests
##########################################################################


@pytest.mark.usefixtures("missing_bar_tolerance")
class TestMissingBarVisualizer(VisualTestCase):
    """
    FeatureImportances visualizer
    """

    @pytest.mark.skipif(pd is None, reason="pandas is required")
    def test_missingvaluesbar_pandas(self):
        """
        Integration test of visualizer with pandas
        """
        X, y = make_classification(
            n_samples=400,
            n_features=20,
            n_informative=8,
            n_redundant=8,
            n_classes=2,
            n_clusters_per_class=4,
            random_state=854,
        )

        # add nan values to a range of values in the matrix
        X[X > 1.5] = np.nan
        X_ = pd.DataFrame(X)

        features = [str(n) for n in range(20)]
        viz = MissingValuesBar(features=features)
        viz.fit(X_)
        viz.finalize()
        
        self.assert_images_similar(viz, tol=self.tol)

    @pytest.mark.skipif(pd is None, reason="pandas is required")
    def test_missingvaluesbar_pandas_no_features_passed(self):
        """
        Integration test of visualizer with pandas
        """
        X, y = make_classification(
            n_samples=400,
            n_features=20,
            n_informative=8,
            n_redundant=8,
            n_classes=2,
            n_clusters_per_class=4,
            random_state=854,
        )

        # add nan values to a range of values in the matrix
        X[X > 1.5] = np.nan
        X_ = pd.DataFrame(X)

        viz = MissingValuesBar()
        viz.fit(X_)
        viz.finalize()

        self.assert_images_similar(viz, tol=self.tol)

    def test_missingvaluesbar_numpy(self):
        """
        Integration test of visualizer with numpy without target y passed in
        """
        X, y = make_classification(
            n_samples=400,
            n_features=20,
            n_informative=8,
            n_redundant=8,
            n_classes=2,
            n_clusters_per_class=4,
            random_state=856,
        )

        # add nan values to a range of values in the matrix
        X[X > 1.5] = np.nan

        features = [str(n) for n in range(20)]
        viz = MissingValuesBar(features=features)
        viz.fit(X)
        viz.finalize()

        self.assert_images_similar(viz, tol=self.tol)

    def test_missingvaluesbar_numpy_no_features_passed(self):
        """
        Integration test of visualizer with numpy without target y passed in
        """
        X, y = make_classification(
            n_samples=400,
            n_features=20,
            n_informative=8,
            n_redundant=8,
            n_classes=2,
            n_clusters_per_class=4,
            random_state=856,
        )

        # add nan values to a range of values in the matrix
        X[X > 1.5] = np.nan

        viz = MissingValuesBar()
        viz.fit(X)
        viz.finalize()

        self.assert_images_similar(viz, tol=self.tol)

    def test_missingvaluesbar_numpy_with_y_target(self):
        """
        Integration test of visualizer with numpy without target y passed in
        but no class labels
        """
        X, y = make_classification(
            n_samples=400,
            n_features=20,
            n_informative=8,
            n_redundant=8,
            n_classes=2,
            n_clusters_per_class=4,
            random_state=856,
        )

        # add nan values to a range of values in the matrix
        X[X > 1.5] = np.nan

        features = [str(n) for n in range(20)]
        viz = MissingValuesBar(features=features)
        viz.fit(X, y)
        viz.finalize()

        self.assert_images_similar(viz, tol=self.tol)

    def test_missingvaluesbar_numpy_with_y_target_with_labels(self):
        """
        Integration test of visualizer with numpy without target y passed in
        but no class labels
        """
        X, y = make_classification(
            n_samples=400,
            n_features=20,
            n_informative=8,
            n_redundant=8,
            n_classes=2,
            n_clusters_per_class=4,
            random_state=856,
        )

        # add nan values to a range of values in the matrix
        X[X > 1.5] = np.nan

        features = [str(n) for n in range(20)]
        viz = MissingValuesBar(features=features, classes=["class A", "class B"])
        viz.fit(X, y)
        viz.finalize()

        self.assert_images_similar(viz, tol=self.tol)

    def test_missingvaluesbar_numpy_with_string_and_bool_cols(self):
        """
        Integration test of visualizer with numpy array with string and boolean columns
        """
        X, y = make_classification(
            n_samples=400,
            n_features=10,
            n_informative=2,
            n_redundant=3,
            n_classes=2,
            n_clusters_per_class=2,
            random_state=854
        )

        # add nan values to a range of values in the matrix
        X[X > 1.5] = np.nan

        rng = np.random.default_rng(2021)
        fruit_choices = np.array(['apples', 'pears', 'peaches', "", np.nan, 'bananas'])
        fruits = rng.choice(fruit_choices, (400, 1))

        bool_choices = np.array([np.nan, False, True])
        booleans = rng.choice(bool_choices, (400, 1))

        X = np.append(X, fruits, axis=1)
        X = np.append(X, booleans, axis=1)

        features = [str(n) for n in range(12)]
        viz = MissingValuesBar(features=features)
        viz.fit(X, y)
        viz.finalize()

        self.assert_images_similar(viz, tol=5)

    @pytest.mark.skipif(pd is None, reason="pandas is required")
    def test_missingvaluesbar_pandas_with_string_and_bool_cols(self):
        """
        Integration test of visualizer with pandas dataframe with string and boolean columns
        """
        X, y = make_classification(
            n_samples=400,
            n_features=10,
            n_informative=2,
            n_redundant=3,
            n_classes=2,
            n_clusters_per_class=2,
            random_state=854
        )

        # add nan values to a range of values in the matrix
        X[X > 1.5] = np.nan

        rng = np.random.default_rng(2021)
        fruit_choices = np.array(['apples', 'pears', 'peaches', "", np.nan, 'bananas'])
        fruits = rng.choice(fruit_choices, (400, 1))

        bool_choices = np.array([np.nan, False, True])
        booleans = rng.choice(bool_choices, (400, 1))

        X = np.append(X, fruits, axis=1)
        X = np.append(X, booleans, axis=1)

        X_ = pd.DataFrame(X)

        features = [str(n) for n in range(12)]
        viz = MissingValuesBar(features=features)
        viz.fit(X_, y)
        viz.finalize()

        self.assert_images_similar(viz, tol=5)

    def test_missingvaluesbar_numpy_with_mixed_dtypes(self):
        """
        Integration test of visualizer with numpy array with mixed dtypes in a single column
        """
        X, y = make_classification(
            n_samples=400,
            n_features=10,
            n_informative=2,
            n_redundant=3,
            n_classes=2,
            n_clusters_per_class=2,
            random_state=854
        )

        # add nan values to a range of values in the matrix
        X[X > 1.5] = np.nan

        rng = np.random.default_rng(2021)
        mixed_dtype_choices = np.array(['apples', 'pears', 'peaches', "", np.nan, 'bananas', 1, 2.4, 5.6, False, True])
        mixed_dtypes = rng.choice(mixed_dtype_choices, (400, 1))

        X_with_mixed_dtypes = np.append(X, mixed_dtypes, axis=1)

        features = [str(n) for n in range(11)]
        viz = MissingValuesBar(features=features)
        viz.fit(X_with_mixed_dtypes, y)
        viz.finalize()

        self.assert_images_similar(viz, tol=5)

    @pytest.mark.skipif(pd is None, reason="pandas is required")
    def test_missingvaluesbar_pandas_with_mixed_dtypes(self):
        """
        Integration test of visualizer with pandas dataframe with mixed dtypes in a single column
        """
        X, y = make_classification(
            n_samples=400,
            n_features=10,
            n_informative=2,
            n_redundant=3,
            n_classes=2,
            n_clusters_per_class=2,
            random_state=854
        )

        # add nan values to a range of values in the matrix
        X[X > 1.5] = np.nan

        rng = np.random.default_rng(2021)
        mixed_dtype_choices = np.array(['apples', 'pears', 'peaches', "", np.nan, 'bananas', 1, 2.4, 5.6, False, True])
        mixed_dtypes = rng.choice(mixed_dtype_choices, (400, 1))

        X_with_mixed_dtypes = np.append(X, mixed_dtypes, axis=1)
        X_ = pd.DataFrame(X_with_mixed_dtypes)

        features = [str(n) for n in range(11)]
        viz = MissingValuesBar(features=features)
        viz.fit(X_, y)
        viz.finalize()

        self.assert_images_similar(viz, tol=5)
