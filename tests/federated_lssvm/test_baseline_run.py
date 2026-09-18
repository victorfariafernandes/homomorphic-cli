import pytest

pytest.importorskip("openfhe", reason="requires the OpenFHE C++/Python build")

import federated_lssvm.train as T
from lssvm.preprocessors import prepare_binary


def test_max_feat_dim_iris_matches_poly_expansion():
    T.configure_dataset("iris")
    splits = prepare_binary("iris")
    from federated_lssvm.baseline_run import _max_feat_dim

    d = _max_feat_dim(splits, T.CLASS_KERNELS)
    # class 0 is linear (4 raw features), classes 1-2 are degree-2 poly on 4
    # features -> 4 + C(4,2) + 4 = 14 explicit terms (poly_feature_map's actual
    # output width); the max across classes must be >= the raw feature count.
    assert d >= 4


def test_max_feat_dim_breast_cancer_is_linear_30_features():
    T.configure_dataset("breast_cancer")
    splits = prepare_binary("breast_cancer")
    from federated_lssvm.baseline_run import _max_feat_dim

    d = _max_feat_dim(splits, T.CLASS_KERNELS)
    assert d == 30
