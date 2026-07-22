from federated_lssvm.infer import _model_dir


def test_model_dir_defaults_to_k_folder():
    assert _model_dir(20, 2) == "models/k=20/class_2"


def test_model_dir_respects_model_root_override():
    assert _model_dir(20, 2, model_root="models/k=20_baseline") == "models/k=20_baseline/class_2"
