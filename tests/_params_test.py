import pytest

import histoencoder.functional as F


def test_no_lr_scale() -> None:
    encoder = F.create_encoder("prostate_small")
    param_groups = F.get_parameter_groups(encoder, lr=0.001, lr_decay=1.0)
    for group in param_groups:
        assert pytest.approx(group["lr"]) == pytest.approx(0.001)


def test_lr_scale() -> None:
    LR, LR_DECAY = 0.001, 0.75
    encoder = F.create_encoder("prostate_small")
    excepted_scale = {15 - i: 0.75**i for i in range(15 + 1)}
    excepted_lr = {k: v * LR for k, v in excepted_scale.items()}
    param_groups = F.get_parameter_groups(encoder, lr=LR, lr_decay=LR_DECAY)
    got_scale = {int(x["name"].split("_")[1]): x["lr_scale"] for x in param_groups}
    got_lr = {int(x["name"].split("_")[1]): x["lr"] for x in param_groups}
    assert excepted_scale == got_scale
    assert excepted_lr == got_lr


def test_param_names() -> None:
    pass
