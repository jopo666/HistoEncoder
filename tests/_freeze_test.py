from timm.models.xcit import XCiT

import histoencoder.functional as F

CLS_ATTN_BLOCK_PARAMS = [
    "cls_attn_blocks.{}.gamma1",
    "cls_attn_blocks.{}.gamma2",
    "cls_attn_blocks.{}.norm1.weight",
    "cls_attn_blocks.{}.norm1.bias",
    "cls_attn_blocks.{}.attn.q.weight",
    "cls_attn_blocks.{}.attn.q.bias",
    "cls_attn_blocks.{}.attn.k.weight",
    "cls_attn_blocks.{}.attn.k.bias",
    "cls_attn_blocks.{}.attn.v.weight",
    "cls_attn_blocks.{}.attn.v.bias",
    "cls_attn_blocks.{}.attn.proj.weight",
    "cls_attn_blocks.{}.attn.proj.bias",
    "cls_attn_blocks.{}.norm2.weight",
    "cls_attn_blocks.{}.norm2.bias",
    "cls_attn_blocks.{}.mlp.fc1.weight",
    "cls_attn_blocks.{}.mlp.fc1.bias",
    "cls_attn_blocks.{}.mlp.fc2.weight",
    "cls_attn_blocks.{}.mlp.fc2.bias",
]
BLOCK_PARAMS = [
    "blocks.{}.gamma1",
    "blocks.{}.gamma3",
    "blocks.{}.gamma2",
    "blocks.{}.norm1.weight",
    "blocks.{}.norm1.bias",
    "blocks.{}.attn.temperature",
    "blocks.{}.attn.qkv.weight",
    "blocks.{}.attn.qkv.bias",
    "blocks.{}.attn.proj.weight",
    "blocks.{}.attn.proj.bias",
    "blocks.{}.norm3.weight",
    "blocks.{}.norm3.bias",
    "blocks.{}.local_mp.conv1.weight",
    "blocks.{}.local_mp.conv1.bias",
    "blocks.{}.local_mp.bn.weight",
    "blocks.{}.local_mp.bn.bias",
    "blocks.{}.local_mp.conv2.weight",
    "blocks.{}.local_mp.conv2.bias",
    "blocks.{}.norm2.weight",
    "blocks.{}.norm2.bias",
    "blocks.{}.mlp.fc1.weight",
    "blocks.{}.mlp.fc1.bias",
    "blocks.{}.mlp.fc2.weight",
    "blocks.{}.mlp.fc2.bias",
]


def test_freeze_layers_small() -> None:
    encoder = F.create_encoder("prostate_small")
    check_encoder_cls_attn_blocks(encoder)
    check_encoder_blocks(encoder)


def test_freeze_layers_medium() -> None:
    encoder = F.create_encoder("prostate_medium")
    check_encoder_cls_attn_blocks(encoder)
    check_encoder_blocks(encoder)


def unfreeze_encoder(encoder: XCiT) -> None:
    for param in encoder.parameters():
        param.requires_grad = True


def collect_liquid_params(encoder: XCiT) -> list[str]:
    output = []
    for name, param in encoder.named_parameters():
        if param.requires_grad:
            output.append(name)
    return output


def check_encoder_blocks(encoder: XCiT) -> None:
    num_cls_blocks = len(encoder.cls_attn_blocks)
    num_blocks = len(encoder.blocks)
    for num_liquid_blocks in range(num_blocks - 1):
        num_liquid = num_cls_blocks + num_liquid_blocks + 1
        F.freeze_encoder(encoder, num_liquid=num_cls_blocks + num_liquid_blocks + 1)
        expected = []
        for cls_block_idx in range(len(encoder.cls_attn_blocks)):
            expected.extend([x.format(cls_block_idx) for x in CLS_ATTN_BLOCK_PARAMS])
        for block_idx in range(num_liquid_blocks + 1):
            expected.extend(
                [x.format(num_blocks - (block_idx + 1)) for x in BLOCK_PARAMS]
            )
        print(f"NUMBER OF LIQUID BLOCKS: {num_liquid}")  # noqa
        assert set(expected) == (set(collect_liquid_params(encoder)))
        unfreeze_encoder(encoder)
    F.freeze_encoder(encoder, num_liquid=10000)
    assert set(collect_liquid_params(encoder)) == {
        name for name, _ in encoder.named_parameters()
    }
    unfreeze_encoder(encoder)


def check_encoder_cls_attn_blocks(encoder: XCiT) -> None:
    # Nothing.
    F.freeze_encoder(encoder, num_liquid=0, freeze_last_mlp_layer=True)
    assert collect_liquid_params(encoder) == []
    unfreeze_encoder(encoder)
    # MLP.
    F.freeze_encoder(encoder, num_liquid=0, freeze_last_mlp_layer=False)
    assert collect_liquid_params(encoder) == [
        x.format(1) for x in CLS_ATTN_BLOCK_PARAMS if ".mlp." in x
    ]
    unfreeze_encoder(encoder)
    # Everything
    for num_liquid in range(1, len(encoder.cls_attn_blocks) + 1):
        F.freeze_encoder(encoder, num_liquid=num_liquid)
        print(f"NUMBER OF LIQUID BLOCKS: {num_liquid}")  # noqa
        expected = []
        for block_idx in range(num_liquid):
            expected.extend(
                [
                    x.format(len(encoder.cls_attn_blocks) - (block_idx + 1))
                    for x in CLS_ATTN_BLOCK_PARAMS
                ]
            )
        assert set(collect_liquid_params(encoder)) == set(expected)
        unfreeze_encoder(encoder)
