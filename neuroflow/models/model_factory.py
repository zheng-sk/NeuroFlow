from typing import List


def available_model_variants() -> List[str]:
    """Architectures reported in the NeuroFlow manuscript.

    `pre_upsample_attention` is the CBAM variant published as CBAM-DNS (x1,
    domain translation) and CBAM-SupRes (x2, joint translation and
    super-resolution). The other three are the reported baselines.

    The cascade, phase-attention, and JiT-SR ablations are preserved on the
    `neuroflow_dev` branch; see the Reproducibility section of the README.
    """
    return [
        "original",
        "residual_skip",
        "pre_res_attention",
        "pre_upsample_attention",
    ]


def normalize_model_variant(model_variant: str) -> str:
    key = str(model_variant or "original").strip().lower()
    aliases = {
        "sr4dflownet": "original",
        "base": "original",
        "baseline": "original",
        "baseline_residual_skip": "residual_skip",
        "velocity_residual_skip": "residual_skip",
        "denoise_residual_skip": "residual_skip",
        "identity_skip": "residual_skip",
        "local_attention": "pre_res_attention",
        "single_local_attention": "pre_res_attention",
        "pre_res_local_attention": "pre_res_attention",
        "preres_attention": "pre_res_attention",
        "attention_before_residuals": "pre_res_attention",
        "local_attention_before_upsample": "pre_upsample_attention",
        "attention_before_upsample": "pre_upsample_attention",
        "preupsample_attention": "pre_upsample_attention",
    }
    return aliases.get(key, key)


def build_sr_model(
    model_variant: str,
    res_increase: int,
    low_resblock: int,
    hi_resblock: int,
    channel_nr: int = 64,
    predict_mag: bool = False,
    use_seg_head: bool = False,
    seg_head_bias_init: float = -4.0,
):
    """Construct one of the manuscript architectures.

    `use_seg_head` / `seg_head_bias_init` apply to `pre_upsample_attention`
    only. They were previously named `cascade_use_seg_head` /
    `cascade_seg_head_bias_init`; the old names are still accepted by the
    trainer CLI flags for checkpoint compatibility.
    """
    variant = normalize_model_variant(model_variant)

    if variant == "original":
        from .SR4DFlowNet import SR4DFlowNet

        return SR4DFlowNet(
            res_increase=res_increase,
            low_resblock=low_resblock,
            hi_resblock=hi_resblock,
            channel_nr=channel_nr,
            predict_mag=predict_mag,
        )

    if variant == "residual_skip":
        from .SR4DFlowNetResidualSkip import SR4DFlowNetResidualSkip

        return SR4DFlowNetResidualSkip(
            res_increase=res_increase,
            low_resblock=low_resblock,
            hi_resblock=hi_resblock,
            channel_nr=channel_nr,
            predict_mag=predict_mag,
        )

    if variant == "pre_res_attention":
        from .SR4DFlowNetPreResAttention import SR4DFlowNetPreResAttention

        return SR4DFlowNetPreResAttention(
            res_increase=res_increase,
            low_resblock=low_resblock,
            hi_resblock=hi_resblock,
            channel_nr=channel_nr,
            predict_mag=predict_mag,
        )

    if variant == "pre_upsample_attention":
        from .SR4DFlowNetPreUpsampleAttention import SR4DFlowNetPreUpsampleAttention

        return SR4DFlowNetPreUpsampleAttention(
            res_increase=res_increase,
            low_resblock=low_resblock,
            hi_resblock=hi_resblock,
            channel_nr=channel_nr,
            predict_mag=predict_mag,
            use_seg_head=use_seg_head,
            seg_head_bias_init=seg_head_bias_init,
        )

    valid = ", ".join(available_model_variants())
    raise ValueError(f"Unknown model_variant={model_variant!r}. Valid options: {valid}")
