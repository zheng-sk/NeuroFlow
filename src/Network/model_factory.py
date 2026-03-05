from typing import List


def available_model_variants() -> List[str]:
    return [
        "original",
        "phase1_attention",
        "phase2_attention",
        "phase3_transformer_cross_attention",
    ]


def normalize_model_variant(model_variant: str) -> str:
    key = str(model_variant or "original").strip().lower()
    aliases = {
        "sr4dflownet": "original",
        "base": "original",
        "baseline": "original",
        "phase1": "phase1_attention",
        "fase1": "phase1_attention",
        "phase2": "phase2_attention",
        "fase2": "phase2_attention",
        "transformer_attention": "phase2_attention",
        "self_attention_transformer": "phase2_attention",
        "phase3": "phase3_transformer_cross_attention",
        "fase3": "phase3_transformer_cross_attention",
        "transformer_cross_attention": "phase3_transformer_cross_attention",
        "cross_attention_transformer": "phase3_transformer_cross_attention",
    }
    return aliases.get(key, key)


def build_sr_model(
    model_variant: str,
    res_increase: int,
    low_resblock: int,
    hi_resblock: int,
    channel_nr: int = 64,
    predict_mag: bool = False,
):
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

    if variant == "phase1_attention":
        from .SR4DFlowNetPhase1Attention import SR4DFlowNetPhase1Attention

        return SR4DFlowNetPhase1Attention(
            res_increase=res_increase,
            low_resblock=low_resblock,
            hi_resblock=hi_resblock,
            channel_nr=channel_nr,
            predict_mag=predict_mag,
        )

    if variant == "phase2_attention":
        from .SR4DFlowNetPhase2Attention import SR4DFlowNetPhase2Attention

        return SR4DFlowNetPhase2Attention(
            res_increase=res_increase,
            low_resblock=low_resblock,
            hi_resblock=hi_resblock,
            channel_nr=channel_nr,
            predict_mag=predict_mag,
        )

    if variant == "phase3_transformer_cross_attention":
        from .SR4DFlowNetPhase3TransformerCrossAttention import SR4DFlowNetPhase3TransformerCrossAttention

        return SR4DFlowNetPhase3TransformerCrossAttention(
            res_increase=res_increase,
            low_resblock=low_resblock,
            hi_resblock=hi_resblock,
            channel_nr=channel_nr,
            predict_mag=predict_mag,
        )

    valid = ", ".join(available_model_variants())
    raise ValueError(f"Unknown model_variant={model_variant!r}. Valid options: {valid}")
