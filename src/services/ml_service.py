from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.schemas import NivelRisco, TelemetriaRequest

logger = logging.getLogger("telemetria.ml")

CAMINHO_PADRAO = "models/predictive_model.joblib"


class ModeloIndisponivelError(RuntimeError):
    """Artefato ausente, corrompido ou incompativel."""


def caminho_modelo() -> Path:
    return Path(os.getenv("MODEL_PATH", CAMINHO_PADRAO))


def limiar_alerta() -> float:
    bruto = os.getenv("LIMIAR_ALERTA", "0.80")
    try:
        limiar = float(bruto)
    except ValueError:
        raise ValueError(f"LIMIAR_ALERTA invalido: {bruto!r}") from None
    # Fora de (0, 1] o alerta dispararia sempre ou nunca, silenciosamente.
    if not 0.0 < limiar <= 1.0:
        raise ValueError(f"LIMIAR_ALERTA deve estar em (0, 1], recebido {limiar}")
    return limiar


# lru_cache nao memoriza excecoes: se o artefato aparecer depois (volume montado
# tarde, retreino), a proxima requisicao ja recarrega sem reiniciar o processo.
@lru_cache(maxsize=1)
def carregar_modelo(caminho: str) -> dict[str, Any]:
    try:
        artefato = joblib.load(caminho)
    except FileNotFoundError as exc:
        raise ModeloIndisponivelError(f"modelo nao encontrado em {caminho}") from exc
    except Exception as exc:  # joblib/pickle levantam tipos variados em arquivo corrompido
        raise ModeloIndisponivelError(f"falha ao desserializar {caminho}: {exc}") from exc

    if not isinstance(artefato, dict) or not {"modelo", "features"} <= artefato.keys():
        raise ModeloIndisponivelError(f"artefato {caminho} fora do formato esperado")
    logger.info("modelo carregado caminho=%s versao=%s", caminho, artefato.get("versao", "?"))
    return artefato


def prever_risco(telemetria: TelemetriaRequest) -> float:
    artefato = carregar_modelo(str(caminho_modelo()))
    features: list[str] = artefato["features"]
    # Sensor ausente vira NaN; o XGBoost segue o ramo "missing" aprendido no treino.
    linha = pd.DataFrame(
        [{f: getattr(telemetria, f, None) for f in features}], columns=features
    ).astype(float)
    return float(artefato["modelo"].predict_proba(linha)[0, 1])


def classificar_risco(probabilidade: float) -> NivelRisco:
    if probabilidade >= limiar_alerta():
        return "critico"
    if probabilidade >= 0.5:
        return "elevado"
    return "normal"
