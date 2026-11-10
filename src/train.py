"""Treino do classificador de falha para mancais/turbinas.

Uso: python -m src.train [--amostras 2500] [--seed 42] [--saida models/predictive_model.joblib]
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

logger = logging.getLogger("telemetria.train")

FEATURES: tuple[str, ...] = (
    "temperatura_mancal_celsius",
    "vibracao_rms_mm_s",
    "pressao_linha_bar",
    "rpm_rotor",
    "horas_operacao_continua",
)

FAIXAS_FISICAS: dict[str, tuple[float, float]] = {
    "temperatura_mancal_celsius": (30.0, 130.0),
    "vibracao_rms_mm_s": (0.5, 15.0),
    "pressao_linha_bar": (1.0, 45.0),
    "rpm_rotor": (0.0, 3600.0),
    "horas_operacao_continua": (0.0, 1000.0),
}

# Leituras perdidas por dropout de sensor/gateway. XGBoost trata NaN nativamente.
TAXA_LEITURA_AUSENTE = 0.015
# Desgaste a partir do qual a falha passa a ser mais provavel que improvavel.
PONTO_MEDIO_DESGASTE = 0.5
# Inclinacao da curva desgaste -> falha. Finita de proposito: dois equipamentos com o
# mesmo desgaste podem ter destinos diferentes, como em planta.
INCLINACAO_FALHA = 20.0
VERSAO_MODELO = "2.0.0"


def gerar_telemetria_sintetica(n_amostras: int, seed: int) -> pd.DataFrame:
    """Gera telemetria a partir de um desgaste latente continuo em [0, 1].

    Classes separadas por construcao (versao 1) produziam AUC ~1 e probabilidades
    sempre perto de 0 ou 1. Aqui os sensores derivam junto com o desgaste e o rotulo
    e sorteado a partir dele, o que cria a zona cinzenta de equipamentos parcialmente
    degradados (~18% de falhas no total).
    """
    rng = np.random.default_rng(seed)
    n = n_amostras
    # Maioria dos equipamentos com pouco desgaste; cauda longa de degradados.
    desgaste = rng.beta(1.3, 3.2, n)
    prob_falha = 1.0 / (1.0 + np.exp(-INCLINACAO_FALHA * (desgaste - PONTO_MEDIO_DESGASTE)))
    falha = rng.random(n) < prob_falha

    temperatura = 68 + 38 * desgaste + rng.normal(0, 4.5, n)
    vibracao = rng.lognormal(np.log(2.0) + 1.6 * desgaste, 0.22, n)
    # Falha de pressao pode ser para cima (obstrucao) ou para baixo (vazamento).
    sentido = rng.choice([-1.0, 1.0], n)
    pressao = 24 + sentido * 16 * desgaste**1.5 + rng.normal(0, 3, n)
    # Rotor degradado oscila mais em torno da rotacao nominal.
    rpm = 3000 + rng.normal(0, 80 + 320 * desgaste, n)
    horas = 150 + 650 * desgaste + rng.normal(0, 140, n)

    # Ruido de medicao independente do estado do equipamento.
    temperatura += rng.normal(0, 1.5, n)
    vibracao *= 1 + rng.normal(0, 0.06, n)
    pressao += rng.normal(0, 0.4, n)
    rpm += rng.normal(0, 8, n)

    telemetria = pd.DataFrame(
        {
            "temperatura_mancal_celsius": temperatura,
            "vibracao_rms_mm_s": vibracao,
            "pressao_linha_bar": pressao,
            "rpm_rotor": rpm,
            "horas_operacao_continua": horas,
            "falha_iminente": falha.astype(int),
        }
    )
    for coluna, (minimo, maximo) in FAIXAS_FISICAS.items():
        telemetria[coluna] = telemetria[coluna].clip(minimo, maximo)
        ausentes = rng.random(n) < TAXA_LEITURA_AUSENTE
        telemetria.loc[ausentes, coluna] = np.nan
    return telemetria


LIMIAR_ALERTA_PADRAO = 0.80


def avaliar(y: np.ndarray, probabilidades: np.ndarray, limiar_alerta: float) -> dict[str, float]:
    """Metricas de ranking, calibracao e decisao nos dois limiares usados pela API."""
    alerta = probabilidades >= limiar_alerta
    elevado = probabilidades >= 0.5
    return {
        "roc_auc": float(roc_auc_score(y, probabilidades)),
        "pr_auc": float(average_precision_score(y, probabilidades)),
        "brier": float(brier_score_loss(y, probabilidades)),
        "precisao_050": float(precision_score(y, elevado, zero_division=0)),
        "recall_050": float(recall_score(y, elevado, zero_division=0)),
        "f1_050": float(f1_score(y, elevado, zero_division=0)),
        "precisao_alerta": float(precision_score(y, alerta, zero_division=0)),
        "recall_alerta": float(recall_score(y, alerta, zero_division=0)),
        "taxa_alerta": float(alerta.mean()),
        "fracao_zona_cinzenta": float(np.mean((probabilidades > 0.05) & (probabilidades < 0.95))),
    }


def treinar(n_amostras: int, seed: int, destino: Path) -> dict[str, float]:
    telemetria = gerar_telemetria_sintetica(n_amostras, seed)
    X = telemetria[list(FEATURES)]
    y = telemetria["falha_iminente"]

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    # Sem scale_pos_weight: ele reponderaria a perda e distorceria as probabilidades, e a
    # API usa a probabilidade em si (limiar de alerta), nao so a classe prevista.
    modelo = XGBClassifier(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.9,
        min_child_weight=5,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
    )
    modelo.fit(X_treino, y_treino)

    probabilidades = modelo.predict_proba(X_teste)[:, 1]
    metricas = avaliar(y_teste.to_numpy(), probabilidades, limiar_alerta=LIMIAR_ALERTA_PADRAO)
    logger.info(
        "amostras=%d roc_auc=%.3f pr_auc=%.3f brier=%.4f",
        n_amostras, metricas["roc_auc"], metricas["pr_auc"], metricas["brier"],
    )
    logger.info(
        "limiar 0.50: precisao=%.3f recall=%.3f | limiar %.2f: precisao=%.3f recall=%.3f",
        metricas["precisao_050"], metricas["recall_050"], LIMIAR_ALERTA_PADRAO,
        metricas["precisao_alerta"], metricas["recall_alerta"],
    )
    logger.info(
        "relatorio de classificacao (limiar 0.50):\n%s",
        classification_report(y_teste, probabilidades >= 0.5, digits=3, zero_division=0),
    )

    destino.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "modelo": modelo,
            "features": list(FEATURES),
            "metricas": metricas,
            "versao": VERSAO_MODELO,
            "treinado_em": datetime.now(timezone.utc).isoformat(),
        },
        destino,
    )
    logger.info("modelo salvo em %s", destino)
    return metricas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Treina o modelo preditivo de falha.")
    parser.add_argument("--amostras", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--saida", type=Path, default=Path("models/predictive_model.joblib"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.amostras < 200:
        logger.error("amostras insuficientes para split estratificado: %d", args.amostras)
        return 2
    try:
        treinar(args.amostras, args.seed, args.saida)
    except OSError:
        logger.exception("falha ao gravar o artefato do modelo")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
