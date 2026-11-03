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
from sklearn.metrics import accuracy_score, classification_report, f1_score
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

# Em campo falha e evento raro; 20% ja deixa o conjunto treinavel sem virar caso trivial.
TAXA_FALHA = 0.20
# Leituras perdidas por dropout de sensor/gateway. XGBoost trata NaN nativamente.
TAXA_LEITURA_AUSENTE = 0.015
VERSAO_MODELO = "1.0.0"


def gerar_telemetria_sintetica(n_amostras: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    falha = rng.random(n_amostras) < TAXA_FALHA

    def por_classe(normal: np.ndarray, degradado: np.ndarray) -> np.ndarray:
        return np.where(falha, degradado, normal)

    n = n_amostras
    temperatura = por_classe(rng.normal(72, 9, n), rng.normal(98, 12, n))
    vibracao = por_classe(
        rng.lognormal(np.log(2.2), 0.35, n),
        rng.lognormal(np.log(7.0), 0.45, n),
    )
    # Falha de pressao pode ser para cima (obstrucao) ou para baixo (vazamento).
    sentido = rng.choice([-1.0, 1.0], n)
    pressao = por_classe(rng.normal(24, 4, n), 24 + sentido * rng.normal(11, 5, n))
    rpm = por_classe(rng.normal(3000, 110, n), rng.normal(3000, 380, n))
    horas = por_classe(rng.uniform(0, 700, n), rng.uniform(280, 1000, n))

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


def treinar(n_amostras: int, seed: int, destino: Path) -> dict[str, float]:
    telemetria = gerar_telemetria_sintetica(n_amostras, seed)
    X = telemetria[list(FEATURES)]
    y = telemetria["falha_iminente"]

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    peso_positivos = float((y_treino == 0).sum() / max((y_treino == 1).sum(), 1))
    modelo = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.9,
        scale_pos_weight=peso_positivos,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
    )
    modelo.fit(X_treino, y_treino)

    previsto = modelo.predict(X_teste)
    metricas = {
        "acuracia": float(accuracy_score(y_teste, previsto)),
        "f1": float(f1_score(y_teste, previsto)),
    }
    logger.info("acuracia=%.4f f1=%.4f amostras=%d", metricas["acuracia"], metricas["f1"], n_amostras)
    logger.info("relatorio de classificacao:\n%s", classification_report(y_teste, previsto, digits=3))

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
