from __future__ import annotations

import joblib
import numpy as np
import pytest

from src.train import FAIXAS_FISICAS, FEATURES, avaliar, gerar_telemetria_sintetica, main


@pytest.fixture(scope="module")
def telemetria():
    return gerar_telemetria_sintetica(n_amostras=5000, seed=123)


def test_gerador_respeita_faixas_fisicas(telemetria):
    for coluna, (minimo, maximo) in FAIXAS_FISICAS.items():
        valores = telemetria[coluna].dropna()
        assert valores.between(minimo, maximo).all(), coluna


def test_gerador_tem_falha_minoritaria_e_leituras_ausentes(telemetria):
    assert 0.12 < telemetria["falha_iminente"].mean() < 0.25
    ausentes = telemetria[list(FEATURES)].isna().mean()
    assert ((ausentes > 0.005) & (ausentes < 0.03)).all()


def test_gerador_e_deterministico_pela_seed():
    a = gerar_telemetria_sintetica(300, seed=1)
    b = gerar_telemetria_sintetica(300, seed=1)
    assert a.equals(b)


def test_sensores_pioram_com_a_falha(telemetria):
    medias = telemetria.groupby("falha_iminente")[["temperatura_mancal_celsius", "vibracao_rms_mm_s"]].mean()
    assert (medias.loc[1] > medias.loc[0]).all()


def test_modelo_discrimina_sem_ser_trivial(modelo_treinado):
    metricas = joblib.load(modelo_treinado)["metricas"]
    # Acima de 0.99 indicaria classes separaveis por construcao (o defeito da versao 1).
    assert 0.80 < metricas["roc_auc"] < 0.99
    assert metricas["brier"] < 0.15
    # O modelo precisa produzir riscos intermediarios, nao so 0% ou 100%.
    assert metricas["fracao_zona_cinzenta"] > 0.15


def test_avaliar_calcula_metricas_nos_limiares():
    y = np.array([0, 0, 1, 1])
    probabilidades = np.array([0.1, 0.6, 0.7, 0.9])
    metricas = avaliar(y, probabilidades, limiar_alerta=0.8)
    assert metricas["roc_auc"] == 1.0
    assert metricas["precisao_050"] == pytest.approx(2 / 3)
    assert metricas["recall_050"] == 1.0
    assert metricas["precisao_alerta"] == 1.0
    assert metricas["recall_alerta"] == 0.5
    assert metricas["taxa_alerta"] == 0.25


def test_cli_treina_e_grava_artefato(tmp_path):
    destino = tmp_path / "m.joblib"
    assert main(["--amostras", "400", "--seed", "3", "--saida", str(destino)]) == 0
    artefato = joblib.load(destino)
    assert artefato["features"] == list(FEATURES)
    assert artefato["versao"] == "2.0.0"


def test_cli_recusa_amostras_insuficientes(tmp_path):
    assert main(["--amostras", "50", "--saida", str(tmp_path / "m.joblib")]) == 2
