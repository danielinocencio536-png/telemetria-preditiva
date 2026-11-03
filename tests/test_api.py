from __future__ import annotations

import asyncio
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from src.main import app
from src.schemas import TelemetriaRequest
from src.services import ml_service
from src.services.llm_service import LLMService, diagnostico_por_regras
from tests.conftest import (
    DIAGNOSTICO_CLAUDE,
    PAYLOAD_CRITICO,
    PAYLOAD_SAUDAVEL,
    bloco_texto,
    resposta_claude,
)

ROTA = "/api/v1/telemetry/analyze"
REQUISICAO_ANTHROPIC = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def test_equipamento_saudavel_nao_aciona_claude(client, chamada_claude):
    resp = client.post(ROTA, json=PAYLOAD_SAUDAVEL)

    assert resp.status_code == 200
    corpo = resp.json()
    assert corpo["alerta"] is False
    assert corpo["diagnostico"] is None
    chamada_claude.assert_not_called()


def test_risco_critico_retorna_diagnostico_do_claude(client, chamada_claude):
    resp = client.post(ROTA, json=PAYLOAD_CRITICO)

    assert resp.status_code == 200
    corpo = resp.json()
    assert corpo["alerta"] is True
    assert corpo["nivel_risco"] == "critico"
    assert corpo["fonte_diagnostico"] == "claude"
    assert corpo["diagnostico"]["componente_afetado"] == DIAGNOSTICO_CLAUDE["componente_afetado"]
    chamada_claude.assert_awaited_once()


def test_requisicao_ao_claude_respeita_restricoes_do_modelo(client, chamada_claude):
    client.post(ROTA, json=PAYLOAD_CRITICO)

    parametros = chamada_claude.await_args.kwargs
    # Modelos atuais rejeitam (HTTP 400) tool_choice forcado e temperature fora do padrao.
    assert "tool_choice" not in parametros
    assert "temperature" not in parametros
    assert parametros["output_config"]["format"]["type"] == "json_schema"
    assert parametros["fallbacks"] == "default"


def test_saida_apos_fallback_de_servidor_usa_texto_do_modelo_substituto(client, chamada_claude):
    chamada_claude.return_value = resposta_claude(
        blocos=[
            bloco_texto('{"causa_provavel": "parcial'),
            SimpleNamespace(type="fallback"),
            bloco_texto(resposta_claude().content[0].text),
        ]
    )

    resp = client.post(ROTA, json=PAYLOAD_CRITICO)

    assert resp.json()["fonte_diagnostico"] == "claude"


def test_timeout_do_claude_cai_para_regras(client, chamada_claude):
    chamada_claude.side_effect = anthropic.APITimeoutError(request=REQUISICAO_ANTHROPIC)

    resp = client.post(ROTA, json=PAYLOAD_CRITICO)

    assert resp.status_code == 200
    corpo = resp.json()
    assert corpo["fonte_diagnostico"] == "regras_fallback"
    assert corpo["diagnostico"]["severidade"] in {"alta", "critica"}


@pytest.mark.parametrize(
    "falha",
    [
        anthropic.APIConnectionError(request=REQUISICAO_ANTHROPIC),
        anthropic.RateLimitError(
            "rate limited",
            response=httpx.Response(429, request=REQUISICAO_ANTHROPIC),
            body=None,
        ),
        anthropic.InternalServerError(
            "overloaded",
            response=httpx.Response(529, request=REQUISICAO_ANTHROPIC),
            body=None,
        ),
    ],
    ids=["conexao", "rate_limit_429", "sobrecarga_529"],
)
def test_indisponibilidade_do_provedor_cai_para_regras(client, chamada_claude, falha):
    chamada_claude.side_effect = falha

    resp = client.post(ROTA, json=PAYLOAD_CRITICO)

    assert resp.status_code == 200
    assert resp.json()["fonte_diagnostico"] == "regras_fallback"


@pytest.mark.parametrize(
    "resposta",
    [
        resposta_claude(stop_reason="refusal", blocos=[]),
        resposta_claude(stop_reason="max_tokens", blocos=[bloco_texto('{"causa_')]),
        resposta_claude(blocos=[]),
        resposta_claude(blocos=[bloco_texto("Nao consigo diagnosticar.")]),
        resposta_claude({"causa_provavel": "x", "severidade": "apocaliptica"}),
    ],
    ids=["recusa", "truncada", "sem_texto", "texto_livre", "fora_do_contrato"],
)
def test_resposta_inutilizavel_do_claude_cai_para_regras(client, chamada_claude, resposta):
    chamada_claude.return_value = resposta

    resp = client.post(ROTA, json=PAYLOAD_CRITICO)

    assert resp.status_code == 200
    assert resp.json()["fonte_diagnostico"] == "regras_fallback"


def test_sensor_ausente_e_reportado_e_nao_derruba_inferencia(client):
    payload = {**PAYLOAD_SAUDAVEL}
    del payload["pressao_linha_bar"]

    resp = client.post(ROTA, json=payload)

    assert resp.status_code == 200
    assert resp.json()["sensores_ausentes"] == ["pressao_linha_bar"]


def test_menos_de_tres_sensores_e_rejeitado(client):
    payload = {
        "equipamento_id": "TURB-09",
        "temperatura_mancal_celsius": 80.0,
        "rpm_rotor": 3000,
    }
    assert client.post(ROTA, json=payload).status_code == 422


def test_sensor_com_nome_errado_e_rejeitado(client):
    payload = {**PAYLOAD_SAUDAVEL}
    payload["temperatura_mancal"] = payload.pop("temperatura_mancal_celsius")

    assert client.post(ROTA, json=payload).status_code == 422


def test_valor_fora_da_faixa_fisica_e_rejeitado(client):
    payload = {**PAYLOAD_SAUDAVEL, "temperatura_mancal_celsius": 400.0}
    assert client.post(ROTA, json=payload).status_code == 422


def test_modelo_ausente_retorna_503(client, monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "inexistente.joblib"))
    ml_service.carregar_modelo.cache_clear()

    resp = client.post(ROTA, json=PAYLOAD_SAUDAVEL)

    assert resp.status_code == 503
    assert resp.headers["retry-after"] == "30"


def test_artefato_fora_do_formato_retorna_503(client, monkeypatch, tmp_path):
    import joblib

    artefato_cru = tmp_path / "modelo_cru.joblib"
    joblib.dump(["nao", "e", "um", "dict"], artefato_cru)
    monkeypatch.setenv("MODEL_PATH", str(artefato_cru))
    ml_service.carregar_modelo.cache_clear()

    assert client.post(ROTA, json=PAYLOAD_SAUDAVEL).status_code == 503


def test_modelo_volta_a_funcionar_sem_reiniciar(client, modelo_treinado, monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "ainda_nao_existe.joblib"))
    ml_service.carregar_modelo.cache_clear()
    assert client.post(ROTA, json=PAYLOAD_SAUDAVEL).status_code == 503

    monkeypatch.setenv("MODEL_PATH", str(modelo_treinado))

    assert client.post(ROTA, json=PAYLOAD_SAUDAVEL).status_code == 200


def test_health_reflete_estado_do_modelo(client, monkeypatch, tmp_path):
    assert client.get("/health").json()["modelo"] == "carregado"

    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "sumiu.joblib"))
    ml_service.carregar_modelo.cache_clear()

    assert client.get("/health").json()["status"] == "degradado"


@pytest.mark.parametrize("valor", ["abc", "0", "1.5"])
def test_limiar_invalido_impede_subida_da_api(monkeypatch, valor):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("LIMIAR_ALERTA", valor)

    with pytest.raises(ValueError, match="LIMIAR_ALERTA"):
        with TestClient(app):
            pass


def test_servico_sem_api_key_usa_regras():
    servico = LLMService(api_key="")
    telemetria = TelemetriaRequest(**PAYLOAD_CRITICO)

    diagnostico, fonte = asyncio.run(servico.diagnosticar(telemetria, 0.95))

    assert fonte == "regras_fallback"
    assert diagnostico.prazo_intervencao_horas == 8


def test_regras_apontam_o_sensor_mais_extrapolado():
    telemetria = TelemetriaRequest(
        equipamento_id="CALD-02",
        temperatura_mancal_celsius=85.0,
        vibracao_rms_mm_s=13.0,
        pressao_linha_bar=25.0,
    )

    diagnostico = diagnostico_por_regras(telemetria, 0.85)

    assert "vibracao_rms_mm_s" in diagnostico.justificativa
    assert diagnostico.componente_afetado == "rotor/mancal"


def test_regras_detectam_queda_de_pressao_por_vazamento():
    telemetria = TelemetriaRequest(
        equipamento_id="CALD-05",
        temperatura_mancal_celsius=75.0,
        vibracao_rms_mm_s=2.5,
        pressao_linha_bar=4.0,
    )

    diagnostico = diagnostico_por_regras(telemetria, 0.85)

    assert diagnostico.componente_afetado == "linha de pressao"
    assert diagnostico.severidade == "critica"
