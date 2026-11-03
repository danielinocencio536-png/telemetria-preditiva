from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.services import ml_service
from src.services.llm_service import LLMService
from src.train import treinar

DIAGNOSTICO_CLAUDE = {
    "causa_provavel": "Desalinhamento do rotor com sobreaquecimento do mancal",
    "componente_afetado": "mancal dianteiro",
    "severidade": "critica",
    "acoes_recomendadas": ["Parar a turbina", "Inspecionar o alinhamento"],
    "prazo_intervencao_horas": 6,
    "justificativa": "Vibracao e temperatura muito acima da faixa operacional.",
}

PAYLOAD_CRITICO = {
    "equipamento_id": "TURB-03",
    "temperatura_mancal_celsius": 118.0,
    "vibracao_rms_mm_s": 12.5,
    "pressao_linha_bar": 41.0,
    "rpm_rotor": 3480,
    "horas_operacao_continua": 950,
}

PAYLOAD_SAUDAVEL = {
    "equipamento_id": "TURB-01",
    "temperatura_mancal_celsius": 68.0,
    "vibracao_rms_mm_s": 2.0,
    "pressao_linha_bar": 24.0,
    "rpm_rotor": 3000,
    "horas_operacao_continua": 120,
}


def bloco_texto(texto: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=texto)


def resposta_claude(
    diagnostico: dict | None = None,
    stop_reason: str = "end_turn",
    blocos: list | None = None,
) -> SimpleNamespace:
    if blocos is None:
        blocos = [bloco_texto(json.dumps(diagnostico or DIAGNOSTICO_CLAUDE))]
    return SimpleNamespace(content=blocos, stop_reason=stop_reason)


@pytest.fixture(scope="session")
def modelo_treinado(tmp_path_factory: pytest.TempPathFactory):
    destino = tmp_path_factory.mktemp("modelos") / "modelo_teste.joblib"
    treinar(n_amostras=800, seed=7, destino=destino)
    return destino


@pytest.fixture
def cliente_anthropic() -> SimpleNamespace:
    criar = AsyncMock(return_value=resposta_claude())
    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=criar)))


@pytest.fixture
def chamada_claude(cliente_anthropic) -> AsyncMock:
    return cliente_anthropic.beta.messages.create


@pytest.fixture
def client(modelo_treinado, cliente_anthropic, monkeypatch):
    from src.main import app, obter_llm_service

    monkeypatch.setenv("MODEL_PATH", str(modelo_treinado))
    # main.py chama load_dotenv(); fixa o limiar para o .env local nao alterar o resultado da suite.
    monkeypatch.setenv("LIMIAR_ALERTA", "0.80")
    ml_service.carregar_modelo.cache_clear()
    servico = LLMService(client=cliente_anthropic, modelo="claude-teste")
    app.dependency_overrides[obter_llm_service] = lambda: servico

    with TestClient(app) as tc:
        yield tc

    app.dependency_overrides.clear()
    ml_service.carregar_modelo.cache_clear()
