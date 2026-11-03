from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

from src.schemas import AnaliseTelemetriaResponse, TelemetriaRequest  # noqa: E402
from src.services.llm_service import LLMService  # noqa: E402
from src.services.ml_service import (  # noqa: E402
    ModeloIndisponivelError,
    caminho_modelo,
    carregar_modelo,
    classificar_risco,
    limiar_alerta,
    prever_risco,
)

logger = logging.getLogger("telemetria.api")


@lru_cache(maxsize=1)
def obter_llm_service() -> LLMService:
    return LLMService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Configuracao invalida derruba a subida; descobrir isso na primeira requisicao seria pior.
    limiar_alerta()
    # Subir sem modelo e aceitavel (container antes do volume); /health reporta e a rota responde 503.
    try:
        carregar_modelo(str(caminho_modelo()))
    except ModeloIndisponivelError as exc:
        logger.warning("api iniciada sem modelo: %s", exc)
    yield


app = FastAPI(
    title="Telemetria Industrial - Manutencao Preditiva",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(ModeloIndisponivelError)
async def modelo_indisponivel_handler(_: Request, exc: ModeloIndisponivelError) -> JSONResponse:
    logger.error("modelo indisponivel: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "modelo preditivo indisponivel"},
        headers={"Retry-After": "30"},
    )


# Sincrono de proposito: o primeiro carregamento le o .joblib do disco e nao pode
# bloquear o event loop; o FastAPI executa rotas sincronas em threadpool.
@app.get("/health")
def health() -> dict[str, str]:
    try:
        carregar_modelo(str(caminho_modelo()))
        return {"status": "ok", "modelo": "carregado"}
    except ModeloIndisponivelError:
        return {"status": "degradado", "modelo": "indisponivel"}


@app.post("/api/v1/telemetry/analyze", response_model=AnaliseTelemetriaResponse)
async def analisar_telemetria(
    payload: TelemetriaRequest,
    llm: LLMService = Depends(obter_llm_service),
) -> AnaliseTelemetriaResponse:
    probabilidade = await run_in_threadpool(prever_risco, payload)
    alerta = probabilidade >= limiar_alerta()

    resposta = AnaliseTelemetriaResponse(
        equipamento_id=payload.equipamento_id,
        probabilidade_falha=round(probabilidade, 4),
        nivel_risco=classificar_risco(probabilidade),
        alerta=alerta,
        sensores_ausentes=payload.sensores_ausentes(),
    )
    if alerta:
        resposta.diagnostico, resposta.fonte_diagnostico = await llm.diagnosticar(
            payload, probabilidade
        )

    logger.info(
        "analise equipamento=%s risco=%.3f alerta=%s fonte=%s ausentes=%d",
        payload.equipamento_id,
        probabilidade,
        alerta,
        resposta.fonte_diagnostico,
        len(resposta.sensores_ausentes),
    )
    return resposta
