from __future__ import annotations

import json
import logging
import os

import anthropic
from pydantic import ValidationError

from src.schemas import (
    SENSORES,
    DiagnosticoTecnicoLLM,
    FonteDiagnostico,
    Severidade,
    TelemetriaRequest,
)

logger = logging.getLogger("telemetria.llm")

# Reenvia a requisicao para outro modelo quando o principal recusa por politica.
# Sem isso a recusa so cairia no fallback por regras, perdendo o diagnostico do LLM.
BETA_FALLBACK_SERVIDOR = "server-side-fallback-2026-07-01"

PROMPT_SISTEMA = (
    "Voce e um engenheiro de confiabilidade especialista em turbinas e caldeiras industriais. "
    "Receba leituras de telemetria de um equipamento com alto risco de falha e produza um "
    "diagnostico tecnico objetivo, em portugues. Nao invente leituras: sensores marcados como "
    "indisponiveis sao lacuna de informacao, e isso deve aparecer na justificativa."
)

# Limiares de alerta usados so no fallback, calibrados de forma conservadora
# (vibracao alinhada ao inicio da zona C da ISO 10816). Pressao falha nos dois
# sentidos: alta indica obstrucao, baixa indica vazamento.
LIMITES_ALERTA: dict[str, tuple[float | None, float, str, str]] = {
    "temperatura_mancal_celsius": (None, 100.0, "mancal", "superaquecimento do mancal"),
    "vibracao_rms_mm_s": (None, 7.1, "rotor/mancal", "desbalanceamento ou desalinhamento do rotor"),
    "pressao_linha_bar": (12.0, 38.0, "linha de pressao", "anomalia de pressao na linha"),
    "rpm_rotor": (None, 3300.0, "rotor", "instabilidade de rotacao"),
    "horas_operacao_continua": (None, 800.0, "conjunto rotativo", "desgaste por operacao continua prolongada"),
}


class LLMServiceError(Exception):
    """Falha ao obter diagnostico estruturado do Claude (rede, cota, recusa ou saida invalida)."""


class LLMService:
    def __init__(
        self,
        api_key: str | None = None,
        modelo: str | None = None,
        timeout_s: float | None = None,
        max_retries: int | None = None,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self.modelo = modelo or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        chave = api_key if api_key is not None else os.getenv("ANTHROPIC_API_KEY", "")
        if client is not None:
            self._client: anthropic.AsyncAnthropic | None = client
        elif chave:
            self._client = anthropic.AsyncAnthropic(
                api_key=chave,
                timeout=timeout_s or float(os.getenv("LLM_TIMEOUT_SEGUNDOS", "15")),
                max_retries=(
                    max_retries
                    if max_retries is not None
                    else int(os.getenv("LLM_MAX_RETRIES", "1"))
                ),
            )
        else:
            self._client = None
            logger.warning("ANTHROPIC_API_KEY ausente; diagnosticos usarao somente regras locais")

    async def diagnosticar(
        self, telemetria: TelemetriaRequest, probabilidade_falha: float
    ) -> tuple[DiagnosticoTecnicoLLM, FonteDiagnostico]:
        # O alerta nunca pode ficar sem diagnostico por causa de indisponibilidade do provedor.
        try:
            return await self._consultar_claude(telemetria, probabilidade_falha), "claude"
        except LLMServiceError as exc:
            logger.warning(
                "fallback_regras equipamento=%s motivo=%s", telemetria.equipamento_id, exc
            )
            return diagnostico_por_regras(telemetria, probabilidade_falha), "regras_fallback"

    async def _consultar_claude(
        self, telemetria: TelemetriaRequest, probabilidade_falha: float
    ) -> DiagnosticoTecnicoLLM:
        if self._client is None:
            raise LLMServiceError("cliente Anthropic nao configurado")

        try:
            resposta = await self._client.beta.messages.create(
                model=self.modelo,
                # O raciocinio do modelo consome o mesmo orcamento; com pouco espaco o JSON sai truncado.
                max_tokens=4096,
                system=PROMPT_SISTEMA,
                output_config={
                    # Diagnostico curto sobre 5 leituras: esforco baixo segura latencia e custo.
                    "effort": "low",
                    "format": {
                        "type": "json_schema",
                        "schema": anthropic.transform_schema(DiagnosticoTecnicoLLM),
                    },
                },
                betas=[BETA_FALLBACK_SERVIDOR],
                fallbacks="default",
                messages=[
                    {"role": "user", "content": _montar_mensagem(telemetria, probabilidade_falha)}
                ],
            )
        except anthropic.APITimeoutError as exc:
            raise LLMServiceError("timeout na chamada ao Claude") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMServiceError("falha de conexao com a API Anthropic") from exc
        except anthropic.RateLimitError as exc:
            raise LLMServiceError("limite de requisicoes excedido (429)") from exc
        except anthropic.APIStatusError as exc:
            raise LLMServiceError(f"API Anthropic retornou HTTP {exc.status_code}") from exc
        except anthropic.APIError as exc:
            raise LLMServiceError(f"resposta inesperada da API Anthropic: {exc}") from exc

        if resposta.stop_reason == "refusal":
            raise LLMServiceError("Claude recusou a requisicao")
        if resposta.stop_reason == "max_tokens":
            raise LLMServiceError("resposta truncada por max_tokens")

        texto = _texto_final(resposta.content)
        if not texto:
            raise LLMServiceError("resposta sem conteudo de texto")
        try:
            return DiagnosticoTecnicoLLM.model_validate_json(texto)
        except ValidationError as exc:
            raise LLMServiceError(
                f"saida do Claude fora do contrato ({exc.error_count()} erros)"
            ) from exc


def _texto_final(blocos: list) -> str:
    # Com fallback de servidor, o texto antes do ultimo bloco "fallback" e a saida
    # parcial do modelo que recusou; so o trecho depois dele e o JSON valido.
    inicio = 0
    for indice, bloco in enumerate(blocos):
        if getattr(bloco, "type", None) == "fallback":
            inicio = indice + 1
    return "".join(
        bloco.text for bloco in blocos[inicio:] if getattr(bloco, "type", None) == "text"
    )


def _montar_mensagem(telemetria: TelemetriaRequest, probabilidade_falha: float) -> str:
    leituras = {
        nome: valor if (valor := getattr(telemetria, nome)) is not None else "indisponivel"
        for nome in SENSORES
    }
    coleta = telemetria.coletado_em.isoformat() if telemetria.coletado_em else "nao informado"
    return (
        f"Equipamento: {telemetria.equipamento_id}\n"
        f"Coletado em: {coleta}\n"
        f"Probabilidade de falha estimada pelo modelo preditivo: {probabilidade_falha:.1%}\n"
        f"Leituras: {json.dumps(leituras, ensure_ascii=False)}"
    )


def _extrapolacao(nome: str, valor: float) -> float:
    """Quanto a leitura passou do limite: 1.0 e exatamente no limite, acima disso e alarme."""
    inferior, superior, _, _ = LIMITES_ALERTA[nome]
    acima = valor / superior
    abaixo = inferior / valor if inferior is not None and valor > 0 else 0.0
    return max(acima, abaixo)


def diagnostico_por_regras(
    telemetria: TelemetriaRequest, probabilidade_falha: float
) -> DiagnosticoTecnicoLLM:
    leituras = telemetria.leituras_validas()
    # Sensor que mais extrapolou o limite relativo e tratado como o suspeito principal.
    sensor, razao = max(
        ((nome, _extrapolacao(nome, valor)) for nome, valor in leituras.items()),
        key=lambda par: par[1],
    )
    _, _, componente, causa = LIMITES_ALERTA[sensor]
    critico = probabilidade_falha >= 0.9 or razao >= 1.15

    return DiagnosticoTecnicoLLM(
        causa_provavel=f"Indicio de {causa} (analise automatica por regras, sem LLM).",
        componente_afetado=componente,
        severidade=Severidade.CRITICA if critico else Severidade.ALTA,
        acoes_recomendadas=[
            f"Inspecionar {componente} e conferir a calibracao do sensor {sensor}",
            "Reduzir carga ate a inspecao presencial",
            "Reavaliar a telemetria apos a intervencao",
        ],
        prazo_intervencao_horas=8 if critico else 48,
        justificativa=(
            f"{sensor}={leituras[sensor]:.2f} ({razao:.0%} do limite de alerta); "
            f"probabilidade de falha {probabilidade_falha:.1%}. "
            "Diagnostico gerado por regras locais porque o Claude esteve indisponivel."
        ),
    )
