from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SENSORES: tuple[str, ...] = (
    "temperatura_mancal_celsius",
    "vibracao_rms_mm_s",
    "pressao_linha_bar",
    "rpm_rotor",
    "horas_operacao_continua",
)

# Com menos que isso a probabilidade do modelo vira chute; melhor recusar a leitura.
MIN_SENSORES_VALIDOS = 3


class TelemetriaRequest(BaseModel):
    # Campo desconhecido e quase sempre sensor com nome errado no gateway; aceitar
    # em silencio faria a leitura virar "ausente" e mascarar o problema.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    equipamento_id: str = Field(min_length=1, max_length=64, examples=["TURB-03"])
    coletado_em: datetime | None = None
    temperatura_mancal_celsius: float | None = Field(default=None, ge=30.0, le=130.0)
    vibracao_rms_mm_s: float | None = Field(default=None, ge=0.5, le=15.0)
    pressao_linha_bar: float | None = Field(default=None, ge=1.0, le=45.0)
    rpm_rotor: float | None = Field(default=None, ge=0.0, le=3600.0)
    horas_operacao_continua: float | None = Field(default=None, ge=0.0, le=1000.0)

    @model_validator(mode="after")
    def exige_sensores_minimos(self) -> TelemetriaRequest:
        recebidos = len(self.leituras_validas())
        if recebidos < MIN_SENSORES_VALIDOS:
            raise ValueError(
                f"telemetria com menos de {MIN_SENSORES_VALIDOS} sensores validos "
                f"({recebidos} recebidos)"
            )
        return self

    def leituras_validas(self) -> dict[str, float]:
        return {
            nome: valor
            for nome in SENSORES
            if (valor := getattr(self, nome)) is not None
        }

    def sensores_ausentes(self) -> list[str]:
        return [nome for nome in SENSORES if getattr(self, nome) is None]


class Severidade(str, Enum):
    BAIXA = "baixa"
    MEDIA = "media"
    ALTA = "alta"
    CRITICA = "critica"


class DiagnosticoTecnicoLLM(BaseModel):
    """Contrato de saida imposto ao Claude via structured outputs."""

    causa_provavel: str = Field(min_length=10, description="Hipotese principal da degradacao.")
    componente_afetado: str = Field(min_length=2, description="Mancal, selo, rotor, linha de pressao etc.")
    severidade: Severidade
    acoes_recomendadas: list[str] = Field(min_length=1, max_length=5)
    prazo_intervencao_horas: int = Field(ge=0, le=720, description="Janela maxima para intervir.")
    justificativa: str = Field(min_length=10, description="Leituras que sustentam o diagnostico.")


NivelRisco = Literal["normal", "elevado", "critico"]
FonteDiagnostico = Literal["claude", "regras_fallback"]


class AnaliseTelemetriaResponse(BaseModel):
    equipamento_id: str
    probabilidade_falha: float = Field(ge=0.0, le=1.0)
    nivel_risco: NivelRisco
    alerta: bool
    sensores_ausentes: list[str] = Field(default_factory=list)
    fonte_diagnostico: FonteDiagnostico | None = None
    diagnostico: DiagnosticoTecnicoLLM | None = None
