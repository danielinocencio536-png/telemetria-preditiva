# Pipeline de Telemetria Industrial para Manutenção Preditiva

Diagnóstico automatizado de falhas em turbinas e caldeiras: um modelo XGBoost estima a probabilidade de falha a partir da telemetria e, quando o risco passa de 80%, o Claude gera um diagnóstico técnico estruturado.

![CI](https://github.com/danielinocencio536-png/telemetria-preditiva/actions/workflows/ci.yml/badge.svg)
![Cobertura](https://img.shields.io/badge/cobertura-96%25-brightgreen)
![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-009688)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange)
![Docker](https://img.shields.io/badge/docker-ready-2496ED)

## Fluxo de dados

```mermaid
flowchart LR
    A[Sensores / gateway] -->|POST /api/v1/telemetry/analyze| B[FastAPI + Pydantic]
    B -->|valida faixas físicas<br/>mín. 3 sensores| C[ml_service<br/>XGBoost em cache]
    C -->|probabilidade de falha| D{risco >= 80%?}
    D -->|não| E[Resposta: nível normal/elevado]
    D -->|sim| F[llm_service<br/>Claude + structured outputs]
    F -->|timeout / 429 / recusa / saída inválida| G[Diagnóstico por regras]
    F -->|DiagnosticoTecnicoLLM| H[Resposta com diagnóstico]
    G --> H
```

## Instalação

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt                   # só runtime: requirements.txt
cp .env.example .env                                  # preencha ANTHROPIC_API_KEY

python -m src.train                                   # gera models/predictive_model.joblib
uvicorn src.main:app --reload
```

Exemplo de chamada:

```bash
curl -X POST localhost:8000/api/v1/telemetry/analyze \
  -H "Content-Type: application/json" \
  -d '{"equipamento_id":"TURB-03","temperatura_mancal_celsius":118,"vibracao_rms_mm_s":12.5,
       "pressao_linha_bar":41,"rpm_rotor":3480,"horas_operacao_continua":950}'
```

Sensores podem ser omitidos (mínimo de 3 válidos); os ausentes voltam em `sensores_ausentes`. Campos desconhecidos são rejeitados com 422: um sensor com nome errado no gateway não vira "ausente" em silêncio.

### Docker

```bash
docker compose run --rm api python -m src.train   # treina e grava em ./models
docker compose up --build
```

No Linux, o container roda como usuário não-root (uid 10001) e pode não ter permissão de escrita em `./models`. Nesse caso, treine com o seu usuário: `docker compose run --rm --user "$(id -u):$(id -g)" api python -m src.train`.

### Testes

```bash
python -m pytest
```

A suíte treina um modelo pequeno em diretório temporário e substitui o cliente Anthropic por um mock: nenhuma chamada de rede, nenhuma chave necessária. Além da API, ela verifica o gerador de dados (faixas físicas, taxa de falha, determinismo pela seed) e a qualidade do modelo: um AUC acima de 0,99 também reprova, porque indicaria dados triviais.

A cobertura é medida em toda execução (`pytest-cov`) e fica em ~96%. Abaixo de 90%, a suíte falha, inclusive no CI. Avisos de depreciação também quebram os testes, para que dependências desatualizadas não passem despercebidas.

## Decisões de Arquitetura & Trade-offs

**Dados sintéticos com desgaste contínuo.** O treino gera 2.500 amostras a partir de um desgaste latente em [0, 1]: os sensores derivam junto com ele (mais temperatura, vibração e oscilação de rotação; pressão subindo por obstrução ou caindo por vazamento), e a falha é sorteada por uma curva logística do desgaste. Dois equipamentos com o mesmo desgaste podem ter destinos diferentes, como em planta. Há ainda ruído de medição e 1,5% de leituras ausentes. A primeira versão separava as classes por construção e chegava a AUC ≈ 1,0 com 99% das previsões abaixo de 5% ou acima de 95%. Esse modelo parece ótimo, mas não informa nada sobre o risco intermediário. Em produção, o primeiro passo é trocar `gerar_telemetria_sintetica` por histórico real rotulado com ordens de manutenção.

## Desempenho do modelo

Conjunto de teste estratificado (20% de 2.500 amostras, ~17% de falhas), `python -m src.train`:

| Métrica | Valor | Leitura |
|---|---|---|
| ROC-AUC | 0,936 | Ordena bem equipamentos por risco |
| PR-AUC | 0,795 | Bom para a classe rara (o acaso daria ~0,17) |
| Brier | 0,069 | Probabilidades calibradas: quando o modelo diz ~30%, falham ~30% |
| Limiar 0,50 (`elevado`) | precisão 0,76 · recall 0,65 | Triagem: pega 2 de cada 3 falhas |
| Limiar 0,80 (`critico`, aciona o LLM) | precisão 0,88 · recall 0,42 | Poucos alarmes falsos; casos menos claros ficam em `elevado` |

O limiar crítico privilegia precisão de propósito: cada alerta custa uma chamada ao LLM e uma ida da equipe ao equipamento. O nível `elevado` existe para não perder o restante das falhas. Baixar `LIMIAR_ALERTA` troca alarmes falsos por recall, sem retreinar.

O treino não usa `scale_pos_weight`: o peso de classe distorceria as probabilidades, e a API decide pelo valor da probabilidade, não só pela classe prevista.

**XGBoost em vez de rede neural.** Dados tabulares com cinco variáveis: árvores treinam em segundos, lidam com `NaN` nativamente (sensor ausente não exige imputação) e dispensam GPU. O custo é não capturar dependência temporal; janelas deslizantes (média/derivada de vibração) seriam a evolução natural.

**LLM só acima do limiar.** O modelo clássico é barato e determinístico; o Claude entra apenas nos casos críticos (`LIMIAR_ALERTA`, padrão 0,80). Reduz custo e latência para a maioria das requisições, ao preço de não haver explicação textual para riscos intermediários.

**Saída estruturada via structured outputs.** O schema Pydantic `DiagnosticoTecnicoLLM` vai em `output_config.format`, e a API restringe a geração a esse JSON. A resposta ainda é revalidada pelo Pydantic, porque limites como `minLength` e `maxItems` não são impostos pela API. Saída fora do contrato é tratada como falha do provedor, não como dado confiável. Tool use com `tool_choice` forçado não serve aqui: os modelos atuais rejeitam essa opção com HTTP 400.

**Fallback em duas camadas.** Se o modelo principal recusar a requisição por política, o fallback de servidor da Anthropic (`fallbacks: "default"`) refaz a chamada em outro modelo, dentro da mesma requisição. Se ainda assim não houver diagnóstico, por timeout, 429, erro de conexão, recusa, resposta truncada ou inválida, ou falta de chave, entra o diagnóstico heurístico (`fonte_diagnostico: "regras_fallback"`). Um alerta crítico nunca fica sem diagnóstico por causa de terceiros. O custo é um texto genérico, sinalizado de forma explícita na resposta.

**Cache do modelo com `lru_cache`.** O artefato carrega uma vez por processo. Como exceções não são memorizadas, um modelo que aparece depois (volume montado, retreino) é carregado na próxima requisição sem reiniciar; enquanto isso a API responde 503 com `Retry-After`. Trocar um modelo já carregado exige reiniciar o processo ou chamar `carregar_modelo.cache_clear()`.

**Modelo fora da imagem Docker.** O `.joblib` é montado por volume: a imagem fica enxuta (multi-stage, usuário não-root) e o modelo é versionado separadamente do código.

**Modelo de LLM configurável.** O identificador vem de `ANTHROPIC_MODEL`; trocar de modelo não exige deploy de código.
