# Uma Arquitetura de Aprendizado Federado Homomórfica para Treinamento de LSSVM usando CKKS e Decomposição QR de Householder

Este trabalho propõe uma arquitetura de Aprendizado Federado Homomórfico
One-Shot para o treinamento de Máquinas de Vetores de Suporte de Mínimos
Quadrados (LSSVM) usando o esquema CKKS. Ao reduzir o treinamento a um
sistema linear, emprega-se a decomposição QR de Householder para resolver o
sistema KKT sobre dados cifrados, evitando ramificações estruturais
incompatíveis com a criptografia homomórfica. A arquitetura é avaliada via
OpenFHE em dois datasets — Iris e o *Breast Cancer Wisconsin Diagnostic*
(WDBC) — sob particionamento IID e não-IID (Dirichlet) entre clientes, com
escalabilidade testada de 40 a 225 clientes. A solução FHE federada acompanha
de perto sua contraparte federada em texto claro em todas as configurações
(erro relativo dos parâmetros abaixo de $3\times10^{-4}$ para o WDBC e abaixo
de $3{,}5\%$ para as classes de kernel polinomial do Iris), atingindo $100\%$
de acurácia na classe linearmente separável do Iris e até $95{,}6\%$ de
acurácia no WDBC sob particionamento não-IID. A abordagem elimina múltiplas
rodadas de comunicação, garantindo privacidade robusta de dados e
estabilidade algorítmica através de diferentes escalas de dataset, números
de clientes e distribuições de dados.

Este artefato reproduz os experimentos apresentados no artigo, incluindo os
pontos levantados na revisão (avaliação em datasets maiores/de maior
dimensionalidade, comparação analítica com métodos iterativos de HE,
escalabilidade e robustez sob distribuições não-IID entre clientes via
particionamento de Dirichlet).

# Estrutura do repositório

```
.
├── README.md, LICENSE, CITATION.cff
├── pyproject.toml          # dependências (pip install -e ".[dev]"), config do pytest e do ruff
├── activate_env.sh         # ativa o venv/ do projeto
├── lssvm/                  # LSSVM em texto claro + criptografado (CKKS/OpenFHE)
│   ├── preprocessors/      # carga/split/escala por dataset (iris, breast_cancer)
│   └── solvers/            # solvers criptografados (CG, QR-Householder)
├── federated_lssvm/        # treinamento + inferência multi-parte (FedAvg sobre CKKS)
├── config/                 # scripts de execução, métricas e helpers de inicialização
├── tests/                  # testes pytest, espelhando lssvm/ e federated_lssvm/
└── paper_results/          # relatórios e métricas dos experimentos do artigo
```

`lssvm/` — núcleo LSSVM:
- `plain.py` — referência de LSSVM em texto claro
- `cipher.py` — LSSVM criptografado com CKKS
- `preprocessing.py` — normalização de features, preparo de kernel, montagem do sistema linear
- `preprocessors/` — `base.py` (split/escala/binarização comuns) e um sub-pacote por dataset (`iris/`, `breast_cancer/`) expondo `prepare_binary(...)`
- `qr_householder.py` — referência em texto claro do QR-Householder
- `inference.py` — motor de inferência criptografada
- `solvers/cg_cipher.py` — solver de Gradiente Conjugado, LHS/RHS criptografados
- `solvers/qr_householder_cipher_{col,row}.py` — variantes do QR-Householder com diferentes trade-offs entre profundidade multiplicativa e empacotamento de slots
- `solvers/_solver_common.py` — contexto CKKS, chaves de rotação, serialização de modelo/checkpoint e predição, compartilhados pelos solvers QR
- `solvers/utils.py` — helpers de rotação/máscara e simulações em texto claro (normas, sinais) usados pelos solvers
- Solvers federados com suporte a checkpoint (`--solver=`): `cg`, `qr_row` (padrão), `qr_col`

`federated_lssvm/` — pipeline federado:
- `train.py` — driver de treinamento multi-parte (k clientes, FedAvg, checkpoints)
- `worker.py` — pool de workers paralelos (fork) sobre os solves por cliente
- `baseline_run.py` — baseline de um único cliente com os parâmetros de segurança do artigo
- `plain_run.py` — mesmo pipeline federado, inteiramente em texto claro (oráculo)
- `infer.py` — inferência federada a partir dos modelos serializados
- `solver_selection.py` — parsing das flags `--solver`, `--partition`, `--alpha`, `--models-root`

`config/` — execução e métricas:
- `paper_run.sh` — reproduz os resultados reportados no artigo (wrapper de `run_campaign.sh`)
- `run_campaign.sh` — campanha completa: iris + breast_cancer, IID e Dirichlet, 128 bits
- `run_parallel.sh` — um experimento: prepara contexto → W workers → finalize (agrega + avalia)
- `run.sh` — executa um módulo Python com o venv e threads configurados
- `parallel.py` — inicialização de threads/OpenMP
- `metrics.py` — coleta de acurácia e tempo de execução
- `report.py` — gera `report.md`/`metrics.csv` a partir dos logs de uma execução
- `baseline_security_params.py` — parâmetros CKKS de referência do artigo (Seção IV-B)
- `omp_smoke.py` — verifica se o build do OpenFHE está usando OpenMP

`tests/` — `pytest` (ver "Teste mínimo"); `paper_results/` — um par `*_report.md` / `*_metrics.csv` por configuração (dataset × k × partição).

# Selos Considerados

Os selos considerados são: Disponível (Selo D).

# Informações básicas

- **Sistemas operacionais testados:** macOS (Apple Silicon, ARM64) e Ubuntu
  Linux (ARM64/x86_64, testado em instância Oracle Cloud).
- **Linguagem/runtime:** Python 3.11 (testado com 3.11.13).
- **CPU:** qualquer x86_64/ARM64 multi-core; o paralelismo (fork-pool +
  OpenMP) se ajusta automaticamente ao número de núcleos disponíveis.
- **RAM:** variável com a profundidade multiplicativa do contexto CKKS
  usado. Como referência, execuções locais deste artefato observaram picos de
  ~5–8 GB de RSS por worker; recomenda-se pelo menos 16 GB de RAM livre para
  rodar a suíte completa com múltiplos workers em paralelo (o script de
  experimentos (`config/run_campaign.sh`) dimensiona o número de workers ao
  orçamento de RAM da máquina automaticamente).
- **Disco:** poucos GB livres para compilar o OpenFHE a partir do
  código-fonte (não há pacote `openfhe` pré-compilado no PyPI).
- **Rede:** apenas para clonar este repositório e as dependências (OpenFHE,
  OpenFHE-Python) via `git clone`; os datasets (iris, breast_cancer) são
  públicos e vêm embutidos no `scikit-learn`, sem necessidade de download
  externo nem credenciais de acesso a terceiros.

# Dependências

- **Python 3.11**, com as dependências fixadas em `pyproject.toml`:
  `numpy==2.1.3`, `scipy==1.16.3`, `scikit-learn==1.8.0`; extras de
  desenvolvimento (`[dev]`): `pytest==9.0.1`, `pytest-cov==7.0.0`, `ruff`.
- **OpenFHE** (núcleo C++), branch `v1.5.1`, compilado com
  `-DWITH_OPENMP=ON` (obrigatório — sem essa flag o OpenFHE roda em thread
  única).
- **openfhe-python** — bindings Python do OpenFHE, compilados a partir do
  código-fonte via `pybind11` (não há build pré-compilado publicado).
- Ferramentas de build (Ubuntu/Debian): `build-essential cmake git
  libssl-dev libomp-dev autoconf python3.11 python3.11-venv python3.11-dev`.
  Nem Ubuntu 22.04 (Python 3.10) nem 24.04 (Python 3.12) trazem Python 3.11
  no repositório padrão — é necessário o PPA `deadsnakes`:
  `sudo add-apt-repository ppa:deadsnakes/ppa` (requer também
  `software-properties-common gnupg dirmngr ca-certificates` instalados
  antes, caso ainda não estejam).
- Nenhum dataset ou serviço de terceiros requer cadastro, chave de API ou
  autenticação — `iris` e `breast_cancer` são carregados diretamente do
  `scikit-learn`.

# Preocupações com segurança

Nenhuma. A execução deste artefato (testes, build do OpenFHE, experimentos)
não oferece risco aos avaliadores — não requer privilégios elevados além dos
pacotes de build padrão (`apt install`), não acessa rede além de clonar
dependências, e não processa dados sensíveis (usa apenas datasets públicos do
scikit-learn).

# Instalação

```bash
git clone https://github.com/victorffernandes/homomorphic-cli.git
cd homomorphic-cli
python3.11 -m venv venv
source activate_env.sh
pip install -e ".[dev]"
```

O caminho criptografado requer, adicionalmente, compilar o OpenFHE e seus
bindings Python a partir do código-fonte:

```bash
sudo apt install software-properties-common gnupg dirmngr ca-certificates
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install build-essential cmake git libssl-dev libomp-dev autoconf python3.11 python3.11-venv python3.11-dev

# 1. Núcleo C++ do OpenFHE
git clone --depth 1 --branch v1.5.1 https://github.com/openfheorg/openfhe-development.git /tmp/openfhe
cmake -S /tmp/openfhe -B /tmp/openfhe/build -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARKS=OFF -DCMAKE_BUILD_TYPE=Release -DWITH_NATIVEOPT=ON -DWITH_OPENMP=ON
cmake --build /tmp/openfhe/build -j"$(nproc)"
sudo cmake --install /tmp/openfhe/build
sudo ldconfig

# 2. Bindings Python (instalados na venv criada por activate_env.sh)
pip install pybind11
git clone --depth 1 https://github.com/openfheorg/openfhe-python.git /tmp/openfhe-python
cmake -S /tmp/openfhe-python -B /tmp/openfhe-python/build \
  -DCMAKE_PREFIX_PATH="/usr/local;$(python -m pybind11 --cmakedir)" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DPYTHON_EXECUTABLE="$(which python)"
cmake --build /tmp/openfhe-python/build -j"$(nproc)"
cp /tmp/openfhe-python/build/openfhe*.so "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-}
python -m config.omp_smoke   # PASS = OpenFHE está paralelo, FAIL = build serial
```

`LD_LIBRARY_PATH` deve ser definido em todo shell que importar `openfhe`. Ao
final deste processo, a aplicação já pode ser executada (veja "Teste mínimo"
abaixo).

# Teste mínimo

Verificação rápida de sanidade, **sem** necessidade do build do OpenFHE
(executa em segundos, usa apenas o caminho em texto claro):

```bash
pytest tests/lssvm
```

Verificação mínima do caminho criptografado (requer o build do OpenFHE
concluído), poucos minutos, com parâmetros criptográficos inseguros
(`notset`) apenas para validar a forma do pipeline:

```bash
pytest tests/federated_lssvm
```

# Experimentos

As três reivindicações abaixo correspondem diretamente às Seções 6.1, 6.2 e
6.3 do artigo (Overhead computacional, Escalabilidade e Precisão do modelo,
respectivamente). Cada uma indica o comando exato usado, os arquivos de
resultado correspondentes em `paper_results/` (já gerados por esta execução
de referência) e o tempo/recursos observados. Os comandos abaixo usam
`security=notset` (parâmetros criptográficos rápidos, mesma profundidade
multiplicativa e mesmo circuito da configuração seguraa usada no artigo —
`HEStd_NotSet` ao invés de padrões de segurança calibrados — apenas para
viabilizar a reprodução em tempo hábil); os números reportados no artigo
(Tabelas 2, 3 e 4) foram obtidos com estes mesmos comandos.

Todos os experimentos do artigo foram executados em um MacBook Air 13"
(Apple M4, 24 GB de RAM), OpenFHE v1.5.1, `OMP_NUM_THREADS` ajustado
automaticamente ao número de workers × threads de cada comando.

## Reivindicação #1 — Overhead computacional do treinamento federado em FHE (§6.1)

O treinamento e a inferência sob CKKS têm overhead de várias ordens de
magnitude em relação ao texto claro (Tabela 2 do artigo: ≈5,6×10⁶× no
treinamento, ≈5,1×10⁶× na inferência), com o tempo por cliente escalando com
a complexidade do kernel (linear vs. polinomial grau 2).

```bash
bash config/run_parallel.sh 40 5 2 --dataset=iris --security=notset --models-root=models_iris
```

- **Tempo esperado:** ~45 min em máquina de 10 núcleos (5 workers × 2 threads).
- **Recursos esperados:** picos de ~2,2 GB de RSS por worker.
- **Resultado esperado:** `paper_results/iris_k40_iid_report.md`, seção "Per
  client" — tempo de treino por cliente da ordem de dezenas de segundos por
  classe (linear mais rápido que polinomial grau 2), consistente com a
  Tabela 2 do artigo (73,8 s / 97,0 s / 111,9 s por classe).

## Reivindicação #2 — Escalabilidade em função do número de clientes (§6.2)

Tempo médio de treino por cliente, pico de RSS e custo de comunicação por
cliente variam com o número de clientes k de forma consistente com a Tabela 3
do artigo, para os dois datasets (Iris: k=40→80; WDBC: k=150→225).

```bash
bash config/run_parallel.sh 80 5 2  --dataset=iris          --security=notset --models-root=models_iris
bash config/run_parallel.sh 150 5 2 --dataset=breast_cancer --security=notset --models-root=models_bc
bash config/run_parallel.sh 225 5 2 --dataset=breast_cancer --security=notset --models-root=models_bc
```

- **Tempo esperado:** ~3–4 h (iris k=80), ~4 h (WDBC k=150), ~3,5 h (WDBC
  k=225), em máquina de 10 núcleos.
- **Recursos esperados:** picos de RSS por worker entre ~2,2 GB (iris k=40) e
  ~6 GB (WDBC), crescendo com k.
- **Resultado esperado:** comparar `paper_results/iris_k40_iid_report.md` vs
  `paper_results/iris_k80_iid_report.md`, e
  `paper_results/breast_cancer_k150_iid_report.md` vs
  `paper_results/breast_cancer_k225_iid_report.md` — tempo/cliente, RSS de
  pico e uplink/cliente reportados em cada arquivo (seções "Per worker" e
  "Communication"), consistentes com a Tabela 3 do artigo (Iris: 73,8 s→199,8
  s; WDBC: 432,9 s→250,9 s; uplink crescente com k em ambos).

## Reivindicação #3 — Precisão do modelo: IID, Dirichlet e escala (§6.3)

A solução FHE federada acompanha de perto a referência em texto claro em
todas as configurações — Padrão (IID), Dirichlet (α=0,5, não-IID) e Escala
(k maior) — com erro relativo dos parâmetros abaixo de 3×10⁻⁴ para o WDBC e
abaixo de 3,5% para as classes de kernel polinomial do Iris (Tabela 4 do
artigo). O "Baseline (média)" da Tabela 4 é o cliente único (N=2 por classe,
semente 42), já incluído automaticamente como linha "Single-client FHE" em
cada `report.md` abaixo — não deve ser confundido com os arquivos
`paper_results/*_wp3baseline_*` (uma ablação interna à parte, não citada no
artigo).

```bash
# Padrão (IID)
bash config/run_parallel.sh 40 5 2  --dataset=iris          --security=notset --models-root=models_iris
bash config/run_parallel.sh 150 5 2 --dataset=breast_cancer --security=notset --models-root=models_bc
# Dirichlet (não-IID, alpha=0.5)
bash config/run_parallel.sh 40 5 2  --dataset=iris          --security=notset --models-root=models_iris_dirichlet  --partition=dirichlet --alpha=0.5
bash config/run_parallel.sh 150 5 2 --dataset=breast_cancer --security=notset --models-root=models_bc_dirichlet   --partition=dirichlet --alpha=0.5
# Escala
bash config/run_parallel.sh 80 5 2  --dataset=iris          --security=notset --models-root=models_iris
bash config/run_parallel.sh 225 5 2 --dataset=breast_cancer --security=notset --models-root=models_bc
```

- **Tempo esperado:** ~45 min a ~4 h por comando, dependendo do dataset/k
  (ver Reivindicações #1 e #2 acima).
- **Resultado esperado**, comparando cada `report.md` em `paper_results/`
  contra a Tabela 4 do artigo:
  - Iris, Classe 0 (setosa, linear): 100% de acurácia em todas as
    configurações (`iris_k40_iid`, `iris_k40_dirichlet_a0.5`, `iris_k80_iid`).
  - Iris, Classes 1/2 (poly grau 2): 86,67%/76,67% (Padrão) →
    83,33%/73,33% (Dirichlet) → 90,00%/73,33% (Escala k=80); erro relativo
    dos pesos na faixa de 10⁻⁶ a 10⁻².
  - WDBC (linear, binário): 91,23% (Padrão, k=150,
    `breast_cancer_k150_iid`) → 95,61% (Dirichlet,
    `breast_cancer_k150_dirichlet_a0.5`) → 89,47% (Escala, k=225,
    `breast_cancer_k225_iid`); erro relativo dos pesos ~2,4–2,9×10⁻⁴ em
    todas as configurações.

# Licença

Este projeto é distribuído sob a licença **MIT** — veja [`LICENSE`](LICENSE).

# Como citar

Os metadados de citação estão em [`CITATION.cff`](CITATION.cff) (o GitHub
exibe o botão "Cite this repository" a partir dele). Equivalente em BibTeX:

```bibtex
@software{fernandes2026lssvm_fhe,
  author     = {Fernandes, Victor Faria},
  title      = {lwe: Privacy-Preserving LSSVM via Homomorphic Encryption (CKKS/OpenFHE)},
  year       = {2026},
  license    = {MIT},
  url        = {https://github.com/victorffernandes/homomorphic-cli},
  note       = {Least-squares SVM training and inference over CKKS-encrypted data
                using OpenFHE, including a federated variant (FedAvg) for
                multi-party training without sharing plaintext.}
}
```
