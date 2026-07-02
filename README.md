# ZeraMatumizi 🇰🇪

![CI](https://github.com/B-Omare/zeramatumizi/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

> *Zera Matumizi — "Eliminate Use" in Swahili*

A longitudinal, open-science, production-grade **Causal AI Early Warning
and Precision Intervention System** for Drug and Substance Use Disorders
in Kenya — combining causal inference, Bayesian modelling, NLP, graph
neural networks, and quantum computing into a unified public health
intelligence platform.

---

## The Problem

Over **1.5 million Kenyan youths** are grappling with drug and substance
abuse. Over 90% of rehabilitation facilities are privately owned, skewed
toward urban centres, and unaffordable to the majority of Kenyans.
**No data-driven early identification system exists at the county level.**

ZeraMatumizi addresses this gap by:
1. **Predicting** which individuals and communities are at highest risk —
   before clinical presentation
2. **Explaining** the causal pathways driving risk across 47 counties
3. **Optimising** allocation of Kenya's scarce treatment resources
4. **Generating** actionable intelligence for NACADA officers in both
   Swahili and English

---

## System Architecture
Raw Data (KDHS 2022, NACADA, DHIS2, OSM)
│
▼
D1: ETL Pipeline ──────────────────────────────────────────────
│  loader.py → validator.py → cleaner.py
│  4,000 respondents, 13 features, Pandera schema validation
▼
D2: Causal Inference ──────────────────────────────────────────
│  dag.py          → Interactive causal DAG (26 nodes)
│  did_analysis.py → NACADA campaigns: -27% disorder rate
│  rdd_analysis.py → Age-18 threshold: +56% disorder risk
│  iv_analysis.py  → Chang'aa proximity IV: β=0.557
▼
D3: Bayesian Hierarchical Model ───────────────────────────────
│  hierarchical_model.py → County risk with credible intervals
│  Nyamira 14.1% [9.7%, 20.7%] ... Homa Bay 11.7% [7.5%, 16.9%]
▼
D4: NLP & LLM Pipeline ────────────────────────────────────────
│  swahili_ner_model.py   → Swahili SUD NER (5 entity types)
│  rag_pipeline.py        → NACADA counsellor RAG assistant
│  topic_modelling.py     → BERTopic emerging trend detection
│  sentiment_analysis.py  → Multilingual distress flagging
│  report_generator.py    → Automated county situation reports
│  message_generator.py   → De-stigmatisation message generation
▼
D5: ML Ensemble & Explainable AI ──────────────────────────────
│  xgboost_classifier.py     → AUROC 0.73, SHAP explainability
│  random_survival_forest.py → C-index 0.70+, time-to-onset
│  gnn_peer_network.py       → AUROC 0.93 with peer network
│  isolation_forest.py       → Anomaly detection (individual + county)
│  quantum_kmeans.py         → Quantum K-Means risk stratification
▼
D6: Quantum Optimisation [Coming Soon] ────────────────────────
D7: API & Dashboard      [Coming Soon] ────────────────────────

---

## Key Results

| Module | Finding |
|---|---|
| DiD Analysis | NACADA campaigns causally reduced SUD registrations by **27%** |
| RDD Analysis | Legal alcohol access at 18 causally increases disorder risk by **56%** |
| IV Analysis | Chang'aa proximity instrument F-stat **290.82** (strong instrument) |
| Bayesian Model | Nyamira highest county risk at **14.1%** [9.7%, 20.7%] |
| XGBoost | AUROC **0.73**, top features: unemployment, wealth, substance use |
| Survival Forest | C-index **0.70+**, identifies cases needing intervention within 26 months |
| GNN | Peer network adds **+0.21 AUROC** over individual features alone |
| Isolation Forest | Kisii and Kisumu flagged as county-level anomalies |
| Quantum K-Means | 10-county risk stratification into Low/Medium/High tiers |
| Swahili NER | 5 entity types: SUBSTANCE, RISK_FACTOR, SEVERITY, GEOGRAPHIC, TREATMENT |
| RAG Pipeline | Bilingual Swahili/English protocol-grounded NACADA counsellor assistant |

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/B-Omare/zeramatumizi.git
cd zeramatumizi

# Create and activate environment
conda create -n zeramatumizi python=3.11
conda activate zeramatumizi

# Install the package
pip install -e .

# Run the ETL pipeline
python src/zeramatumizi/ingestion/loader.py
python src/zeramatumizi/ingestion/validator.py

# Run causal inference
python src/zeramatumizi/causal/dag.py
python src/zeramatumizi/causal/did_analysis.py

# Run ML ensemble
python src/zeramatumizi/models/xgboost_classifier.py

# Run tests
pytest tests/ -v
```

---
## Containerisation

The project includes a `Dockerfile` for containerised deployment.

### Using Podman (recommended — rootless, no daemon):
```bash
# Build
podman build -t localhost/zeramatumizi:latest .

# Run
podman run -d --name zeramatumizi \
  -p 8000:8000 -p 8501:8501 \
  -e GROQ_API_KEY=your-key-here \
  -v ./data:/app/data \
  -v ./docs:/app/docs \
  localhost/zeramatumizi:latest
```

### Using Docker:
```bash
docker build -t zeramatumizi:latest .
docker run -d --name zeramatumizi \
  -p 8000:8000 -p 8501:8501 \
  -e GROQ_API_KEY=your-key-here \
  zeramatumizi:latest
```

### Access:
- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs

## Project Structure
zeramatumizi/
├── src/zeramatumizi/
│   ├── ingestion/          # D1: ETL pipeline
│   │   ├── loader.py
│   │   └── validator.py
│   ├── causal/             # D2: Causal inference
│   │   ├── dag.py
│   │   ├── did_analysis.py
│   │   ├── rdd_analysis.py
│   │   └── iv_analysis.py
│   ├── bayesian/           # D3: Bayesian models
│   │   └── hierarchical_model.py
│   ├── nlp/                # D4: NLP & LLM
│   │   ├── swahili_ner_data.py
│   │   ├── swahili_ner_model.py
│   │   ├── rag_pipeline.py
│   │   ├── topic_modelling.py
│   │   ├── sentiment_analysis.py
│   │   ├── report_generator.py
│   │   └── message_generator.py
│   ├── models/             # D5: ML ensemble
│   │   ├── xgboost_classifier.py
│   │   ├── random_survival_forest.py
│   │   ├── gnn_peer_network.py
│   │   ├── isolation_forest.py
│   │   └── quantum_kmeans.py
│   ├── quantum/            # D6: Quantum optimisation [Coming Soon]
│   ├── api/                # D7: FastAPI backend [Coming Soon]
│   └── dashboard/          # D7: Streamlit dashboard [Coming Soon]
├── data/
│   ├── raw/                # KDHS sample, social media posts
│   └── chroma_db/          # RAG vector store
├── docs/
│   ├── reports/            # All generated plots and reports
│   │   └── shap/           # SHAP explainability plots
│   └── model_card.md       # Responsible AI model card
├── tests/
│   └── unit/               # 9 automated unit tests
├── .github/workflows/      # CI/CD (GitHub Actions)
├── configs/
└── pyproject.toml

---

## Data Sources

| Source | Data | Access |
|---|---|---|
| KDHS 2022 (KNBS) | National substance use prevalence | Open access |
| NACADA National Survey 2022 | County-level disorder estimates | Open access |
| Kenya DHIS2 | Treatment registrations | Government open data |
| OpenStreetMap | Facility locations, bar density | Fully open |

> **Note:** This repository uses synthetic sample data matching the
> statistical properties of the above sources for development.
> Production deployment requires the real datasets.

---

## Generated Outputs

All pipeline outputs are saved to `docs/reports/`:

| Output | Description |
|---|---|
| `causal_dag.html` | Interactive causal DAG (open in browser) |
| `did_analysis.png` | DiD treatment effect plot |
| `rdd_analysis.png` | RDD age-18 threshold plot |
| `iv_analysis.png` | IV 2SLS vs OLS comparison |
| `bayesian_county_risk.png` | County risk ranking with credible intervals |
| `topic_modelling.png` | BERTopic emerging trend clusters |
| `sentiment_analysis.png` | Distress signal distribution |
| `shap/shap_beeswarm.png` | SHAP feature impact summary |
| `shap/shap_waterfall_*.png` | Individual risk explanations |
| `survival_curves.png` | High vs low risk survival curves |
| `gnn_results.png` | GNN vs MLP ablation study |
| `isolation_forest.png` | Anomaly detection PCA and county rates |
| `quantum_kmeans.png` | Classical vs quantum clustering |
| `county_report_*.md` | Auto-generated county situation reports |
| `destigma_messages_*.md` | Generated de-stigmatisation messages |

---

## Responsible AI

This project includes a **Model Card** (`docs/model_card.md`) documenting:
- Performance metrics and limitations
- Fairness audit across gender and HIV status subgroups
- Data provenance and synthetic data disclaimer
- Recommended use and misuse warnings

**NACADA helpline: 1192 (toll-free, 24 hours)**

---

## Roadmap

- [x] D1 — ETL Pipeline
- [x] D2 — Causal Inference (DAG, DiD, RDD, IV)
- [x] D3 — Bayesian Hierarchical Model
- [x] D4 — NLP & LLM Pipeline (NER, RAG, Topics, Sentiment, Reports)
- [x] D5 — ML Ensemble (XGBoost, RSF, GNN, Isolation Forest, Quantum K-Means)
- [ ] D6 — Quantum Optimisation (QAOA resource allocator)
- [ ] D7 — FastAPI + Streamlit Bilingual Dashboard

---

## Author

**Brian Omare**
- Email: omarebrian4@gmail.com
- GitHub: [B-Omare](https://github.com/B-Omare)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

If you use this work in research, please cite:
Omare, B. (2026). ZeraMatumizi: A Causal AI Early Warning and
Precision Intervention System for Drug and Substance Use Disorders
in Kenya. GitHub. https://github.com/B-Omare/zeramatumizi