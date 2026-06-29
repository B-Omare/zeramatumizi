# ZeraMatumizi 🇰🇪

![CI](https://github.com/YOUR_USERNAME/zeramatumizi/actions/workflows/ci.yml/badge.svg)

> *Zera Matumizi — "Eliminate Use" / "Zero Consumption" in Swahili*

A longitudinal, open-science, production-grade Causal AI Early Warning 
and Precision Intervention System for Drug and Substance Use Disorders in Kenya.

## The Problem
Over 1.5 million Kenyan youths are grappling with drug and substance abuse.
Over 90% of rehabilitation facilities are privately owned, skewed in urban 
centres, and unaffordable to the majority of Kenyans. No data-driven early 
identification system exists at the county level.

## What ZeraMatumizi Does
1. **Predicts** which individuals and communities are at highest risk — before clinical presentation
2. **Explains** the causal pathways driving risk in Kisumu, Kakamega, Siaya, Homa Bay, and Migori
3. **Optimises** allocation of Kenya's scarce treatment resources across 47 counties

## System Architecture

| Deliverable | Module | Status |
|---|---|---|
| D1 — ETL Pipeline | `src/zeramatumizi/ingestion/` | ✅ Complete |
| D2 — Causal Inference | `src/zeramatumizi/causal/` | 🔄 In Progress |
| D3 — Bayesian Models | `src/zeramatumizi/bayesian/` | 🔄 In Progress |
| D4 — NLP & LLM | `src/zeramatumizi/nlp/` | 🔄 In Progress |
| D5 — ML Ensemble | `src/zeramatumizi/models/` | 🔄 In Progress |
| D6 — Quantum Optimisation | `src/zeramatumizi/quantum/` | 🔄 In Progress |
| D7 — API & Dashboard | `src/zeramatumizi/api/` | 🔄 In Progress |

## Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/zeramatumizi.git
cd zeramatumizi

# Create and activate environment
conda create -n zeramatumizi python=3.11
conda activate zeramatumizi

# Install the package
pip install -e .

# Run the ETL pipeline
python src/zeramatumizi/ingestion/loader.py
python src/zeramatumizi/ingestion/validator.py

# Run tests
pytest tests/ -v
```

## Data Sources

| Source | Data | Access |
|---|---|---|
| KDHS 2022 (KNBS) | National substance use prevalence | Open access |
| NACADA National Survey 2022 | County-level disorder estimates | Open access |
| Kenya DHIS2 | Treatment registrations | Government open data |
| OpenStreetMap | Facility locations, bar density | Fully open |

## Author
**Brian Omare** — omarebrian4@gmail.com

## License
MIT License