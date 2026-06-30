# ZeraMatumizi - XGBoost Disorder Progression Model Card

## Model Overview
Predicts 12-month substance use disorder progression risk using demographic, substance use, and socioeconomic features.

## Performance Metrics
- AUROC: 0.7284
- AUPRC: 0.6472

## Hyperparameters (Optuna-tuned)
- max_depth: 2
- learning_rate: 0.04473481595038693
- n_estimators: 181
- subsample: 0.6876417350588806
- colsample_bytree: 0.8076618331010319
- min_child_weight: 4
- reg_alpha: 0.006970214407390649
- reg_lambda: 0.1556486280212068

## Top Features (by SHAP importance)
- unemployed_num: 0.3844
- any_substance: 0.3132
- wealth_num: 0.2556
- age_of_initiation: 0.1166
- alcohol_use: 0.1082

## Fairness Audit
| Subgroup | Category | AUROC | N | Flag |
|---|---|---|---|---|
| gender | female | 0.7138 | 381 | OK |
| gender | male | 0.7402 | 419 | OK |
| hiv_status | HIV negative/unknown | 0.7326 | 759 | OK |
| hiv_status | HIV positive | 0.6789 | 41 | OK |

## Limitations
- Trained on synthetic data for development purposes
- Production deployment requires retraining on real KDHS 2022, NACADA survey, and DHIS2 data
- Performance should be re-validated on real Kenyan population data
