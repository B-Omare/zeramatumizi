import sys
sys.path.insert(0, 'src')
from zeramatumizi.api.main import get_xgboost_model, engineer_individual_features
from zeramatumizi.api.schemas import IndividualRiskRequest

req = IndividualRiskRequest(
    age=24, gender='male', education_level='secondary',
    wealth_index='poor', alcohol_use=1, cannabis_use=0,
    khat_use=1, age_of_initiation=16,
    hiv_status='negative', employment_status='unemployed'
)

try:
    model = get_xgboost_model()
    print("Model loaded OK")
    X = engineer_individual_features(req)
    print("Features engineered OK")
    print(X)
    result = model.predict_proba(X)
    print(f"Prediction OK: {result}")
except Exception as e:
    import traceback
    traceback.print_exc()