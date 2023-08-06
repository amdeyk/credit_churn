include *.txt
include *.md
include *.pkl
recursive-include ./bank_churn_model *

include bank_churn_model/datasets/BankChurners.csv
include bank_churn_model/trained_models/*.pkl
include bank_churn_model/VERSION
include bank_churn_model/config.yml

include ./requirements/requirements.txt
include ./requirements/test_requirements.txt
exclude *.log
exclude *.cfg

recursive-exclude * __pycache__
recursive-exclude * *.py[co]
