rm .cov*
rm -r htmlcov

pytest -n 10 -m "parallel" --cov=./scallop/  --cov-append
pytest -n 1 -m "parallel_individual" --cov=./scallop/ --cov-append --cov-report=html

