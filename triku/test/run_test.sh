rm -f .cov*
rm -rf htmlcov

pytest -s -m "general" --cov=./triku/  --cov-append
mv .coverage .coverage.general

pytest -s -n 4 -m "output_check" --cov=./triku/  --cov-append
mv .coverage .coverage.ouput_check

pytest -s -n 4 -m "var_check" --cov=./triku/  --cov-append
mv .coverage .coverage.var_check

pytest -s -n 4 -m "calc_check" --cov=./triku/  --cov-append
mv .coverage .coverage.calc_check

pytest -s -n 4 -m "exception_check" --cov=./triku/  --cov-append
mv .coverage .coverage.exception_check

coverage combine .cov*
pytest -s -n 1 -m "end" --cov=./triku/ --cov-append --cov-report=html
codecov --token=6e1967cb-4cf2-4eee-b32a-82b0ca1725a0