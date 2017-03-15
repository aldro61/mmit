set -o errexit
set -o pipefail
python setup.py test | tee python_test.out
