test: test_R.out test_python.out
	cat test_R.out test_python.out
test_R.out: Rpackage/src/*.cpp Rpackage/src/*.h Rpackage/R/*.R Rpackage/man/*.Rd Rpackage/DESCRIPTION Rpackage/NAMESPACE Rpackage/tests/testthat/*.R test_R.sh 
	bash test_R.sh 
test_python.out: mmit/tests/*.py mmit/core/*.cpp mmit/core/*.h setup.py test_python.sh
	bash test_python.sh
