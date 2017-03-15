Rpackage.check: Rpackage/src/*.cpp Rpackage/src/*.h Rpackage/R/*.R Rpackage/man/*.Rd Rpackage/DESCRIPTION Rpackage/NAMESPACE Rpackage/tests/testthat/*.R R_CMD_build_check.sh 
	bash R_CMD_build_check.sh 
