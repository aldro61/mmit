set -o errexit
set -o pipefail
pkg_targz=$(R CMD build Rpackage|grep tar.gz|sed 's/[*] building ‘//'|sed 's/’//')
R CMD check $pkg_targz | tee test_R.out
