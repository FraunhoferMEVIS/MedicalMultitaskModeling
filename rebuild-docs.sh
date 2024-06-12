# You might need to run
export LC_ALL=C
rm -f docs/source/auto -r
sphinx-apidoc -o docs/source/auto mmm/
sphinx-build -b html docs/source/ docs/build/
