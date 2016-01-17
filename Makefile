PKGDIR := ./neuralnet

default: all

all: clean install test

install: clean
	python setup.py build_ext --inplace

test:
	nosetests -v --nocapture

speed:
	python tests/speed_benchmark.py

.PHONY: clean

clean:
	\rm -f $(PKGDIR)/*.c
	\rm -f $(PKGDIR)/*.so
