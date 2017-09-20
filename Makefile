
entrega-tp1.zip:
	make -C informe informe.pdf
	mv informe/informe.pdf informe.pdf
	zip -r -9 entrega-tp1.zip informe.pdf requirements.txt src/*.py
	rm -f informe.pdf
