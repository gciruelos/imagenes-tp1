
informe.pdf: informe.tex estudio_cantidad_histogramas.tex estudio_variacion_parametros.tex ejemplos.tex
	latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make informe.tex

view-informe: informe.pdf
	zathura informe.pdf &

clean:
	latexmk -CA
