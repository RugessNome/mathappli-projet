taskkill /im AcroRd32.exe
pdflatex main
bibtex main
pdflatex main
start main.pdf
copy /y main.pdf "../rapport.pdf"