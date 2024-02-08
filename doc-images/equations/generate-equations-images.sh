for file in *.tex; do
    if [ "$file" != "_preamble.tex" ]; then
        echo "Processing file: $file"
        pdflatex -shell-escape -synctex=1 "$file" | grep '^!.*' -A200
        pdf2svg "${file%.tex}.pdf" "${file%.tex}.svg"
        rm -f "${file%.tex}.aux" "${file%.tex}.log" "${file%.tex}.pdf" "${file%.tex}.pdf" "${file%.tex}.synctex.gz"
    else
        echo "Excluded file: $file"
    fi
done
