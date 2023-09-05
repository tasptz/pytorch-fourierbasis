isort .
black .
jupyter nbconvert --to markdown --execute --output ReadMe example.ipynb
jupyter nbconvert --clear-output --inplace example.ipynb
