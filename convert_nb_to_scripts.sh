# script to convert certain notebooks to .py files

jupyter nbconvert --to script --no-prompt read_and_calculate.ipynb
jupyter nbconvert --to script --no-prompt plotting_functions.ipynb
jupyter nbconvert --to script --no-prompt read_mem1_functions.ipynb
jupyter nbconvert --to script --no-prompt custom_moc_functions.ipynb
