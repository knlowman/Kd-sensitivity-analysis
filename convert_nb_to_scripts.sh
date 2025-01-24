# script to convert certain notebooks to .py files

jupyter nbconvert --to script --no-prompt notebooks/read_and_calculate.ipynb
jupyter nbconvert --to script --no-prompt notebooks/plotting_functions.ipynb
jupyter nbconvert --to script --no-prompt notebooks/read_mem1_functions.ipynb
jupyter nbconvert --to script --no-prompt notebooks/custom_moc_functions.ipynb
