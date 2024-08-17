When you download Abell 2385 Use the following on IDE such as Spyder or Jupyter Notebook:
1. Run "layer_remover.py" to remove some unnecessry layers that give outliers)
2. Run "separation_of_cube.py" to split into slices.
3. "amani_Functions.py" is to be directly imported and used in the "source_code_<studentnumber>" for further use and shouldnt be run as a lone script.
4. Significant changes have been made with radioflux.py, so please use this one instead.
    Exmample usage:
        python3 radioflux.py freq_layer* -f sources.reg -b background.reg -s -i -o results_with_sigma.txt
    Will output all results into "results_with_sigma.txt" file which is automatically read in the source code for further plotting.
5. I have included a "requirements.txt" file that lists all my packages installed in my conda environment incase of any incompatibilties.
