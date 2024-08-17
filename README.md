Test code with fits file : "Abell 2385" from : https://archive-gw-1.kat.ac.za/public/repository/10.48479/7epd-w356/data/enhanced_products/bucket_contents.html

## When you download Abell 2385 Use the following on Jupyter Notebook or IDE such as Spyder:
### 1. Run "layer_remover.py" to remove some unnecessry layers that give outliers)
### 2. Run "separation_of_cube.py" to split into slices.
### 3. "amani_Functions.py" is to be directly imported and used in the main.py for further use and shouldnt be run as a lone script.

### 4. Significant changes made with radioflux.py from https://github.com/mhardcastle/radioflux/radioflux/radioflux.py

Example usage:

        python3 radioflux.py freq_layer* -f sources.reg -b background.reg -s -i -o results_with_sigma.txt

Will output all results into "results_with_sigma.txt" file which is automatically read in the source code for further plotting.

### Changes made to original radioflux.py:

#### Imported `units` from astropy for unit conversions
        from astropy import units as u

#### Changed frequency unit conversion to display frequencies in Hz
        print('Frequencies are',[frq*u.Hz.to(u.GHz) for frq in self.frq],'GHz')

#### Added `save_to_file` parameter to `printflux` function signature
        def printflux(filename, rm, region, noise, bgsub, background=0, label='', verbose=False, save_to_file=None):

#### Included `output_lines` list and file saving code
        print('RMS values used:', fg.rms)
        output_lines.append(f'RMS values used: {fg.rms}\n')
        header = "Filename\tLabel\tFrequency(Hz)\tFlux(Jy)\tError(Jy)\tSigma_3(Jy)\tSigma_5(Jy)\n"
        print(filename,label,'%8.4g %10.6g %10.6g' % (freq,fg.flux[i],fg.error[i]))
        print(filename,label,'%8.4g %10.6g' % (freq,fg.flux[i]))

 #### Added `save_to_file` to `flux_for_files` function signature
        def flux_for_files(files, fgr, bgr=None, individual=False, bgsub=False, action=printflux, verbose=False, save_to_file=None):

#### Added file clearing to automatically overwrite the output file of results, in case it exists
        if save_to_file: #clear the old part of the file
            with open(save_to_file, 'w') as f: pass #overwrite

#### Added `fg_ir` for handling individual regions
        fg_ir = pyregion.open(fgr).as_imagecoord(rm.headers[0])
        for n, reg in enumerate(fg_ir):
            fg = pyregion.ShapeList([reg])
            r = action(filename, rm, fg, noise, bgsub, background, label=n+1, verbose=verbose, save_to_file=save_to_file)
        return r

#### Modified argument parsing to include `save_to_file` option
        parser.add_argument('-o', '--output', dest='output', action='store', default=None, help='Output file to save results')

#### Added file saving to `flux_for_files` function call
        flux_for_files(args.files, args.fgr, args.bgr, args.indiv, args.bgsub, verbose=args.verbose, save_to_file=args.output)

## 5.Check "requirements.txt" file in case of any compatibility issues


