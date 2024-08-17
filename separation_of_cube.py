from astropy.io import fits
import numpy as np
import astropy.units as u

def splitFrequencies(data, header, layers_in_header):
    # Extract frequency axis length
    freq_axis_length = data.shape[1]  # Assuming frequency is the second axis

    # Loop through each frequency layer
    for i in range(freq_axis_length):
        # Extract the ith frequency layer
        freq_layer = data[:, i:i+1, :, :]
        
        # Create a new FITS HDU (Header/Data Unit)
        hdu = fits.PrimaryHDU(freq_layer)
        
        # Update header to reflect the current frequency slice
        new_header = header.copy()
        
        frq = header['CRVAL3'] + i*header['CDELT3']
        
        # Update header for the frequency information
        new_header['CRPIX3'] = 1  # Reference pixel for the frequency axis
        new_header['CRPIX3'] = 1  # Reference pixel for the frequency axis
        new_header['CRPIX3'] = 1  # Reference pixel for the frequency axis
        new_header['CRVAL3'] = frq # Frequency value for the slice
        new_header['CDELT3'] = header['CDELT3']  # Frequency increment per slice
        new_header['CTYPE3'] = 'FREQ'  # Frequency axis type
        new_header['CUNIT3'] = 'Hz'  # Frequency unit
        new_header['NSPEC'] = (1, 'Number of layers of frequency')  # Number of Spectral frequencies shouls just be 1.

        output_filename = f'freq_layer_({frq*u.Hz.to(u.GHz):.2f}GHz).fits'
        hdu = fits.PrimaryHDU(freq_layer, header=new_header)
        hdu.writeto(output_filename, overwrite=True)
        print(f'Saved {output_filename}')
        
        #Removing unnecessary frequency info (or renaming)
        with fits.open(output_filename, mode='update') as hdul:
            hdr = hdul[0].header
            for j in range(1, layers_in_header+1):
                if j == i + 1:
                     hdr["N_FREQ"] = (j, "New Slice position, after stripping some")
                     continue
                # Attempt to delete keys only if they exist
                for key in [f'FREQ{str(j).zfill(4)}', f'FREH{str(j).zfill(4)}', f'FREL{str(j).zfill(4)}']:
                    if key in hdr:
                        del hdr[key]
            hdul.flush()  # Save changes to file
   

def get_headerinfo(fnum):
    fnum= str(fnum)
    fits_file = f'freq_layer_({fnum}GHz).fits' 
    
    # Open the FITS file
    hdu_list = fits.open(fits_file)
    hdu = hdu_list[0] #Ignoring the bin Table HDU
    print('__________________')
    print(f'{hdu.data.shape=}')
    print('__________________')

    return hdu.header

# Open the FITS file
fits_file = "Abell_2485_new.fits"
hdulist = fits.open(fits_file) 

# Get the data from the FITS file
hdat = hdulist[0].data
hdr = hdulist[0].header

splitFrequencies(data = hdat, header = hdr, layers_in_header=14)

hdulist.close()

#Just to see if all dimensions are kept and this can store the header information into a variable fo rfurther use
head_info = get_headerinfo(1.92)

