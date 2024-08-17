from astropy.io import fits
import numpy as np

def remove_layers(filename: str, layers_to_remove: list):

    hdul = fits.open(filename)
    data = hdul[0].data

    #removing specified layers
    new_data = np.delete(data, np.array(layers_to_remove)-1, axis=1)

    #updating FITS file with modified data
    hdul[0].data = new_data
    hdul.writeto(f"{filename.split('.fits')[0]}_new.fits", overwrite=True)

    print(fits.open("Abell_2485.fits")[0].data.shape)
    print(fits.open("Abell_2485_new.fits")[0].data.shape)
    
    hdul.close()

#Change name to required fits file
#Choose specific layers to strip off if needed
remove_layers(filename="Abell_2485.fits", layers_to_remove=[1, 2, 9, 10])

