#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created on Sun Jul 28 12:10:10 2024

@author: amani
"""
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker
from astropy.io import fits
import astropy.units as u
import astropy.constants as c
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from matplotlib.patches import Circle
import warnings #Used for suppressing specific Astropy warnings (Like WCS warnings)
from astropy.utils.exceptions import AstropyWarning
from astropy.cosmology import Planck18 as cosmo #To calculate comobing distance
import amani_functions as amF #My personal functions
import pandas as pd

warnings.simplefilter('ignore', AstropyWarning)

filename = "Abell_2485_new.fits"

hdu_obj = fits.open(filename)

#print(hdu_obj) gives --> [<astropy.io.fits.hdu.image.PrimaryHDU object at 0x74f608124510>, 
#                         <astropy.io.fits.hdu.table.BinTableHDU object at 0x74f6081f5b10>]

hdu = hdu_obj[0]
hdu.header #(or fits.getheader(filename))

#getting the world coordinate system (wcs) from the fit file header of interest (the default one):
wcs= WCS(hdu.header)

random_coord = SkyCoord(ra="22:48:30", dec="-16:15:00", unit=(u.hourangle, u.deg), frame="fk5")

cbars_pad = .05 #Padding for colorbars
labelpad = 25 #Padding for colorbar labels
rotation = -90 #Rotation for colorbar labels
#Looping through all 12 layers
#for h in range(300, 280, -20):

def get_csv_dat(filename): #For extracting data from csv files downloaded from online sourecs.
    df = pd.read_csv(filename)
    filtered_df = df.dropna(subset=['Frequency', 'Flux Density']) #Keeping only the rows that have them both
    return list(zip(filtered_df['Frequency'], filtered_df['Flux Density']))

def see_layer(layer, objs_collected, chosen=(0, 1, 2, 3, 4), manual=False, loop=False, **kw):
    def _see_layer(layer):
        #Creating figure and axis with WCS projection
        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_subplot(111, projection=wcs, slices=('x', 'y', 0, 0))

        n = kw["cap_flux"] if "cap_flux" in kw else 1e-4  #Capping the JY/beam
        #Plotting data with dynamic color scaling
        norm = Normalize(vmin=0, vmax=n)
        img = ax.imshow(hdu.data[0, layer, :, :], origin='lower', cmap='inferno', norm=norm)

        im_big_cbar = fig.colorbar(img, ax=ax, fraction=.03, pad=cbars_pad)
        im_big_cbar.set_label(fits.getval(filename, 'BUNIT'), rotation=rotation,
                              labelpad=labelpad, fontsize=16)

        #Formatting numbering of colorbar
        formatter = plt.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 0))
        im_big_cbar.ax.yaxis.set_major_formatter(formatter)

        ax.grid(ls='solid', alpha=.3)
        ax.coords[0].set_axislabel('Right Ascension (J2000)', fontsize=16)
        ax.coords[1].set_axislabel('Declination (J2000)', fontsize=16)

        h2 = 5e-2  #Random position locator variable
        rsltn = fits.getval(filename, "CLEANBMJ")

        #Plotting the circle to show the maximum resolution
        Resolution = Circle((random_coord.ra.deg - h2, random_coord.dec.deg - h2),
                            radius=rsltn,
                            color='white', fill=False,
                            transform=ax.get_transform('world'), linewidth=2,
                            label='Resolution =' + "{:.3e}".format(rsltn) + " deg \n (14.97 arcsecs)")
        ax.add_patch(Resolution)

        #Create insets for each selected object
        for ind, (obj, _id) in enumerate(zip([objs_collected[i] for i in chosen],
                                             np.arange(1, len(chosen) + 1))):
            #Setting the limits for the inset
            center = wcs.world_to_pixel_values(obj.ra, obj.dec, 0, 0)[:2]
            h = 100  #Scaling used in zooming in or out RA and Dec
            x1, x2 = center[0] - h, center[0] + h
            y1, y2 = center[1] - h, center[1] + h

            #Axins idea retrieved from: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/zoom_inset_axes.html
            #Creating the inset
            #n=1.04 #Scale factor to push them outwards
            #axins_pos = [(.8*n,.3*(-n)),(.05*(-n),.1*(-n)),
            #             (.7*n,.1*(-n)),(.55*n,.75*n),
            #             (.2*(-n),.55*n)]#Manual Positions of the axins
            
            #Sorting them out in a different way than their ID:
            ind_old = ind #Storing to get back on track later for the loop
            #ID_2 goes to position 1, ID_5 to pos 2, ID_4 to pos 3, ID_1 to pos 4, ID_3 to pos 5
            pos = [2, 1, 4, 3, 0] #Switching positions list (where its pos-1 for correct indexing)
            ind = pos.index(ind) #Doing the switcharoooo!
            axins = ax.inset_axes(
                #[pos_x, pos_y, width, height]
                [-0.18 + ind*0.3, 1.03,  0.2, 0.2],  #Positioning and size of the inset
                xlim=(x1, x2), ylim=(y1, y2),
                projection=wcs, slices=('x', 'y', 0, 0), xticklabels=[], yticklabels=[]) 
            ind = ind_old #Going back on track
            axins.imshow(hdu.data[0, layer, :, :], origin='lower', cmap='inferno', norm=norm)
            axins.scatter(obj.ra.deg, obj.dec.deg, marker='',
                          transform=axins.get_transform('world'), s=20)

            #Removing tickmarks from axins plot for x axis
            axins.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False, which="major", axis="x") 
            #Removing tickmarks from axins plot for y axis
            axins.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False, which="major", axis="y") 
            
            #Set the edge color of the inset axis to white
            sp_col = "red"
            axins.spines['top'].set_color(sp_col)
            axins.spines['bottom'].set_color(sp_col)
            axins.spines['left'].set_color(sp_col)
            axins.spines['right'].set_color(sp_col)
        
            #Draw lines towards the inset with a white color 
            ax.indicate_inset_zoom(axins, edgecolor=sp_col, linewidth=1.5)
            
            axins.set_title(f"ID: {_id}", fontsize=12, color="k")
            
            
            #Hide tick labels on the inset
            axins.set_xticklabels([])
            axins.set_yticklabels([])

        plt.tight_layout()
        plt.show()
    
    if loop:
        for layer in range(12):
            _see_layer(layer)
    else:
        _see_layer(layer-1) #easier to interprate when using for individual layers
        
#Zooms in on fit file areas that have been identified by sources collected from online database
#Limit of number of sources can be increased or decreased by 'limit_num' when calling amF.get_sources().
def zoom_chosen(objs_collected, chosen=(0,1,2,3,4), manual=False):
    #Chosen -> The index of the objects that have been found.
    if manual: #If sources were selected manually rather than using astroquery.
        ID = np.arange(1,6) #Creating manual ID of 5 sources.
        
    for ind, (obj, _id) in enumerate(zip([objs_collected[i] for i in chosen], 
                                         [ID[i] for i in chosen]
                                         )
                                     ):
        layer=0
        
        #Creating figure and axis with WCS projection
        fig = plt.figure(figsize=(8,7))
        ax = fig.add_subplot(111, projection=wcs, slices=('x', 'y', 0, 0))
        
        n=1e-4 #Capping the JY/beam
        #Plotting data with dynamic color scaling
        norm = Normalize(vmin=0, vmax=n)
        img = ax.imshow(hdu.data[0, layer, :, :], origin='lower', cmap='inferno', norm=norm)
        
        im_big_cbar = fig.colorbar(img, ax=ax, fraction=.03,  pad = cbars_pad)
        im_big_cbar.set_label(fits.getval(filename, 'BUNIT'), rotation=rotation, 
                              labelpad=labelpad, fontsize=16)
        
        #Formatting numbering of colorbar
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 0))
        im_big_cbar.ax.yaxis.set_major_formatter(formatter)
    
        ax.grid(ls='solid', alpha=.3)
        ax.coords[0].set_axislabel('Right Ascension (J2000)', fontsize=16)
        ax.coords[1].set_axislabel('Declination (J2000)', fontsize=16)
    

        ##Adding a red star to clasrify point of zoom focus by using SkyCoord
        #ax.scatter(obj.ra, 
        #           obj.dec, 
        #           color='r', 
        #           marker='*',  
        #           transform=ax.get_transform('world'), 
        #           s=20)
    
        #Setting axis limits
        center = wcs.world_to_pixel_values(obj.ra, obj.dec, 0 ,0)[:2]
        h=300 #scaling used in zooming in or out RA and Dec
        ax.set_xlim(center[0]-h, center[0]+h)
        ax.set_ylim(center[1]-h, center[1]+h)
        
    
        #Plottting the circle to show the maximum resolution
        Resolution = Circle((obj.ra.deg, obj.dec.deg), 
                            radius=0.03,
                            color='white', fill=False, 
                            transform=ax.get_transform('world'), linewidth=2)
        ax.add_patch(Resolution)
    
        if manual: plt.title(f"ID: {_id}", fontsize=20)
        else: plt.title(f"(ind={chosen[ind]}) ID: {_id}", fontsize=20)
        
        plt.tight_layout()
        plt.show()

#this funciton looks at the header information in the FITs file and
#Identifies the upper and lower frequencies of each slice
#This helps determine the upper and lower uncertainty of each frequency layer
#this will mostly be run on ipython when quoting the fitfile name.
def get_freq_errors(filename: str, **kw):
    #Get the upper and lower boundaries of each frequency
    def get_freq_errors(filename: str):
        layers_count = fits.getval(filename, "NSPEC")
        freq_s = []
        if fits.open(filename)[0].data.shape[1] == 1: #Only a slice
            freq = []
            for typ in ['Q', 'L', 'H']:  #form is in tuple(F, L_unc, H_unc)
                frq = fits.getval(filename, 'FRE' + typ + str(kw['layer']).zfill(4))
                freq.append(frq)
            freq_s.append(tuple(freq))
        else:
            for s in [str(num).zfill(4) for num in range(1, layers_count + 1)]:
                freq = []
                for typ in ['Q', 'L', 'H']:  #form is in tuple(F, L_unc, H_unc)
                    frq = fits.getval(filename, 'FRE' + typ + s)
                    freq.append(frq)
                freq_s.append(tuple(freq))

        freq_s = list(zip(*freq_s))
        return [np.array(f) for f in freq_s]

    #Get the uncertainties after extracting the upper and lower boundaries
    fQ, fL, fH = get_freq_errors(filename)
    f_unc_low = np.abs(fQ - fL)
    f_unc_high = np.abs(fQ - fH)

    #Convert units if needed
    if 'units' in kw:
        if kw['units'].lower() in ['ghz', 'gigahertz', 'giga hertz']:
            f_unc_low = (f_unc_low * u.Hz).to(u.GHz).value
            f_unc_high = (f_unc_high * u.Hz).to(u.GHz).value

    #Apply rounding if specified
    if 'rounding' in kw:
        decimal_places = kw['rounding']
        f_unc_low = np.round(f_unc_low, decimal_places)
        f_unc_high = np.round(f_unc_high, decimal_places)

    return *f_unc_high, *f_unc_low

#Zooms in on fit file areas that have been identified by sources collected from online database
#Limit of number of sources can be increased or decreased by 'limit_num' when calling amF.get_sources().
def searching_objects():
    for ind, (obj, _id) in enumerate(zip(objs_collected[LIM:], ID[LIM:])):
        
        layer=0
        
        #Creating figure and axis with WCS projection
        fig = plt.figure(figsize=(8,7))
        ax = fig.add_subplot(111, projection=wcs, slices=('x', 'y', 0, 0))
        #=============================================================================
        #Printing data shape to understand its dimensions (gives same information as fits.open(filename).info())
        # print(f'Data shape: {hdu.data.shape}') 
        # This gives --> Data shape: (1, 12, 3617, 3617)
        #Meaning that i have to loop (with looping variable 'var') 
        #through hdu.data[0, var, :, :] rather than hdu.data[:, :, var,0]
                 
        #Additonally: hdu.data[0, 0, :, :].shape ---output--> (3617, 3617)
        #=============================================================================
        
        n=1e-4 #Capping the JY/beam
        #Plotting data with dynamic color scaling
        norm = Normalize(vmin=0, vmax=n)
        img = ax.imshow(hdu.data[0, layer, :, :], origin='lower', cmap='inferno', norm=norm)
        
        im_big_cbar = fig.colorbar(img, ax=ax, fraction=.03,  pad = cbars_pad)
        im_big_cbar.set_label(fits.getval(filename, 'BUNIT'), rotation=rotation, 
                              labelpad=labelpad, fontsize=16)
        
        #Formatting numbering of colorbar
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 0))
        im_big_cbar.ax.yaxis.set_major_formatter(formatter)
    
        ax.grid(ls='solid', alpha=.3)
        ax.coords[0].set_axislabel('Right Ascension (J2000)', fontsize=16)
        ax.coords[1].set_axislabel('Declination (J2000)', fontsize=16)
    
        #Adding a red star to clasrify point of zoom focus by using SkyCoord
        ax.scatter(obj.ra, 
                    obj.dec, 
                    color='r', 
                    marker='*',  
                    transform=ax.get_transform('world'), 
                    s=20)
    
        #Setting axis limits
        #center = wcs.world_to_pixel_values(random_coord.ra, random_coord.dec, 0 ,0)[:2]
        center = wcs.world_to_pixel_values(obj.ra, obj.dec, 0 ,0)[:2]
    
        h=300 #scaling used in zooming in or out RA and Dec
        ax.set_xlim(center[0]-h, center[0]+h)
        ax.set_ylim(center[1]-h, center[1]+h)
    
        h2 = 5e-2 #random position locator variable
        rsltn = fits.getval(filename, "CLEANBMJ")
    
        #Plottting the circle to show the maximum resolution
        Resolution = Circle((random_coord.ra.deg-h2, random_coord.dec.deg-h2), 
                            radius=rsltn,
                            color='white', fill=False, 
                            transform=ax.get_transform('world'), linewidth=2, 
                            label='Resolution =' + "{:.3e}".format(rsltn) + " deg \n (14.97 arcsecs)")
        ax.add_patch(Resolution)
    
        #plt.legend(fontsize = 16, loc = 'upper center', bbox_to_anchor = (.5, 1.21), 
        #            framealpha = 0, ncols = 2)
    
        plt.title(f"(ind={ind+LIM}) ID: {_id}", fontsize=20)
        #plt.title(f"layer = {layer+1}", fontsize=20)
        plt.tight_layout()
        plt.show()

def plot_SpectralIndex(Fr, Fl, Err, old=False, separate=True, print_spInd=False, with_online=False):
    if not separate:
        plt.figure(figsize=(9, 8))
    else:
        # Data from online sources
        data_sources = {
            # online sources: Click link and go to Photometry & SEDs
            # ID 1: https://ned.ipac.caltech.edu/byname?objname=WISEA+J224727.36-155403.2&hconst=67.8&omegam=0.308&omegav=0.692&wmap=4&corr_z=1
            1: get_csv_dat("ID_1_online_Sources.csv"),
            # ID 2: https://ned.ipac.caltech.edu/byname?objname=NVSS+J224936-161552&hconst=67.8&omegam=0.308&omegav=0.692&wmap=4&corr_z=1
            2: [(1.4e9, 0.169), (7.38e7, 2.61)],
            # ID_3: https://ned.ipac.caltech.edu/byname?objname=WISEA+J225111.55-153819.1&hconst=67.8&omegam=0.308&omegav=0.692&wmap=4&corr_z=1
            3: get_csv_dat("ID_3_online_Sources.csv"),
            # ID_4: https://ned.ipac.caltech.edu/byname?objname=WISEA+J224812.37-154811.5&hconst=67.8&omegam=0.308&omegav=0.692&wmap=4&corr_z=1
            4: get_csv_dat("ID_4_online_Sources.csv"),
            # ID_5: https://ned.ipac.caltech.edu/byname?objname=NVSS+J224845-160148&hconst=67.8&omegam=0.308&omegav=0.692&wmap=4&corr_z=1
            5: get_csv_dat("ID_5_online_Sources.csv")
        }
    
    for ind, (freq_s, flux_s, err_s) in enumerate(zip(Fr, Fl, Err)):
        all_x, all_y = [], []
        x, y, err_s = np.log10(freq_s), np.log10(flux_s), np.abs(np.log10(err_s))
        coeff, cov = np.polyfit(x, y, 1, cov=True)  # Getting coefficients of m and c of linear fit
        coeff_errs = np.sqrt(np.diag(cov))  # Getting errors of coeff
        
        if separate:
            plt.figure(figsize=(8, 6))
            
            # Plotting online sources
            if ind + 1 in data_sources:
                freq_data, flux_data = zip(*data_sources[ind + 1])
                x_data, y_data = np.log10(freq_data), np.log10(flux_data)
                plt.scatter(x_data, y_data, marker="x", color='red')
                
        plt.scatter(x, y, marker="o")  # Show scatter plot for individual from personal fitfile
        
        if separate:
            # Accumulate data points for overall fit
            all_x.extend(x);  all_y.extend(y)
            all_x.extend(x_data);  all_y.extend(y_data)

            # plt.title(f"Source {ind+1}", fontsize=18, pad=15)
            plt.xlabel(r"$\log_{10}(\nu)$" + " (Hz)", fontsize=20, labelpad=20)
            plt.ylabel(r"$\log_{10}(S_{\nu})$" + " (Jy)", fontsize=20, labelpad=30)
            plt.grid(True, which="both")
            plt.tight_layout()

            # Fit a linear regression line through all data points
            coeff_corr, cov_corr = np.polyfit(all_x, all_y, 1, cov=True)  # Getting coefficients of m and c of linear fit
            coeff_errs_corr = np.sqrt(np.diag(cov))  # Getting errors of coeff
            fit_line = np.poly1d(coeff_corr)
            plt.plot(all_x, fit_line(all_x), color='black')
            plt.legend(fontsize=17)
            plt.show() # Make sure to split into separate plots
            
        else:  # Show the plots all together
            plt.xlabel(r"$\log_{10}(\nu)$" + " (Hz)", fontsize=20, labelpad=20)
            plt.ylabel(r"$\log_{10}(S_{\nu})$" + " (Jy)", fontsize=20, labelpad=20)
            plt.grid(True, which="both")
            plt.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.1),
                       framealpha=0, ncols=5, markerscale=1.5)
            plt.tight_layout()

        
        if print_spInd:
            if with_online: print(f"Overall α = {coeff_corr[0]:.2f} ± {coeff_errs_corr[0]:.2f}")
            else:
                print(f"Overall α = {coeff[0]:.2f} ± {coeff_errs[0]:.2f}")
        
        plt.show()  # Show all at once if there is no return

        

def plot_SED(Fr, Fl, Err, old=False):
    plt.figure(figsize=(8, 8))
    col = ["r","b","g","orange","cyan"]
    Fitcurves=[]
    for ind, (freq_s, flux_s, err_s) in enumerate(zip(Fr, Fl, Err)):
        x, y, err_s = freq_s, np.array(flux_s), err_s
        x = (x*u.Hz).to(u.GHz).value
        plt.scatter(x, y, marker="o", color=col[ind])  #Show scatter plot for individual plots
        plt.errorbar(x, y, yerr=err_s, marker='',
                     linestyle='', capsize=4, ecolor="k")

        #plt.xlim(20.95, 21.42)
        #plt.ylim(3, 7.5)

        #Fit a linear regression line
        coeff = np.polyfit(x, y, 4)
        fit_line = np.poly1d(coeff)
        x=np.linspace(x[0], x[-1], 400) #generate more points within x-range for smoother curved fitting
        Fitcurves.append((x,fit_line(x)))

        plt.plot(x, fit_line(x), label=f"ID {ind+1}", color=col[ind])


        plt.xlabel(r"$\nu$" + " (GHz)", fontsize=25, labelpad=10)
        plt.ylabel(r"$S_{\nu}$" + " (Jy)", fontsize=25, labelpad=16)
        plt.grid(True, which="both")

    X, Y = list(zip(*Fitcurves))
    
    #exact value of the HI line wavelength in meters
    f_HI = (1420405751.768*u.Hz).to(u.GHz).value #https://tf.nist.gov/general/pdf/13.pdf
    plt.axvline(x=f_HI, linestyle="--", color="blue", label='HI 21cm line')
    
    plt.legend(fontsize = 14, loc = 'upper center', bbox_to_anchor = (0.5, 1.11), 
                framealpha = 0, ncols = (2,4), markerscale=1.5)
    plt.tight_layout()
    plt.show()
    
#Save time by retrieving pickle data when new_sources=False
#If limit_num is changed then set new_sources=True.
#After runningscript once make sure to set new_sources=False
objs_collected, df, ID = amF.get_sources(random_coord, 
                                 search_radius=2*u.deg,
                                 limit_num=2000, 
                                 new_sources=False)

LIM = 0 #Parametre used in one or more of the defined functions thats meant for indexing 
#to skip already printed information (which can be about 2000 or more sources and must 
#have to be stopped midway and kernelrestart dur to low space to handle printing all the sources)  

#searching_objects()

manual_objects = [("22:47:28.6651", "-15:53:36.805"), #ID=1
                  ("22:49:36.6921", "-16:15:51.181"), #"22:50:55.5132", "-16:27:09.645"), #ID=2
                  ("22:51:11.5347", "-15:38:06.427"), #ID=3
                  ("22:48:13.3184", "-15:47:36.732"), #ID=4
                  ("22:48:44.5971", "-16:01:24.086") #ID=5
                  ]

#Creating skyCoord objects from sources
manual_objects = [SkyCoord(ra=RA, dec= DEC, 
                           unit=(u.hourangle, u.deg), 
                           frame="fk5") 
                  for RA, DEC in manual_objects]

#if loop = True, then layer=<anything> is ignored.
see_layer(layer=12, objs_collected = manual_objects, 
          loop=True, mark_astroquery_objects=False)#, cap_flux=2e-5) #Lower cap size shows more

# # zoom_chosen(objs_collected=objs_collected, 
#             # chosen=(148, 316, 507, 615))

# zoom_chosen(objs_collected=manual_objects, manual=True)

# FREQ, FLUX, Err, Sig_3, Sig_5  = amF.extract_data('results_with_sigma.txt')

# plot_SED(FREQ, FLUX, Err)
# plot_SpectralIndex(FREQ, FLUX, Err, print_spInd=True)
# plot_SpectralIndex(FREQ, FLUX, Err, print_spInd=True, with_online=True) #Corrected indeces when using catalog data

# #Prints if all sources selected in the region file are above the sigma threshold.
# amF.above_sigma_detection(FLUX, Sig_5, "5")

# plot_SpectralIndex(FREQ, FLUX, Err, separate=False)

