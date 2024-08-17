#these are extra functions that asssit with analysis but dont need to be seen in main.py.
# However, they are bing referenced in the main.py

#Suppressing specific Astropy warnings
def get_sources(skycoord, search_radius, limit_num=5, new_sources=True):
    
    #Setting limits for RA and Dec search based on limits of the Fit File
    #I had to manually find an estimate of these limtis by looking at the graph
    high_point = SkyCoord(ra="22h53m", dec="-15:00:00", unit=(u.hourangle, u.deg), frame="fk5")
    low_point = SkyCoord(ra="22h44m", dec="-17:13:00", unit=(u.hourangle, u.deg), frame="fk5")
    ra_min, dec_min = low_point.ra, low_point.dec
    ra_max, dec_max = high_point.ra, high_point.dec
    
    if new_sources:
        
        #adding a parameter "Object type" into my search query
        #"otype" was identified by seeing doing Simbad.list_votable_fields() in terminal to view all other parametres to search by
        #Simbad.add_votable_fields('otype') #, 'otypes')

        #Query the Simbad database
        all_res = Simbad.query_region(skycoord, radius=search_radius)
        
        #Getting only the galaies
        #res = all_res[(all_res['OTYPE'] == 'Galaxy')] #& ('Rad' in all_res['OTYPES'])]
        #res = all_res[all_res['OTYPE'] == 'Galaxy']
        
        #Convert response to DataFrame
        df = all_res.to_pandas()
        df_old = all_res.to_pandas()
        
        if df.empty:
            print("No results found.")
            return [], df, []
        
        Found = []
        ID = []
    
#        c = 0
#        for i in range(len(df)):
#            obj_found = SkyCoord(ra=df.loc[i].RA, dec=df.loc[i].DEC, unit=(u.hourangle, u.deg), frame='fk5')
            
#            #If within RA and DEC boundary of the fit file record it
#            if ra_min < obj_found.ra < ra_max and dec_min < obj_found.dec < dec_max:
#                Found.append(obj_found)
#                ID.append(df.loc[i].MAIN_ID)
#                c += 1
    
#            if c == limit_num: #this wont show if pickle data is chosen
#                print(f"-Limit of {limit_num} galaxies limit reached")
#                break
    
#        print()
#        print(
#f"""
#  Found {c} Galaxies within Fit-file RA and Dec limits out of {len(df)} Galaxies within {search_radius}.
#  (Note total sources (galaxy or not) within {search_radius} is {len(df_old)})
#  """
#              )
        
        c = 0
        print()
        print("When Getting sources the following occurred:")
        for i in range(len(df)):
            if df.loc[i].MAIN_ID[:5] == "6dFGS" or df.loc[i].MAIN_ID[:4] == "NVSS":
                ID.append(df.loc[i].MAIN_ID)
                obj_found = SkyCoord(ra=df.loc[i].RA, dec=df.loc[i].DEC, unit=(u.hourangle, u.deg), frame='fk5')
                
                #If within RA and DEC boundary of the fit file record it
                if ra_min < obj_found.ra < ra_max and dec_min < obj_found.dec < dec_max:
                    Found.append(obj_found)
                    c += 1
        
                if c == limit_num: #this wont show if pickle data is chosen
                    print(f"-Limit of {c} sources reached")
                    break
        
            #If sources in entire radial catalog is smaller than custom number of sources wanted
            if i == len(df) - 1:  #this wont show if pickle data is chosen
                print(f"-Found {c} sources out of {len(df)}")
                break

        #Storing data in pickle before returning it
        with open("sources_found.pkl", "wb") as f:
            pickle.dump((Found, df, ID), f)

    else:
        #Unpickle the data
        with open("sources_found.pkl", "rb") as f:
            Found, df, ID = pickle.load(f)

    return Found, df, ID

def extract_data(Filename):
    data = {}

    with open(Filename, 'r') as file:
        lines = file.readlines()

        for line in lines:
            line = line.strip()

            if line and not line.startswith("Filename"):
                parts = line.split()
                if len(parts) == 7:  #Ensuring the line has all expected data fields
                    label = int(parts[1])  #Label
                    frequency = float(parts[2])  #Frequency (Hz)
                    flux = float(parts[3])  #Flux (Jy)
                    error = float(parts[4])  #Error (Jy)
                    sig_3 = float(parts[5])  #sigma_3(Jy)
                    sig_5 = float(parts[6])  #sigma_5(Jy)

                    #If the label is not already in the dictionary, add it
                    if label not in data:
                        data[label] = {"FREQ": [], "FLUX": [], "err": [], 'sigma_3':[], 'sigma_5':[]}

                    #Append the frequency, flux, and error to the respective label
                    data[label]["FREQ"].append(frequency)
                    data[label]["FLUX"].append(flux)
                    data[label]["err"].append(error)
                    data[label]["sigma_3"].append(sig_3)
                    data[label]["sigma_5"].append(sig_5)

    #Converting dictionary to lists of lists format
    FREQ = [data[label]["FREQ"] for label in sorted(data.keys())]
    FLUX = [data[label]["FLUX"] for label in sorted(data.keys())]
    err = [data[label]["err"] for label in sorted(data.keys())]
    Sig_3 = [data[label]["sigma_3"] for label in sorted(data.keys())]
    Sig_5 = [data[label]["sigma_5"] for label in sorted(data.keys())]

    return FREQ, FLUX, err, Sig_3, Sig_5 

def above_sigma_detection(Flux, Sig, num):
    not_above, above = set(), set() #A set to only include the ID of the source. No repeating values.
    #If source ends up in the not_above, then it should not be used for spectral analysis.
    
    for ind, (flux_s, sig_s) in enumerate(zip(Flux, Sig)):
        calc = np.array(flux_s) - np.array(sig_s) #Get difference
        
        #Checking if all values in calc are greater than 0
        if np.all(calc > 0): 
            above.add(ind)
        else: 
            not_above.add(ind)
    
    sig = '\u03C3'
    #If empty then just print that all chosen regions definitely have sources.
    if len(not_above) == 0: 
        print(f'All Sources in region file are above {num}' + sig)
    else: print(f"The following sources are not above {num}" + sig + f" : {np.array(list(not_above))+1}")
    
#D_l = (1+z)*D_c---> idea from https://phys.libretexts.org/Courses/University_of_California_Davis/UCD%3A_Physics_156_-_A_Cosmology_Workbook/Workbook/08._The_Distance-Redshift_Relation

def get_lum_distance_from_z(z, z_err, rounding=1) -> tuple: #(D_L, D_L uncertianty) 

    #central difference method, which provides a good approximation of the 
    #derivative by considering small changes around the point of interest

    dz = 1e-4  #A small step for numerical derivative
    D_L_plus = cosmo.luminosity_distance(z + dz).value
    D_L_minus = cosmo.luminosity_distance(z - dz).value

    #Compute numerical derivatives
    dD_L_dz = (D_L_plus - D_L_minus) / (2 * dz)

    #Calculate distances at perturbed redshifts for error propagation
    #Also return the comoving dtiance itself
    return np.round([cosmo.luminosity_distance(z).value, 
                    dD_L_dz * z_err], decimals=rounding) #Give back luminosity distance_err after conducting

#this funciton looks at the header information in the FITs file and
#Identifies the upper and lower frequencies of each slice
#This helps determine the upper and lower uncertainty of each frequency layer
#this will mostly be run on ipython when quoting the fitfile name.
def get_freq_errors(filename: str, **kw):
    #Get the upper and lower boundaries of each frequency
    def get_freq_errors(filename: str):
        layers_count = fits.getval(filename, "NSPEC")
        freq_s = []
        if layers_count == 1: #Only a slice
            freq = []
            new_slice_num = fits.getval(filename, "N_FREQ")
            for typ in ['Q', 'L', 'H']:  #form is in tuple(F, L_unc, H_unc)
                frq = fits.getval(filename, 'FRE' + typ + str(new_slice_num).zfill(4))
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
