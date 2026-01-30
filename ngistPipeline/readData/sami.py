import logging
import os

import numpy as np
from astropy.io import fits
from printStatus import printStatus
from astropy.wcs import WCS
import h5py
import pandas as pd
from scipy.interpolate import interp1d
from ppxf.ppxf_util import log_rebin



# ======================================
# Routine to interp blue cube onto red
# ======================================
def interp_cube_lambda(cube, wave_in, wave_out, fill_value=np.nan):
    """
    Interpolate a spectral cube along wavelength.
    """
    interp = interp1d(
        wave_in,
        cube,
        axis=0,
        kind='linear',
        bounds_error=False,
        fill_value=fill_value,
        assume_sorted=True
    )
    return interp(wave_out)

# ======================================
# Routine to set DEBUG mode
# ======================================
def set_debug(cube, xext, yext):
    logging.info(
        "DEBUG mode is activated. Instead of the entire cube, only one line of spaxels is used."
    )
    cube["x"] = cube["x"][int(yext / 2) * xext : (int(yext / 2) + 1) * xext]
    cube["y"] = cube["y"][int(yext / 2) * xext : (int(yext / 2) + 1) * xext]
    cube["snr"] = cube["snr"][int(yext / 2) * xext : (int(yext / 2) + 1) * xext]
    cube["signal"] = cube["signal"][int(yext / 2) * xext : (int(yext / 2) + 1) * xext]
    cube["noise"] = cube["noise"][int(yext / 2) * xext : (int(yext / 2) + 1) * xext]

    cube["spec"] = cube["spec"][:, int(yext / 2) * xext : (int(yext / 2) + 1) * xext]
    cube["error"] = cube["error"][:, int(yext / 2) * xext : (int(yext / 2) + 1) * xext]

    return cube

def vactoair(vacwl):
    """Calculate the approximate wavelength in air for vacuum wavelengths.

    Parameters
    ----------
    vacwl : ndarray
       Vacuum wavelengths.

    This uses an approximate formula from the IDL astronomy library
    https://idlastro.gsfc.nasa.gov/ftp/pro/astro/vactoair.pro

    """
    wave2 = vacwl * vacwl
    n = 1.0 + 2.735182e-4 + 131.4182 / wave2 + 2.76249e8 / (wave2 * wave2)

    # Do not extrapolate to very short wavelengths.
    if not isinstance(vacwl, np.ndarray):
        if vacwl < 2000:
            n = 1.0
    else:
        ignore = np.where(vacwl < 2000)
        n[ignore] = 1.0

    return vacwl / n

def save_table(
    config,
    x,
    y,
    signal,
    snr,
    binNum_new,
    xNode,
    yNode,
    sn,
    nPixels,
    pixelsize,
    wcshdr,
    ubins,
):
    """
    Save all relevant information about the Voronoi binning to disk. In
    particular, this allows to later match spaxels and their corresponding bins.

    Args:
        config (dict): Configuration settings.
        x (ndarray): X-coordinates of the spaxels.
        y (ndarray): Y-coordinates of the spaxels.
        signal (ndarray): Flux values of the spaxels.
        snr (ndarray): Signal-to-noise ratio values of the spaxels.
        binNum_new (ndarray): Array of bin IDs for each spaxel.
        ubins (ndarray): Unique bin IDs.
        xNode (ndarray): X-coordinates of the bin nodes. - flux-weighted centres ok?
        yNode (ndarray): Y-coordinates of the bin nodes.
        sn (ndarray): Signal-to-noise ratio values of the bins.
        nPixels (ndarray): Number of spaxels in each bin.
        pixelsize (float): Size of each pixel.
        wcshdr (Header): WCS header information.

    Returns:
        None
    """
    outfits_table = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_table.fits"
    )
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_table.fits")
    xshape = x.shape
    yshape = y.shape
    printStatus.running("size of the input array=" + str(xshape) + str(yshape))
    # Expand data to spaxel level 
    xNode_new = np.zeros(len(x))
    yNode_new = np.zeros(len(x))
    nPixels_new = np.zeros(len(x))
    for i in range(len(ubins)-1):
        idx = np.where(ubins[i] == np.abs(binNum_new))[0]
        xNode_new[idx] = xNode[i]
        yNode_new[idx] = yNode[i]
        nPixels_new[idx] = nPixels[i]

    # Primary HDU
    priHDU = fits.PrimaryHDU()
    # Table HDU with output data

    cols = []
    cols.append(fits.Column(name="ID", format="J", array=np.arange(len(x))))
    cols.append(fits.Column(name="BIN_ID", format="J", array=binNum_new-1))
    cols.append(fits.Column(name="X", format="D", array=x))
    cols.append(fits.Column(name="Y", format="D", array=y))
    cols.append(fits.Column(name="FLUX", format="D", array=signal))
    cols.append(fits.Column(name="SNR", format="D", array=snr))
    cols.append(fits.Column(name="XBIN", format="D", array=xNode_new))
    cols.append(fits.Column(name="YBIN", format="D", array=yNode_new))
    cols.append(fits.Column(name="SNRBIN", format="D", array=sn))
    cols.append(fits.Column(name="NSPAX", format="J", array=nPixels_new)) #Attempt to shift bin 0 to -1

    tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    tbhdu.name = "TABLE"

    # create empty imageHDU with wcs header info
    imghdu = fits.ImageHDU(data=None, header=wcshdr)

    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, tbhdu, imghdu])
    HDUList.writeto(outfits_table, overwrite=True)
    fits.setval(outfits_table, "PIXSIZE", value=pixelsize)

    printStatus.updateDone("Writing: " + config["GENERAL"]["RUN_ID"] + "_table.fits")
    logging.info("Wrote Voronoi table: " + outfits_table)

# -- Create a mask file - needed for emission line mapping.
# --- For now, just make a single wavelength-independent mask
def saveMask(config, combinedMask, maskedDefunct, maskedSNR, maskedMask):
    """Save the mask to disk."""
    outfits = (os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])+ "_mask.fits")
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_mask.fits")

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with output data
    # This is an integer array! 0 means unmasked, 1 means masked!
    cols = []
    cols.append(
        fits.Column(
            name="MASK", format="I", array=np.array(combinedMask, dtype=np.int32)
        )
    )
    cols.append(
        fits.Column(
            name="MASK_DEFUNCT",
            format="I",
            array=np.array(maskedDefunct, dtype=np.int32),
        )
    )
    cols.append(
        fits.Column(
            name="MASK_SNR", format="I", array=np.array(maskedSNR, dtype=np.int32)
        )
    )
    cols.append(
        fits.Column(
            name="MASK_FILE", format="I", array=np.array(maskedMask, dtype=np.int32)
        )
    )
    tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    tbhdu.name = "MASKFILE"

    # Create HDU list and write to file
    tbhdu.header["COMMENT"] = "Value 0  -->  unmasked"
    tbhdu.header["COMMENT"] = "Value 1  -->  masked"
    HDUList = fits.HDUList([priHDU, tbhdu])
    HDUList.writeto(outfits, overwrite=True)

    printStatus.updateDone("Writing: " + config["GENERAL"]["RUN_ID"] + "_mask.fits")
    logging.info("Wrote mask file: " + outfits)

    return None


def saveAllSpectra(config, log_spec, log_error, velscale, logLam):
    """
    Save all logarithmically rebinned spectra to file.
    Currently not used for MaNGA as we're using the binned cubes. 
    We'd have to do a completely different run when we could really
    just take them from the DAP.

    Args:
        config (dict): Configuration parameters.
        log_spec (numpy.ndarray): Logarithmically rebinned spectra.
        log_error (numpy.ndarray): Logarithmically rebinned error spectra.
        velscale (float): Velocity scale.
        logLam (numpy.ndarray): Logarithmically rebinned wavelength array.

    Returns:
        None
    """

    outfn_spectra = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_all_spectra.hdf5"
    )
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_all_spectra.hdf5")

    # Create a new HDF5 file
    with h5py.File(outfn_spectra, 'w') as f:
        # Create datasets for the spectra and error spectra
        spec_dset = f.create_dataset('SPEC', shape=log_spec.shape, dtype=log_spec.dtype)
        espec_dset = f.create_dataset('ESPEC', shape=log_error.shape, dtype=log_error.dtype)

        # Write the data in chunks
        chunk_size = 1000  # Adjust this value to fit your memory capacity
        for i in range(0, len(log_spec), chunk_size):
            spec_dset[i:i+chunk_size] = log_spec[i:i+chunk_size]
            espec_dset[i:i+chunk_size] = log_error[i:i+chunk_size]

        # Create a dataset for LOGLAM
        f.create_dataset('LOGLAM', data=logLam)

        # Set attributes
        f.attrs['VELSCALE'] = velscale
        f.attrs["CRPIX1"] = 1.0
        f.attrs["CRVAL1"] = logLam[0]
        f.attrs["CDELT1"] = logLam[1] - logLam[0]

    printStatus.updateDone(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_all_spectra.hdf5"
    )
    logging.info("Wrote: " + outfn_spectra)

def saveBinSpectra(config, logLam, log_spec, log_error, lsf, velscale=70): #, log_spec, log_error, velscale, logLam, flag):
    """Save spatially binned spectra and error spectra are saved to disk.
    For SAMI, we will also stitch the two spectra together... I think"""
    outfile = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])

#    if flag == "log":
    outfn_spectra = outfile + "_bin_spectra.hdf5"
    printStatus.running(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_bin_spectra.hdf5"
    )
    # elif flag == "lin":
    #     outfn_spectra = outfile + "_bin_spectra_linear.hdf5"
    #     printStatus.running(
    #         "Writing: " + config["GENERAL"]["RUN_ID"] + "_bin_spectra_linear.hdf5"
    #     )
    # load orig cube and get it into the format needed
    # Create a new HDF5 file
    with h5py.File(outfn_spectra, 'w') as f:
        # Create datasets for the spectra and error spectra
        spec_dset = f.create_dataset('SPEC', shape=log_spec.shape, dtype=log_spec.dtype)
        espec_dset = f.create_dataset('ESPEC', shape=log_error.shape, dtype=log_error.dtype)
        #lsf_dset = f.create_dataset('LSF', shape=lsf.shape, dtype=lsf.dtype) # Not needed for SAMI

        # Write the data in chunks
        chunk_size = 1000  # Adjust this value to fit your memory capacity
        for i in range(0, len(log_spec), chunk_size):
            spec_dset[i:i+chunk_size] = log_spec[i:i+chunk_size]
            espec_dset[i:i+chunk_size] = log_error[i:i+chunk_size]
            #lsf_dset[i:i+chunk_size] = lsf[i:i+chunk_size]


        # Create a dataset for LOGLAM
        f.create_dataset('LOGLAM', data=logLam)

        # Set attributes
        f.attrs['VELSCALE'] = velscale
        f.attrs['CRPIX1'] = 1.0
        f.attrs['CRVAL1'] = logLam[0]
        f.attrs['CDELT1'] = logLam[1] - logLam[0]

    # if flag == "log":
    printStatus.updateDone(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_bin_spectra.hdf5"
        )
    # elif flag == "lin":
    #     printStatus.updateDone(
    #         "Writing: " + config["GENERAL"]["RUN_ID"] + "_bin_spectra_linear.hdf5"
        #)
    logging.info("Wrote: " + outfn_spectra)

# ======================================
# Routine to load SAMI-cubes
# Based on CALIFA_V500.py
# For now, assuming that you are using the 
# Voronoi-binned cubes (red + blue are separate cubes) at:
# https://datacentral.org.au/services/sov/
# Batch download also possible from DC.
# ======================================
def readCube(config):
    loggingBlanks = (len(os.path.splitext(os.path.basename(__file__))[0]) + 33) * " "

    # Read SAMI-cube
    printStatus.running("Reading the SAMI red and blue cubes")
    logging.info("Reading the SAMI R&B cubes" + config["GENERAL"]["INPUT"])

    # Reading the red cube
    hdured = fits.open(config["GENERAL"]["INPUT"]+ '_red.fits.gz')
    hdrr = hdured['PRIMARY'].header
    cube_red = hdured['PRIMARY'].data # FLUX extension
    var_red = hdured['VARIANCE'].data # VARIANCE extension
    s = np.shape(cube_red) # 2048, 50, 50
    # Getting the wavelength info ## CRVAL3 for SAMI is the mid-point of the wavelength range, not the start!!  
    startwav = hdrr["CRVAL3"] - (hdrr["CDELT3"] * (hdrr["CRPIX3"]-1)) # Starting wavelength in angstrom.
    wave_red = startwav + (np.arange(s[0])) * hdrr["CDELT3"] # These are linearly spaced.
    wcshdr = WCS(hdrr).to_header()
    obs_date = hdrr['ORIGFILE'][-15:-5]

    if int(obs_date[0:4]) == 2014 and int(obs_date[5:7]) < 11:
        # Use the old LSF
        lsf_red = 1.62 * np.ones(len(wave_red)) # Accounting for the CCD change in October, 2014
    elif int(obs_date[0:4]) < 2014 :
        lsf_red = 1.62 * np.ones(len(wave_red)) # Accounting for the CCD change in October, 2014
    else:
        # Use the new LSF
        lsf_red = 1.57 * np.ones(len(wave_red))
    
    # Reading the blue cube
    hdublue = fits.open(config["GENERAL"]["INPUT"]+ '_blue.fits.gz')
    hdrb = hdublue['PRIMARY'].header
    cube_blue = hdublue['PRIMARY'].data # FLUX extension
    var_blue = hdublue['VARIANCE'].data # VARIANCE extension
    s = np.shape(cube_blue) # 2048, 50, 50
    # Getting the wavelength info ## CRVAL3 for SAMI is the mid-point of the wavelength range, not the start!! 
    startwav = hdrb["CRVAL3"] - (hdrb["CDELT3"] * (hdrb["CRPIX3"]-1)) # Starting wavelength in angstrom.
    wave_blue = startwav + (np.arange(s[0])) * hdrb["CDELT3"] # These are linearly spaced.
    lsf_blue = 2.66 * np.ones(len(wave_blue))
    wcshdrb = WCS(hdrb).to_header()

    # Stitch the red and blue cubes together, by oversampling the blue
    # Red spectral sampling
    dlam_red = np.median(np.diff(wave_red))

    # New blue-extended wavelength grid
    wave_ext = np.arange(wave_blue.min(), wave_red.max() + dlam_red, dlam_red)
    cube_blue_ext = interp_cube_lambda(cube_blue, wave_blue, wave_ext)
    var_blue_ext = interp_cube_lambda(var_blue, wave_blue, wave_ext)
    lsf_blue_ext = interp_cube_lambda(lsf_blue, wave_blue, wave_ext)

    # Index mapping of red wavelengths onto extended grid
    red_indices = np.round((wave_red - wave_ext[0]) / dlam_red).astype(int)
    ny, nx = cube_red.shape[1:]
    cube_stitched = np.full((len(wave_ext), ny, nx), np.nan)
    var_stitched = np.full((len(wave_ext), ny, nx), np.nan)
    lsf_stitched = np.full((len(wave_ext)), np.nan)
    mask_blue = np.isfinite(cube_blue_ext)
    mask_blue_var = np.isfinite(var_blue_ext)
    mask_blue_lsf = np.isfinite(lsf_blue_ext) #
    cube_stitched[mask_blue] = cube_blue_ext[mask_blue]
    var_stitched[mask_blue_var] = var_blue_ext[mask_blue_var]
    lsf_stitched[mask_blue_lsf] = lsf_blue_ext[mask_blue_lsf]

    # Insert red cube (overwrite)
    cube_stitched[red_indices, :, :] = cube_red
    var_stitched[red_indices, :, :] = var_red
    lsf_stitched[red_indices] = lsf_red
    wave_stitched = wave_ext

    median_val = np.nanmedian(cube_stitched, axis=0) # 
    # 2. Find NaN positions & Replace
    cube_stitched = np.where(np.isnan(cube_stitched), median_val, cube_stitched)
    median_val = np.nanmedian(var_stitched, axis=0) # 
    # 2. Find NaN positions & Replace
    var_stitched = np.where(np.isnan(var_stitched), median_val, var_stitched)
    lsf_stitched = np.where(np.isnan(lsf_stitched), 1, lsf_stitched) # Just set this to 1.


    ##################################################
    ## Everything is in linear wavelength spacing.  ##
    ## Time to log-rebin everything!                ##
    ##################################################
    specNew, ln_lam, velscale = log_rebin(wave_stitched, cube_stitched) #log_rebinned, log spacing.
    lam = np.exp(ln_lam) # log_rebinned, linear spacing. 
    noiseNew, _, _ = log_rebin(wave_stitched, var_stitched) #log_rebinned, log spacing.
    lsfNew, _, _ = log_rebin(wave_stitched, lsf_stitched) #log_rebinned, log spacing.

    # Save LSF for future use 
    lsf_save = np.array([lam, lsf_stitched]).T # 
    lsf_savepath = os.path.join(config["GENERAL"]["CONFIG_DIR"], config["GENERAL"]["LSF_DATA"])
    np.savetxt(lsf_savepath, lsf_save, delimiter=' ') # Save it here, as I don't want to add it to the cube as I do with MaNGA.

    logLam_full = ln_lam  # # For ppxf, you need ln(λ), not λ itself Natural log, not log10!

    # Calculate velscale from the logarithmic spacing
    dlog = np.mean(np.diff(logLam_full))
    C = 299792.458 # km/s
    velscale = C * dlog  # C = 299792.458 km/s, should be ~33 km/s for SAMI. 


##### ---------------- Save spaxel-based stuff now, before you start ditching bins
    # Getting the spatial coordinates
    # Can assume that all SAMI targets are close enough to centred in the IFU
    origin = [hdrr['CRPIX1'], hdrr['CRPIX2']] # This is 25.5, 25.5 for SAMI.
    xaxis = (np.arange(s[2]) - origin[0]) *  hdrr["CDELT1"] * 3600.0 # Changed CD2_2 --> CDELT2 as  In simple, non-rotated images, CDELT1 and CDELT2 are equivalent to CD1_1 and CD2_2, respectively.
    yaxis = (np.arange(s[1]) - origin[1]) * hdrr["CDELT2"] * 3600.0 ## 
    x, y = np.meshgrid(xaxis, yaxis)

    pixelsize = np.abs(hdrr["CDELT2"]) * 3600.0 #0.5 arcsec in degrees.
    binid = hdured['BIN_MASK'].data.astype('int') # 0 = mask, >1 = binid
    Hamask = np.copy(binid)
    Hamask[Hamask>0] = 1
    Hamask = Hamask.astype('int')
    nPixels = np.bincount(binid.flatten(), minlength=np.max(binid)+1) #. Now remove bin 0, which is actually the masked region
    ubins, idxx = np.unique(binid.flatten(), return_index=True)

    # logging.info("Saving table.fits!")
    unique_bins = np.unique(binid)

    xs = []
    ys = []
    bins = []

    for b in unique_bins:
        # Find indexes where binid == b
        i, j = np.where(binid == b)
        
        # Pick the first match (any is fine)
        xs.append(x[i[0], j[0]])
        ys.append(y[i[0], j[0]])
        bins.append(b)

    xs = np.array(xs)
    ys = np.array(ys)
    bins = np.array(bins)
    x = np.reshape(x, [s[1] * s[2]])
    y = np.reshape(y, [s[1] * s[2]])
    sn = np.ones(len(Hamask.flatten())) # fudge this at SN=10 throughout for now.
    snr = np.ones(len(Hamask.flatten()))
    signal = np.sum(specNew, axis=0).flatten()
    
    save_table(config, x.flatten(), y.flatten(), signal, snr, binid.flatten(), xs, ys, sn, nPixels, pixelsize, wcshdr, ubins)

    saveMask(config, np.ravel(Hamask), np.ravel(Hamask), np.ravel(Hamask), np.ravel(Hamask))

##### ----------------

    # Now, extra step because the MaNGA Voronoi binned cubes are nSpax*nSpax, not nBins in size 
    nwave, nx, ny = specNew.shape
    data_flat = specNew.reshape(nwave, nx*ny).T # Shape 2500 * 6394 (lam)
    stat_flat = noiseNew.reshape(nwave, nx*ny).T
    # tile the lsf for now, even though we assume it is spatially-invariant 
    lsf_tile = np.tile(lsfNew[:,None, None], (1,nx,ny))
    lsf_flat = lsf_tile.reshape(nwave, nx*ny).T
    binid_flat = binid.ravel()           # shape: (nx*ny,)

    # 2. Find unique BINIDs and the first occurrence index
    unique_binids, idxx = np.unique(binid_flat, return_index=True)
    # However, there is also binid = -1; these are the masked regions, so ditch these. 

    # 3. Select one spectrum per BINID
    spec = data_flat[idxx, :]           # shape: (n_unique_binid, nwave)
    spec = spec[1::, :]
    spec = spec.T
    espec = stat_flat[idxx, :]
    espec = espec[1::, :]
    espec = espec.T
    lsf = lsf_flat[idxx, :]
    lsf = lsf[1::, :]
    lsf = lsf.T

    logging.info(
        "Extracting spatial information:\n"
        + loggingBlanks
        + "* Spatial coordinates are centred to "
        + str(origin)
        + "\n"
        + loggingBlanks
        + "* Spatial pixelsize is "
        + str(pixelsize)
    )

    # De-redshift spectra
    wave = np.exp(logLam_full)
    wavesave = wave / (1 + config["GENERAL"]["REDSHIFT"])
    logging.info(
        "Shifting spectra to rest-frame, assuming a redshift of "
        + str(config["GENERAL"]["REDSHIFT"])
    )
    wave = wave_stitched
    # Shorten spectra to required wavelength range
    lmin = config["READ_DATA"]["LMIN_TOT"]
    lmax = config["READ_DATA"]["LMAX_TOT"]
    idx = np.where(np.logical_and(wave >= lmin, wave <= lmax))[0]
    spec = spec[idx, :]
    espec = espec[idx, :]
    lsf = lsf[idx, :]
    wave = wave[idx]
    wavesave = wavesave[idx]
    logging.info(
        "Shortening spectra to the wavelength range from "
        + str(config["READ_DATA"]["LMIN_TOT"])
        + "A to "
        + str(config["READ_DATA"]["LMAX_TOT"])
        + "A."
    )

    # Pass error spectra as variances instead of stddev - Skipped for SAMI because its variance already
    #espec = espec**2
    #espec = 1/espec # (As it was IVAR in the MaNGA case.)

    # Computing the SNR per spaxel
    idx_snr = np.where(
        np.logical_and(
            wave >= config["READ_DATA"]["LMIN_SNR"],
            wave <= config["READ_DATA"]["LMAX_SNR"],
        )
    )[0]

    signal = np.nanmedian(spec[idx_snr, :], axis=0)
    noise = np.abs(np.nanmedian(np.sqrt(espec[idx_snr, :]), axis=0))
    snr = signal / noise # Some will be nans, as there are 0/0s.

    logging.info(
        "Computing the signal-to-noise ratio in the wavelength range from "
        + str(config["READ_DATA"]["LMIN_SNR"])
        + "A to "
        + str(config["READ_DATA"]["LMAX_SNR"])
        + "A."
    )

    # Storing everything into a structure
    cube = {
        "x": x,
        "y": y,
        "wave": wave,
        "spec": spec,
        "error": espec,
        "lsf": lsf,
        "snr": snr,
        "signal": signal,
        "noise": noise,
        "pixelsize": pixelsize,
        "wcshdr": wcshdr,
    }

    # Constrain cube to one central row if switch DEBUG is set
    if config["READ_DATA"]["DEBUG"] == True:
        cube = set_debug(cube, s[2], s[1])

    # Print out bin_spectra.hdf5 for use with Mapviewer
    # Must be done here as prepareSpectra is not run for SAMI 
    # Voronoi-binned cubes 
    saveBinSpectra(config, np.log(wavesave), spec, espec, lsf, velscale =velscale) #

    printStatus.updateDone("Reading the MaNGA cube")
    print("             Read " + str(len(cube["x"])) + " spectra!")
    print("             From " + str(len(idxx)-1) + " bins!")

    logging.info("Finished reading the SAMI cubes!")

    return cube


