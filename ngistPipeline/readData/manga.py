import logging
import os

import numpy as np
from astropy.io import fits
from printStatus import printStatus
from astropy.wcs import WCS
import h5py
import pandas as pd


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
    # Expand data to spaxel level - not needed because already have it!
    # xNode_new = np.zeros(len(x))
    # yNode_new = np.zeros(len(x))
    # sn_new = np.zeros(len(x))
    # nPixels_new = np.zeros(len(x))
    # for i in range(len(ubins)):
    #     idx = np.where(ubins[i] == np.abs(binNum_new))[0]
    #     xNode_new[idx] = xNode[i]
    #     yNode_new[idx] = yNode[i]
    #     sn_new[idx] = sn[i]
    #     nPixels_new[idx] = nPixels[i]

    # Primary HDU
    priHDU = fits.PrimaryHDU()
    # Table HDU with output data
    cols = []
    cols.append(fits.Column(name="ID", format="J", array=np.arange(len(x))))
    cols.append(fits.Column(name="BIN_ID", format="J", array=binNum_new))
    cols.append(fits.Column(name="X", format="D", array=x))
    cols.append(fits.Column(name="Y", format="D", array=y))
    cols.append(fits.Column(name="FLUX", format="D", array=signal))
    cols.append(fits.Column(name="SNR", format="D", array=snr))
    cols.append(fits.Column(name="XBIN", format="D", array=xNode))
    cols.append(fits.Column(name="YBIN", format="D", array=yNode))
    cols.append(fits.Column(name="SNRBIN", format="D", array=sn))
    cols.append(fits.Column(name="NSPAX", format="J", array=nPixels))

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
    """Save spatially binned spectra and error spectra are saved to disk."""
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
        lsf_dset = f.create_dataset('LSF', shape=lsf.shape, dtype=lsf.dtype)

        # Write the data in chunks
        chunk_size = 1000  # Adjust this value to fit your memory capacity
        for i in range(0, len(log_spec), chunk_size):
            spec_dset[i:i+chunk_size] = log_spec[i:i+chunk_size]
            espec_dset[i:i+chunk_size] = log_error[i:i+chunk_size]
            lsf_dset[i:i+chunk_size] = lsf[i:i+chunk_size]


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
# Routine to load MaNGA-cubes
# Based on CALIFA_V500.py
# For now, assuming that you are using the 
# Voronoi-binned cubes at 
# https://data.sdss.org/sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0/VOR10-MILESHC-MASTARSSP/ 
# Data model: https://data.sdss.org/datamodel/files/MANGA_SPECTRO_ANALYSIS/DRPVER/DAPVER/DAPTYPE/PLATE/IFU/manga-LOGCUBE-DAPTYPE.html
# ======================================
def readCube(config):
    loggingBlanks = (len(os.path.splitext(os.path.basename(__file__))[0]) + 33) * " "

    # Read MaNGA-cube
    printStatus.running("Reading the MaNGA cube")
    logging.info("Reading the MaNGA cube" + config["GENERAL"]["INPUT"])

    # Reading the cube
    hdu = fits.open(config["GENERAL"]["INPUT"])
    hdr = hdu[1].header
    data = hdu[1].data # FLUX extension
    s = np.shape(data)
    wcshdr = WCS(hdr).to_header()

    # Read the error spectra
    logging.info("Reading the error spectra from the cube")
    stat = hdu[2].data # IVAR extension. 

    logging.info("Reading the binned lsf from the cube")
    lsf = hdu[4].data # LSF extension. 

    # Getting the wavelength info
    wave = hdu['WAVE'].data # Straight from the cube, units are Angstrom. Already log-spaced
    wave = vactoair(wave) #MaNGA wavelengths are in vacuum. MUSE (and nGIST therefore assumes) they are in air. So convert.
    logLam_full = np.log(wave)  # # For ppxf, you need ln(λ), not λ itself Natural log, not log10!

    # Calculate velscale from the logarithmic spacing
    dlog = np.mean(np.diff(logLam_full))
    C = 299792.458 # km/s
    velscale = C * dlog  # C = 299792.458 km/s

    # Here save an central LSF for use in run_ppxf_firsttime
    # Save in the format of 'lsf_manga'
    lsf_cen =  lsf[:, int(lsf.shape[1]/2), int(lsf.shape[2]/2)] #LSF at a central value (i.e. unlikely to be 0)
    lsf_save = np.array([wave, lsf_cen*2.355]).T # The *2.355 is to take account of the fact that MaNGA LSF = 1sigma dispersion, not FWHM
    lsf_savepath = lsfTempFile = os.path.join(config["GENERAL"]["CONFIG_DIR"], config["GENERAL"]["LSF_DATA"])
    np.savetxt(lsf_savepath, lsf_save, delimiter=' ')

##### ---------------- Save spaxel-based stuff now, before you start ditching bins
    # Getting the spatial coordinates
    # Can assume that all MaNGA targets are close enough to centred in the IFU
    origin = [hdr['CRPIX1'], hdr['CRPIX2']]
    xaxis = (np.arange(s[2]) - origin[0]) *  hdr["PC2_2"] * 3600.0 # Changed CD2_2 --> CDELT2 as  In simple, non-rotated images, CDELT1 and CDELT2 are equivalent to CD1_1 and CD2_2, respectively.
    yaxis = (np.arange(s[1]) - origin[1]) * hdr["PC2_2"] * 3600.0
    x, y = np.meshgrid(xaxis, yaxis)
    x = np.reshape(x, [s[1] * s[2]])
    y = np.reshape(y, [s[1] * s[2]])
    pixelsize = hdr["PC2_2"] * 3600.0 # 0.5" spaxels, as expected.

    # Open the corresponding MAPS file, as we need some information from that, too.
    logcubefile = config["GENERAL"]["INPUT"]
    mapsfile = logcubefile.replace("LOGCUBE", "MAPS") #This should now be the filename of the MAPS file 
    hdumaps = fits.open(mapsfile)
    sn = hdumaps['BIN_SNR'].data.flatten()
    nPixels = hdumaps['BIN_AREA'].data.flatten() / pixelsize 
    bin_skycoo = hdumaps['BIN_LWSKYCOO'].data  
    xNode = bin_skycoo[0,:,:].flatten()
    yNode = bin_skycoo[1,:,:].flatten()
    signal = hdumaps['SPX_MFLUX'].data.flatten()
    snr = hdumaps['SPX_SNR'].data.flatten()
    binid = hdumaps['BINID'].data[0,:,:].flatten()
    Hamask = hdumaps['EMLINE_GFLUX_MASK'].data[23,:,:] #23rd channel is Halpha. Use for now.
    Hamask[Hamask>0] = 1
    Hamask = Hamask.astype('int')
    # # Save _table.fits here too, as spatialBinning in not run for MaNGA

    save_table(config, x, y, signal, snr, binid, xNode, yNode, sn, nPixels, pixelsize, wcshdr)

    saveMask(config, np.ravel(Hamask), np.ravel(Hamask), np.ravel(Hamask), np.ravel(Hamask))

##### ----------------

    # Now, extra step because the MaNGA Voronoi binned cubes are nSpax*nSpax, not nBins in size 
    BINID = hdu['BINID'].data
    binid = BINID[0,:,:]
    nwave, nx, ny = data.shape
    data_flat = data.reshape(nwave, nx*ny).T
    stat_flat = stat.reshape(nwave, nx*ny).T
    lsf_flat = lsf.reshape(nwave, nx*ny).T
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
    wave = wave / (1 + config["GENERAL"]["REDSHIFT"])
    logging.info(
        "Shifting spectra to rest-frame, assuming a redshift of "
        + str(config["GENERAL"]["REDSHIFT"])
    )

    # Shorten spectra to required wavelength range
    lmin = config["READ_DATA"]["LMIN_TOT"]
    lmax = config["READ_DATA"]["LMAX_TOT"]
    idx = np.where(np.logical_and(wave >= lmin, wave <= lmax))[0]
    spec = spec[idx, :]
    espec = espec[idx, :]
    lsf = lsf[idx, :]
    wave = wave[idx]
    logging.info(
        "Shortening spectra to the wavelength range from "
        + str(config["READ_DATA"]["LMIN_TOT"])
        + "A to "
        + str(config["READ_DATA"]["LMAX_TOT"])
        + "A."
    )

    # Pass error spectra as variances instead of stddev
    #espec = espec**2
    espec = 1/espec # (As it was IVAR in the MaNGA case.)

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
    # Must be done here as prepareSpectra is not run for MaNGA 
    # Voronoi-binned cubes 
    saveBinSpectra(config, np.log(wave), spec, espec, lsf, velscale =velscale) #

    printStatus.updateDone("Reading the MaNGA cube")
    print("             Read " + str(len(cube["x"])) + " spectra!")
    print("             From " + str(len(idxx)) + " bins!")

    logging.info("Finished reading the MaNGA cube!")

    return cube


