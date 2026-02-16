import code
import logging
import os
import time

import numpy as np
from astropy.io import fits
from printStatus import printStatus
from collections import defaultdict



def SortGasMaps(mapfile, sortby):
    """
    Automatically detect and swap narrow/broad component data where narrow sigma > broad sigma.
    
    Parameters:
    -----------
    hdulist : astropy.io.fits.HDUList
        The FITS file with extensions to correct
    """
    
    hdulist = fits.open(mapfile)
    
    # First, identify all unique lines that have both NARROW and BROAD components
    lines_with_components = defaultdict(lambda: {'narrow': {}, 'broad': {}})
    
    for i, hdu in enumerate(hdulist):
        name = hdu.name
        
        if 'NARROW' in name:
            # Extract the base line name (e.g., 'NII_NARROW6548' -> 'NII6548')
            parts = name.split('_NARROW')
            if len(parts) == 2:
                line_base = parts[0] + parts[1].split('_')[0]  # e.g., 'NII6548'
                property_name = '_'.join(parts[1].split('_')[1:])  # e.g., 'SIGMA', 'FLUX'
                lines_with_components[line_base]['narrow'][property_name] = i
                
        elif 'BROAD' in name:
            # Extract the base line name
            parts = name.split('_BROAD')
            if len(parts) == 2:
                line_base = parts[0] + parts[1].split('_')[0]
                property_name = '_'.join(parts[1].split('_')[1:])
                lines_with_components[line_base]['broad'][property_name] = i
        
    # Filter to only lines that have both narrow AND broad components
    lines_to_process = {}
    for line, components in lines_with_components.items():
        if components['narrow'] and components['broad']:
            # Check if both have SIGMA
            if 'SIGMA' in components['narrow'] and 'SIGMA' in components['broad']:
                lines_to_process[line] = components
            else:
                print(f"Warning: {line} missing SIGMA in narrow or broad component")
    
    print(f"Found {len(lines_to_process)} lines with both narrow and broad components:")
    for line in sorted(lines_to_process.keys()):
        print(f"  - {line}")
    
    # Now swap components where needed
    for line, components in lines_to_process.items():
        
        if sortby == 'sigma':
            narrow_sigma_idx = components['narrow']['SIGMA']
            broad_sigma_idx = components['broad']['SIGMA']
            
            # Get sigma data
            narrow_sigma = hdulist[narrow_sigma_idx].data
            broad_sigma = hdulist[broad_sigma_idx].data
            
            # Create mask where swap is needed (narrow > broad)
            swap_mask = narrow_sigma > broad_sigma
            n_swaps = np.sum(swap_mask)
        
        elif sortby == 'amplitude':
            narrow_flux_idx = components['narrow']['FLUX']
            broad_flux_idx = components['broad']['FLUX']
            
            # Get flux data
            narrow_flux = hdulist[narrow_flux_idx].data
            broad_flux = hdulist[broad_flux_idx].data
            
            # Create mask where swap is needed (narrow > broad)
            swap_mask = narrow_flux > broad_flux
            n_swaps = np.sum(swap_mask)
            
        elif sortby == 'velocity':
            narrow_vel_idx = components['narrow']['VEL']
            broad_vel_idx = components['broad']['VEL']
            
            # Get velocity data
            narrow_vel = hdulist[narrow_vel_idx].data
            broad_vel = hdulist[broad_vel_idx].data
            
            # Create mask where swap is needed (narrow > broad)
            swap_mask = np.abs(narrow_vel) > np.abs(broad_vel)
            n_swaps = np.sum(swap_mask)
        
        if n_swaps > 0:
            print(f"\n{line}: Swapping {n_swaps} pixels where narrow > broad")
            
            # Find all properties to swap (intersection of narrow and broad properties)
            properties_to_swap = set(components['narrow'].keys()) & set(components['broad'].keys())
            
            for prop in sorted(properties_to_swap):
                narrow_idx = components['narrow'][prop]
                broad_idx = components['broad'][prop]
                
                # Swap the data where mask is True
                narrow_data = hdulist[narrow_idx].data.copy()
                broad_data = hdulist[broad_idx].data.copy()
                
                hdulist[narrow_idx].data[swap_mask] = broad_data[swap_mask]
                hdulist[broad_idx].data[swap_mask] = narrow_data[swap_mask]
                
                print(f"  Swapped {prop}: {hdulist[narrow_idx].name} <-> {hdulist[broad_idx].name}")
        else:
            print(f"\n{line}: All narrow <= broad, no swaps needed")
        
    return hdulist


def create_sorted_gas_maps(config):
    """
    Sort the gas maps (if any) in the correct order for output.
    """
    runname = config["GENERAL"]["RUN_ID"]
    sortby = config["UMOD"]["GAS_MAP_SORTBY"]
    if config["UMOD"]["LEVEL"] != "BOTH":
        inputfits = os.path.join(config["GENERAL"]["OUTPUT"], runname) + "_gas_" + config["UMOD"]["LEVEL"] + "_maps.fits"
        if not os.path.isfile(inputfits):
            printStatus.warning(
                "The file "
                + runname
                + "_gas_" + config["UMOD"]["LEVEL"] + "_maps.fits does not exist. Cannot sort gas maps."
            )
            logging.warning(
                "The file "
                + runname
                + "_gas_" + config["UMOD"]["LEVEL"] + "_maps.fits does not exist. Cannot sort gas maps."
            )
            return None
        else:
            hdulist = SortGasMaps(inputfits, sortby=sortby)
            outfits = os.path.join(config["GENERAL"]["OUTPUT"], runname) + "_gas_" + config["UMOD"]["LEVEL"] + "_maps_" + sortby + "Sorted.fits"
            hdulist.writeto(outfits, overwrite=True)
            printStatus.updateDone(
                "Sorting gas maps by " + sortby + " for level " + config["UMOD"]["LEVEL"]
            )
            logging.info(
                "Sorted gas maps by " + sortby + " for level " + config["UMOD"]["LEVEL"]
            )
    else:
        for config["UMOD"]["LEVEL"] in ["BIN", "SPAXEL"]:
            inputfits = os.path.join(config["GENERAL"]["OUTPUT"], runname) + "_gas_" + config["UMOD"]["LEVEL"] + "_maps.fits"
            if not os.path.isfile(inputfits):
                printStatus.warning(
                    "The file "
                    + runname
                    + "_gas_" + config["UMOD"]["LEVEL"] + "_maps.fits does not exist. Cannot sort gas maps."
                )
                logging.warning(
                    "The file "
                    + runname
                    + "_gas_" + config["UMOD"]["LEVEL"] + "_maps.fits does not exist. Cannot sort gas maps."
                )
                return None
            else:
                hdulist = SortGasMaps(inputfits, sortby=sortby)
                outfits = os.path.join(config["GENERAL"]["OUTPUT"], runname) + "_gas_" + config["UMOD"]["LEVEL"] + "_maps_" + sortby + "Sorted.fits"
                hdulist.writeto(outfits, overwrite=True)
                printStatus.updateDone(
                    "Sorting gas maps by " + sortby + " for level " + config["UMOD"]["LEVEL"]
                )
                logging.info(
                    "Sorted gas maps by " + sortby + " for level " + config["UMOD"]["LEVEL"]
                )