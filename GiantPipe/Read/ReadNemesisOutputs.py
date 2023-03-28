import numpy.typing as npt
import numpy as np
import operator
import Globals
from typing import List, Tuple
import time
from sorcery import dict_of

class ReadNemesisOutputs:

    def __init__():
        return
    
    def RetrieveGasesNames(gas_id):

        if gas_id == 11: 
            gas_name = r'NH$_{3}$'
        if gas_id == 28:
            gas_name = r'PH$_{3}$'
        if gas_id == 26: 
            gas_name = r'C$_{2}$H$_{2}$'
        if gas_id == 32:
            gas_name = r'C$_{2}$H$_{4}$'
        if gas_id == 27: 
            gas_name = r'C$_{2}$H$_{6}$'
        if gas_id == 30:
            gas_name = r'C$_{4}$H$_{2}$'
        if gas_id == 39: 
            gas_name = r'H$_{2}$'
        if gas_id == 40:
            gas_name = r'He'
        if gas_id == 6: 
            gas_name = r'CH$_{4}$'

        return gas_name


    @staticmethod
    def get_mre_header(lines: List[str]) -> List[int]:
        line_1 = lines[0].split()
        line_2 = lines[1].split()
        line_3 = lines[2].split()
        number_of_retrievals = line_1[0]
        ngeom, ny, nx = line_2[1], line_2[2], line_2[3]
        latitude, longitude = line_3[0], line_3[1]
        return int(ngeom), int(ny), int(nx), float(latitude), float(longitude)
    
    @staticmethod
    def get_nvar(lines: List[str]) -> int:
        nvar = [line.split() for _, line in enumerate(lines) if 'nvar' in line].pop()
        return int(nvar[-1])
    
    @staticmethod
    def find_vars(lines: List[str]) -> List[str]:
        return [i for i, line in enumerate(lines) if 'Variable' in line]
    
    @staticmethod
    def get_apr(vars_idx: List[int], lines: List[str]) -> List[str]:
        return [lines[i+1].split() for i in vars_idx]
    
    @staticmethod
    def get_profile_header(vars_idx: List[int], lines: List[str]) -> List[str]:
        return [lines[i+3] for i in vars_idx]
    
    @classmethod
    def get_retrieved_profiles(cls, lines: List[str]) -> List[str]:
        
        # Number of variables (e.g. temperature, aerosols gases)
        nvar = cls.get_nvar(lines)
        
        # Find where each variable starts in the file
        vars_idx = cls.find_vars(lines)
        
        # Extract retrieval configuration and output columns
        apr_config = cls.get_apr(vars_idx, lines)
        header = cls.get_profile_header(vars_idx, lines)
        
        data = []
        for i, _ in enumerate(vars_idx):
            if i < nvar-1:
                data.append(lines[vars_idx[i]+4:vars_idx[i+1]])
            else:
                data.append(lines[vars_idx[i]+4:])
        return apr_config, header, data

    @classmethod
    def convert_profiles_to_data(cls, lines: List[str], profiles: List[str]) -> List[npt.ArrayLike]:
                
        # Number of variables (e.g. temperature, aerosols gases)
        nvar = cls.get_nvar(lines)

        # Prepare list of lists to store float values
        data = [[] for _ in range(nvar)]
        for iprofile, profile in enumerate(profiles):
            for level in profile:
                elements = level.split()
                data[iprofile].append(elements[2:6])
        return data
    
    @staticmethod
    def get_spectrum_header(lines: List[str]) -> List[str]:
        return [[i, line] for i, line in enumerate(lines) if 'lambda' in line].pop()
    
    @classmethod
    def get_retrieved_spectrum(cls, lines: List[str], ngeom: int, ny: int) -> List[str]:
        """Follows NEMESIS manual conventions.
        NCONV: int = number of convolution wavenumbers per geometry
        NGEOM: int = number geometries
        NY: int = number of spectral points (NCONV * NGEOM)"""

        # Find where each variable starts in the file
        header = cls.get_spectrum_header(lines)
        header_start = header[0]
        # Construct spectrum nested list
        nconv = int(ny / (ngeom))
        spectrum = [[] for _ in range(ngeom)]
        for igeom in range(ngeom):
            geom_start = header_start + (nconv * igeom)
            geom_end = header_start + (nconv * (igeom + 1))
            geom = lines[geom_start+1:geom_end+1]
            values = (conv.split() for conv in geom)
            reshape_values = [list(value) for value in zip(*values)]
            spectrum[igeom].extend(reshape_values)
        return header, spectrum
    
    @staticmethod
    def get_spx_header(lines: List[str]) -> list:
        null, latitude, longitude, ngeom = [float(value) for value in lines[0].split()]
        return null, latitude, longitude, int(ngeom)
    
    @staticmethod
    def find_angles_line(lines: List[str]) -> list:
       return [i for i, line in enumerate(lines) if len(line.split()) == 6]

    @classmethod
    def get_spx_block(cls, lines: List[str], ngeom: int) -> list:
        """Follows NEMESIS manual conventions.
        NCONV: int = number of convolution wavenumbers per geometry
        NGEOM: int = number geometries
        NY: int = number of spectral points (NCONV * NGEOM)"""

        # Specify where geometries start in the file (always the second line)
        block_idx = cls.find_angles_line(lines)

        # Construct lists to hold data
        nconv, nav, angles, spectrum = [[] for _ in range(4)] 
        for i, igeom in enumerate(block_idx):
            nconv.append(int(lines[igeom-2]))
            nav.append(int(lines[igeom-1]))
            angles.append([value for value in lines[igeom].split()])
            geom_start = igeom
            geom_end = igeom + nconv[i]
            geom = lines[geom_start+1:geom_end+1]
            spectrum.append((conv.split() for conv in geom))
        return nconv, nav, angles, spectrum
    
    @classmethod
    def read_mre(cls, filepath: str) -> dict:

        with open(filepath) as f:
            
            # Read file
            lines = f.readlines()

            # Get .mre header
            ngeom, ny, nx, latitude, longitude = cls.get_mre_header(lines)

            # Get spectral information
            spectrum_header, spectrum = cls.get_retrieved_spectrum(lines, ngeom, ny)

            # Get retrieved profiles and convert to float type
            apr_config, profiles_header, raw_profiles = cls.get_retrieved_profiles(lines)
            profiles = cls.convert_profiles_to_data(lines, raw_profiles)

        return dict_of(spectrum, profiles, apr_config, ngeom, ny, latitude)
    
    @classmethod
    def read_spx(cls, filepath: str) -> dict:
        
        with open(filepath) as f:
            
            # Read file
            lines = f.readlines()

            # Get .spx header
            _, latitude, longitude, ngeom = cls.get_spx_header(lines)

            # Read in spectrum for each block (geometry)
            nconv, nav, angles, spectrum = cls.get_spx_block(lines, ngeom)

        return dict_of(nconv, ngeom, angles)
    
    @classmethod
    def read_kk(cls, filepath: str) -> dict:

        return

    @classmethod
    def check_core(cls, filepaths: List[str], latitude: float) -> Tuple[bool, str]:
        
        check = []
        for filepath in filepaths:
            # Get latitudes .mre file
            mre = cls.read_mre(f"{filepath}/nemesis.mre")
            check.append(mre["latitude"])
        
        # Return a tuple containing the location of the match
        find = check == latitude
        found = [filepath for i, filepath in enumerate(filepaths) if find[i] == True]
        if not found:
            return False
        else:
            return found.pop()