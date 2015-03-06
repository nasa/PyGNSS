"""
Title/Version
-------------
Python CYGNSS Toolkit (PyGNSS)
pygnss v0.2
Developed & tested with Python 2.7.6-2.7.8
Last changed 2/24/2015
    
    
Author
------
Timothy Lang
NASA MSFC
timothy.j.lang@nasa.gov
(256) 961-7861


Overview
--------
This module enables the ingest, analysis, and plotting of Cyclone Global
Navigation Satellite System (CYGNSS) End-to-End Simulator (E2ES) input and 
output data. To use, place in PYTHONPATH and use the following import command:
import pygnss


Notes
-----
Requires - numpy, matplotlib, Basemap, netCDF4, warnings, os


Change Log
----------
v0.2 Major Changes (2/24/2015)
1. Fixed miscellaneous bugs related to data subsectioning and plotting.
2. Added histogram plot for CYGNSS vs. Truth winds.

v0.1 Functionality:
1. Reads netCDFs for input to CYGNSS E2ES as well as output files.
2. Can ingest single-satellite data or merge all 8 together.
3. Capable of masking L2 wind data by RangeCorrectedGain.
4. Basic display objects and plotting routines exist for input/output data, with
   support for combined input/output plots.


Planned Updates
---------------
1. Enable subsectioning of output data by time.
2. Support for DDM file analysis/plotting
3. Merged input/output display object for 1-command combo plots given proper
   files.

"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
from warnings import warn
import os

VERSION = '0.2'

#########################

class NetcdfFile(object):
    
    def __init__(self, filename=None):
        try:
            self.read_netcdf(filename)
        except:
            warn('Please provide a correct filename as argument')

    def read_netcdf(self, filename):
        """variable_list = holds all the variable key strings """
        volume = Dataset(filename, 'r')
        self.filename = os.path.basename(filename)
        self.fill_variables(volume)
    
    def fill_variables(self, volume):
        self.variable_list = volume.variables.keys()
        for key in self.variable_list:
            new_var = np.array(volume.variables[key][:])
            setattr(self, key, new_var)

#########################

class CygnssSingleSat(NetcdfFile):
    
    """In theory works with both L2 Wind and DDM files"""
    
    def get_gain_mask(self, number=4):
        """
        With L2 wind data, identifies top 4 specular points in terms of
        range corrected gain.
        """
        if hasattr(self, 'RangeCorrectedGain'):
            self.GoodData = 0 * np.int16(self.RangeCorrectedGain)
            indices = np.argsort(self.RangeCorrectedGain, axis=1)
            max4 = indices[:,-1*number:]
            for i in np.arange(np.shape(self.RangeCorrectedGain)[0]):
                self.GoodData[i,max4[i]] = 1
            self.variable_list.append('GoodData')

#########################

class CygnssMultiSat(object):
    
    """In theory works with both L2 Wind and L1 DDM files"""
    
    def __init__(self, l2list, number=4):
        """
        l2list = list of CygnssL2SingleSat objects or files
        number = Number of maximum Gain slots to consider
        """
        warntxt = 'Requires input list of CygnssSingleSat '+\
                  'objects or L2 wind files'
        try:
            test = l2list[0].WindSpeed #Not available in DDM, find common var
        except:
            try:
                if isinstance(l2list[0], str):
                    tmplist = []
                    for filen in l2list:
                        sat = CygnssSingleSat(filen)
                        sat.get_gain_mask(number=number)
                        tmplist.append(sat)
                    l2list = tmplist
                else:
                    warn(warntxt)
                    return
            except:
                warn(warntxt)
                return
        self.satellites = l2list
        self.merge_cygnss_data()
        
    def merge_cygnss_data(self):    
        for i, sat in enumerate(self.satellites):
            if i == 0:
                self.variable_list = sat.variable_list
                for var in sat.variable_list:
                    setattr(self, var, getattr(sat, var))
            else:    
                for var in sat.variable_list:
                    array = getattr(sat, var)
                    if np.rank(array) == 1:
                        new_array = np.append(getattr(self, var), array)
                        setattr(self, var, new_array)
                    elif np.rank(array) == 2:
                        new_array = np.append(getattr(self, var), array, axis=1)
                        setattr(self, var, new_array)
                    else:
                        pass #for now ...

#########################

class CygnssL2WindDisplay(object):
    
    def __init__(self, cygnss_sat_object):
        #If passed string(s), try to read the file(s) & make the wind data object
        flag = check_for_strings(cygnss_sat_object)
        if flag == 1:
            cygnss_sat_object = CygnssSingleSat(cygnss_sat_object)
        if flag == 2:
            cygnss_sat_object = CygnssMultiSat(cygnss_sat_object)
        for var in cygnss_sat_object.variable_list:
            setattr(self, var, getattr(cygnss_sat_object, var))
        if hasattr(cygnss_sat_object, 'satellites'):
            setattr(self, 'satellites', getattr(cygnss_sat_object, 'satellites'))
            self.multi_flag = True
        else:
            self.multi_flag = False
        
    def specular_plot(self, cmap='YlOrRd', title='CYGNSS data', vmin=0, vmax=30,
                      ms=50, marker='o', bad=-500, fig=None, ax=None,
                      colorbar_flag=False, basemap=None, edge_flag=False,
                      axis_label_flag=False, title_flag=True, indices=None,
                      save=None, lonrange=None, latrange=None):
        """Function docs here"""
        ws, lon, lat, gd = self.subsection_data(indices)
        good = self.get_good_data_mask(ws, lon, lat, gd, bad=bad)
        if np.size(lon[good]) == 0:
            print 'No good specular points, not plotting'
            return
        fig, ax = parse_fig_ax(fig, ax)
        if edge_flag:
            ec = 'black'
        else:
            ec = 'none'
        if basemap is None:
            sc = ax.scatter(lon[good], lat[good], c=ws[good], vmin=vmin,
                            vmax=vmax, cmap=cmap, s=ms, marker=marker,
                            edgecolors=ec)
            if lonrange is not None:
                ax.set_xlim(lonrange)
            if latrange is not None:
                ax.set_ylim(latrange)
        else:
            x, y = basemap(lon[good], lat[good])
            sc = basemap.scatter(x, y, c=ws[good], vmin=vmin, vmax=vmax,
                                 cmap=cmap, s=ms, marker=marker, edgecolors=ec)
        if colorbar_flag:
            plt.colorbar(sc, label='CYGNSS Wind Speed (m/s)')
        if axis_label_flag:
            plt.xlabel('Longitude (deg E)')
            plt.ylabel('Latitude (deg N)')
        if title_flag:
            plt.title(title)
        if save is not None:
            plt.savefig(save)

    def get_good_data_mask(self, ws, lon, lat, gd, bad=-500):
        good1 = np.logical_and(gd == 1, ws >= 0)
        good2 = np.logical_and(lon > bad, lat > bad)
        return np.logical_and(good1, good2)

    def subsection_data(self, indices, truth_flag=False):
        if indices is None:
            if not truth_flag:
                return self.WindSpeed, self.Longitude, self.Latitude,\
                       self.GoodData
            else:
                return self.WindSpeed, self.Longitude, self.Latitude,\
                       self.GoodData, self.TruthWindSpeed
        else:
            if not truth_flag:
                return self.WindSpeed[indices[0]:indices[1]][:],\
                       self.Longitude[indices[0]:indices[1]][:],\
                       self.Latitude[indices[0]:indices[1]][:],\
                       self.GoodData[indices[0]:indices[1]][:]
            else:
                return self.WindSpeed[indices[0]:indices[1]][:],\
                       self.Longitude[indices[0]:indices[1]][:],\
                       self.Latitude[indices[0]:indices[1]][:],\
                       self.GoodData[indices[0]:indices[1]][:],\
                       self.TruthWindSpeed[indices[0]:indices[1]][:]

    def histogram_plot(self, title='CYGNSS Winds vs. True Winds', fig=None,
                       ax=None, axis_label_flag=False, title_flag=True,
                       indices=None, bins=10, bad=-500):
        ws, lon, lat, gd, tws = self.subsection_data(indices, truth_flag=True)
        good = self.get_good_data_mask(ws, lon, lat, gd, bad=bad)
        if np.size(lon[good]) == 0:
            print 'No good specular points, not plotting'
            return
        fig, ax = parse_fig_ax(fig, ax)
        ax.hist(ws[good].ravel()-tws[good].ravel(), bins=bins, normed=True)
        if axis_label_flag:
            plt.xlabel('CYGNSS Wind Speed - True Wind Speed (m/s)')
            plt.ylabel('Frequency')
        if title_flag:
            plt.title(title)


#########################

class E2esInputData(NetcdfFile):
    
    def get_wind_speed(self):
        self.WindSpeed = np.sqrt(self.eastward_wind**2 + self.northward_wind**2)
        self.variable_list.append('WindSpeed')

#########################

class InputWindDisplay(object):
    
    def __init__(self, input_winds_object):
        #If passed a string, try to read the file and make the input data object
        if isinstance(input_winds_object, str):
            input_winds_object = E2esInputData(input_winds_object)
        if not hasattr(input_winds_object, 'WindSpeed'):
            input_winds_object.get_wind_speed()
        for var in input_winds_object.variable_list:
            setattr(self, var, getattr(input_winds_object, var))

    def basemap_plot(self, fill_color='#ACACBF', ax=None, fig=None,
                     time_index=0, cmap='YlOrRd', vmin=0, vmax=30,
                     colorbar_flag=True, return_flag=True, save=None,
                     title='Input Wind Speed', title_flag=True):
        fig, ax = parse_fig_ax(fig, ax)
        m = get_basemap(lonrange=[np.min(self.longitude),
                                  np.max (self.longitude)],
                        latrange=[np.min(self.latitude), np.max(self.latitude)])
        m.fillcontinents(color=fill_color)
        x, y = m(self.longitude, self.latitude)
        cs = m.pcolormesh(x, y, self.WindSpeed[time_index],
                          vmin=vmin, vmax=vmax, cmap=cmap)
        if title_flag:
            plt.title(title)
        if colorbar_flag:
            m.colorbar(cs, label='Wind Speed (m/s)', location='bottom',
                       pad="7%")
        if save is not None:
            plt.savefig(save)
        if return_flag:
            return m

#########################

#########################

#########################

#########################
#Independent Functions Follow
#########################

def parse_fig_ax(fig, ax):
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()
    return fig, ax

def get_basemap(latrange=[-90,90], lonrange=[-180,180], resolution='l',
                area_thresh=1000):
    lon_0 = np.mean(lonrange)
    lat_0 = np.mean(latrange)
    m = Basemap(projection='merc', lon_0=lon_0, lat_0=lat_0, lat_ts=lat_0,
                llcrnrlat=np.min(latrange), urcrnrlat=np.max(latrange),
                llcrnrlon=np.min(lonrange), urcrnrlon=np.max(lonrange),
                rsphere=6371200., resolution=resolution, area_thresh=area_thresh)
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    return m

def check_for_strings(var):
    """0 = non-string, 1 = string scalar, 2 = string array"""
    if np.size(var) == 1:
        if isinstance(var, str):
            return 1
        else:
            return 0
    else:
        for val in var:
            if not isinstance(val, str):
                return 0
        return 2

