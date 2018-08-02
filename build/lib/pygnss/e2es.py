"""
Title/Version
-------------
Python CYGNSS Toolkit (PyGNSS)
pygnss v0.7
Developed & tested with Python 2.7 and 3.4


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
Requires - numpy, matplotlib, Basemap, netCDF4, warnings, os, six, datetime,
           sklearn, copy


Change Log
----------
v0.7 Major Changes (04/20/2016)
1. Added CygnssSubsection class to consolidate and simplify data subsectioning.
   Can be called independently, but is also used by the plotting methods in
   CygnssL2WindDisplay class. Moved the subsection_data and get_good_data_mask
   methods to this new class, and greatly modifed them to eliminate
   method returns.
2. Added ability to subsection data by CYGNSS and GPS satellite numbers, as
   well as by range-corrected gain threshold or interval. All plotting routines
   now support these subsectioning capabilities.
3. Added get_datetime indpendent function to derive datetime objects in the
   same array shape as the WindSpeed data. Added datetime module dependency.
4. Added CygnssTrack class to leverage CygnssSubsection to help isolate
   and contain all the data from an individual track. Also enables filtering
   of wind speeds along a track.
5. Added get_tracks independent function to isolate all individual tracks, and
   return a list of CygnssTrack objects from a CygnssSingleSat, CygnssMultiSat,
   or CygnssL2WindDisplay object.

v0.6 Major Changes (11/20/2015)
1. Added hist2d_plot method to CygnssL2WindDisplay.
2. Added threshold keyword to allow filtering of histrogram figures by
   RangeCorrectedGain windows.

v0.5 Major Changes (08/10/2015)
1. Supports Python 3 now.

v0.4 Major Changes (07/02/2015)
1. Made all code pep8 compliant.

v0.3 Major Changes (03/19/2015)
1. Documentation improvements. Doing help(pygnss) should be more useful now.
2. Fixed bug where GoodData attribute was not getting set for CygnssSingleSat
   objects after they were input into CygnssL2WindDisplay.
3. Fixes to ensure CygnssSingle/MultiSat classes can ingest L1 DDM files w/out
   errors. This provides a basis for adding L1 DDM functionality to PyGNSS.
4. Swapped out np.rank for np.ndim due to annoying deprecation warnings.

v0.2 Major Changes (02/24/2015)
1. Fixed miscellaneous bugs related to data subsectioning and plotting.
2. Added histogram plot for CYGNSS vs. Truth winds.

v0.1 Functionality:
1. Reads netCDFs for input to CYGNSS E2ES as well as output files.
2. Can ingest single-satellite data or merge all 8 together.
3. Capable of masking L2 wind data by RangeCorrectedGain.
4. Basic display objects & plotting routines exist for input/output data, w/
   support for combined input/output plots.


Planned Updates
---------------
1. Enable subsectioning of output data specifically by time rather than index.
2. Support for DDM file analysis/plotting
3. Merged input/output display object for 1-command combo plots given proper
   inputs.
4. Get CygnssL2WindDisplay.specular_plot() to automatically adjust size of
   points to reflect actual CYGNSS spatial resolution on Basemap. Right now,
   user just has manual control of the marker size and would need to guess at
   this if they want the specular points to be truly spatially accurate.
5. Incorporate land/ocean flag in non-Basemap CYGNSS plots

"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
from warnings import warn
import os
from six import string_types
import datetime as dt
from sklearn.cluster import DBSCAN
from copy import deepcopy

VERSION = '0.7'

#########################


class NetcdfFile(object):

    """Base class used for reading netCDF-format L1 and L2 CYGNSS data files"""

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
        """Loop thru all variables and store them as attributes"""
        self.variable_list = []
        for key in volume.variables.keys():
            new_var = np.array(volume.variables[key][:])
            setattr(self, key, new_var)
            self.variable_list.append(key)

#########################


class CygnssSingleSat(NetcdfFile):

    """
    Child class of NetcdfFile. Can ingest both L2 Wind and DDM files.
    All variables within the files are incorporated as attributes of the
    class. This class forms the main building block of PyGNSS.
    """

    def get_gain_mask(self, number=4):
        """
        With L2 wind data, identifies top specular points in terms of
        range corrected gain. Creates the GoodData attribute, which provides
        a mask that analysis and plotting routines can use to only consider
        specular points with the highest RangeCorrectedGain
        number = Number of specular points to consider in the rankings
        """
        if hasattr(self, 'RangeCorrectedGain'):
            self.GoodData = 0 * np.int16(self.RangeCorrectedGain)
            indices = np.argsort(self.RangeCorrectedGain, axis=1)
            max4 = indices[:, -1*number:]
            for i in np.arange(np.shape(self.RangeCorrectedGain)[0]):
                self.GoodData[i, max4[i]] = 1
            self.variable_list.append('GoodData')

#########################


class CygnssMultiSat(object):

    """
    Can ingest both L2 Wind and L1 DDM files. Merges the CYGNSS constellation's
    individual satellites' data together into a class structure very similar to
    CygnssSingleSat, just with bigger array dimensions.
    """

    def __init__(self, l2list, number=4):
        """
        l2list = list of CygnssL2SingleSat objects or files
        number = Number of maximum RangeCorrectedGain slots to consider
        """
        warntxt = 'Requires input list of CygnssSingleSat ' + \
            'objects or L2 wind files'
        try:
            test = l2list[0].WindSpeed
        except:
            try:
                if isinstance(l2list[0], string_types):
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
        """
        Loop over each satellite and append its data to the master arrays
        """
        for i, sat in enumerate(self.satellites):
            if i == 0:
                self.variable_list = sat.variable_list
                for var in sat.variable_list:
                    setattr(self, var, getattr(sat, var))
            else:
                for var in sat.variable_list:
                    array = getattr(sat, var)
                    if np.ndim(array) == 1:
                        new_array = np.append(getattr(self, var), array)
                        setattr(self, var, new_array)
                    elif np.ndim(array) == 2:
                        new_array = np.append(
                            getattr(self, var), array, axis=1)
                        setattr(self, var, new_array)
                    else:
                        pass  # for now ...

#########################


class CygnssL2WindDisplay(object):

    """
    This display class provides an avenue for making plots from CYGNSS L2 wind
    data.
    """

    def __init__(self, cygnss_sat_object, number=4):

        """
        cygnss_sat_object = CygnssSingle/MultiSat object, single L2 file,
                            or list of files.
        number = Number of specular points to consider in RangeCorrectedGain
                 rankings.
        """
        # If passed string(s), try to read file(s) & make the wind data object
        flag = check_for_strings(cygnss_sat_object)
        if flag == 1:
            cygnss_sat_object = CygnssSingleSat(cygnss_sat_object)
        if flag == 2:
            cygnss_sat_object = CygnssMultiSat(cygnss_sat_object,
                                               number=number)
        if not hasattr(cygnss_sat_object, 'GoodData'):
            try:
                cygnss_sat_object.get_gain_mask(number=number)
            except:
                pass
        # Try again to confirm L2, this avoids problems
        # caused by ingest of L1 DDM
        if not hasattr(cygnss_sat_object, 'GoodData'):
            warn('Not a CYGNSS L2 wind object most likely, failing ...')
            return
        for var in cygnss_sat_object.variable_list:
            setattr(self, var, getattr(cygnss_sat_object, var))
        if hasattr(cygnss_sat_object, 'satellites'):
            setattr(self, 'satellites', getattr(cygnss_sat_object,
                                                'satellites'))
            self.multi_flag = True
        else:
            self.multi_flag = False
        self.variable_list = cygnss_sat_object.variable_list

    def specular_plot(self, cmap='YlOrRd', title='CYGNSS data', vmin=0,
                      vmax=30, ms=50, marker='o', bad=-500, fig=None, ax=None,
                      colorbar_flag=False, basemap=None, edge_flag=False,
                      axis_label_flag=False, title_flag=True, indices=None,
                      save=None, lonrange=None, latrange=None,
                      truth_flag=False, return_flag=False, gpsid=None,
                      gain=None, sat=None, **kwargs):
        """
        Plots CYGNSS specular points on lat/lon axes using matplotlib's scatter
        object, which colors each point based on its wind speed value.

        cmap = matplotlib or user-defined colormap
        title = Title of plot
        vmin = Lowest wind speed value to display on color table
        vmax = Highest wind speed value to display on color table
        ms = Size of marker used to plot each specular point
        marker = Marker shape to use ('o' is best)
        bad = Bad value of Lat/Lon to throw out
        fig = matplotlib Figure object to use
        ax = matplotlib Axes object to use
        colorbar_flag = Set to True to show the colorbar
        basemap = Basemap object to use in plotting the specular points
        edge_flag = Set to True to show a black edge to make each specular
                    point more distinctive
        axis_label_flag = Set to True to label lat/lon axes
        title_flag = Set to False to suppress title
        indices = Indices (2-element tuple) to use to limit the period of data
                  shown (i.e., limit by time)
        save = Name of image file to save plot to
        lonrange = 2-element tuple to limit longitude range of plot
        latrange = 2-element tuple to limit latitude range of plot
        gpsid = Integer ID number for GPS satellite to examine
        sat = CYGNSS satellite number (0-7)
        gain = Threshold for range-corrected gain (RCG). Can be 2-ele. tuple.
               If so, use only RCG within that range. If scalar, then use
               data above the given RCG value.
        return_flag = Set to True to return Figure, Axes, and Basemap objects
                      (in that order)
        """
        ds = CygnssSubsection(self, indices=indices, bad=bad,
                              gpsid=gpsid, gain=gain, sat=sat)
        if truth_flag:
            ws = ds.tws
        else:
            ws = ds.ws
        if np.size(ds.lon[ds.good]) == 0:
            print('No good specular points, not plotting')
            return
        fig, ax = parse_fig_ax(fig, ax)
        if edge_flag:
            ec = 'black'
        else:
            ec = 'none'
        if basemap is None:
            sc = ax.scatter(ds.lon[ds.good], ds.lat[ds.good], c=ws[ds.good],
                            vmin=vmin, vmax=vmax, cmap=cmap, s=ms,
                            marker=marker, edgecolors=ec, **kwargs)
            if lonrange is not None:
                ax.set_xlim(lonrange)
            if latrange is not None:
                ax.set_ylim(latrange)
        else:
            x, y = basemap(ds.lon[ds.good], ds.lat[ds.good])
            sc = basemap.scatter(x, y, c=ws[ds.good], vmin=vmin, vmax=vmax,
                                 cmap=cmap, s=ms, marker=marker, edgecolors=ec,
                                 **kwargs)
        if colorbar_flag:
            plt.colorbar(sc, label='CYGNSS Wind Speed (m/s)')
        if axis_label_flag:
            plt.xlabel('Longitude (deg E)')
            plt.ylabel('Latitude (deg N)')
        if title_flag:
            plt.title(title)
        if save is not None:
            plt.savefig(save)
        if return_flag:
            return fig, ax, basemap, sc

    def histogram_plot(self, title='CYGNSS Winds vs. True Winds', fig=None,
                       ax=None, axis_label_flag=False, title_flag=True,
                       indices=None, bins=10, bad=-500, save=None,
                       gain=None, sat=None):
        """
        Plots a normalized histogram of CYGNSS wind speed vs. true wind speed
        (as provided by the input data to the E2ES).

        bins = Number of bins to use in the histogram
        title = Title of plot
        bad = Bad value of Lat/Lon to throw out
        fig = matplotlib Figure object to use
        ax = matplotlib Axes object to use
        axis_label_flag = Set to True to label lat/lon axes
        title_flag = Set to False to suppress title
        indices = Indices (2-element tuple) to use to limit the period of data
                  shown (i.e., limit by time)
        gain = Threshold for range-corrected gain (RCG). Can be 2-ele. tuple.
               If so, use only RCG within that range. If scalar, then use
               data above the given RCG value.
        save = Name of image file to save plot to
        sat = CYGNSS satellite number (0-7)
        """
        ds = CygnssSubsection(self, indices=indices, gain=gain, bad=bad,
                              sat=sat)
        if np.size(ds.lon[ds.good]) == 0:
            print('No good specular points, not plotting')
            return
        fig, ax = parse_fig_ax(fig, ax)
        ax.hist(ds.ws[ds.good].ravel()-ds.tws[ds.good].ravel(), bins=bins,
                normed=True)
        if axis_label_flag:
            plt.xlabel('CYGNSS Wind Speed - True Wind Speed (m/s)')
            plt.ylabel('Frequency')
        if title_flag:
            plt.title(title)
        if save is not None:
            plt.savefig(save)

    def hist2d_plot(self, title='CYGNSS Winds vs. True Winds', fig=None,
                    ax=None, axis_label_flag=False, title_flag=True,
                    indices=None, bins=20, bad=-500, save=None,
                    gain=None, colorbar_flag=True,
                    cmap='YlOrRd', range=(0, 20), ls='--',
                    add_line=True, line_color='r', sat=None,
                    colorbar_label_flag=True, **kwargs):
        """
        Plots a normalized 2D histogram of CYGNSS wind speed vs. true wind spd
        (as provided by the input data to the E2ES). This information can be
        thresholded by RangeCorrectedGain

        bins = Number of bins to use in the histogram
        title = Title of plot
        bad = Bad value of Lat/Lon to throw out
        fig = matplotlib Figure object to use
        ax = matplotlib Axes object to use
        axis_label_flag = Set to True to label lat/lon axes
        title_flag = Set to False to suppress title
        indices = Indices (2-element tuple) to use to limit the period of data
                  shown (i.e., limit by time)
        gain = Threshold for range-corrected gain (RCG). Can be 2-ele. tuple.
               If so, use only RCG within that range. If scalar, then use
               data above the given RCG value.
        save = Name of image file to save plot to
        sat = CYGNSS satellite number (0-7)
        **kwargs = Whatever else pyplot.hist2d will accept
        """
        ds = CygnssSubsection(self, indices=indices, gain=gain,
                              bad=bad, sat=sat)
        if np.size(ds.lon[ds.good]) == 0:
            print('No good specular points, not plotting')
            return
        fig, ax = parse_fig_ax(fig, ax)
        H, xedges, yedges, img = ax.hist2d(
            ds.ws[ds.good].ravel(), ds.tws[ds.good].ravel(), bins=bins,
            normed=True, cmap=cmap, zorder=1, range=[range, range],
            **kwargs)
        # These 2 lines are a hack to get color bar, but not plot anything new
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(H.T, cmap=cmap, extent=extent, zorder=0)
        ax.set_xlim(range)
        ax.set_ylim(range)
        if add_line:
            ax.plot(range, range, ls=ls,
                    color=line_color, lw=2, zorder=2)
        if axis_label_flag:
            ax.set_xlabel('CYGNSS Wind Speed (m/s)')
            ax.set_ylabel('True Wind Speed (m/s)')
        if title_flag:
            ax.set_title(title)
        if colorbar_flag:
            if colorbar_label_flag:
                label = 'Frequency'
            else:
                label = ''
            plt.colorbar(im, label=label, ax=ax, shrink=0.75)
        if save is not None:
            plt.savefig(save)

#########################


class CygnssSubsection(object):

    """
    Class to handle subsectioning CYGNSS data. Subsectioning by
    satellite (via CygnssSingleSat input), time indices, GPS satellite ID,
    range-corrected gain, etc. is supported.

    Main Attributes
    ---------------
    ws = Wind speed array
    lon = Longitude array
    lat = Latitude array
    gd = GoodData array
    rcg = RangeCorrectedGain array
    gps = GpsID array
    """

    def __init__(self, data, indices=None, gpsid=None, gain=None, bad=-500,
                 sat=None):
        """
        data = CygnssSingleSat, CygnssMultiSat, or CygnssL2WindDisplay object
        gpsid = Integer ID number for GPS satellite to examine
        gain = Threshold by range-corrected gain, values below will be masked
        bad = Value to compare against lat/lon to mask out missing data
        sat = CYGNSS satellite number (0-7)
        indices = Indices (2-element tuple) to use to limit the period of data
                  shown (i.e., limit by time)
        """
        # Set basic attributes based on input data object
        if sat is not None and hasattr(data, 'satellites'):
            data = data.satellites[sat]
        self.ws = data.WindSpeed
        self.tws = data.TruthWindSpeed
        self.lon = data.Longitude
        self.lat = data.Latitude
        self.gps = data.GpsID
        self.rcg = data.RangeCorrectedGain
        self.gd = data.GoodData

        # Set keyword-based attributes
        self.gpsid = gpsid
        self.gain = gain
        self.bad = bad
        self.indices = indices

        # Now subsection the data
        self.subsection_data()
        self.get_good_data_mask()

    def subsection_data(self):
        """
        This method subsections the L2 wind data and returns these as arrays
        ready to plot.
        """
        if self.indices is not None:
            self.ws = self.ws[self.indices[0]:self.indices[1]][:]
            self.tws = self.tws[self.indices[0]:self.indices[1]][:]
            self.lon = self.lon[self.indices[0]:self.indices[1]][:]
            self.lat = self.lat[self.indices[0]:self.indices[1]][:]
            self.gd = self.gd[self.indices[0]:self.indices[1]][:]
            self.gps = self.gps[self.indices[0]:self.indices[1]][:]
            self.rcg = self.rcg[self.indices[0]:self.indices[1]][:]

    def get_good_data_mask(self):
        """
        Sets a mask used to limit the data plotted. Filtered out are data
        masked out by the GoodData mask (based on RangeCorrectedGain), missing
        lat/lon values, and bad data (ws < 0)
        """
        good1 = np.logical_and(self.gd == 1, self.ws >= 0)
        good2 = np.logical_and(self.lon > self.bad, self.lat > self.bad)
        if self.gpsid is not None and type(self.gpsid) is int:
            good2 = np.logical_and(good2, self.gps == self.gpsid)
        if self.gain is not None:
            if np.size(self.gain) == 2:
                cond = np.logical_and(self.rcg >= self.gain[0],
                                      self.rcg < self.gain[1])
                good2 = np.logical_and(good2, cond)
            else:
                good2 = np.logical_and(good2, self.rcg >= self.gain)
        self.good = np.logical_and(good1, good2)

#########################


class CygnssTrack(object):

    """
    Class to facilitate extraction of a single track of specular points
    from a CygnssSingleSat, CygnssMultiSat, or CygnssL2WindDisplay object.

    Attributes
    ----------
    input = CygnssSubsection object
    ws = CYGNSS wind speeds
    tws = Truth wind speeds
    lon = Longitudes of specular points
    lat = Latitudes of specular points
    rcg = Range-corrected gains of specular points
    datetimes = Datetime objects for specular points

    The following attributes are created by filter_track method:
    fws = Filtered wind speeds
    flon = Filtered longitudes
    flat = Filtered latitudes
    These attributes are shorter than the main attributes by the window length
    """

    def __init__(self, data, datetimes=None, **kwargs):
        """
        data = CygnssSingleSat, CygnssMultiSat, or CygnssL2WindDisplay object
        datetimes = List of datetime objects from get_datetime function.
                    If None, this function is called.
        """
        self.input = CygnssSubsection(data, **kwargs)
        self.ws = self.input.ws[self.input.good]
        self.tws = self.input.tws[self.input.good]
        self.lon = self.input.lon[self.input.good]
        self.lat = self.input.lat[self.input.good]
        self.rcg = self.input.rcg[self.input.good]
        if datetimes is None:
            dts = get_datetime(data)
        else:
            dts = datetimes
        if self.input.indices is not None:
            self.datetimes = dts[
                self.input.indices[0]:self.input.indices[1]][self.input.good]
        else:
            self.datetimes = dts[self.input.good]

    def filter_track(self, window=5):
        """
        Applies a running-mean filter to the track.

        window = Number of specular points in the running mean window.
                 Must be odd.
        """
        if window % 2 == 0:
            raise ValueError('Window must be odd length, not even.')
        hl = int((window - 1) / 2)
        self.fws = np.convolve(
            self.ws, np.ones((window,))/window, mode='valid')
        self.flon = self.lon[hl:-1*hl]
        self.flat = self.lat[hl:-1*hl]

#########################


class E2esInputData(NetcdfFile):

    """Base class for ingesting E2ES input data. Child class of NetcdfFile."""

    def get_wind_speed(self):
        """
        Input E2ES data normally don't have wind speed as a field. This method
        fixes that.
        """
        self.WindSpeed = np.sqrt(self.eastward_wind**2 +
                                 self.northward_wind**2)
        self.variable_list.append('WindSpeed')

#########################


class InputWindDisplay(object):

    """Display object for the E2ES input data"""

    def __init__(self, input_winds_object):
        """
        input_winds_object = Input E2esInputData object or wind file
        """
        # If passed a string, try to read the file & make input data object
        if isinstance(input_winds_object, string_types):
            input_winds_object = E2esInputData(input_winds_object)
        if not hasattr(input_winds_object, 'WindSpeed'):
            input_winds_object.get_wind_speed()
        for var in input_winds_object.variable_list:
            setattr(self, var, getattr(input_winds_object, var))
        self.make_coordinates_2d()

    def basemap_plot(self, fill_color='#ACACBF', ax=None, fig=None,
                     time_index=0, cmap='YlOrRd', vmin=0, vmax=30,
                     colorbar_flag=True, return_flag=True, save=None,
                     title='Input Wind Speed', title_flag=True,
                     show_grid=False):
        """
        Plots E2ES input wind speed data on a Basemap using matplotlib's
        pcolormesh object. Defaults to return the Basemap object so other
        things (e.g., CYGNSS data) can be overplotted.

        fill_color = Color to fill continents
        time_index = If the input data contain more than one time step, this
                     index selects the time step to display
        cmap = matplotlib or user-defined colormap
        title = Title of plot
        vmin = Lowest wind speed value to display on color table
        vmax = Highest wind speed value to display on color table
        fig = matplotlib Figure object to use
        ax = matplotlib Axes object to use
        return_flag = Set to False to suppress Basemap object return
        title_flag = Set to False to suppress title
        save = Name of image file to save plot to
        colorbar_flag = Set to False to suppress the colorbar
        show_grid = Set to True to show the lat/lon grid and label it
        """
        fig, ax = parse_fig_ax(fig, ax)
        m = get_basemap(lonrange=[np.min(self.longitude),
                                  np.max(self.longitude)],
                        latrange=[np.min(self.latitude),
                                  np.max(self.latitude)])
        m.fillcontinents(color=fill_color)
        x, y = m(self.longitude, self.latitude)
        cs = m.pcolormesh(x, y, self.WindSpeed[time_index],
                          vmin=vmin, vmax=vmax, cmap=cmap)
        if show_grid:
            m.drawmeridians(
                np.arange(-180, 180, 5), labels=[True, True, True, True])
            m.drawparallels(
                np.arange(-90, 90, 5), labels=[True, True, True, True])
        if title_flag:
            plt.title(title)
        if colorbar_flag:
            m.colorbar(cs, label='Wind Speed (m/s)', location='bottom',
                       pad="7%")
        if save is not None:
            plt.savefig(save)
        if return_flag:
            return m

    def make_coordinates_2d(self):
        if np.ndim(self.longitude) == 1:
            lon2d, lat2d = np.meshgrid(self.longitude, self.latitude)
            self.longitude = lon2d
            self.latitude = lat2d

##############################
# Independent Functions Follow
##############################


def get_tracks(data, indices=None, min_samples=10, verbose=False,
               filter=False, window=5):
    """
    Returns a list of CygnssTrack objects from a CYGNSS data or display object

    data = CygnssSingleSat, CygnssMultiSat, or CygnssL2WindDisplay object
    indices = Indices (2-element tuple) to use to limit the period of data
              shown (i.e., limit by time). Not usually necessary unless
              processing more than one day's worth of data.
    min_samples = Minimum allowable track size (number of specular points)
    verbose = Set to True for some text updates while running
    filter = Set to True to filter each track
    window = Window length of filter, in # of specular points. Must be odd.
    """
    trl = []
    dts = get_datetime(data)
    # For some reason range works but np.arange doesn't.
    for csat in range(8):
        if not hasattr(data, 'satellites'):
            if csat > 0:
                break
        if verbose:
            print('CYGNSS satellite', csat)
        for gsat in range(np.max(data.GpsID)+1):
            # This will isolate most tracks, improving later cluster analysis
            ds = CygnssTrack(data, datetimes=dts, indices=indices, gpsid=gsat,
                             sat=csat)
            if np.size(ds.lon) > 0:
                # Cluster analysis separates out any remaining grouped tracks
                X = list(zip(ds.lon, ds.lat))
                db = DBSCAN(min_samples=min_samples).fit(X)
                labels = db.labels_
                uniq = np.unique(labels)
                for element in uniq[uniq >= 0]:
                    # A bit clunky, but make a copy of the CygnssTrack object
                    # to help separate out remaining tracks in the scene
                    dsc = deepcopy(ds)
                    dsc.lon = ds.lon[labels == element]
                    dsc.lat = ds.lat[labels == element]
                    dsc.ws = ds.ws[labels == element]
                    dsc.tws = ds.tws[labels == element]
                    dsc.rcg = ds.lon[labels == element]
                    dsc.datetimes = ds.datetimes[labels == element]
                    dsc.sat = csat
                    dsc.prn = gsat
                    trl.append(dsc)
    if filter:
        for tr in trl:
            tr.filter_track(window=window)
    return trl


def get_datetime(data):
    if hasattr(data, 'satellites'):
        data = data.satellites[0]
    dts = []
    for i in np.arange(len(data.Year)):
        dti = dt.datetime(data.Year[i], data.Month[i], data.Day[i],
                          data.Hour[i], data.Minute[i], data.Second[i])
        tmplist = [dti for i in np.arange(15)]
        dts.append(tmplist)
    return np.array(dts)


def parse_fig_ax(fig, ax):
    """
    Parse matplotlib Figure and Axes objects, if provided, or just grab the
    current ones in memory.
    """
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()
    return fig, ax


def get_basemap(latrange=[-90, 90], lonrange=[-180, 180], resolution='l',
                area_thresh=1000):
    """
    Function to create a specifically formatted Basemap provided the input
    parameters.

    latrange = Latitude range of the plot (2-element tuple)
    lonrange = Longitude range of the plot (2-element tuple)
    resolution = Resolution of the Basemap
    area_thresh = Threshold (in km^**2) for displaying small features, such as
                  lakes/islands
    """
    lon_0 = np.mean(lonrange)
    lat_0 = np.mean(latrange)
    m = Basemap(projection='merc', lon_0=lon_0, lat_0=lat_0, lat_ts=lat_0,
                llcrnrlat=np.min(latrange), urcrnrlat=np.max(latrange),
                llcrnrlon=np.min(lonrange), urcrnrlon=np.max(lonrange),
                rsphere=6371200., resolution=resolution,
                area_thresh=area_thresh)
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    return m


def check_for_strings(var):
    """
    Given an input var, check to see if it is a string (scalar or array of
    strings), or something else.

    Output:
    0 = non-string, 1 = string scalar, 2 = string array
    """
    if np.size(var) == 1:
        if isinstance(var, string_types):
            return 1
        else:
            return 0
    else:
        for val in var:
            if not isinstance(val, string_types):
                return 0
        return 2
