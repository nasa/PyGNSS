import numpy as np
import datetime as dt
from copy import deepcopy
import xarray
from sklearn.cluster import DBSCAN
import h5py
import scipy
import os

array_list = ['lon', 'lat', 'ws', 'rcg', 'ws_yslf_nbrcs',
              'ws_yslf_les', 'datetimes', 'sod']
scalar_list = ['antenna', 'prn', 'sat']


def read_cygnss_l2(fname):
    return xarray.open_dataset(fname)


def read_imerg(fname):
    return Imerg(fname)


def split_tracks_in_time(track, gap=60):
    """
    This function will split a CygnssTrack object into two separate tracks
    if there is a significant gap in time. Currently can split a track up to
    three times.

    Parameters
    ----------
    track : CygnssTrack object
        CYGNSS track that needs to be checked for breaks in time

    Other Parameters
    ----------------
    gap : int
        Number of seconds in a gap before a split is forced

    Returns
    -------
    track_list : list
        List of CygnssTrack objects broken up from original track
    """
    indices = np.where(np.diff(track.sod) > gap)[0]
    if len(indices) > 0:
        if len(indices) == 1:
            track1 = subset_track(deepcopy(track), 0, indices[0]+1)
            track2 = subset_track(deepcopy(track), indices[0]+1,
                                  len(track.sod))
            return [track1, track2]
        if len(indices) == 2:
            track1 = subset_track(deepcopy(track), 0, indices[0]+1)
            track2 = subset_track(deepcopy(track), indices[0]+1,
                                  indices[1]+1)
            track3 = subset_track(deepcopy(track), indices[1]+1,
                                  len(track.sod))
            return [track1, track2, track3]
        if len(indices) == 3:
            track1 = subset_track(deepcopy(track), 0, indices[0]+1)
            track2 = subset_track(deepcopy(track), indices[0]+1, indices[1]+1)
            track3 = subset_track(deepcopy(track), indices[1]+1, indices[2]+1)
            track4 = subset_track(deepcopy(track), indices[2]+1,
                                  len(track.sod))
            return [track1, track2, track3, track4]
        else:
            print('Found more than four tracks!')
            return 0


def subset_track(track, index1, index2):
    """
    This function subsets a CYGNSS track to only include data from a range
    defined by two indexes.

    Parameters
    ----------
    track : CygnssTrack object
        CygnssTrack to be subsetted
    index1 : int
        Starting index
    index2 : int
        Ending index

    Returns
    -------
    track : CygnssTrack object
        Subsetted CygnssTrack object
    """
    for arr in array_list:
        setattr(track, arr, getattr(track, arr)[index1:index2])
    for scalar in scalar_list:
        setattr(track, scalar, getattr(track, scalar))
    return track


def get_tracks(data, sat, min_samples=10, verbose=False,
               filter=False, window=5, eps=1, gap=60):
    """
    Returns a list of isolated CygnssTrack objects from a CYGNSS data object.

    Parameters
    ----------
    data : xarray.core.dataset.Dataset object
        CYGNSS data object as read by xarray.open_dataset
    sat : int
        CYGNSS satellite to be analyzed.

    Other Parameters
    ----------------
    min_samples : int
        Minimum allowable track size (number of specular points)
    verbose : bool
        True - Provide text updates while running

        False - Don't do this

    filter : bool
        True - Each track will receive a filter

        False - Don't do this

    window : int
        Window length of filter, in number of specular points. Must be odd.
    eps : scalar
        This is the eps keyword to be passed to DBSCAN. It is the max distance
        (in degrees lat/lon) between two tracks for them to be considered as
        part of the same track.
    gap : int
        Number of seconds in a track gap before a split is forced

    Returns
    -------
    trl : list
        List of isolated CygnssTrack objects
    """
    trl = []
    dts = get_datetime(data)
    # Currently only works for one satellite at a time due to resource issues
    if type(sat) is not int or sat < 1 or sat > 8:
        raise ValueError('sat must be integer between 1 and 8')
    else:
        csat = sat
    if verbose:
        print('CYGNSS satellite', csat)
        print('GPS code (max =', str(int(np.max(data.prn_code.data)))+'):',
              end=' ')
    for gsat in range(np.int16(np.max(data.prn_code.data)+1)):
        if verbose:
            print(gsat, end=' ')
        # This will isolate most tracks, improving later cluster analysis
        for ant in range(np.int16(np.max(data.antenna.data)+1)):
            ds = CygnssTrack(data, datetimes=dts, gpsid=gsat,
                             sat=csat, antenna=ant)
            if np.size(ds.lon) > 0:
                # Cluster analysis separates out additional grouped tracks
                # Only simplistic analysis of lat/lon gaps in degrees needed
                X = list(zip(ds.lon, ds.lat))
                db = DBSCAN(min_samples=min_samples, eps=eps).fit(X)
                labels = db.labels_
                uniq = np.unique(labels)
                for element in uniq[uniq >= 0]:
                    # A bit clunky, but make a copy of the CygnssTrack object
                    # to help separate out remaining tracks in the scene
                    dsc = deepcopy(ds)
                    for key in array_list:
                        setattr(dsc, key, getattr(ds, key)[labels == element])
                    dsc.lon[dsc.lon > 180] -= 360.0
                    for key in scalar_list:
                        setattr(dsc, key, np.array(getattr(ds, key))[0])
                    # Final separation by splitting about major time gaps
                    test = split_tracks_in_time(dsc, gap=gap)
                    if test is None:  # No time gap, append the original track
                        trl.append(dsc)
                    # Failsafe - Ignore difficult-to-split combined tracks
                    elif test == 0:
                        pass
                    else:  # Loop thru split-up tracks and append separately
                        for t in test:
                            trl.append(t)
                    del dsc
                del db, labels, uniq, X
            del ds  # This function is a resource hog, forcing some cleanup
    if filter:
        for tr in trl:
            tr.filter_track(window=window)
    return trl


def get_datetime(cyg):
    epoch_start = np.datetime64('1970-01-01T00:00:00Z')
    tdelta = np.timedelta64(1, 's')
    return np.array([dt.datetime.utcfromtimestamp((st - epoch_start) / tdelta)
                     for st in cyg.sample_time.data])


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
    rcg = RangeCorrectedGain array
    gps = GpsID array
    """

    def __init__(self, data, gpsid=None, gain=None, sat=None, antenna=None):
        """
        data = CygnssSingleSat, CygnssMultiSat, or CygnssL2WindDisplay object
        gpsid = Integer ID number for GPS satellite to examine
        gain = Threshold by range-corrected gain, values below will be masked
        bad = Value to compare against lat/lon to mask out missing data
        sat = CYGNSS satellite number (1-8)
        """
        # Set basic attributes based on input data object
        self.ws = data.wind_speed.data
        self.ws_yslf_nbrcs = data.yslf_nbrcs_wind_speed.data
        self.ws_yslf_les = data.yslf_les_wind_speed.data
        self.lon = data.lon.data
        self.lat = data.lat.data
        self.gps = np.int16(data.prn_code.data)
        self.antenna = np.int16(data.antenna.data)
        self.rcg = data.range_corr_gain.data
        self.cygnum = np.int16(data.spacecraft_num.data)

        # Set keyword-based attributes
        self.gpsid = gpsid
        self.gain = gain
        self.sat = sat
        self.ant_num = antenna

        # Now subsection the data
        self.get_good_data_mask()

    def get_good_data_mask(self):
        """
        Sets a mask used to limit the data plotted. Filtered out are data
        masked out by the GoodData mask (based on RangeCorrectedGain), missing
        lat/lon values, and bad data (ws < 0)
        """
        good1 = self.ws >= 0
        good2 = np.logical_and(np.isfinite(self.lon), np.isfinite(self.lat))
        if self.gpsid is not None and type(self.gpsid) is int:
            good2 = np.logical_and(good2, self.gps == self.gpsid)
        if self.gain is not None:
            if np.size(self.gain) == 2:
                cond = np.logical_and(self.rcg >= self.gain[0],
                                      self.rcg < self.gain[1])
                good2 = np.logical_and(good2, cond)
            else:
                good2 = np.logical_and(good2, self.rcg >= self.gain)
        if self.sat is not None and type(self.sat) is int:
            good2 = np.logical_and(good2, self.cygnum == self.sat)
        if self.ant_num is not None and type(self.sat) is int:
            good2 = np.logical_and(good2, self.antenna == self.ant_num)
        self.good = np.logical_and(good1, good2)


class CygnssTrack(object):

    """
    Class to facilitate extraction of a single track of specular points
    from a CygnssSingleSat, CygnssMultiSat, or CygnssL2WindDisplay object.

    Attributes
    ----------
    input = CygnssSubsection object
    ws = CYGNSS wind speeds
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
        self.ws_yslf_nbrcs = self.input.ws_yslf_nbrcs[self.input.good]
        self.ws_yslf_les = self.input.ws_yslf_les[self.input.good]
        self.lon = self.input.lon[self.input.good]
        self.lat = self.input.lat[self.input.good]
        self.rcg = self.input.rcg[self.input.good]
        self.antenna = self.input.antenna[self.input.good]
        self.prn = self.input.gps[self.input.good]
        self.sat = self.input.cygnum[self.input.good]
        if datetimes is None:
            dts = get_datetime(data)
        else:
            dts = datetimes
        self.datetimes = dts[self.input.good]
        sod = []
        for dt1 in self.datetimes:
            sod.append((dt1 - dt.datetime(
                self.datetimes[0].year, self.datetimes[0].month,
                self.datetimes[0].day)).total_seconds())
        self.sod = np.array(sod)

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


class Imerg(object):

    def __init__(self, filen):
        self.read_imerg(filen)

    def read_imerg(self, filen):
        imerg = h5py.File(filen, 'r')
        self.datetime = dt.datetime.strptime(os.path.basename(filen)[23:39],
                                             '%Y%m%d-S%H%M%S')
        self.precip = np.ma.masked_where(
            np.transpose(imerg['Grid']['precipitationCal']) <= 0,
            np.transpose(imerg['Grid']['precipitationCal']))
        self.lon = np.array(imerg['Grid']['lon'])
        self.lat = np.array(imerg['Grid']['lat'])
        self.filename = os.path.basename(filen)
        imerg.close()

    def downsample(self):
        filled_precip = self.precip.filled(fill_value=0.0)
        dummy = scipy.ndimage.interpolation.zoom(filled_precip, 0.5)
        self.coarse_precip = np.ma.masked_where(dummy <= 0, dummy)
        self.coarse_lon = self.lon[::2]
        self.coarse_lat = self.lat[::2]


def add_imerg(trl, ifiles, dt_imerg):
    for ii in range(len(trl)):
        check_dt = trl[ii].datetimes[len(trl[ii].sod)//2]
        # diff = np.abs(check_dt - dt_imerg)
        index = np.where(dt_imerg <= check_dt)[0][-1]  # np.argmin(diff)
        imerg = Imerg(ifiles[index])
        if ii % 50 == 0:
            print(ii, end=' ')
        precip = []
        for j in range(len(trl[ii].lon)):
            ilon = int(np.round((trl[ii].lon[j] - imerg.lon[0]) / 0.10))
            ilat = int(np.round((trl[ii].lat[j] - imerg.lat[0]) / 0.10))
            precip.append(imerg.precip[ilat, ilon])
        precip = np.array(precip)
        precip[~np.isfinite(precip)] = 0.0
        setattr(trl[ii], 'precip', precip)
        setattr(trl[ii], 'imerg', os.path.basename(ifiles[index]))
    print()
    return trl


def write_netcdfs(trl, path):
    for i, track in enumerate(trl):
        fname = 'track_' + str(track.sat).zfill(2) + '_' + \
            str(track.prn).zfill(2) + \
            '_' + str(track.antenna).zfill(2) + '_' + str(i).zfill(4) + \
            track.datetimes[0].strftime('_%Y%m%d_s%H%M%S_') + \
            track.datetimes[-1].strftime('e%H%M%S.nc')
        ds = xarray.Dataset(
            {'ws': (['nt'], track.ws),
             'ws_yslf_nbrcs': (['nt'], track.ws_yslf_nbrcs),
             'ws_yslf_les': (['nt'], track.ws_yslf_les),
             'lat': (['nt'], track.lat),
             'lon': (['nt'], track.lon),
             'datetimes': (['nt'], track.datetimes),
             'rcg': (['nt'], track.rcg),
             'precip': (['nt'], track.precip),
             'sod': (['nt'], track.sod)},
            coords={'nt': (['nt'], np.arange(len(track.ws)))},
            attrs={'imerg': track.imerg})
        ds.to_netcdf(path + fname, format='NETCDF3_CLASSIC')
        ds.close()
        del(ds)
