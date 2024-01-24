This module enables the ingest, analysis, and plotting of Cyclone Global Navigation Satellite System (CYGNSS) on-orbit data as well as pre-launch CYGNSS End-to-End Simulator (E2ES) data.

Notable features include the ability to identify contiguous tracks of specular reflections associated with the same pair of CYGNSS and Global Positioning System (GPS) satellites. The winds along these tracks can then be filtered to reduce noise. Precipitation from the Global Precipitation Measurement (GPM) constellation can also be added.

Code example:
```
cyg = pygnss.orbit.read_cygnss_l2(files[0])
for sat in range(8):
    trl = pygnss.orbit.get_tracks(cyg, sat, verbose=True, eps=2.0)
    print('\nAdding IMERG to', len(trl), 'tracks')
    trl = pygnss.orbit.add_imerg(trl, ifiles, dt_imerg)
    print('Saving Files')
    pygnss.orbit.write_netcdfs(trl, tr_path + sdate + '/')
cyg.close()
```

<b>References</b>
<ul>Hoover, K. E., J. R. Mecikalski, T. J. Lang, X. Li, T. J. Castillo, and T. Chronis, 2018: Use of an End-to-End-Simulator to analyze CYGNSS. J. Atmos. Ocean. Technol., doi: 10.1175/JTECH-D-17-0036.1.</ul>
<ul>Ruf, C. S., Chew, C., Lang, T., Morris, M. G., Nave, K., Ridley, A., & Balasubramaniam, R. (2018). A New Paradigm in Earth Environmental Monitoring with the CYGNSS Small Satellite Constellation. Scientific reports, 8(1), 8782.</ul>
<ul>Lang, T.J. Comparing Winds Near Tropical Oceanic Precipitation Systems with and without Lightning. Remote Sens. 2020, 12, 3968. https://doi.org/10.3390/rs12233968</ul>
