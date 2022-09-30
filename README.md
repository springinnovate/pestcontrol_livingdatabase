# pestcontrol_livingdatabase
For the USDA FACT project!

ee_sampler originally came from https://github.com/therealspring/earth_engine_scripts

lingling_scripts are also forked on https://github.com/lingling-liu/pestcontrol_livingdatabase

Process for building table:

* Inputs
    * CSV table of points with associated years
    * list of landcover datasets to build additional masks

* Outputs
    * CSV table, by point, of aggregated values of MODIS parameters:
        * MODIS phenology parameters: Greenup_1, MidGreenup_1, Peak_1, Maturity_1, MidGreendown_1,
          Senescence_1, Dormancy_1, EVI_Minimum_1, EVI_Amplitude_1, EVI_Area_1, QA_Overall_1.
        * MODIS parameters masked by landcover type for each of the landcover datasets passed as input
          (currently these landcover types are *cultivated* and *natural*, but dataset configuration
          files can easily take more

Action:
1) Build GEE MODIS multi-band raster for each of the parameters described above
2) For each of the bands in #1
    2a) For each of the datasets listed in inputs
        2aa) each of the landcover types listed above, create a new band that is masked by the
             landcover type from the given dataset
3) Load points from input CSV table
4) Use points from #3 to sample ALL the bands created by #2
5) For each point sample in #4 write a single row to a table with headers that correspond to all the
   bands created in #1 and #2.
