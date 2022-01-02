The code held in this folder builds datasets that answer the questions for Technical Areas 1 and 2. The details are below.
-------------------------------------
To access the data:

Go to the "src" folder and open the file "postgresql_db_access.ipynb". Ensure that the sslrootcert, sslcert, and sslkey
variables are assigned the path to the associated keys in the "static" folder. Once done, open the "dataset_documentation.xls"
file and determine the dataset to load. Once selected, edit the "sSQL" string to query the correct database by name and 
run the code.

All data sources are filtered for the dates from 2018 Jan to 2020 Jan and for the region bounded by the lat/lon poly:
(24.78015, -83.00653)
(25.04137, -80.73785)
(27.12657, -80.16182)
(27.10291, -79.75860)
(25.51416, -79.86051)
(24.38124, -80.57390)
(24.19141, -82.07159)
(24.58690, -83.16162)
(24.78015, -83.00653)

-------------------------------------
To answer Technical Area 1: Cross-Validate the Open-Source Reef Databases 

Deliverable:  Students  will  provide  one  unified  data  source  with  fused  data  from each of the data sources and the 
scripts that were used for downloading and data fusion. 

The final deliverable that answers the challenge for Technical Area 1 is the wcmc_labels data set. It combines truth
data from wcmc, reefbase, bleachwatch, and allen coral atlas to give a comprehensive truth dataset with labels for 
coral, bleaching, and other classes.
--------------------------------------
To solve Technical Area 2: Time-Align and Geo-Align with Corresponding CALIPSO Data 
Deliverables: Students will provide scripts for downloading the necessary data for training and inference and all the 
specific training/inference data used to produce the final report.

The GOTECH team stored all relevant datasets in a postgresql database which can be accessed with code found in the Jupyter 
notebook at  "src/postgresql_db_access.ipynb". This notebook will utilize permanent keys from the "static" folder to access 
the database. In addition, the file "dataset_documentation.xls" defines each dataset in the database, provides background 
for its purpose, and defines the columns/units for each.

