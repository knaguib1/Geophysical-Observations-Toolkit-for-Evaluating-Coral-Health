FF_NN_50m_MaxAbs_Undersample_results.csv is the fused dataset with labels from our best performing model (Feed Forward NN with Allen Coral Atlas labels filtered to be withn 50m.)

html_visualiztion.ipynb will define functions and generate the html timeseries graphic. It references FF_NN_50m_MaxAbs_Undersample_results.csv for the data and will need the
filepath to be updated if this file structure is changed.

timeseries.html is the final graphic that run on your internet browser and display CALIPSO datapoints, their labels, and bleaching information for the state of Florida from
2006-2021.