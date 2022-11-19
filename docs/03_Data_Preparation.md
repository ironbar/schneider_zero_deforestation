# Data Preparation

<!--- --->

## Select Data

<!---Decide on the data to be used for analysis. Criteria include relevance to
the data mining goals, quality, and technical constraints such as limits on data
volume or data types. Note that data selection covers selection of attributes
(columns) as well as selection of records (rows) in a table.
List the data to be included/excluded and the reasons for these decisions.--->

All the available data will be used for this challenge.

## Clean Data

<!---Raise the data quality to the level required by the selected analysis techniques.
This may involve selection of clean subsets of the data, the insertion of suitable
defaults, or more ambitious techniques such as the estimation of missing data by
modeling. Describe what decisions and actions were taken to address the data
quality problems reported during the Verify Data Quality task of the Data
Understanding phase. Transformations of the data for cleaning purposes and the
possible impact on the analysis results should be considered. --->

There is no time to clean the data. Since the dataset is very small I will assume that the data is clean.

## Construct Data

<!---This task includes constructive data preparation operations such as the
production of derived attributes or entire new records, or transformed values
for existing attributes. --->

I will only use the images for training the model.

Since my plan is to use different pretrained models I will have to do different preprocessing for
each of them.
