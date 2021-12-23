#%%
### Created by Jack Sovich and Divya Menon (jack.sovich@jmsmucker.com / divya.menon@jmsmucker.com)
### Check "NOTE"s throughout code for clarification

#NOTE: Ensure Camelot and PyPDF have been installed (pip install "camelot-py[base]" AND pip install PyPDF2)
# IMPORTS / USER VARIABLES: Run prior to other kernels in order to make sure variables are all assigned properly
# PDF AUTOMATED SCRAPING TOOL: Run this code for your specified PDF's in a specified directory to pull in linear space dataframe by POG
# BUILDING MAIN DATAFRAME: Run this code to pull Sales data and merge data frames for export to clustering script

# If Excel is available, data must include columns of each brand's linear space AND share of linear space for each POG
# Resulting dataframe will be merged with Main_df based on POG/dbkey

#%%
### IMPORTS
import PyPDF2
import camelot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.core.reshape.merge import merge
import glob
import os
import re

### USER VARIABLES
# Master Excel file with dimensions for each product (merged on UPC with data pulled from PDF's)
master_file = 'Y:\\Private\\Category Analytics Perm10\\Category Segmentation\\Data Science Segmentation Project (Intern Access)\\Documentation_Files\\MASTER - PB Item Dims.xlsx'
master_df_col = ['UPC', 'Brand', 'Width']
master_df = pd.read_excel(master_file, header=1, usecols=master_df_col,
                          dtype=str).convert_dtypes(convert_string=True)
width = "Width"
master_df[width] = master_df[width].str.extract('(\d+.\d+)').astype(float)
master_match_on = "UPC"

# PDF extractor tool settings for POG's
# NOTE: Change pixel numbers accordingly depending on results of camelot.plot
column_seperator = ['63,120,155,199,219,335,362,389,419,450,487,513']
path = 'Y:\\Private\\Category Analytics Perm10\\Category Segmentation\\Data Science Segmentation Project (Intern Access)\\Documentation_Files\\POGs\\'
# path = 'Y:\\Private\\Category Analytics Perm10\\Category Segmentation\\Data Science Segmentation Project (Intern Access)\\Documentation_Files\\test POGs\\'
linear_space_df = pd.DataFrame()
linear_space_measure = 'Space'
table_df_col = ['UPC#', 'Fac...', 'Long Description']
table_match_on = 'UPC#'
table_brand_col = "Brand"
facing = "Fac..."

# Main dataframe for information to be appended to (including POG, store number, sales, etc)
main_file = "Y:\\Private/Category Analytics Perm10\\Category Segmentation\\Data Science Segmentation Project (Intern Access)\\Documentation_Files\\Food Lion PB Store Rank by Brand_L104wks 6.12.21.xlsx"
header_num = 8
main_df_demographics = ['Geography', 'Store Number', 'Price Zone']
main_df_key = 'Store Number'
main_df_measure = 'Dollar Sales'
main_brand_col = 'Product'  # NOTE: Will be used in the case of a pivoted table

# POG additional info (includes POG and dbkey along with store number to merge with IRI data)
POG_additional_info = "Y:\\Private\\Category Analytics Perm10\\Category Segmentation\\Data Science Segmentation Project (Intern Access)\\Documentation_Files\\PBJ Store List by POG 5.28.2021.xlsx"
POG_info_col = ['Store Number', 'dbkey', 'Desc9', 'name']
POG_key = 'Store Number'
POG_info_df = pd.read_excel(
    POG_additional_info, header=0, usecols=POG_info_col).dropna()
# NOTE: Change column names for preference / clarity
POG_info_df.rename(columns={'Desc9': 'State', 'name': 'POG'}, inplace=True)

# Variables under investigation -- used to ensure data is merged appropriately
variables = [main_df_measure, linear_space_measure]
merge_on = 'dbkey'

### FUNCTIONS


def renameColumns(expression):
    for col in export_df.filter(regex=expression):
        brand = col.split(expression)[0]
        brand_match = col.split(brand)[1]
        brand = brand.replace(" ", "")

        # Edge cases: Should not affect script if above procedure is followed in column naming
        if brand.upper() == "SMUCKER" or brand.upper() == "SMUCKERS":
            brand = "SMUCKERS"
        if brand == 'PVL':
            brand = 'PRIVATELABEL'

        new_name = brand + " " + brand_match
        export_df.rename(columns={col: new_name}, inplace=True)


#%%
### PDF AUTOMATED SCRAPING TOOL
pdffiles = glob.glob(os.path.join(path, "*.pdf"))
print('Scraping PDFs... \n')
for file in pdffiles:
    print('Evaluating File: ' + file + '\n')

    pages = PyPDF2.PdfFileReader(open(file, 'rb')).getNumPages()
    columns = column_seperator
    columns *= pages
    # NOTE: Camelot can guess columns without the use of "columns" if that is preferred
    tables = camelot.read_pdf(
        file, pages='all', flavor='stream', columns=columns, split_text=True)

    table_df = pd.DataFrame()
    for table in range(0, len(tables)):
        table_df = pd.concat(objs=[table_df, tables[table].df], axis=0)

        # #NOTE: Visualization of column splitting, use cursor to find pixel number where table lines seperate values -- edit "columns" accordingly
        # #NOTE: Run in terminal and use cursor to select pixel demarcations line using x-coordinate (tablePlot plots each page of POG)
        # tablePlot = camelot.plot(tables[table], kind='text')
        # plt.show()

    # Cleaning table in preparation to merge with master_df with size dimensions
    table_df.replace('', np.nan, inplace=True)
    table_df.dropna(axis=0, inplace=True)
    # NOTE: Filtering out rows where header repeated for each new table / repeated UPCs from Camelot data pull
    table_df.drop_duplicates(inplace=True)
    new_header = table_df.iloc[0]
    table_df = table_df[1:]
    table_df.columns = new_header

    # table_df = table_df[table_df[table_df.columns[0]] != table_df.columns[0]]
    table_df = pd.DataFrame(table_df, columns=table_df_col,
                            dtype=str).convert_dtypes(convert_string=True)
    # NOTE: Change column name depending on data
    table_df["Facing"] = table_df[facing].astype(float)

    # Merging with Master_df in preparation to calculate linear space
    final_result = pd.merge(left=table_df, right=master_df, how='inner',
                            left_on=table_match_on, right_on=master_match_on)
    final_result['Linear Space'] = final_result["Facing"]*final_result["Width"]
    final_result['dbkey'] = file.split('.')[0].split(
        '\\')[-1].split('-')[-1]  # NOTE: Pulling POG ID number from .pdf name
    final_result['Brand'] = final_result[table_brand_col].str.upper()
    brands = final_result['Brand'].unique()

    # Adding new row of information to linear_space_df which will eventually be merged with Main_df
    new_row = pd.DataFrame(data=final_result.groupby('Brand', as_index=False).sum()[
                           'Linear Space'].tolist(), columns=[final_result['dbkey'][0]], index=brands+' Space').transpose()
    linear_space_df = linear_space_df.append(new_row)
print('--- Scraping Complete ---')

# Data table to be merged with main_df
linear_space_df = linear_space_df.reset_index().rename(columns={
    'index': merge_on})
linear_space_df.columns = linear_space_df.columns.get_level_values(
    0)  # NOTE: Reduces multi-level index to first level index
linear_space_df[merge_on] = linear_space_df[merge_on].astype(str).astype(int)

# appending to df share of space for each brand
linear_space_df['TOTAL '+linear_space_measure] = linear_space_df.filter(
    regex=linear_space_measure).sum(axis=1)
for col in linear_space_df.filter(regex=linear_space_measure):
    linear_space_df[col + ' Share'] = linear_space_df[col] / \
        linear_space_df['TOTAL '+linear_space_measure]
linear_space_df.drop(linear_space_df.filter(
    regex='TOTAL'), axis=1, inplace=True)

#%%
### BUILDING MAIN DATAFRAME
#NOTE: Clean up more, POG scraper DONE!

if len(pd.ExcelFile(main_file).sheet_names) > 1:
    # NOTE: Create a combined dataframe for the many tabs in your Excel
    print('Reading Legend Table...')

    for sheet in pd.ExcelFile(main_file).sheet_names:
        if "INDEX" in sheet.upper():
            print("Index Skipped")
        elif "TOTAL" in sheet.upper():
            main_df = pd.read_excel(
                main_file, sheet_name=sheet, header=header_num, usecols=main_df_demographics).dropna()
        else:
            temp_df = pd.read_excel(main_file, sheet_name=sheet, header=header_num, usecols=[
                                    main_df_key, main_df_measure])
            main_df = pd.merge(left=main_df, right=temp_df,
                               how='outer', on=main_df_key)
            sheet = sheet.upper() + ' '
            main_df.rename(columns={main_df_measure: (
                sheet+main_df_measure)}, inplace=True)
    main_df.dropna(axis=0, inplace=True)

elif len(pd.ExcelFile(main_file).sheet_names) == 1:
    print('Reading Pivoted Table...')
    main_df = pd.read_excel(main_file).dropna()
    main_df = main_df.pivot(index=main_df_demographics,
                            columns=main_brand_col, values=main_df_measure).reset_index()

    for col in main_df.drop(main_df_demographics, axis=1).columns:
        # NOTE: Use if in the format Product - BRAND
        brand = col.split(' - ')[1].replace(" ", "").upper()
        main_df.rename(
            columns={col: (brand+" "+main_df_measure)}, inplace=True)

else:
    print('ERROR: It appears there is something wrong with your file.')

# appending to df share of sales for each brand
main_df['TOTAL ' +
        main_df_measure] = main_df.filter(regex=main_df_measure).sum(axis=1)

# calculating the share of each brand by store
for col in main_df.filter(regex=main_df_measure):
    main_df[col + ' Share'] = main_df[col]/main_df['TOTAL '+main_df_measure]
main_df.drop(main_df.filter(regex='TOTAL'), axis=1, inplace=True)

# merging additional data from POG info with main_df
# merging main_df and linear_space_df for final export to clustering script
main_df = pd.merge(left=main_df, right=POG_info_df, how='inner',
                   left_on=main_df_key, right_on=POG_key)  # NOTE: Appending POG info to main_df
export_df = pd.merge(left=main_df, right=linear_space_df,
                     how='inner', on=merge_on)

# cleaning columns names and appending to final df the index for each brand
for expression in variables:
    renameColumns(expression)

# calculating index of each brand and appending to export_df
for col in export_df.filter(regex=(main_df_measure + " Share")):
    brand = col.split(" ")[0]
    export_df[brand + " Index"] = export_df[col] / \
        export_df[brand+" "+linear_space_measure+" Share"]*100

print("DONE")
