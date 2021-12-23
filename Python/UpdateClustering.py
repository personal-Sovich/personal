### Segmentation Opportunity Calculator Python Script ###
# Divya Menon (divya.menon@jsmucker.com) and Jack Sovich (jack.sovich@jmsmucker.com)
# Working Final Version as of 8/03/2021

# Takes a transformed data frame from script titled ExtractPOGTables.py
# Note: you must make all manipulations for extracting your data in ExtractPOGTables.py before running this script
# See notes in code or documentation file to follow logic in the script

#%%
### IMPORTING and FUNCTIONS ###
from ExtractPOGTables import export_df
import tkinter
from seaborn import palettes
import matplotlib.pyplot as plt
from numpy import exp, i0
from numpy.core.numeric import full
from scipy.sparse import data
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
from matplotlib import rcParams
from functools import reduce
import re
from sklearn.decomposition import PCA
from matplotlib.pyplot import axis, figure, plot
from kneed import KneeLocator
import numpy as np
import warnings
import matplotlib.colors as mcolors
from tkinter import *
from tkinter import filedialog
from math import pi
warnings.filterwarnings("ignore")

# pre-process data using minmax scaler before doing KMeans/PCA


def scaleData(data):
    mms = MinMaxScaler()
    mms.fit(data)
    return mms.transform(data)
# plot the elbow graph and automate how many clusters to use in analysis


def plotSSE(data):
    inertia_vals = []
    kmax = 25
    range_n_clusters = list(range(1, kmax+1))
    # data = scaleData(data)
    for i in range_n_clusters:
        kmeans = KMeans(n_clusters=i)
        kmeans = kmeans.fit(data)
        inertia_vals.append(kmeans.inertia_)
    # using kneed package to get k
    k1 = KneeLocator(range(1, len(inertia_vals)+1), inertia_vals,
                     curve='convex', direction="decreasing")
    print('Clustering using ' + str(k1.knee) + ' clusters.')
    global k_val
    k_val = k1.knee
    return k1.knee

# do kmeans clustering on the scaled/transformed data, returns a df w/ cluster labels and a list of the clusters


def clusterData(df):
    # non_numeric_cols = df.select_dtypes(exclude='number')
    # new_df = df.drop(non_numeric_cols, axis=1)
    k_val = plotSSE(df)
    kmeans = KMeans(n_clusters=k_val)
    clusters = kmeans.fit_predict(df)
    # df['Cluster'] = clusters
    score = silhouette_score(df, clusters)
    print('For ' + str(k_val) + ' clusters: silhouette score is ' + str(score))
    return df, clusters, kmeans.cluster_centers_, k_val

# creates a dictionary of all the seperate clusters after the kmeans analysis


def filterByCluster(data, k):
    df_dict = {}
    clust_nums = list(range(0, k))
    for i in clust_nums:
        df = data.loc[data['Cluster'] == i]
        df_dict[i] = df
    return df_dict

# use PCA prior to clustering for better results


def pcaVariance(data_trans):
    # data_trans = scaleData(data)
    pca = PCA()
    pca.fit(data_trans)
    ratio_list = pca.explained_variance_ratio_.cumsum()*100
    plt.plot(range(1, len(ratio_list)+1), ratio_list)
    plt.show()
    # n_components = int(input('How many principle components do you want to track? \
    #     (The rule of thumb is preserve 80% of the variance) --> '))
    n_components = 2
    return n_components


def reduceDimensions(df_main):
    # non_numeric_cols = df_main.select_dtypes(exclude='number')
    # data_trans = df_main.drop(non_numeric_cols, axis=1)
    data_trans = df_main
    pca = PCA(n_components=pcaVariance(data_trans))
    pca.fit(data_trans)
    x_reduced_df = pd.DataFrame(pca.transform(data_trans))
    return x_reduced_df

# takes kmeans results df and plot dimension reduction


def pcaClusterPlot(reduced_df):
    sns.scatterplot(x=0, y=1, data=reduced_df, hue='Cluster', palette='Set2').set_title(
        'Clustering results (with reduced dimensions)')
    plt.show()

# function to take in dictionary of all the cluster subsets and write to an excel sheet


def writeToExcel(dict):
    writer = pd.ExcelWriter('Y:\\Private\\Category Analytics Perm10\\Category Segmentation\\Data Science Segmentation Project (Intern Access)\\PCA_Store_Clusters_edit.xlsx',
                            engine='xlsxwriter')
    for key, value in dict.items():
        value.to_excel(writer, index=False, sheet_name='Cluster ' + str(key))
    writer.save()

# function used to select Excel file using GUI


def getExcel():
    # ask user to select file from the interactive file explorer
    filepath = filedialog.askopenfilename(initialdir='Y:/Private/Category Analytics Perm10/Category Segmentation',
                                          title='Select a File', filetypes=[('excel files', '*.xlsx')])
    global df_sheets
    df_sheets = pd.read_excel(filepath, sheet_name=None, skiprows=[1])
# return a df of overindexing stores for a specific brand


def getOverIndexedStores(df, brand):
    stores = df[df[brand] >= 120]
    if len(stores) == 0:
        return None
    return stores
# return a df of underindexing stores for a specific brand


def getUnderIndexedStores(df, brand):
    stores = df[df[brand] <= 80]
    if len(stores) == 0:
        return None
    return stores
# function to extract the brand name from a column (returns a string)


def getBrandName(name):
    brand = re.findall(r'\b[A-Z][A-Z]+\b', name)
    if len(brand) > 1:
        brand = brand[0] + ' ' + brand[1]
        return brand
    else:
        brand = brand[0]
        return brand
# method to calculate the wasted space from underindexing brands


def calculatedRec(df):
    for index, row in df.iterrows():
        index_dict = row.filter(regex='Index').to_dict()
        for key, value in index_dict.items():
            if value <= 80 and ('SMUCKERS' not in key) and (priority_brand not in key):
                brand = getBrandName(key)
                new_space_share = row[brand + ' Dollar Sales Share']
                current_space_share = row[brand + ' Space Share']
                df.loc[index, brand +
                       ' Rec Space Share'] = current_space_share - new_space_share

    df.fillna(0, inplace=True)
    for col in df.filter(regex='Rec').columns:
        brand_name = getBrandName(col)
        # df[brand_name + ' Wasted Space'] = df['Total Space'] * df[col]
        df[brand_name + ' Wasted Space'] = df[brand_name+' Space']
    return df


#%%
### CREATING  GUI for entering specific analysis information ###

# Note: the only thing that needs to be manually typed is the priority_brand variable below
# change the priority_brand variable to a brand to always calculate the recommendation for (when overindexed)
priority_brand = 'JIF'

# the GUI will prompt user to select a folder to export all the excel files to
root = Tk()
root.withdraw()
export_path = filedialog.askdirectory()  # ask user to select folder
export_path = export_path + '/'


#%%

### CLUSTERING ANALYSIS ###

# in this specific analysis we used PCA without scaling, but function scaleData() can be used to scale data before
main_df = export_df

# Approach for this analysis: use PCA on dollar share df and plotting clustering after kmeans
dollar_share_df = main_df.filter(regex='Sales Share')
x_reduced_df = reduceDimensions(dollar_share_df)
x_reduced_df, cluster_labels, centroids, k_val = clusterData(x_reduced_df)
x_reduced_df['Cluster'] = cluster_labels
pcaClusterPlot(x_reduced_df)
main_df['Cluster'] = cluster_labels

# Case 1: raw share data clustering
# dollar_share_df = export_df.filter(regex='Sales Share')
# clustered_df, cluster_labels, centroids, k_val = clusterData(dollar_share_df)

# Case 2: share data w/ scaling
# dollar_share_df = scaleData(export_df.filter(regex='Sales Share'))
# clustered_df, cluster_labels, centroids, k_val = clusterData(dollar_share_df)

# Case 3: PCA w/ scaling
# dollar_share_df = scaleData(export_df.filter(regex='Sales Share'))
# x_reduced_df = reduceDimensions(dollar_share_df)
# x_reduced_df, cluster_labels, centroids, k_val = clusterData(x_reduced_df)
# x_reduced_df['Cluster'] = cluster_labels
# pcaClusterPlot(x_reduced_df)

# Calculate dollars per linear inch for each brand
for i in main_df.filter(regex='Space Share').columns:
    brand_name = getBrandName(i)
    main_df[brand_name + ' Dollars per Linear Inch'] = main_df[brand_name +
                                                               ' Dollar Sales'] / main_df[brand_name + ' Space']

# EXPORTING FILE 1: all data (including cluster labels)
main_df.to_excel(
    export_path + 'Main - Food Lion PB Store Rank.xlsx', index=False)

#%%
### Opportunity Calculator Work ###

# Note: See documentation for details dollar opportunity calculation

# adding the total sales column onto main_df so that it can calculate recommendation
sales_cols = main_df.filter(regex='Sales').columns
raw_sales_cols = []
for col in sales_cols:
    if 'Share' not in col:
        raw_sales_cols.append(col)
main_df['Total Sales'] = main_df[raw_sales_cols].sum(axis=1)

# adding the total space column onto main_df so that it can calculate recommendation
space_cols = main_df.filter(regex='Space').columns
raw_space_cols = []
for col in space_cols:
    if 'Share' not in col:
        raw_space_cols.append(col)
main_df['Total Space'] = main_df[raw_space_cols].sum(axis=1)

# dollar opportunity = (Recommended Brand's $ per linear inch) * (sum of the linear space gained)

# apply recommendation to stores where prioritized brand is overindexing
# recommended store will always be JIF when it is overindexing (120 or above)

df1 = getOverIndexedStores(main_df, priority_brand+' Index')
res1 = calculatedRec(df1)
res1['Recommended Brand'] = priority_brand
res1['Recommended Space Allocation'] = res1.filter(regex='Wasted').sum(axis=1)

# Calculating dollar opportunity
# = (dollars per linear inch of the recommended brand) * (sum of the wasted space)
# res1['Dollar Opportunity'] = res1['JIF Dollars per Linear Inch'] * res1['Sum of Linear Space']
res1['Dollar Opportunity'] = res1[priority_brand +
                                  ' Dollars per Linear Inch'] * res1['Recommended Space Allocation']

# apply to stores where priority brand is not overindexing
# recommended space will be a different overindexing brand
df2 = main_df[main_df[priority_brand+' Index'] < 120]
res2 = calculatedRec(df2)
res2['Recommended Brand'] = None
index_cols = res2.filter(regex='Index').columns
for index, row in res2.iterrows():
    max_index = {}
    for i in index_cols:
        max_index[i] = row[i]
    max_brand = getBrandName(max(max_index, key=max_index.get))
    res2.loc[index, 'Recommended Brand'] = max_brand
# sum the linear space lost (wasted space) from each brand
res2['Recommended Space Allocation'] = res2.filter(regex='Wasted').sum(axis=1)
# Calculate dollar opportunity
# = (dollars per linear inch of the recommended brand) * (sum of the wasted space)
res2['Dollar Opportunity'] = None
for index, row in res2.iterrows():
    rec_brand = row['Recommended Brand']
    res2.loc[index, 'Dollar Opportunity'] = row[rec_brand +
                                                ' Dollars per Linear Inch'] * row['Recommended Space Allocation']

# Combine dfs together
export_df = pd.concat([res1, res2])

# Export_df includes all information from the analysis with the extra opportunity info
export_df['Dollar Opportunity'] = export_df['Dollar Opportunity'].astype(
    float).round(2)

# EXPORTING FILE 2: for store level recommendation tab on dashboard
store_level_df = export_df.filter(
    regex='Geography|Number|Rec|Wasted|POG|Estimated|Dollars|Opportunity')
store_level_df.to_excel(
    export_path + 'Store Level Recommendation.xlsx', index=False)

# EXPORTING FILE 3: for radar chart (unpivoted table)
to_unpivot_df = export_df.filter(
    regex='Geography|State|Cluster|Index|POG|Number')
unpivoted_df = pd.melt(to_unpivot_df, id_vars=['Store Number', 'Geography', 'State', 'POG', 'Cluster'],
                       value_vars=to_unpivot_df.filter(
                           regex='Index').columns.tolist(),
                       value_name='Index', var_name='Brand')

# Calculations for ploting the polar coordinates for the radar chart
# R = index
# Theta = (rank_of_brand/max_rank_brand) * 2 * PI
# rad_x = R * cos(theta)
# rad_y = R * sin(theta)
unique_vals = unpivoted_df['Brand'].unique().tolist()
max_rank = len(unique_vals)
ranks = unpivoted_df.groupby('Brand', as_index=False)['Index'].mean()
ranks['Rank'] = range(1, max_rank+1)
unpivoted_df = pd.merge(ranks, unpivoted_df, on='Brand')
unpivoted_df.rename(columns={'Index_y': 'R'}, inplace=True)
unpivoted_df['Theta'] = (unpivoted_df['Rank'] / max_rank) * 2 * pi
unpivoted_df['rad_x'] = unpivoted_df['R'] * np.cos(unpivoted_df['Theta'])
unpivoted_df['rad_y'] = unpivoted_df['R'] * np.sin(unpivoted_df['Theta'])
unpivoted_df['rad_x'] = unpivoted_df['rad_x'].round(2)
unpivoted_df['rad_y'] = unpivoted_df['rad_y'].round(2)
unpivoted_df.drop(['Rank', 'Index_x'], axis=1, inplace=True)

unpivoted_df.to_excel(export_path + 'Radar Plot Table.xlsx', index=False)

# EXPORTING FILE 4: for POG level recommendation tab on dashboard
new_space_cols = export_df.filter(regex='Wasted').columns.tolist()
# rec_df = export_df.groupby(['POG' ,'Recommended Brand'], as_index=False).agg({'Recommended Space Allocation': 'mean', new_space_cols[0]:'mean', new_space_cols[1]:'mean', new_space_cols[2]:'mean' })

rec_df = export_df.groupby(['POG', 'Recommended Brand'], as_index=False).agg(
    {'Recommended Space Allocation': 'mean', new_space_cols[0]: 'mean', new_space_cols[1]: 'mean'})

lst = []
for index, row in rec_df.iterrows():
    rec_brand = row['Recommended Brand']
    pog = row['POG']
    brand_space_dollars = str(rec_brand) + ' Dollars per Linear Inch'
    avg = export_df.loc[(export_df['POG'] == pog) & (
        export_df['Recommended Brand'] == rec_brand), [brand_space_dollars]].mean()
    val = avg.iloc[0]
    lst.append(val)
rec_df['Avg Dollars per Linear Inch'] = lst
rec_df.rename(columns={
              'Recommended Space Allocation': 'Avg Recommended Wasted Space'}, inplace=True)
rec_df['Avg POG Dollar Opportunity'] = rec_df['Avg Recommended Wasted Space'] * \
    rec_df['Avg Dollars per Linear Inch']

rec_df.to_excel(export_path + 'POG Level Recommendation.xlsx', index=False)

#%%
