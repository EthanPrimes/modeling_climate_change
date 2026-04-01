from time import perf_counter
import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
import sklearn as sk
from scipy.special import softmax
import matplotlib


# Global Temperature
# gt_data = pd.read_csv("datasets/global_temp/GlobalTemperatures.csv")
# gt_data["year"] = gt_data["dt"].str[:4].astype(int)

# gt_data_filtered = []
# for year in np.arange(1750, 2016):
#     gt_year_data = gt_data[gt_data["year"] == year]
#     land_temp = gt_year_data["LandAverageTemperature"].to_numpy()
#     ocean_temp = gt_year_data["LandAndOceanAverageTemperature"].to_numpy()
#     yearly_data = np.array((land_temp, ocean_temp)) # append land and ocean monthly temperatures together
#     gt_data_filtered.append(yearly_data)
#     # print(yearly_data.shape)

# gt_data_filtered = np.dstack(gt_data_filtered)

def get_year(year: int):
    """
    Converts a year to an index offset from 1750.

    Args:
        year (int): A year between 1750 and 2015 inclusive.

    Returns:
        int: The index of the year (e.g. 1750 → 0, 1751 → 1).
    """
    return year - 1750

# np.save("datasets/gt_data_filtered.npy", gt_data_filtered)
gt_data_filtered = np.load("datasets/gt_data_filtered.npy")


# # eg: gt_data_filtered[0, :, get_year(2015)]

# # [ (0, 1) # {0: land temperatures, 1: ocean temperatures}
# #   (0-11) # month of the year
# #   (0-266) # year 1750 - 2015
# #  ]

gt_data_noNaNs = gt_data_filtered[:, :, 100:]

def get_year_noNaNs(year: int):
    """
    Converts a year to an index offset from 1850.

    Note: This is for the noNaNs data since the first 100 years have NaNs

    Args:
        year (int): A year between 1850 and 2015 inclusive.

    Returns:
        int: The index of the year (e.g. 1850 → 0, 1851 → 1).
    """
    return year - 1850

gt_data_land_temps = gt_data_filtered[0, :, 3:]

# # eg: gt_data_land_temps[:, get_year_land(1800)] # returns the land average temperatures for every month in the year 1800

# # [ (0-11) # month of the year
# #   (0-263) # year 1753 - 2015
# #  ]

def get_year_land(year: int):
    """
    Converts a year to an index offset from 1753.

    Note: This is for the land temps data since the first 3 years had NaNs for land

    Args:
        year (int): A year between 1753 and 2015 inclusive.

    Returns:
        int: The index of the year (e.g. 1753 → 0, 1754 → 1).
    """
    return year - 1753

# Average Temperature
avtemp_df = pd.read_csv("datasets/Average_Temperature_1900_2023.csv", index_col=0)
avtemp_df = avtemp_df.astype(float)  # Setting the temperature datatypes to floats
avtemp_df["temp"] = avtemp_df["Average_Fahrenheit_Temperature"]
avtemp_df = avtemp_df.drop(columns=["Average_Fahrenheit_Temperature"])  # Renaming the column

# greenhouse gases dataframe by country. Each row has the greenhouse gas emissions for a country by year
co2ghg_df = pd.read_csv("datasets/co2_ghg.csv")


# ========= I did a lot of work to average out when a country recorded 0... I think the 0s are legit and this dataset seems really clean actually =========

# # add the mean value of the countries that didn't record 0 to those that did record 0 for each year
# co2_nocountries = co2ghg_df.drop(columns="Country")
# year_means = co2_nocountries.replace(0, pd.NA).mean()
# co2ghg_avg0s_df = co2_nocountries.mask(co2_nocountries == 0, year_means, axis="columns")
# co2ghg_avg0s_df.insert(0, "Country", co2ghg_df["Country"])


# # co2ghg_df is the df unaltered
# # co2ghg_avg0s_df is the df with yearly means added to 0 entries
# country_to_filter = "Yemen"
# afghan_co2data = co2ghg_df[co2ghg_df["Country"] == country_to_filter].to_numpy()[0][1:]
# afghan_co2data_altered = co2ghg_avg0s_df[co2ghg_avg0s_df["Country"] == country_to_filter].to_numpy()[0][1:]

# plt.plot(np.arange(len(afghan_co2data)) + 1751, afghan_co2data, label="unaltered")
# plt.plot(np.arange(len(afghan_co2data)) + 1751, afghan_co2data_altered, label="avg value added")
# plt.xlabel("Year")
# plt.ylabel("Greenhouse Gas Emission (tonnes)")
# plt.title(f"Averaging comparison {country_to_filter}")
# plt.legend()
# plt.show()

world_ghg_df = co2ghg_df[co2ghg_df["Country"] == "World"].drop(columns="Country")
worldghg_numpy = world_ghg_df.to_numpy()[0]

# Sea Ice Levels
seaice_df = pd.read_csv("datasets/seaice.csv")
seaice_df.columns = seaice_df.columns.str.strip()  # Stripping whitespace from column names
seaice_df["date"] = pd.to_datetime(seaice_df[["Year", "Month", "Day"]])  # Making a "date" column
seaice_df = seaice_df.drop(columns=["Month", "Day", "Source Data"])
seaice_df = seaice_df.set_index("date")  # Setting the date as the index

# Computing the averages for each year
seaice_north_yearly_df = seaice_df[seaice_df["hemisphere"] == "north"].groupby("Year")["Extent"].agg(["mean", "std"])  # Computing the mean and std
seaice_north_yearly_df = seaice_north_yearly_df.drop(index=[1978, 2019])  # Don't have data for the whole of each year; changes std value

seaice_south_yearly_df = seaice_df[seaice_df["hemisphere"] == "south"].groupby("Year")["Extent"].agg(["mean", "std"])
seaice_south_yearly_df = seaice_south_yearly_df.drop(index=[1978, 2019])

print(avtemp_df)

X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(avtemp_df, avtemp_df['temp'])

def mixture_model_normal(x):
    # Define variables
    V = np.array([[-20,0],[10,0]])
    B = np.array([[1,3],[4,-1]])
    sig = np.array([.1,.2])
    
    # Draw z
    xbar = np.array([1, x])
    val = softmax(V.T @ xbar)
    z = np.random.choice(val.shape[0], 1, p=val).item()
    
    # Draw y
    meanval = xbar @ (B[z, :])
    
    y = np.random.normal(meanval, sig[z], 1).item()
    
    return y

xt = X_train.index.to_numpy().flatten()

yt = y_train.to_numpy().flatten()
x_full = np.vstack([xt, yt]).T

print(x_full)

# Scatter your results
plt.scatter(xt, yt)

model = sk.mixture.GaussianMixture(2).fit(x_full)
# out = model.predict(X_test.index.to_numpy().reshape(-1, 1))
# plt.scatter(X_test.index.to_numpy().reshape(-1, 1), y_test.to_numpy(), c=out)

# print(model.means_)
# covs = model.covariances_.flatten()
# Plot the line segment y = x^TB_1 and y = x^TB_2
ts = np.linspace(1900, 2020, 10)
# plt.plot(ts, (ts - model.means_[0]) / covs[0])
# plt.plot(ts, (ts - model.means_[1]) / covs[1])



# display predicted scores by the model as a contour plot
x = np.linspace(1900, 2040)
y = np.linspace(45, 60)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -model.score_samples(XX)
Z = Z.reshape(X.shape)


CS = plt.contour(
    X, Y, Z, norm=matplotlib.colors.LogNorm(vmin=.001, vmax=1000.0), levels=np.logspace(-3, 3, 100)
)
CB = plt.colorbar(CS, shrink=0.8, extend="both")

plt.title("Negative log-likelihood predicted by a GMM")
plt.axis("tight")
plt.show()

lbls = model.predict(xt)

a=5
plt.scatter(xt, yt, c=lbls)

grp1 = xt[lbls]
grp0 = xt[~lbls]
model1 = sk.linear_model.LinearRegression()
model2 = sk.linear_model.LinearRegression()
model1.fit(grp1, yt[lbls])
model2.fit(grp0, yt[~lbls])
yout = np.where(model.predict(x), model1.predict(x), model2.predict(x))
plt.plot(x, yout)
plt.show()