import pandas as pd
import numpy as np
from scipy.stats import pearsonr, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(suppress=True, precision = 2)
nba = pd.read_csv('nba_games.csv')
nba.head()


nba_2010 = nba[nba.year_id == 2010]
nba_2014 = nba[nba.year_id == 2014]

knicks_pts = nba_2010[nba_2010.fran_id == 'Knicks']['pts']
nets_pts = nba_2010[nba_2010.fran_id == 'Nets']['pts']


diff_means_2010 = knicks_pts.mean() - nets_pts.mean()
diff_means_2010


plt.hist(knicks_pts, alpha = .8, density = True, label = 'Knicks')
plt.hist(nets_pts, alpha = .8, density = True, label = 'Nets')
#note that density is used for newer version of matplotlib
plt.legend()
plt.title("2010 Season")
plt.show()


knicks_pts_14 = nba_2014[nba_2014.fran_id == 'Knicks']['pts']
nets_pts_14 = nba_2014[nba_2014.fran_id == 'Nets']['pts']

diff_means_2014 = knicks_pts_14.mean() - nets_pts_14.mean()
print(diff_means_2014)

plt.hist(knicks_pts_14, alpha = .8, density = True, label = 'Knicks')
plt.hist(nets_pts_14, alpha = .8, density = True, label = 'Nets')
plt.legend()
plt.title("2014 Season")
plt.show()


sns.boxplot(data = nba_2010, x = 'fran_id', y = 'pts')
plt.show()


location_result_freq = pd.crosstab(nba_2010.game_result, nba_2010.game_location)
location_result_freq

location_result_proportions = location_result_freq/len(nba_2010)
location_result_proportions

chi2, pval, dof, expected = chi2_contingency(location_result_freq)
print(expected)
print(chi2)

point_diff_forecast_cov = np.cov(nba_2010.point_diff, nba_2010.forecast)
point_diff_forecast_cov

point_diff_forecast_corr = pearsonr(nba_2010.forecast, nba_2010.point_diff)
point_diff_forecast_corr


plt.clf() #to clear the previous plot
plt.scatter('forecast', 'point_diff', data=nba_2010)
plt.xlabel('Forecasted Win Prob.')
plt.ylabel('Point Differential')
plt.show()
