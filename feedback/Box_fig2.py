import os
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ================================================================
# GLOBAL STYLE SETTINGS (FIXED & NOT OVERRIDDEN BY SEABORN)
# ================================================================
plt.switch_backend('agg')

plt.rcParams.update({
    'font.size': 5,
    'font.family': 'sans-serif',
    'axes.titlesize': 6,
    'axes.labelsize': 6,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 5,
    'figure.titlesize': 6,
    'axes.grid': True,
})

mao_scaling =  0.7
# Do NOT let seaborn scale fonts:
sns.set_theme(context='paper', font_scale=1.0)

# Colorblind-friendly palette
palette_cb = {
    'ERA5':   '#117733',
    'MERRA2': '#332288',
    'OBS':    '#DDCC77',
    'COMBO':  '#CC6677'
}
hue_order_cb = ['MERRA2', 'ERA5', 'COMBO', 'OBS']


# ================================================================
# (Functions unchanged from your previous script)
# ================================================================
def load_obs_dataset(obspath, filename, offset=0):
    ds = xr.open_dataset(os.path.join(obspath, filename))
    d = ds.sel(Variables=0).to_array().squeeze().to_dataset(name='campaign')
    d['Wstd'] = ds.sel(Variables=2).to_array().squeeze()
    d['temp'] = ds.sel(Variables=3).to_array().squeeze()
    d = d.assign_coords(Time=d.Time).where(d.campaign > 0, drop=True).drop_vars('variable')
    d['campaign'] += offset
    return d


def process_model(exp, rean, expspath, dobs, campaign_id):
    ds = xr.open_dataset(f"{expspath}{exp}_Wstd_{rean}.nc4")
    ds = ds.where((ds.lev > 10) & (ds.lev < 600), drop=True)
    ds['T'] = ds['T'] - 273.0

    tobs = dobs.where(dobs.campaign == campaign_id, drop=True)
    tmin = tobs.temp.mean() - 2 * tobs.temp.std()
    tmax = tobs.temp.mean() + 2 * tobs.temp.std()

    W = ds['Wstd'].where((ds['Wstd'] > 0.001) & (ds['Wstd'] < 10))
    W = W.where((ds['T'] > tmin) & (ds['T'] < tmax)).dropna(dim='lev', how='all')
    W['campaign'] = W * 0 + campaign_id
    return W


def get_obs(site, label):
    path = f"/discover/nobackup/dbarahon/DATA/W_ASRsites/all/resampled2/to_test/Wstd_asr_resampled_stdev30min_72lv_{site}.nc"
    obs = xr.open_mfdataset(path, parallel=True).where(lambda d: d != -9999).where(lambda d: d < 10)

    if site == 'manus':
        obs = obs.where((obs["time.year"] < 2005) | (obs["time.year"] > 2007), drop=True)
    if site == 'twp':
        obs = obs.where(obs["time.year"] < 2014, drop=True)
    if site == 'mao':
        obs = xr.concat([
            obs.sel(time=slice('2014-02-01', '2015-01-10')),
            obs.sel(time=slice('2015-04-01', '2016-02-01'))
        ], dim='time')
        obs = obs*mao_scaling
    if site == 'ena':
        obs = obs.where((obs["time.year"] < 2017) | (obs["time.year"] >= 2018), drop=True)

    obs = obs.where(obs > 0.001)
    obs_std = obs.groupby('time.month').map(lambda x: abs((x - x.mean()) / x.std()))
    obs = obs.where(obs_std < 2.0)
    obs = obs.dropna('time', how='all', thresh=2)
    obs = obs.rename({'height': 'lev', 'W_asr_std': 'Wstd'}).stack(s=('time', 'lev'))
    obs = obs.assign_coords(source=('s', np.full(obs.sizes['s'], 'OBS')),
                             site_label=('s', np.full(obs.sizes['s'], label)))

    return obs[['Wstd', 'source', 'site_label']].to_dataframe().reset_index(drop=True)


def get_model(site, label, name, paths):
    p = os.path.join(paths[name], f"{name}_{site}.nc")
    if not os.path.exists(p):
        return None   
    	
    ds = xr.open_dataset(p).stack(s=('time', 'lev'))
    if site == "mao":
    	ds = ds*mao_scaling
    ds = ds.assign_coords(source=('s', np.full(ds.sizes['s'], name)),
                          site_label=('s', np.full(ds.sizes['s'], label)))
    return ds[['Wstd', 'source', 'site_label']].to_dataframe().reset_index(drop=True)


def get_avg_model(site, label, paths):
    e = os.path.join(paths["ERA5"], f"ERA5_{site}.nc")
    m = os.path.join(paths["MERRA2"], f"MERRA2_{site}.nc")
    if not (os.path.exists(e) and os.path.exists(m)):
        return None

    W1 = xr.open_dataset(e)['Wstd']
    W2 = xr.open_dataset(m)['Wstd']

    W1i = W1.interp(time=W2.time, lev=W2.lev)
    W1a, W2a = xr.align(W1i, W2, join='inner')

    Wc = (W1a + W2a) / 2
    if site == "mao":
    	Wc = Wc*mao_scaling 
    Wc = Wc.to_dataset(name='Wstd').stack(s=('time','lev'))
    Wc = Wc.assign_coords(source=('s', np.full(Wc.sizes['s'], 'COMBO')),
                          site_label=('s', np.full(Wc.sizes['s'], label)))

    return Wc[['Wstd','source','site_label']].to_dataframe().reset_index(drop=True)


def build_df(sites, labels, paths):
    out = []
    for site, lab in zip(sites, labels):
        out.append(get_obs(site, lab))
        mcount = 0
        for src in paths:
            d = get_model(site, lab, src, paths)
            if d is not None:
                out.append(d)
                mcount += 1
        if mcount == 2:
            d = get_avg_model(site, lab, paths)
            if d is not None:
                out.append(d)
    return pd.concat(out, ignore_index=True)


# ================================================================
# LOAD DATASETS (unchanged)
# ================================================================
obspath = "/discover/nobackup/dbarahon/RESMOD/ADDFKB/MingHui/"
paths_campaign = {
    "ERA5": "/discover/nobackup/dbarahon/ML_param/W_NET/single_level/ERA5/Wnet_era5/prediction/downd_predict/flight_campaigns/",
    "MERRA2": "/discover/nobackup/dbarahon/ML_param/W_NET/single_level/full_dataset/campaigns/"
}

exps = [
    'CONTRAST','NSF_DC3','HIPPO','ORCAS','PREDICT','START08','TORERO',
    'ATTREX','MACPEX','NASA_DC3','POSIDON','SEAC4RS'
]

d1 = load_obs_dataset(obspath, "NSF_sig_w_T_100km.nc")
d2 = load_obs_dataset(obspath, "NASA_sig_w_T_100km.nc", offset=7)
dobs = xr.concat([d1, d2], dim="Time")

era_list = []
mer_list = []
combo_list = []

for i, exp in enumerate(exps):
    W1 = process_model(exp, "ERA5", paths_campaign["ERA5"], dobs, i+1)
    W2 = process_model(exp, "MERRA2", paths_campaign["MERRA2"], dobs, i+1)

    W1i = W1.interp(lev=W2.lev, x=W2.x)
    W1a, W2a = xr.align(W1i, W2, join='inner')

    Wc = (W1a + W2a) / 2
    Wc = Wc.to_dataset(name='Wstd')
    Wc['campaign'] = Wc['Wstd']*0 + (i+1)

    df1 = W1.to_dataframe().reset_index()[['campaign','Wstd']].dropna()
    df1['source'] = 'ERA5'

    df2 = W2.to_dataframe().reset_index()[['campaign','Wstd']].dropna()
    df2['source'] = 'MERRA2'

    df3 = Wc.to_dataframe().reset_index()[['campaign','Wstd']].dropna()
    df3['source'] = 'COMBO'

    era_list.append(df1)
    mer_list.append(df2)
    combo_list.append(df3)

df_obs = dobs.to_dataframe().reset_index()[['campaign','Wstd']].dropna()
df_obs['source'] = 'OBS'

df_campaign = pd.concat([pd.concat(era_list), pd.concat(mer_list),
                         pd.concat(combo_list), df_obs])

# Ground-based
paths_ground = {
    "ERA5": "/discover/nobackup/dbarahon/ML_param/W_NET/single_level/ERA5/Wnet_era5/prediction/downd_predict/ground_based/",
    "MERRA2": "/discover/nobackup/dbarahon/ML_param/W_NET/single_level/full_dataset/ground_based/"
}

cirrus_sites = ['sgp_cirrus','manus','lei','lim']
cirrus_labels = ['SGP','MAN','LEI','LIM']

pbl_sites = ['sgp_pbl','nsa','asi','twp','cor','pgh','ena','mao']
pbl_labels = ['SGP','NSA','ASI','TWP','COR','PGH','ENA','MAO']

df_cirrus = build_df(cirrus_sites, cirrus_labels, paths_ground)
df_pbl = build_df(pbl_sites, pbl_labels, paths_ground)


# ================================================================
# ================================================================
fig = plt.figure(figsize=(7, 7))

outer = gridspec.GridSpec(
    2, 2,
    height_ratios=[1, 1],
    width_ratios=[0.3, 0.7],
    hspace=0.38,
    wspace=0.25
)

# ---------------------
# (a) FIELD CAMPAIGNS
# ---------------------
axA = fig.add_subplot(outer[0, :])

sns.boxplot(
    x='campaign', y='Wstd', hue='source',
    data=df_campaign, ax=axA,
    palette=palette_cb, hue_order=hue_order_cb,
    showfliers=False, linewidth=0.5
)

axA.set_xticklabels(exps, rotation=45)
axA.set_xlabel("")
axA.set_ylabel(r"$\sigma_W~(\rm{m~s}^{-1})$")
axA.set_title("(a) Field campaigns", loc='left')

axA.grid(True, linestyle="--", linewidth=0.3)
axA.legend(title=None, ncol=2, loc='upper left')


# ---------------------
# (b) CIRRUS SITES
# ---------------------
axB = fig.add_subplot(outer[1, 0])

sns.boxplot(
    x='site_label', y='Wstd', hue='source',
    data=df_cirrus, ax=axB,
    palette=palette_cb, hue_order=hue_order_cb,
    showfliers=False, linewidth=0.5
)

axB.set_xlabel("")
axB.set_ylabel(r"$\sigma_W~(\rm{m~s}^{-1})$")
axB.set_title("(b) Cirrus sites", loc='left')
axB.grid(True, linestyle="--", linewidth=0.3)

if axB.get_legend():
    axB.get_legend().remove()


# ---------------------
# (c) PBL SITES
# ---------------------
axC = fig.add_subplot(outer[1, 1])

sns.boxplot(
    x='site_label', y='Wstd', hue='source',
    data=df_pbl, ax=axC,
    palette=palette_cb, hue_order=hue_order_cb,
    showfliers=False, linewidth=0.5
)

axC.set_xlabel("")
axC.set_ylabel(r"$\sigma_W~(\rm{m~s}^{-1})$")
axC.set_title("(c) PBL sites", loc='left')
axC.grid(True, linestyle="--", linewidth=0.3)

# Remove legend here too
if axC.get_legend():
    axC.get_legend().remove()



# Final layout + save
fig.tight_layout()
fig.savefig("Wstd_campaigns_cirrus_PBL_final.eps", format='eps', bbox_inches='tight')


