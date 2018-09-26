#  -*- coding: utf-8 -*-
#  vim: tabstop=4 shiftwidth=4 softtabstop=4

#  Copyright (c) 2018, GEM Foundation

#  OpenQuake is free software: you can redistribute it and/or modify it
#  under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.

#  OpenQuake is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.

#  You should have received a copy of the GNU Affero General Public License
#  along with OpenQuake.  If not, see <http://www.gnu.org/licenses/>.
import csv
import getpass
import configparser
import pandas as pd
import numpy as np
import scipy
from datetime import datetime
from scipy.spatial.distance import cdist

from openquake.baselib.datastore import hdf5new, extract_calc_id_datadir
from openquake.baselib import sap
from openquake.hazardlib.imt import PGA, SA
from openquake.hazardlib.gsim.base import (
    RuptureContext, SitesContext, DistancesContext)
from openquake.commonlib.readinput import (
    get_oqparam, get_site_collection, get_gsim_lt,
    get_sitecol_assetcol, get_risk_model)
from openquake.hazardlib import const
from openquake.commonlib import logs
from openquake.commonlib import source
from openquake.server import dbserver

U16 = np.uint16
U32 = np.uint32
U64 = np.uint64
F32 = np.float32


def read_input_gmf(gmf_file, gmf_file_gmpe_rate):
    df_gmf = pd.read_csv(gmf_file, header=0)
    df_gmf_gmpe_rate = pd.read_csv(gmf_file_gmpe_rate, header=0)
    gmfs_median = []
    # If the file needs to be divided in two:
    # for event in range(int(len(df_gmf_gmpe_rate)/2),len(df_gmf_gmpe_rate)):
    for event in range(len(df_gmf_gmpe_rate)):
        gmf_median = {}  # gmpe -> [gmv_PGA, gmv_SA(0.3)]
        gmf_median['rate'] = df_gmf_gmpe_rate['rate'][event]
        gmf_median[df_gmf_gmpe_rate['gmpe'][event]] = (
            [df_gmf[df_gmf.event_id == event][['gmv_PGA']].values,
             df_gmf[df_gmf.event_id == event][['gmv_SA(0.3)']].values])
        gmfs_median.append(gmf_median)
    return gmfs_median


def calculate_total_std(gsim_list, imts, vs30):
    std_total = {}
    std_inter = {}
    std_intra = {}
    for gsim in gsim_list:
        rctx = RuptureContext()
        # The calculator needs these inputs but they are not used
        # in the std calculation
        rctx.mag = 5
        rctx.rake = 0
        rctx.hypo_depth = 0
        dctx = DistancesContext()
        dctx.rjb = np.copy(np.array([1]))  # I do not care about the distance
        dctx.rrup = np.copy(np.array([1]))  # I do not care about the distance
        sctx = SitesContext()
        sctx.vs30 = vs30 * np.ones_like(np.array([0]))
        for imt in imts:
            gm_table, [gm_stddev_inter, gm_stddev_intra] = (
                gsim.get_mean_and_stddevs(
                    sctx, rctx, dctx, imt,
                    [const.StdDev.INTER_EVENT, const.StdDev.INTRA_EVENT]))
            std_total[gsim, imt] = (
                np.sqrt(gm_stddev_inter[0] ** 2 + gm_stddev_intra[0] ** 2))
            std_inter[gsim, imt] = gm_stddev_inter[0]
            std_intra[gsim, imt] = gm_stddev_intra[0]
    return (std_total, std_inter, std_intra)


def calc_inter_residuals(mean_shift_inter_residuals, realizations_inter,
                         std_inter):
    # Importance Sampling
    rates_inter = np.array([1./realizations_inter]*realizations_inter)
    cumulative_rates = np.cumsum(rates_inter)-rates_inter/2
    distr_values = scipy.stats.norm.ppf(
        cumulative_rates, loc=mean_shift_inter_residuals, scale=1)
    p_distr_values = scipy.stats.norm.pdf(distr_values, loc=0, scale=1)
    q_distr_values = scipy.stats.norm.pdf(
        distr_values, loc=mean_shift_inter_residuals, scale=1)
    weights = (p_distr_values/q_distr_values)
    rates_inter = weights / sum(weights)
    # Calculate distribution mean - needs to be approximately zero
    np.mean(distr_values * rates_inter)
    # get std_inter values from gmpe
    gmpe_imt = list(std_inter)
    inter_residual = {}
    for gmpe, imt in gmpe_imt:
        stddev_inter = [std_inter[gmpe, imt]]
        inter_residual[gmpe, imt] = stddev_inter * distr_values
    inter_residual['rates_inter'] = rates_inter
    return inter_residual, gmpe_imt


def calc_intra_residuals(sp_correlation, realizations_intra, intra_files_name,
                         intra_files, sites, gmpe_imt, std_intra):
    # Only works when rates are equal (Not applicable for IS)!!!
    intra_residual = {}
    intra_residual['rates_intra'] = ([1. / realizations_intra]
                                     * realizations_intra)
    df_0 = pd.read_csv(intra_files[0], header=None)
    number_cols = len(df_0.columns)
    num_intra_matrices = len(df_0.columns)-2
    df_coords = pd.DataFrame({'lons': sites.lons, 'lats': sites.lats})
    intra_residual_no_coords = {}
    # Find indeces of rows to extract from Matrices
    coords_matrix = np.array(df_0.iloc[:, 0:2])
    exposure_coords = np.array(list(zip(*[sites.lons, sites.lats])))
    rows_to_extract = np.argmin(
        cdist(coords_matrix, exposure_coords, 'sqeuclidean'), axis=0)
    cols = np.array(range(2, number_cols))

    for gmpe, imt in gmpe_imt:
        if sp_correlation == 'True':
            file_name = intra_files_name + str(imt) + '.csv'
            df = pd.read_csv(file_name, usecols=cols, header=None)
            df_part = (df.loc[rows_to_extract].reset_index(drop=True)).values
        else:  # Intra-event residuals: No Correlation
            mu = 0.0
            sigma = 1.0
            df_part = np.random.normal(
                mu, sigma, len(sites) *
                num_intra_matrices).reshape((len(sites), num_intra_matrices))

        # get std_intra values from gmpe
        stddev_intra = [std_intra[gmpe, imt]]
        intra_residual_no_coords[gmpe, imt] = stddev_intra * df_part
        intra_residual[gmpe, imt] = (
                np.concatenate(
                    [df_coords.values,
                     intra_residual_no_coords[gmpe, imt]], axis=1))

    return intra_residual, num_intra_matrices


def create_indices(N, f, num_gmfs):
    lst = []
    offset = 0
    for sid in range(N):
        lst.append((offset, offset + num_gmfs))
        offset += num_gmfs
    f['gmf_data/indices'] = np.array(lst, U32)


def create_gmdata(f, num_gmfs, imts):
    dtlist = [(imt, F32) for imt in imts]
    gmdata_dt = np.dtype(dtlist + [('events', U32), ('nbytes', U32)])
    dset1 = f.create_dataset('gmdata', (1,), dtype=gmdata_dt)
    dset1['events'] = int(num_gmfs)
    # 0.03 is a random number, will not be used
    dset1['PGA'] = 0.03
    dset1['SA(0.3)'] = 0.03


def create_events(f, num_gmfs):
    events_dt = np.dtype([('eid', U64), ('rup_id', U32), ('grp_id', U16),
                          ('year', U32), ('ses', U32), ('sample', U32)])
    dset2 = f.create_dataset('events', (num_gmfs,), dtype=events_dt)
    dset2['eid'] = np.arange(0, num_gmfs)
    dset2['ses'] = np.ones(num_gmfs)
    dset2['rup_id'] = np.ones(num_gmfs)
    dset2['year'] = np.ones(num_gmfs)


def create_zip_intra(gsim_list, imts, intra_residual):
    zip_intra = {}
    for gsim in gsim_list:
        for imt in imts:
            try:
                zip_intra[gsim, imt] = list(zip(*intra_residual[gsim, imt]))
            except Exception:  # don't care about missing gsim in the files
                pass
    return zip_intra


def read_config_file(cfg):
    gmf_file = cfg['input']['gmf_file']
    gmf_file_gmpe_rate = cfg['input']['gmf_file_gmpe_rate']
    job_ini = cfg['input']['job_ini']
    oq_param = get_oqparam(job_ini)
    get_risk_model(oq_param)  # read risk functions and set imtls
    haz_sitecol = get_site_collection(oq_param)
    sites, assets_by_site = get_sitecol_assetcol(oq_param, haz_sitecol)
    gsimlt = get_gsim_lt(oq_param)
    gsim_list = [br.uncertainty for br in gsimlt.branches]
    cinfo = source.CompositionInfo.fake(gsimlt)
    mean_shift_inter_residuals = float(
                    cfg['input']['mean_shift_inter_residuals'])
    realizations_inter = int(cfg['input']['realizations_inter'])
    realizations_intra = int(cfg['input']['realizations_intra'])
    intra_files_name = cfg['input']['intra_files_name']
    intra_files = cfg['input']['intra_files'].split()
    csv_rate_gmf_file = cfg['output']['csv_rate_gmf_file']
    seed = int(cfg['input']['seed'])
    return (gmf_file, gmf_file_gmpe_rate, sites, gsim_list, cinfo, oq_param,
            mean_shift_inter_residuals, realizations_inter, realizations_intra,
            intra_files_name, intra_files, csv_rate_gmf_file, seed)


def create_parent_hdf5(N, num_gmfs, sites, cinfo, oq_param):
    parent_hdf5 = f = hdf5new()
    calc_id, datadir = extract_calc_id_datadir(parent_hdf5.path)
    logs.dbcmd('import_job', calc_id, 'event_based',
               'eb_test_hdf5', getpass.getuser(),
               'complete', None, datadir)
    create_gmdata(f, num_gmfs, oq_param.imtls)
    create_events(f, num_gmfs)
    f['csm_info'] = cinfo
    f['sitecol'] = sites
    f['oqparam'] = oq_param
    return f, calc_id


def save_hdf5_rate(num_gmfs, csv_rate_gmf_file, gmfs_median, gsim_list,
                   inter_residual, intra_residual, seed, num_intra_matrices,
                   realizations_intra, N, imts, zip_intra, f):
    row1_rate = 'event_id,rate' + '\n'
    with open(csv_rate_gmf_file, 'w') as text_fi_2:
        text_fi_2.write(row1_rate)
    gmv_data_dt = np.dtype(
        [('rlzi', U16), ('sid', U32), ('eid', U64),
         ('gmv', (F32, (len(imts),)))])
    shape_val = num_gmfs * N
    dset3 = f.create_dataset(
        'gmf_data/data', (shape_val,), dtype=gmv_data_dt, chunks=True)
    eid = -1
    num_sid_per_gmf = []
    with open(csv_rate_gmf_file, 'a', newline='') as text_fi_2:
        ab = csv.writer(text_fi_2, delimiter=',')
        for index_gmf in range(len(gmfs_median)):
            keys_gmfs = gmfs_median[index_gmf].keys()
            gmf_gmpe = [i for i in keys_gmfs if i in gsim_list][0]
            for d in range(len(inter_residual['rates_inter'])):
                for e in range(len(intra_residual['rates_intra'])):
                    np.random.seed(seed + index_gmf + d * 1000 + e * 10000)
                    aleatoryIntraMatrices = np.random.choice(
                        range(num_intra_matrices), realizations_intra,
                        replace=False)
                    eid += 1
                    col1_txt = np.zeros((N, 1)).flatten()
                    col2_txt = np.arange(N)
                    col3_txt = np.full((1, N), eid, dtype=int)[0]
                    gmf_to_txt = np.transpose(np.array(
                        [col1_txt, col2_txt, col3_txt]))
                    rate = (gmfs_median[index_gmf]['rate'] *
                            inter_residual['rates_inter'][d] *
                            intra_residual['rates_intra'][e])
                    for imti, imt in enumerate(imts):
                        gmf_total_part = {}
                        gmv = np.ma.log(gmfs_median[index_gmf][gmf_gmpe][imti])
                        gmf_total_part[gmf_gmpe, imt, d, e] = np.exp(
                           gmv.filled(0) +
                           inter_residual[gmf_gmpe, imt][d] +
                           np.array(zip_intra[gmf_gmpe, imt][
                             2 + aleatoryIntraMatrices[e]]).reshape((-1, 1)))
                        gmf_to_txt = np.c_[
                            gmf_to_txt,
                            gmf_total_part[gmf_gmpe, imt, d, e].flatten()]

                    ab.writerow([eid, rate])

                    dset3['rlzi', index_gmf:len(gmfs_median) * N + index_gmf:
                          len(gmfs_median)] = gmf_to_txt[:, 0]
                    dset3['sid', index_gmf:len(gmfs_median) * N + index_gmf:
                          len(gmfs_median)] = gmf_to_txt[:, 1]
                    dset3['eid', index_gmf:len(gmfs_median) * N + index_gmf:
                          len(gmfs_median)] = gmf_to_txt[:, 2]
                    dset3['gmv', index_gmf:len(gmfs_median) * N + index_gmf:
                          len(gmfs_median)] = gmf_to_txt[:, 3:3 + len(imts)]

                    num_sid_per_gmf.append(len(gmf_to_txt[:, 1]))

    dset3.resize((sum(num_sid_per_gmf),))


@sap.Script
def main(cfg_file):
    dbserver.ensure_on()
    startTime = datetime.now()
    cfg = configparser.ConfigParser()
    cfg.read(cfg_file)
    (gmf_file, gmf_file_gmpe_rate, sites, gsim_list, cinfo, oq_param,
        mean_shift_inter_residuals, realizations_inter, realizations_intra,
        intra_files_name, intra_files, csv_rate_gmf_file,
        seed) = read_config_file(cfg)
    gmfs_median = read_input_gmf(gmf_file, gmf_file_gmpe_rate)
    imts = [PGA(), SA(0.3)]
    vs30 = 180
    (std_total, std_inter, std_intra) = calculate_total_std(
        gsim_list, imts, vs30)

    inter_residual, gmpe_imt = calc_inter_residuals(
        mean_shift_inter_residuals, realizations_inter, std_inter)

    sp_correlation = cfg['input']['sp_correlation']
    intra_residual, num_intra_matrices = calc_intra_residuals(
        sp_correlation, realizations_intra, intra_files_name, intra_files,
        sites, gmpe_imt, std_intra)
    N = len(sites)
    num_gmfs = (
        len(gmfs_median) * len(inter_residual['rates_inter']) *
        len(intra_residual['rates_intra']))
    f, calc_id = create_parent_hdf5(N, num_gmfs, sites, cinfo, oq_param)

    zip_intra = create_zip_intra(gsim_list, imts, intra_residual)

    save_hdf5_rate(num_gmfs, csv_rate_gmf_file, gmfs_median,
                   gsim_list, inter_residual, intra_residual,
                   seed, num_intra_matrices, realizations_intra,
                   N, imts, zip_intra, f)

    create_indices(N, f, num_gmfs)

    f.close()
    print('Saved', calc_id)
    print(datetime.now() - startTime)


main.arg('cfg_file', 'configuration file')


if __name__ == '__main__':
    main.callfunc()
