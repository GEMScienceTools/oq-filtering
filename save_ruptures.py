
# coding: utf-8

# Each trt needs to have only one GMPE
import csv
import configparser
import numpy as np
from datetime import datetime
from itertools import compress

from openquake.baselib import sap
from openquake.commonlib.readinput import (
    get_oqparam, get_site_collection, get_gsim_lt,
    get_sitecol_assetcol)
from openquake.hazardlib.nrml import to_python
from openquake.hazardlib.sourceconverter import SourceConverter
from openquake.hazardlib.gsim.base import (
    RuptureContext, SitesContext, DistancesContext, ContextMaker)
from openquake.hazardlib.geo.mesh import Mesh, RectangularMesh
from openquake.hazardlib.calc.filters import SourceFilter
from openquake.hazardlib.imt import PGA, SA
from openquake.hazardlib import const
from openquake.hazardlib.calc.gmf import GmfComputer


def read_config_file(cfg):
    job_ini = cfg['input']['job_ini']
    # Parse Job configuration file
    oq_param = get_oqparam(job_ini)
    # for keyval in oq_param.__dict__.items():
    #     print (keyval)
    source_model_file = cfg['input']['source_model_file']
    matrixMagsMin = float(cfg['input']['matrixMagsMin'])
    matrixMagsMax = float(cfg['input']['matrixMagsMax'])
    matrixMagsStep = float(cfg['input']['matrixMagsStep'])
    matrixDistsMin = float(cfg['input']['matrixDistsMin'])
    matrixDistsMax = float(cfg['input']['matrixDistsMax'])
    matrixDistsStep = float(cfg['input']['matrixDistsStep'])
    limitIM = float(cfg['input']['limitIM'])
    imt_filtering_ = cfg['input']['imt_filtering']
    if imt_filtering_ == 'PGA':
        imt_filtering = PGA()
    elif imt_filtering_[0:2] == 'SA':
        period = float(imt_filtering_[-4:-1])
        imt_filtering = SA(period)
    else:
        print('Error: Invalid IM for filtering')
    trunc_level = float(cfg['input']['trunc_level'])
    im_filter = cfg['input']['im_filter']
    gmf_file = cfg['output']['gmf_file']
    gmf_file_gmpe_rate = cfg['output']['gmf_file_gmpe_rate']
    rup_mesh_spac = float(cfg['input']['rupture_mesh_spacing'])
    complex_mesh_spac = float(cfg['input']['complex_fault_mesh_spacing'])
    mfd_bin = float(cfg['input']['width_of_mfd_bin'])
    area_discre = float(cfg['input']['area_source_discretization'])
    limit_max_mag = float(cfg['input']['limit_max_mag'])
    limit_min_mag = float(cfg['input']['limit_min_mag'])

    return (oq_param, source_model_file, matrixMagsMin, matrixMagsMax,
            matrixMagsStep, matrixDistsMin, matrixDistsMax,
            matrixDistsStep, limitIM, imt_filtering, trunc_level,
            im_filter, gmf_file, gmf_file_gmpe_rate, rup_mesh_spac,
            complex_mesh_spac, mfd_bin, area_discre, limit_max_mag,
            limit_min_mag)


def build_gmpe_table(matrixMagsMin, matrixMagsMax, matrixMagsStep,
                     matrixDistsMin, matrixDistsMax, matrixDistsStep,
                     imt_filtering, limitIM, gsim_list, limit_max_mag,
                     limit_min_mag):
    # Define the magnitude range of interest, 5.0 - 9.0 every 0.1
    mags = np.arange(matrixMagsMin, matrixMagsMax, matrixMagsStep)
    # Define the distance range of interest, 0.0 - 300.0 every 1 km
    dists = np.arange(matrixDistsMin, matrixDistsMax, matrixDistsStep)
    # Define the Vs30 range of interest, 180.0 - 1000.0 every 1 m/s
    vs30s = np.arange(180.0, 181., 1.)
    gm_table = np.zeros([len(dists), len(mags), len(vs30s)])
    stddevs = [const.StdDev.TOTAL]
    gsim_tables = []
    for gsim in gsim_list:
        for i, mag in enumerate(mags):
            for j, vs30 in enumerate(vs30s):
                # The RuptureContext object holds all of the
                # rupture related attributes (e.g. mag, rake, ztor, hypo_depth)
                rctx = RuptureContext()
                rctx.mag = mag
                rctx.rake = 0.0
                rctx.hypo_depth = 10
                # The DistancesContext object holds all of the distance
                # calculations (e.g. rjb, rrup, rx, ry0)
                # OQ GMPEs are vectorised by distance - so this needs
                # to be an array
                dctx = DistancesContext()
                dctx.rjb = np.copy(dists)
                dctx.rrup = np.copy(dists)
                # dctx.rhypo = np.copy(dists)
                # The SitesContext object holds all of the site
                # attributes - also an array
                sctx = SitesContext()
                # The attributes of the site array must be of the
                # same size as the distances
                sctx.vs30 = vs30 * np.ones_like(dists)
                # GMPE produces 2 outputs, the means (well their
                # natural logarithm) and standard deviations
                gm_table[:, i, j], gm_stddevs = gsim.get_mean_and_stddevs(
                                        sctx, rctx, dctx, imt_filtering,
                                        stddevs)
        gm_table_exp = np.exp(gm_table)
        gsim_tables.append(gm_table_exp)

    if len(gsim_list) == 1:
        gm_table_final = gsim_tables[0]
    else:
        gm_table_final = np.maximum(gsim_tables[0], gsim_tables[1])
    # These "if" exclude all ruptures above and below the limit magnitude
    if limit_max_mag < matrixMagsMax:
        indexMag = int((limit_max_mag - matrixMagsMin) / matrixMagsStep)
        list_mag_to_exclude = np.arange(indexMag+1, len(mags))
        gm_table_final[:, list_mag_to_exclude, 0] = 0.001
    
    if limit_min_mag > matrixMagsMin:
        indexMinMag = int((limit_min_mag - matrixMagsMin) / matrixMagsStep)
        list_min_mag_to_exclude = np.arange(0, indexMinMag)
        gm_table_final[:, list_min_mag_to_exclude, 0] = 0.001

    gm_mask = gm_table_final >= limitIM
    GMPEmatrix = gm_mask[:, :, 0]
    return GMPEmatrix


def calculate_gmfs_filter(source_model, gsimlt, filter1, cmake,
                          gsim_list, recMeshExposure, matrixMagsMin,
                          matrixMagsStep, matrixDistsMin, matrixDistsStep,
                          GMPEmatrix, imts, trunc_level):
    # The source filter will return only sites within the integration distance
    gmfs_median = []
    # properties = []
    # Source-Site Filtering
    for source in source_model:
        trt = source.trt
        gmpe_lt_index = gsimlt.all_trts.index(trt)

        for src, s_sites in filter1(source[:]):
            hypo_list = []
            Mag = []
            for rup in src.iter_ruptures():
                Mag.append(round(rup.mag, 1))
                hypo_rup = rup.hypocenter
                hypo_list.append(hypo_rup)

            rupsHypo = Mesh.from_points_list(hypo_list)
            if gsim_list[gmpe_lt_index].REQUIRES_DISTANCES == {'rjb'}:
                distRupExposure = np.around(
                        RectangularMesh.get_joyner_boore_distance(
                                recMeshExposure, rupsHypo))
            elif gsim_list[gmpe_lt_index].REQUIRES_DISTANCES == {'rrup'}:
                distRupExposure = np.around(RectangularMesh.get_min_distance(
                        recMeshExposure, rupsHypo))

            filteringIndex = []
            for i in range(len(Mag)):
                indexMag = int((Mag[i] - matrixMagsMin) / matrixMagsStep)
                indexDist = int((distRupExposure[i] - matrixDistsMin)
                                / matrixDistsStep)
                filteringIndex.append(GMPEmatrix[indexDist, indexMag])

            src_iter = src.iter_ruptures()
            filteredRup = list(compress(src_iter, filteringIndex))

            for rup in filteredRup:
                gmf_computer = GmfComputer(rup, s_sites, imts, cmake,
                                           truncation_level=trunc_level)
                gmf_median = {}
                # if we have more than one gsim per trt, we need to do this
                # for gsim in gsim_list:
                #     gmf_median[gsim] = gmf_computer.compute(gsim,
                # num_events=1)
                gmf_median[gsim_list[gmpe_lt_index]] = gmf_computer.compute(
                                    gsim_list[gmpe_lt_index], num_events=1)
                gmf_median['rate'] = rup.occurrence_rate
                gmf_median['sites'] = s_sites
                gmfs_median.append(gmf_median)
                # FiltMag = str(rup.mag)
                # FiltHypo = str(rup.hypocenter)
                # FiltRate = str(rup.occurrence_rate)
                # properties.append([FiltMag,FiltHypo,FiltRate])
    return gmfs_median


def calc_gmfs_no_IM_filter(source_model, imts, gsim_list, trunc_level,
                           gsimlt, filter1, cmake):
    gmfs_median = []
    for source in source_model:
        trt = source.trt
        gmpe_lt_index = gsimlt.all_trts.index(trt)
        for src, s_sites in filter1(source[:]):
            src_iter = src.iter_ruptures()
            for rup in src_iter:
                gmf_computer = GmfComputer(rup, s_sites, imts, cmake,
                                           truncation_level=trunc_level)
                gmf_median = {}
                gmf_median[gsim_list[gmpe_lt_index]] = gmf_computer.compute(
                                    gsim_list[gmpe_lt_index], num_events=1)
                gmf_median['rate'] = rup.occurrence_rate
                gmf_median['sites'] = s_sites

                gmfs_median.append(gmf_median)
    return gmfs_median


def save_gmfs(gmf_file, gmf_file_gmpe_rate, gmfs_median, exposureCoords,
              gsim_list, imts):
    row1 = 'event_id,gmv_'+str(imts[0])+',gmv_'+str(imts[1]) + '\n'
    with open(gmf_file, 'w') as text_fi:
        text_fi.write(row1)

    row1_rate = 'event_id,rate,gmpe' + '\n'
    with open(gmf_file_gmpe_rate, 'w') as text_fi_2:
        text_fi_2.write(row1_rate)

    eid = -1
    for index_gmf in range(len(gmfs_median)):
        eid += 1
        rate = gmfs_median[index_gmf]['rate']
        keys_gmfs = gmfs_median[index_gmf].keys()
        col3_txt = np.full((1, len(exposureCoords)), eid, dtype=int)[0]
        gmf_to_txt = np.transpose(np.array([col3_txt]))
        gmf_gmpe = [i for i in keys_gmfs if i in gsim_list][0]

        for c in range(len(imts)):
            gmf_total_part = {}
            gmf_gmpe = [i for i in keys_gmfs if i in gsim_list][0]
            gmv = gmfs_median[index_gmf][gmf_gmpe][c]
            if len(gmfs_median[index_gmf]['sites'].lats) != len(
                        gmfs_median[index_gmf]['sites'].complete):
                indices_zero = np.setdiff1d(
                    gmfs_median[index_gmf]['sites'].complete.sids,
                    gmfs_median[index_gmf]['sites'].sids)
                value = 0
                for b in range(len(indices_zero)):
                    gmv_a = np.insert(gmv, indices_zero[b], value)
                    gmv = np.swapaxes([gmv_a], 0, 1)

            gmf_total_part[gmf_gmpe, imts[c]] = gmv
            gmf_to_txt = np.c_[gmf_to_txt, gmf_total_part
                               [gmf_gmpe, imts[c]].flatten()]

        with open(gmf_file, 'a', newline='') as text_fi:
            aa = csv.writer(text_fi, delimiter=',')
            aa.writerows(map(lambda t: ("%i" % t[0], "%.3f" % t[1],
                                        "%.3f" % t[2]), gmf_to_txt))

        with open(gmf_file_gmpe_rate, 'a', newline='') as text_fi_2:
            aa = csv.writer(text_fi_2, delimiter=',')
            aa.writerow([eid,rate,gmf_gmpe])  # Do NOT add spaces after commas


@sap.Script
def main(cfg_file):
    startTime = datetime.now()
    cfg = configparser.ConfigParser()
    cfg.read(cfg_file)

    (oq_param, source_model_file, matrixMagsMin, matrixMagsMax,
     matrixMagsStep, matrixDistsMin, matrixDistsMax,
     matrixDistsStep, limitIM, imt_filtering, trunc_level,
     im_filter, gmf_file, gmf_file_gmpe_rate, rup_mesh_spac,
     complex_mesh_spac, mfd_bin, area_discre, limit_mag) = read_config_file(cfg)

    # Set up the source model configuration
    conv1 = SourceConverter(1.0,  # Investigation time
                            rup_mesh_spac,   # Rupture mesh spacing
                            complex_fault_mesh_spacing=complex_mesh_spac,
                            width_of_mfd_bin=mfd_bin,
                            area_source_discretization=area_discre)
    # Parse the source Model
    if source_model_file:  # only one source model file
        source_model = to_python(source_model_file, conv1)
    else:  # source model has many files (in this case 2 - adapt for more)
        source_model_file2 = "demo_data/SA_RA_CATAL1_05.xml"
        source_model2 = to_python(source_model_file2, conv1)
        source_model = source_model+source_model2

    # Calculate total number of ruptures in the erf
    # num_rup = 0
    # rate_rup = []
    # for a in range(len(source_model)):
        # model_trt = source_model[a]
        # for b in range(len(model_trt)):
            # num_rup = num_rup + len(list(model_trt[b].iter_ruptures()))
            # for rup in model_trt[b].iter_ruptures():
                # rate_rup.append(rup.occurrence_rate)
    # print(num_rup)
    # print(sum(rate_rup))
    # print(rate_rup[0:10])
    
    # If exposure model is provided:
    haz_sitecol = get_site_collection(oq_param)
    sites, assets_by_site = get_sitecol_assetcol(oq_param, haz_sitecol)
    # If region coordinates are provided:
    # sites = get_site_collection(oq_param)

    gsimlt = get_gsim_lt(oq_param)
    gsim_list = [br.uncertainty for br in gsimlt.branches]
    GMPEmatrix = build_gmpe_table(matrixMagsMin, matrixMagsMax, matrixMagsStep,
                                  matrixDistsMin, matrixDistsMax,
                                  matrixDistsStep, imt_filtering, limitIM,
                                  gsim_list, limit_max_mag, limit_min_mag)

    # Calculate minimum distance between rupture and assets
    # Import exposure from .ini file
    depths = np.zeros(len(sites))
    exposureCoords = Mesh(sites.lons, sites.lats, depths)
    # To calculate Joyner Boore distance:
    exposurePoints = (exposureCoords, exposureCoords)
    recMeshExposure = RectangularMesh.from_points_list(exposurePoints)
    imts = ['PGA', 'SA(0.3)']
    cmake = ContextMaker(gsim_list)
    filter1 = SourceFilter(sites, oq_param.maximum_distance)

    if im_filter == 'True':  # Here we consider the IM and the MaxDist filter
        gmfs_median = calculate_gmfs_filter(source_model, gsimlt, filter1,
                                            cmake, gsim_list, recMeshExposure,
                                            matrixMagsMin, matrixMagsStep,
                                            matrixDistsMin, matrixDistsStep,
                                            GMPEmatrix, imts, trunc_level)
    else:  # No IM filter, just the MAxDist filter
        gmfs_median = calc_gmfs_no_IM_filter(source_model, imts, gsim_list,
                                             trunc_level, gsimlt,
                                             filter1, cmake)

    print("%s Ground Motion Fields" % len(gmfs_median))

    save_gmfs(gmf_file, gmf_file_gmpe_rate, gmfs_median, exposureCoords,
              gsim_list, imts)
    print(datetime.now() - startTime)


main.arg('cfg_file', 'configuration file')


if __name__ == '__main__':
    main.callfunc()
