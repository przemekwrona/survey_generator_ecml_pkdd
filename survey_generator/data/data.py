import os.path

import pandas
import numpy as np
import zipfile
from scipy.io.arff import loadarff

CONTEXT = 'src/main/resources'
SURVEY_CONTEXT = 'src/main/resources/surveys'

exclude_columns = ['id_SURVEY', 'describedDay_SURVEY', 'homeGeocoding_SURVEY', 'homeBL_SURVEY',
                   'cnTransportModes_WALK', 'cnTransportModes_TRANSIT', 'cnTransportModes_TRANSIT_REAL',
                   'cnTransportModes_BICYCLE', 'cnTransportModes_CAR', 'date_SURVEY', 'time_SURVEY',
                   'endTime_SURVEY', 'fromPlaceGeocoding_SURVEY', 'fromPlaceBL_SURVEY', 'toPlaceGeocoding_SURVEY',
                   'toPlaceBL_SURVEY', 'firstStopBL_TRANSIT', 'firstStopBL_TRANSIT_REAL', 'firstStop_TRANSIT',
                   'firstStop_TRANSIT_REAL', 'cnPath_WALK', 'cnPath_TRANSIT', 'cnPath_TRANSIT_REAL', 'cnPath_BICYCLE',
                   'cnPath_CAR',

                   # Added to citizens
                   'FirstStopName_TRANSIT', 'FirstStopName_TRANSIT_REAL'
                   ]

include_columns = ['travelAggregation', 'sex_SURVEY', 'yearOfBirth_SURVEY', 'education_SURVEY', 'childrenNumber_SURVEY',
                   'origin_SURVEY', 'destination_SURVEY', 'peopleInCar_SURVEY']

include_school_surveys = ['travelAggregation', 'sex_SURVEY', 'yearOfBirth_SURVEY', 'education_SURVEY',
                          'childrenNumber_SURVEY',
                          'destination_SURVEY', 'additionalStops_SURVEY', 'nextTravel_SURVEY',
                          'Routes_WALK', 'Distance_WALK', 'Duration_WALK', 'ElevationLost_WALK', 'ElevationGained_WALK',
                          'Routes_TRANSIT', 'minPeriod_TRANSIT', 'avgPeriod_TRANSIT', 'maxPeriod_TRANSIT',
                          'minDistance_TRANSIT', 'avgDistance_TRANSIT', 'maxDistance_TRANSIT',
                          'minWalkDistance_TRANSIT', 'avgWalkDistance_TRANSIT', 'maxWalkDistance_TRANSIT',
                          'minWalkDuration_TRANSIT', 'avgWalkDuration_TRANSIT', 'maxWalkDuration_TRANSIT',
                          'minWaitingTime_TRANSIT', 'avgWaitingTime_TRANSIT', 'maxWaitingTime_TRANSIT',
                          'minDuration_TRANSIT', 'avgDuration_TRANSIT', 'maxDuration_TRANSIT',
                          'minTransfersNumber_TRANSIT', 'avgTransfersNumber_TRANSIT', 'maxTransfersNumber_TRANSIT',
                          'minTransitTime_TRANSIT', 'avgTransitTime_TRANSIT', 'maxTransitTime_TRANSIT',
                          'Routes_TRANSIT_REAL', 'minPeriod_TRANSIT_REAL', 'avgPeriod_TRANSIT_REAL',
                          'maxPeriod_TRANSIT_REAL', 'minDistance_TRANSIT_REAL', 'avgDistance_TRANSIT_REAL',
                          'maxDistance_TRANSIT_REAL', 'minWalkDistance_TRANSIT_REAL', 'avgWalkDistance_TRANSIT_REAL',
                          'maxWalkDistance_TRANSIT_REAL', 'minWalkDuration_TRANSIT_REAL',
                          'avgWalkDuration_TRANSIT_REAL', 'maxWalkDuration_TRANSIT_REAL', 'minWaitingTime_TRANSIT_REAL',
                          'avgWaitingTime_TRANSIT_REAL', 'maxWaitingTime_TRANSIT_REAL', 'minDuration_TRANSIT_REAL',
                          'avgDuration_TRANSIT_REAL', 'maxDuration_TRANSIT_REAL', 'minTransfersNumber_TRANSIT_REAL',
                          'avgTransfersNumber_TRANSIT_REAL', 'maxTransfersNumber_TRANSIT_REAL',
                          'minTransitTime_TRANSIT_REAL', 'avgTransitTime_TRANSIT_REAL', 'maxTransitTime_TRANSIT_REAL',
                          'Routes_BICYCLE', 'Distance_BICYCLE', 'WalkDistance_BICYCLE', 'WalkDuration_BICYCLE',
                          'Duration_BICYCLE', 'ElevationLost_BICYCLE', 'ElevationGained_BICYCLE', 'Routes_CAR',
                          'Distance_CAR', 'Duration_CAR', 'DurationInTraffic_CAR', 'minutesSinceMidnight']

include = ['Distance_WALK', 'drivingLicence_SURVEY', 'householdMembers_SURVEY', 'ElevationLost_BICYCLE',
           'minSpeed_TRANSIT', 'WalkDistance_BICYCLE', 'maxDuration_TRANSIT', 'avgWaitingTime_TRANSIT',
           'travelAggregation', 'minutesSinceMidnight']

clazzName = 'travelAggregation'


def citizensW1W2():
    return get_surveys("CITIZENS_W1_W2_5_1_1_fixed_merged_sort.csv")


def citizensW1():
    return get_surveys(
        "CITIZENSW1_trips_sorted_[WALK, TRANSIT, TRANSIT_REAL, BICYCLE, CAR]_W0_version_5.1_[OTP, PARKING, ENV, LOW, TRAFFIC, ZONE, URBAN, DIFF, BIKE_RENTAL].csv_OTPFixed_merged_sort.csv")


def citizensW2():
    return get_surveys(
        "CITIZENSW2_trips_sorted_[WALK, TRANSIT, TRANSIT_REAL, BICYCLE, CAR]_W0_version_5.1_[OTP, PARKING, ENV, LOW, TRAFFIC, ZONE, URBAN, DIFF, BIKE_RENTAL].csv_OTPFixed_merged_sort.csv")


def parentsW1():
    return get_surveys(
        "PARENTSW1_trips_sorted_[WALK, TRANSIT, TRANSIT_REAL, BICYCLE, CAR]_W0_version_5.1.2_[OTP, PARKING, ENV, LOW, TRAFFIC, ZONE, URBAN, DIFF, BIKE_RENTAL]_sort.csv")


def parentsW2():
    return get_surveys(
        "PARENTSW2_trips_sorted_[WALK, TRANSIT, TRANSIT_REAL, BICYCLE, CAR]_W0_version_5.1_[OTP, PARKING, ENV, LOW, TRAFFIC, ZONE, URBAN, DIFF, BIKE_RENTAL].csv_OTPFixed_merged_sort.csv")


def citizense1():
    return get_surveys('CITIZENSW1_trips_sorted_[WALK, TRANSIT, BICYCLE, CAR]_version_4.10_otp.csv')


def citizense1_v50():
    return get_surveys(
        'CITIZENSW1_trips_sorted_(WALK, TRANSIT, TRANSIT_REAL, BICYCLE, CAR)_W0_version_5.0_(OTP, PARKING, ENV, LOW, TRAFFIC, ZONE, URBAN, DIFF).csv')


def citizens_W1_W2_5_1_1():
    return get_surveys('CITIZENS_W1_W2_5_1_1_fixed_merged_sort.csv')


def generated_citizense1_distributed():
    # return get_surveys('warsaw_generated_citizensw1_distributed_2023-08-28T15_23.csv')
    resource = 'resources'
    csv_file = 'warsaw_generated_citizensw1_distributed_2023-09-04T22_49.csv'
    if not os.path.exists('{}/{}'.format(resource, csv_file)):
        with zipfile.ZipFile('{}/{}.zip'.format(resource, csv_file), 'r') as zip_ref:
            zip_ref.extractall(resource)

    surveys = get_surveys('warsaw_generated_citizensw1_distributed_2023-09-04T22_49.csv')

    surveys.loc[:, 'series_INTERNAL'] = 'GENERATED'

    return surveys


def generated_citizense1_total_randomized():
    # return get_surveys('warsaw_generated_citizensw1_distributed_2023-08-28T15_23.csv')
    resource = 'resources'
    csv_file = 'warsaw_generated_all_ones.csv'

    if not os.path.exists('{}/{}'.format(resource, csv_file)):
        with zipfile.ZipFile('{}/{}.zip'.format(resource, csv_file), 'r') as zip_ref:
            zip_ref.extractall(resource)

    surveys = get_surveys('warsaw_generated_all_ones.csv')
    surveys.loc[:, 'series_INTERNAL'] = 'GENERATED'

    return surveys


def get_surveys(csv_filename, sep=';'):
    csv_path = 'resources/{}'.format(csv_filename)

    surveys = pandas.read_csv(csv_path, sep=sep, encoding='ISO-8859-1', low_memory=False)

    surveys['time_OTP_SinceMidnight'] = surveys.apply(
        lambda x: int(x['time_OTP'].split(':')[0]) * 60 + int(x['time_OTP'].split(':')[1]),
        axis=1)

    surveys['endTime_OTP_SinceMidnight'] = surveys.apply(
        lambda x: int(x['endTime_OTP'].split(':')[0]) * 60 + int(x['endTime_OTP'].split(':')[1]),
        axis=1)

    surveys['endingTime_SURVEY_SinceMidnight'] = surveys.apply(
        lambda x: int(x['endingTime_SURVEY'].split(':')[0]) * 60 + int(x['endingTime_SURVEY'].split(':')[1]),
        axis=1)

    surveys['startingTime_SURVEY_SinceMidnight'] = surveys.apply(
        lambda x: int(x['startingTime_SURVEY'].split(':')[0]) * 60 + int(x['startingTime_SURVEY'].split(':')[1]),
        axis=1)

    surveys.ElevationLost_WALK = surveys.ElevationLost_WALK.astype(float)

    preprocessed_surveys = survey_preprocessing(surveys)

    return preprocessed_surveys


def citizens_surveys():
    csv_filename = 'CITIZENSW1_trips_sorted_[WALK, TRANSIT, TRANSIT_REAL, BICYCLE, CAR]_version_4.7_otp.csv'
    citizens_path = '{}/{}'.format(SURVEY_CONTEXT, csv_filename)

    if not os.path.exists(citizens_path):
        zip_filename = 'CITIZENSW1_trips_sorted_[WALK, TRANSIT, TRANSIT_REAL, BICYCLE, CAR]_version_4.7_otp.csv.zip'
        with zipfile.ZipFile('{}/{}'.format(SURVEY_CONTEXT, zip_filename), 'r') as zip_ref:
            zip_ref.extractall(SURVEY_CONTEXT)

    surveys = pandas.read_csv(citizens_path, sep=';')
    preprocessed_surveys = survey_preprocessing(surveys)

    return preprocessed_surveys


def school_surveys():
    arff_file = 'travels_schools_otp_v2.csv_complete.arff'
    arff_file_v2 = 'travels_schools_otp_W2_journeysSchoolVarExcluded_complete_only_raw.arff'
    schools_path = '{}/{}'.format(SURVEY_CONTEXT, arff_file_v2)
    surveys = loadarff(schools_path)

    surveys_df = pandas.DataFrame(surveys[0])
    surveys_df = surveys_df[include_school_surveys]

    surveys_df['travelAggregation'] = surveys_df['travelAggregation'].str.decode('utf-8')
    surveys_df['sex_SURVEY'] = surveys_df['sex_SURVEY'].str.decode('utf-8')
    surveys_df['education_SURVEY'] = surveys_df['education_SURVEY'].str.decode('utf-8')
    surveys_df['destination_SURVEY'] = surveys_df['destination_SURVEY'].str.decode('utf-8')

    surveys_df['additionalStops_SURVEY'] = surveys_df['additionalStops_SURVEY'].str.decode('utf-8').str.title()
    surveys_df['nextTravel_SURVEY'] = surveys_df['nextTravel_SURVEY'].str.decode('utf-8').str.title()

    return surveys_df


def survey_preprocessing(surveys):
    surveys['minutesSinceMidnight'] = surveys.apply(
        lambda x: int(x['startingTime_SURVEY'].split(':')[0]) * 60 + int(x['startingTime_SURVEY'].split(':')[1]),
        axis=1)

    surveys.replace([np.inf, -np.inf], None, inplace=True)

    # exclude columns
    sub_surveys = surveys.loc[:, ~surveys.columns.isin(exclude_columns)]

    if 'temperature_24_0_ENV' in sub_surveys.columns:
        sub_surveys.loc[:, ('temperature_24_0_ENV')] = pandas.to_numeric(sub_surveys.temperature_24_0_ENV, errors='coerce')

    if 'avgCurrentStopDelay_LOW_TRANSIT' in sub_surveys.columns:
        sub_surveys.loc[:, ('avgCurrentStopDelay_LOW_TRANSIT')] = pandas.to_numeric(sub_surveys.avgCurrentStopDelay_LOW_TRANSIT, errors='coerce')

    values = {
        'Routes_CAR': 0.0,
        'Duration_CAR': 0.0,
        'Distance_CAR': 0.0,
        'WalkDuration_BICYCLE': 0.0,
        'WalkDistance_BICYCLE': 0.0,
        'Speed_BICYCLE': 0.0,
        'Routes_BICYCLE': 0.0
    }
    pandas.set_option("future.no_silent_downcasting", True)
    sub_surveys.fillna(value=values)

    convert_dict = {
        'Routes_CAR': float,
        'Duration_CAR': float,
        'Distance_CAR': float,
        'WalkDuration_BICYCLE': float,
        'WalkDistance_BICYCLE': float,
        'Speed_BICYCLE': float,
        'ElevationLost_WALK': float
    }
    sub_surveys = sub_surveys.astype(convert_dict)

    if 'ParkingDifficultyIntersectionArea_CAR' in sub_surveys.columns:
        sub_surveys.loc[:, ('ParkingDifficultyIntersectionArea_CAR')] = sub_surveys.fillna(value={'ParkingDifficultyIntersectionArea_CAR': 0.0}).astype(
            {'ParkingDifficultyIntersectionArea_CAR': float})

    if 'Distance_CAR' in sub_surveys.columns:
        sub_surveys.loc[:, ('Distance_CAR')] = sub_surveys.fillna(value={'Distance_CAR': 0.0}).astype({'Distance_CAR': float})

    if 'Duration_CAR' in sub_surveys.columns:
        sub_surveys.loc[:, ('Duration_CAR')] = sub_surveys.fillna(value={'Duration_CAR': 0.0}).astype({'Duration_CAR': float})

    if 'Routes_CAR' in sub_surveys.columns:
        sub_surveys.loc[:, ('Routes_CAR')] = sub_surveys.fillna(value={'Routes_CAR': 0.0}).astype({'Routes_CAR': float})

    if 'Speed_CAR' in sub_surveys.columns:
        sub_surveys.loc[:, ('Speed_CAR')] = sub_surveys.fillna(value={'Speed_CAR': 0.0}).astype({'Speed_CAR': float})

    if 'Routes_WALK' in sub_surveys.columns:
        sub_surveys.loc[:, ('Routes_WALK')] = sub_surveys.fillna(value={'Routes_WALK': 0.0}).astype({'Routes_WALK': float})

    if 'parking_SURVEY' in sub_surveys.columns:
        sub_surveys.loc[:, ('parking_SURVEY')] = sub_surveys.fillna(value={'parking_SURVEY': False}).astype({'parking_SURVEY': bool})

    if 'parkingTime_SURVEY' in sub_surveys.columns:
        sub_surveys.loc[:, ('parkingTime_SURVEY')] = sub_surveys.fillna(value={'parkingTime_SURVEY': False}).astype({'parkingTime_SURVEY': int})

    if 'usingCar_SURVEY' in sub_surveys.columns:
        sub_surveys.loc[:, ('usingCar_SURVEY')] = sub_surveys.fillna(value={'usingCar_SURVEY': False}).astype({'usingCar_SURVEY': bool})

    # if 'Speed_BICYCLE' in sub_surveys.columns:
    #     sub_surveys.loc(["Speed_BICYCLE"]) = pandas.to_numeric(sub_surveys["Speed_BICYCLE"])
    # sub_surveys.loc['Speed_BICYCLE'] =
    # sub_surveys.Speed_BICYCLE.astype(float)

    # include columns
    # sub_surveys = sub_surveys[include]

    # Set first column as class
    cols = list(sub_surveys)
    cols.insert(0, cols.pop(cols.index(clazzName)))
    sub_surveys = sub_surveys.loc[:, cols]

    # sub_surveys[['parkingTime_SURVEY']] = sub_surveys[['parkingTime_SURVEY']].fillna(value=0)

    return sub_surveys
