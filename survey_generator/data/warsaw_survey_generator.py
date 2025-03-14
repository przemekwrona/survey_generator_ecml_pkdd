import os
import re
import subprocess
import numpy as np

import pandas

from survey_generator.data import data
from survey_generator.semisupervised import stream
from sklearn.model_selection import train_test_split

CONTEXT = 'src/main/resources'

CSV_REGEXP = re.compile('[\S]+.csv')
TEMP_ARFF_REGEXP = re.compile('[\S]+_temp.arff')


def learn_and_tail_dataset(dataset, num):
    learn_dataset = dataset.head(num)
    tail_dataset = dataset.drop(learn_dataset.index)

    return learn_dataset, tail_dataset


def learn_and_test_surveys(dataset):
    learn_dataset = dataset[dataset.series_INTERNAL == 'TRAIN']
    test_dataset = dataset.drop(learn_dataset.index)

    return learn_dataset, test_dataset


def survey_to_arff(warsaw_surveys, test_surveys, surveys_generated, target, shuffle=False):
    if not os.path.exists(target):
        os.mkdir(target)

    warsaw_surveys = warsaw_surveys.dropna(axis=0, subset=['minTransitTime_TRANSIT'])
    warsaw_surveys = warsaw_surveys.dropna(axis=0, subset=['Routes_WALK'])
    warsaw_surveys = warsaw_surveys.dropna(axis=0, subset=['Routes_BICYCLE'])
    warsaw_surveys = warsaw_surveys.loc[warsaw_surveys['Duration_CAR'] != 0.0]
    warsaw_surveys = warsaw_surveys[warsaw_surveys.columns.values]
    # warsaw_surveys = warsaw_surveys.dropna()
    warsaw_surveys.loc[:, 'transit_car_duration_ratio'] = warsaw_surveys.minDuration_TRANSIT / warsaw_surveys.Duration_CAR
    # warsaw_surveys['car_transit_duration_ratio'] = warsaw_surveys.Duration_CAR / warsaw_surveys.minDuration_TRANSIT
    warsaw_surveys.index.name = 'survey_id'
    warsaw_surveys = warsaw_surveys.sort_values(by=['minutesSinceMidnight', 'survey_id'])

    test_surveys = test_surveys.dropna(axis=0, subset=['minTransitTime_TRANSIT'])
    test_surveys = test_surveys.dropna(axis=0, subset=['Routes_WALK'])
    test_surveys = test_surveys.dropna(axis=0, subset=['Routes_BICYCLE'])
    test_surveys = test_surveys.loc[test_surveys['Duration_CAR'] != 0.0]
    test_surveys = test_surveys[test_surveys.columns.values]
    # test_surveys = test_surveys.dropna()
    test_surveys.loc[:, 'transit_car_duration_ratio'] = test_surveys.minDuration_TRANSIT / test_surveys.Duration_CAR
    # test_surveys['car_transit_duration_ratio'] = test_surveys.Duration_CAR / test_surveys.minDuration_TRANSIT
    test_surveys.index.name = 'survey_id'
    test_surveys = test_surveys.sort_values(by=['minutesSinceMidnight', 'survey_id'])

    surveys_generated = surveys_generated.dropna(axis=0, subset=['minTransitTime_TRANSIT'])
    surveys_generated = surveys_generated.dropna(axis=0, subset=['Routes_WALK'])
    surveys_generated = surveys_generated.dropna(axis=0, subset=['Routes_BICYCLE'])
    surveys_generated = surveys_generated.loc[surveys_generated['Duration_CAR'] != 0.0]
    surveys_generated = surveys_generated[surveys_generated.columns.values]
    surveys_generated = surveys_generated.dropna()
    surveys_generated['minDuration_TRANSIT'] = pandas.to_numeric(surveys_generated['minDuration_TRANSIT'], errors='coerce')
    surveys_generated['Duration_CAR'] = pandas.to_numeric(surveys_generated['Duration_CAR'], errors='coerce')
    surveys_generated.loc[:, 'transit_car_duration_ratio'] = surveys_generated.minDuration_TRANSIT / surveys_generated.Duration_CAR
    # surveys_generated['car_transit_duration_ratio'] = surveys_generated.Duration_CAR / surveys_generated.minDuration_TRANSIT
    surveys_generated.index.name = 'survey_id'
    # surveys_generated = surveys_generated.sort_values(by=['minutesSinceMidnight', 'survey_id'])
    surveys_generated = surveys_generated.sample(frac=1)

    if shuffle:
        for column_name in surveys_generated.columns:
            if column_name != 'travelAggregation':
                surveys_generated[column_name] = surveys_generated[column_name].sample(frac=1).values

    print("After filtering use {} of surveys and {} generated".format(len(warsaw_surveys), len(surveys_generated)))
    print("Test surveys (window): {}".format(len(test_surveys)))
    # percentage_of_generated_vectors = [0.1]
    percentage_of_generated_vectors = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    number_of_learn_vectors = np.arange(0, 1600 if len(warsaw_surveys) >= 1600 else len(warsaw_surveys), 50, dtype=int)
    # number_of_learn_vectors = [0, 200]

    for percentage_of_generated_vector in percentage_of_generated_vectors:
        frac, num = get_number_of_generated_instances(percentage_of_generated_vector, warsaw_surveys)
        print("Use {:.2f}% of surveys ({:.0f} out of {})".format(frac * 100, num, len(surveys_generated)))

        sample_of_surveys_generated = surveys_generated.head(num).assign(travelAggregation='?')

        for number_of_learn_vector in number_of_learn_vectors:
            learn_dataset, tail_dataset = learn_and_tail_dataset(warsaw_surveys, number_of_learn_vector)

            tail_dataset_with_generated_surveys = stream.drift_stream(tail_dataset, sample_of_surveys_generated)
            tail_dataset_with_generated_surveys = tail_dataset_with_generated_surveys.sort_values(by=['minutesSinceMidnight', 'survey_id'])

            dataset = pandas.concat([learn_dataset, tail_dataset_with_generated_surveys, test_surveys], ignore_index=True)

            # Save to CSV and ARFF
            filename = "warsaw_survey_{:.0f}_init_classificator_{:.0f}_generated".format(number_of_learn_vector,
                                                                                         percentage_of_generated_vector * 100)

            # Save dataset as csv
            csv_file_path = '{}/{}.csv'.format(target, filename)
            dataset.to_csv(csv_file_path, sep=';', index=False)
            # arff.dump('generated/semisupervised/{}.arff'.format(filename), dataset.values, relation='warsaw surveys',
            #           names=dataset.columns)

            arff_file_name = "warsaw_survey_{:.0f}_init_classificator_{:.0f}_generated".format(
                number_of_learn_vector, percentage_of_generated_vector * 100)
            arff_file_path_temp = "{}/{}_temp.arff".format(target, arff_file_name)
            subprocess.call(['csv2arff', csv_file_path, arff_file_path_temp, '--delimiter', ';'])

            # Replace null in arff file to ?
            with open(arff_file_path_temp, "rt") as fin:
                arff_file_path_final = "{}/{}.arff".format(target, arff_file_name)
                with open(arff_file_path_final, "wt") as fout:
                    for line in fin:
                        line = line.replace("{'?',", "{")
                        line = line.replace("'?'", "?")
                        line = line.replace("'STAY_AT_HOME_AND_UNEMPLOYE,", "'STAY_AT_HOME_AND_UNEMPLOYED',")
                        fout.write(line)

            # Delete file with extension .csv and *_temp.arff
            target_dir = '{}'.format(target)
            for path in os.listdir(target_dir):
                if os.path.isfile(os.path.join(target_dir, path)):
                    # if CSV_REGEXP.match(path):
                    #     os.remove('{}/{}'.format(target_dir, path))

                    if TEMP_ARFF_REGEXP.match(path):
                        os.remove('{}/{}'.format(target_dir, path))


def save_w1_w2_5_1_1_as_arff(dataset):
    if not os.path.exists('survey_generator/generated'):
        os.mkdir('survey_generator/generated')

    if not os.path.exists('survey_generator/generated/semisupervised'):
        os.mkdir('survey_generator/generated/semisupervised')

    if not os.path.exists('survey_generator/generated/semisupervised/citizens_w1_w2'):
        os.mkdir('survey_generator/generated/semisupervised/citizens_w1_w2')

    # Save dataset as csv
    csv_file_name = 'survey_generator/generated/semisupervised/citizens_w1_w2/citizens_w1_w2.csv'
    dataset.to_csv(csv_file_name, sep=';', index=False)
    # arff.dump('generated/semisupervised/{}.arff'.format(filename), dataset.values, relation='warsaw surveys',
    #           names=dataset.columns)

    arff_file_name = "citizens_w1_w2_generated"
    arff_file_path_temp = "survey_generator/generated/semisupervised/citizens_w1_w2/{}.arff".format(arff_file_name)
    subprocess.call(['csv2arff', csv_file_name, arff_file_path_temp, '--delimiter', ';'])

    # Delete file with extension .csv and *_temp.arff
    target_dir = 'survey_generator/generated/semisupervised/citizens_w1_w2'
    for path in os.listdir(target_dir):
        if os.path.isfile(os.path.join(target_dir, path)):
            if CSV_REGEXP.match(path):
                os.remove('{}/{}'.format(target_dir, path))

            if TEMP_ARFF_REGEXP.match(path):
                os.remove('{}/{}'.format(target_dir, path))


def get_number_of_generated_instances(percentage_of_generated_vector, warsaw_surveys):
    num = (len(warsaw_surveys) * percentage_of_generated_vector) / (1 - percentage_of_generated_vector)
    return percentage_of_generated_vector, int(num)


def to_arff_all(randomized, shuffle, no_sample_train, no_sample_test):
    datasets = [
        ('citizens_W1_W2', 'CITIZENS_W1_W2.csv'),
        ('citizens_W1', 'CITIZENSW1.csv'),
        ('citizens_W2', 'CITIZENSW2.csv'),
        ('parents_W1', 'PARENTSW1.csv'),
        ('parents_W2', 'PARENTSW2.csv')
    ]

    if not os.path.exists('generated'):
        os.mkdir('generated')

    if not os.path.exists('generated/semisupervised'):
        os.mkdir('generated/semisupervised')

    for name, dataset_file_name in datasets:
        directory = 'generated/semisupervised/{}'.format(name)
        if not os.path.exists(directory):
            os.mkdir(directory)

        to_arff(directory=directory, dataset_file_name=dataset_file_name, randomized=randomized, shuffle=shuffle, no_sample_train=no_sample_train,
                no_sample_test=no_sample_test)


def mix_train_with_generated(learn_surveys, surveys_generated):
    exclude = [
        'sex_SURVEY', 'yearOfBirth_SURVEY', 'lineNumber_SURVEY',
        'monthOfBirth_SURVEY', 'education_SURVEY', 'hasBike_SURVEY',
        'carAvailability_SURVEY', 'cyclingLimitations_SURVEY', 'childrenNumber_SURVEY', 'journeyCar_SURVEY', 'journeyTaxi_SURVEY',
        'journeyPublicTransport_SURVEY', 'publicJourneyByCar_SURVEY',
        'drivingLicence_SURVEY', 'usingCar_SURVEY', 'parking_SURVEY',
        'cityCard_SURVEY', 'varsovianCard_SURVEY', 'income_SURVEY',
        'carsNumber_SURVEY', 'scooterNumber_SURVEY', 'bicycleNumber_SURVEY', 'motorNumber_SURVEY',
        'exoticMeansNumber_SURVEY', 'householdMembers_SURVEY',
        'householdBelow15_SURVEY', 'beliefsWarsawS1_SURVEY',
        'beliefsWarsawS2_SURVEY', 'beliefsWarsawS3_SURVEY',
        'beliefsWarsawS4_SURVEY', 'beliefsWarsawS5_SURVEY', 'beliefsWarsawS6_SURVEY',
        'beliefsWarsawS7_SURVEY', 'beliefsWarsawS8_SURVEY',
        'beliefsWarsawS9_SURVEY', 'beliefsWarsawS10_SURVEY',
        'beliefsWarsawS11_SURVEY', 'beliefsWarsawS12_SURVEY',
        'beliefsWarsawS13_SURVEY', 'occupationMainS1_SURVEY',
        'occupationMainS2_SURVEY', 'carFrequencyWarsaw_SURVEY',
        'bicycleFrequency_SURVEY', 'destination_SURVEY', 'transport_SURVEY', 'peopleInCar_SURVEY',
        'additionalStops_SURVEY', 'parkingTime_SURVEY',
        'nextTravel_SURVEY'
    ]
    surveys_generated = surveys_generated.loc[:, ~surveys_generated.columns.isin(exclude)]

    # Generate result using pandas
    result = pandas.DataFrame()
    for value in surveys_generated.index:
        random_survey = learn_surveys.sample(1)
        random_survey = random_survey.loc[:, random_survey.columns.isin(exclude)]
        result = pandas.concat([result, random_survey])

    return pandas.concat([surveys_generated, result.set_index(surveys_generated.index)], axis=1)


def to_arff(directory, dataset_file_name, randomized, shuffle, no_sample_train, no_sample_test):
    include = [
        'travelAggregation',
        'yearOfBirth_SURVEY',
        'varsovianCard_SURVEY',
        'usingCar_SURVEY',
        # 'transport_SURVEY',
        'time_OTP_SinceMidnight',
        'startingTime_SURVEY_SinceMidnight',
        'startingAtHome_SURVEY',
        'sex_SURVEY',
        'scooterNumber_SURVEY',
        # 'peopleInCar_SURVEY',
        'parking_SURVEY',
        # 'parkingTime_SURVEY',
        # 'journeyCar_SURVEY',
        'origin_SURVEY',
        'occupationMainS2_SURVEY',
        'occupationMainS1_SURVEY',
        'nextTravel_SURVEY',
        'motorNumber_SURVEY',
        'minutesSinceMidnight',
        'minWalkDuration_TRANSIT',
        'minWalkDistance_TRANSIT',
        'minWaitingTime_TRANSIT',
        'minTransitTime_TRANSIT',
        'minTransfersNumber_TRANSIT',
        'minStops_TRANSIT',
        'minSpeed_TRANSIT',
        'minPeriod_TRANSIT',
        'minDuration_TRANSIT',
        'minDistance_TRANSIT',
        'maxWalkDuration_TRANSIT',
        'maxWalkDistance_TRANSIT',
        'maxWaitingTime_TRANSIT',
        'maxTransitTime_TRANSIT',
        'maxTransfersNumber_TRANSIT',
        'maxStops_TRANSIT',
        # 'maxSpeed_TRANSIT',
        'maxPeriod_TRANSIT',
        'maxDuration_TRANSIT',
        'maxDistance_TRANSIT',
        'income_SURVEY',
        'householdMembers_SURVEY',
        'householdBelow15_SURVEY',
        'hasBike_SURVEY',
        'exoticMeansNumber_SURVEY',
        'endingTime_SURVEY_SinceMidnight',
        'endTime_OTP_SinceMidnight',
        'education_SURVEY',
        'drivingLicence_SURVEY',
        'destination_SURVEY',
        'cyclingLimitations_SURVEY',
        'cityCard_SURVEY',
        'childrenNumber_SURVEY',
        'carsNumber_SURVEY',
        'carFrequencyWarsaw_SURVEY',
        # 'carAvailability_SURVEY',
        'bicycleNumber_SURVEY',
        'bicycleFrequency_SURVEY',
        'beliefsWarsawS9_SURVEY',
        'beliefsWarsawS8_SURVEY',
        'beliefsWarsawS7_SURVEY',
        'beliefsWarsawS6_SURVEY',
        'beliefsWarsawS5_SURVEY',
        'beliefsWarsawS4_SURVEY',
        'beliefsWarsawS3_SURVEY',
        'beliefsWarsawS2_SURVEY',
        'beliefsWarsawS1_SURVEY',
        'beliefsWarsawS13_SURVEY',
        'beliefsWarsawS12_SURVEY',
        'beliefsWarsawS11_SURVEY',
        'beliefsWarsawS10_SURVEY',
        'avgWalkDuration_TRANSIT',
        'avgWalkDistance_TRANSIT',
        'avgWaitingTime_TRANSIT',
        'avgTransitTime_TRANSIT',
        'avgTransfersNumber_TRANSIT',
        'avgStops_TRANSIT',
        # 'avgSpeed_TRANSIT',
        'avgPeriod_TRANSIT',
        'avgDuration_TRANSIT',
        'avgDistance_TRANSIT',
        'additionalStops_SURVEY',
        'WalkDuration_BICYCLE',
        'WalkDistance_BICYCLE',
        'Starting_ZONE',
        'Speed_WALK',
        'Speed_CAR',
        'Speed_BICYCLE',
        'Routes_WALK',
        'Routes_TRANSIT',
        'Routes_CAR',
        'Routes_BICYCLE',
        'Ending_ZONE',
        'ElevationLost_WALK',
        'ElevationLost_BICYCLE',
        'ElevationGained_WALK',
        'ElevationGained_BICYCLE',
        'Duration_WALK',
        'Duration_CAR',
        'Duration_BICYCLE',
        'Distance_WALK',
        'Distance_CAR',
        'Distance_BICYCLE',
        'toPlaceLongitude_OTP',
        'toPlaceLatitude_OTP',
        'startingAddressLongitude_SURVEY',
        'startingAddressLatitude_SURVEY',
        'monthOfBirth_SURVEY',
        'localisationLongitude_SURVEY',
        'localisationLatitude_SURVEY',
        'fromPlaceLongitude_OTP',
        'fromPlaceLatitude_OTP',
        # 'transit_car_duration_ratio' JEST DODAWANE NA PÓZNIEJSZYM ETAPIE
        'series_INTERNAL'
    ]

    surveys = data.get_surveys(dataset_file_name)
    surveys = surveys.dropna(axis=0, subset=['minTransitTime_TRANSIT'])
    surveys = surveys.dropna(axis=0, subset=['Routes_TRANSIT'])
    surveys = surveys.dropna(axis=0, subset=['Routes_WALK'])
    surveys = surveys.dropna(axis=0, subset=['Routes_BICYCLE'])
    surveys = surveys.loc[surveys['Duration_CAR'] != 0.0]
    surveys = surveys[surveys['travelAggregation'] != 'MIXED_CAR_AND_OTHER']
    surveys = surveys[surveys['travelAggregation'] != 'MULTIMODE']
    surveys = surveys[surveys['travelAggregation'] != 'MIXED_BIKE_AND_OTHER']
    surveys.loc[surveys['travelAggregation'] == 'PRIVATE_BIKE', 'travelAggregation'] = 'BIKE'
    surveys.loc[surveys['travelAggregation'] == 'CITY_BIKE', 'travelAggregation'] = 'BIKE'

    surveys_train = surveys[surveys['series_INTERNAL'] == 'TRAIN']
    surveys_test = surveys[surveys['series_INTERNAL'] == 'TEST']

    number_of_samples_train = min(no_sample_train, len(surveys_train))
    number_of_samples_test = min(no_sample_test, len(surveys_test))

    # surveys_train = surveys_train.sample(number_of_samples_train)
    # surveys_test = surveys_test.sample(number_of_samples_test)

    if len(surveys_train) > number_of_samples_train:
        surveys_train, _ = train_test_split(surveys_train, train_size=number_of_samples_train, stratify=surveys_train['travelAggregation'], random_state=42)

    if len(surveys_test) > number_of_samples_test:
        surveys_test, _ = train_test_split(surveys_test, train_size=number_of_samples_test, stratify=surveys_test['travelAggregation'], random_state=42)

    surveys = pandas.concat([surveys_train, surveys_test], ignore_index=True)

    # surveys_w1 = data.get_surveys(
    #     'CITIZENSW1_trips_sorted_[WALK, TRANSIT, TRANSIT_REAL, BICYCLE, CAR]_W0_version_5.1_[OTP, PARKING, ENV, LOW, TRAFFIC, ZONE, URBAN, DIFF, BIKE_RENTAL].csv_OTPFixed_merged_sort.csv', no_sample_train, no_sample_test)
    surveys_generated = data.generated_citizense1_total_randomized() if randomized == True else data.generated_citizense1_distributed()
    surveys_generated = surveys_generated.dropna(axis=0, subset=['minTransitTime_TRANSIT'])
    surveys_generated = surveys_generated.dropna(axis=0, subset=['Routes_TRANSIT'])
    surveys_generated = surveys_generated.dropna(axis=0, subset=['Routes_WALK'])
    surveys_generated = surveys_generated.dropna(axis=0, subset=['Routes_BICYCLE'])
    surveys_generated = surveys_generated.loc[surveys_generated['Duration_CAR'] != 0.0]

    # surveys_generated = surveys_generated[surveys_generated['origin_SURVEY'] != 'BIKE']
    surveys_generated = surveys_generated[surveys_generated['travelAggregation'] != 'MIXED_CAR_AND_OTHER']
    surveys_generated = surveys_generated[surveys_generated['travelAggregation'] != 'MULTIMODE']
    surveys_generated = surveys_generated[surveys_generated['travelAggregation'] != 'MIXED_BIKE_AND_OTHER']

    surveys_generated.loc[surveys_generated['travelAggregation'] == 'PRIVATE_BIKE', 'travelAggregation'] = 'BIKE'
    surveys_generated.loc[surveys_generated['travelAggregation'] == 'CITY_BIKE', 'travelAggregation'] = 'BIKE'

    max_number_of_surveys = int(len(surveys_train) / 0.05)
    surveys_generated = surveys_generated.sample(max_number_of_surveys)

    common_elements = list(np.intersect1d(list(surveys.columns), list(surveys_generated.columns)))
    common_elements = list(np.intersect1d(common_elements, list(include)))
    print("Dataset {}. Found {} of surveys ({} attributes) and {} generated ({} attributes). {} attributes are common"
          .format(dataset_file_name, len(surveys), len(surveys.columns), len(surveys_generated), len(surveys_generated.columns), len(common_elements)))
    common_elements.remove('travelAggregation')
    common_elements.append('travelAggregation')
    common_elements.reverse()

    surveys = surveys[common_elements]
    # surveys_w1 = surveys_w1[common_elements]
    surveys_generated.loc[:, 'series_INTERNAL'] = 'GENERATED'
    surveys_generated = surveys_generated[common_elements]

    # for col in exclude:
    #     if col in common_elements:
    #         common_elements.remove(col)

    learn_surveys, test_surveys = learn_and_test_surveys(surveys)
    # learn_surveys_w1, test_surveys_w1 = learn_and_test_surveys(surveys_w1)

    surveys_generated = mix_train_with_generated(learn_surveys, surveys_generated)

    print(directory)

    survey_to_arff(learn_surveys[common_elements], test_surveys[common_elements], surveys_generated[common_elements], directory, shuffle=shuffle)


def w1_w2_5_1_1_to_arff():
    citizens_w1_w2 = data.citizens_W1_W2_5_1_1()

    include = [
        # 'id_SURVEY',
        'travelAggregation',
        'C6H6_2_0_ENV', 'O3_2_0_ENV', 'PM25_2_0_ENV', 'PM10_2_0_ENV', 'NO2_2_0_ENV',
        'CO_2_0_ENV', 'C6H6_24_0_ENV', 'O3_24_0_ENV', 'PM25_24_0_ENV', 'PM10_24_0_ENV', 'NO2_24_0_ENV', 'CO_24_0_ENV',
        'temperature_2_0_ENV', 'rain_volume_2_0_ENV', 'windspeed_2_0_ENV', 'cloudiness_2_0_ENV', 'no_fall_2_0_ENV',
        'fog_2_0_ENV', 'rain_2_0_ENV', 'snow_2_0_ENV', 'hail_2_0_ENV', 'temperature_24_0_ENV', 'rain_volume_24_0_ENV',
        'windspeed_24_0_ENV', 'cloudiness_24_0_ENV', 'no_fall_24_0_ENV', 'fog_24_0_ENV', 'rain_24_0_ENV',
        'snow_24_0_ENV', 'hail_24_0_ENV', 'MetroDistance_URBAN', 'BusDistance_URBAN', 'TramDistance_URBAN',
        'StopDistance_URBAN', 'LandEntropy_URBAN', 'RoadDensity_URBAN', 'AddressDensity_URBAN', 'GreenDensity_URBAN',
        'PopulationDensity_URBAN', 'BikeDistance_BIKE_RENTAL',
        # 'ClosestStation_BIKE_RENTAL',
        'Capacity_BIKE_RENTAL',
        'StationsIn1km_BIKE_RENTAL', 'StationsIn2km_BIKE_RENTAL', 'minCurrentStopDelay_LOW_TRANSIT',
        'avgCurrentStopDelay_LOW_TRANSIT', 'maxCurrentStopDelay_LOW_TRANSIT', 'minDeltaDelay_LOW_TRANSIT',
        'avgDeltaDelay_LOW_TRANSIT', 'maxDeltaDelay_LOW_TRANSIT', 'minAverageSpeedInKmPerH_LOW_TRANSIT',
        'avgAverageSpeedInKmPerH_LOW_TRANSIT', 'maxAverageSpeedInKmPerH_LOW_TRANSIT',
        'minCountShortStopsT30_LOW_TRANSIT', 'avgCountShortStopsT30_LOW_TRANSIT', 'maxCountShortStopsT30_LOW_TRANSIT',
        'minCountLongStopsT30_LOW_TRANSIT', 'avgCountLongStopsT30_LOW_TRANSIT', 'maxCountLongStopsT30_LOW_TRANSIT',
        'minCountShortStopsT60_LOW_TRANSIT', 'avgCountShortStopsT60_LOW_TRANSIT', 'maxCountShortStopsT60_LOW_TRANSIT',
        'minCountLongStopsT60_LOW_TRANSIT', 'avgCountLongStopsT60_LOW_TRANSIT', 'maxCountLongStopsT60_LOW_TRANSIT',
        'minCountShortStopsT90_LOW_TRANSIT', 'avgCountShortStopsT90_LOW_TRANSIT', 'maxCountShortStopsT90_LOW_TRANSIT',
        'minCountLongStopsT90_LOW_TRANSIT', 'avgCountLongStopsT90_LOW_TRANSIT', 'maxCountLongStopsT90_LOW_TRANSIT',
        'minCountShortStopsT120_LOW_TRANSIT', 'avgCountShortStopsT120_LOW_TRANSIT',
        'maxCountShortStopsT120_LOW_TRANSIT', 'minCountLongStopsT120_LOW_TRANSIT', 'avgCountLongStopsT120_LOW_TRANSIT',
        'maxCountLongStopsT120_LOW_TRANSIT', 'ParkingCost_CAR', 'ParkingDifficultyIntersectionArea_CAR',
        'ParkingDifficultyIntersectionCentroid_CAR', 'ParkingDifficultyZoneCentroid_CAR', 'sex_SURVEY',
        'yearOfBirth_SURVEY', 'monthOfBirth_SURVEY', 'education_SURVEY', 'hasBike_SURVEY', 'carAvailability_SURVEY',
        'cyclingLimitations_SURVEY',
        # 'homeAddressGeocoding_SURVEY',
        'homeAddressLatitude_SURVEY',
        # 'homeAddressLongitude_SURVEY', 'describedDay_SURVEY', 'startingAtHome_SURVEY'
        # 'startingAddressGeocoding_SURVEY',
        'startingAddressLatitude_SURVEY', 'startingAddressLongitude_SURVEY',
        'childrenNumber_SURVEY', 'drivingLicence_SURVEY', 'usingCar_SURVEY', 'parking_SURVEY', 'cityCard_SURVEY',
        'varsovianCard_SURVEY', 'income_SURVEY', 'carsNumber_SURVEY', 'scooterNumber_SURVEY', 'bicycleNumber_SURVEY',
        'motorNumber_SURVEY', 'exoticMeansNumber_SURVEY', 'householdMembers_SURVEY', 'householdBelow15_SURVEY',
        'beliefsWarsawS1_SURVEY', 'beliefsWarsawS2_SURVEY', 'beliefsWarsawS3_SURVEY', 'beliefsWarsawS4_SURVEY',
        'beliefsWarsawS5_SURVEY', 'beliefsWarsawS6_SURVEY', 'beliefsWarsawS7_SURVEY', 'beliefsWarsawS8_SURVEY',
        'beliefsWarsawS9_SURVEY', 'beliefsWarsawS10_SURVEY', 'beliefsWarsawS11_SURVEY', 'beliefsWarsawS12_SURVEY',
        'beliefsWarsawS13_SURVEY', 'occupationMainS1_SURVEY', 'occupationMainS2_SURVEY', 'carFrequencyWarsaw_SURVEY',
        'bicycleFrequency_SURVEY', 'journeyReasonsWarsawI1_SURVEY', 'journeyReasonsWarsawI2_SURVEY',
        'journeyReasonsWarsawI3_SURVEY', 'journeyReasonsWarsawI4_SURVEY', 'journeyReasonsWarsawI5_SURVEY',
        'journeyReasonsWarsawI6_SURVEY', 'journeyReasonsWarsawI7_SURVEY', 'journeyReasonsWarsawI8_SURVEY',
        'journeyReasonsWarsawI9_SURVEY', 'journeyReasonsWarsawI10_SURVEY', 'journeyReasonsWarsawI11_SURVEY',
        'journeyReasonsWarsawI12_SURVEY', 'journeyReasonsWarsawI13_SURVEY', 'journeyReasonsWarsawI14_SURVEY',
        'journeyReasonsWarsawI15_SURVEY', 'journeyReasonsWarsawI16_SURVEY', 'factorsPublicTransportWarsawI1_SURVEY',
        'factorsPublicTransportWarsawI2_SURVEY', 'factorsPublicTransportWarsawI3_SURVEY',
        'factorsPublicTransportWarsawI4_SURVEY', 'factorsPublicTransportWarsawI5_SURVEY',
        'factorsPublicTransportWarsawI6_SURVEY', 'factorsPublicTransportWarsawI7_SURVEY',
        'factorsPublicTransportWarsawI8_SURVEY', 'factorsPublicTransportWarsawI9_SURVEY',
        'factorsPublicTransportWarsawI10_SURVEY', 'factorsPublicTransportWarsawI11_SURVEY',
        'factorsPublicTransportWarsawI12_SURVEY', 'factorsBicycleWarsawI1_SURVEY', 'factorsBicycleWarsawI2_SURVEY',
        'factorsBicycleWarsawI3_SURVEY', 'factorsBicycleWarsawI4_SURVEY', 'factorsBicycleWarsawI5_SURVEY',
        'factorsBicycleWarsawI6_SURVEY', 'factorsBicycleWarsawI7_SURVEY', 'factorsBicycleWarsawI8_SURVEY',
        'factorsBicycleWarsawI9_SURVEY', 'factorsBicycleWarsawI10_SURVEY', 'factorsBicycleWarsawI11_SURVEY',
        'factorsBicycleWarsawI12_SURVEY', 'factorsBicycleWarsawI13_SURVEY', 'factorsBicycleWarsawI14_SURVEY',
        'factorsBicycleWarsawI15_SURVEY', 'factorsBicycleWarsawI16_SURVEY', 'factorsWalkingWarsawI1_SURVEY',
        'factorsWalkingWarsawI2_SURVEY', 'factorsWalkingWarsawI3_SURVEY', 'factorsWalkingWarsawI4_SURVEY',
        'factorsWalkingWarsawI5_SURVEY', 'factorsWalkingWarsawI6_SURVEY', 'factorsWalkingWarsawI7_SURVEY',
        'factorsWalkingWarsawI8_SURVEY', 'factorsWalkingWarsawI9_SURVEY', 'factorsWalkingWarsawI10_SURVEY',
        'factorsWalkingWarsawI11_SURVEY', 'factorsWalkingWarsawI12_SURVEY', 'factorsWalkingWarsawI13_SURVEY',
        'carReduceNumberReasonWarsawI1_SURVEY', 'carReduceNumberReasonWarsawI2_SURVEY',
        'carReduceNumberReasonWarsawI3_SURVEY', 'carReduceNumberReasonWarsawI4_SURVEY',
        'carReduceNumberReasonWarsawI5_SURVEY', 'carReduceNumberReasonWarsawI6_SURVEY',
        'carReduceNumberReasonWarsawI7_SURVEY', 'carOwnershipReasonWarsawI1_SURVEY',
        'carOwnershipReasonWarsawI2_SURVEY', 'carOwnershipReasonWarsawI3_SURVEY', 'carOwnershipReasonWarsawI4_SURVEY',
        'carOwnershipReasonWarsawI5_SURVEY', 'carOwnershipReasonWarsawI6_SURVEY', 'carOwnershipReasonWarsawI7_SURVEY',
        'opinionsS1_SURVEY', 'opinionsS2_SURVEY', 'opinionsS3_SURVEY', 'opinionsS4_SURVEY', 'opinionsS5_SURVEY',
        'opinionsS6_SURVEY', 'howImportantTravelQualityS1_SURVEY', 'howImportantTravelQualityS2_SURVEY',
        'howImportantTravelQualityS3_SURVEY', 'howImportantTravelQualityS4_SURVEY',
        'howImportantTravelQualityS5_SURVEY', 'howImportantTravelQualityS6_SURVEY',
        'howImportantTravelQualityS7_SURVEY', 'howImportantTravelQualityS8_SURVEY',
        'howImportantTravelQualityS9_SURVEY', 'howImportantTravelQualityS10_SURVEY',
        'howImportantCustomerServiceS1_SURVEY', 'howImportantCustomerServiceS2_SURVEY',
        'howImportantCustomerServiceS3_SURVEY', 'howImportantServiceDeliveryS1_SURVEY',
        'howImportantServiceDeliveryS2_SURVEY', 'howImportantServiceDeliveryS3_SURVEY',
        'howImportantServiceDeliveryS4_SURVEY', 'howImportantServiceDeliveryS5_SURVEY',
        'howImportantServiceDeliveryS6_SURVEY', 'howImportantServiceDeliveryS7_SURVEY',
        'howImportantServiceDeliveryS8_SURVEY', 'howImportantWaitingConditionsS1_SURVEY',
        'howImportantWaitingConditionsS2_SURVEY', 'howImportantWaitingConditionsS3_SURVEY',
        'howImportantWaitingConditionsS4_SURVEY', 'howImportantTicketSalesS1_SURVEY',
        'howImportantTicketSalesS2_SURVEY', 'howImportantTicketSalesS3_SURVEY', 'howImportantChangesS1_SURVEY',
        'howImportantChangesS2_SURVEY', 'howImportantImageS1_SURVEY', 'bicycleMainReason_SURVEY', 'safetyCar_SURVEY',
        'safetyBicycle_SURVEY', 'safetyPublicTransport_SURVEY', 'safetyWalking_SURVEY',
        'drivingStyleDiscouragesCycling_SURVEY',
        # 'startingTime_SURVEY',
        'startingTime_SURVEY_SinceMidnight',
        'origin_SURVEY', 'destination_SURVEY',
        # 'localisationGeocoding_SURVEY',
        'localisationLatitude_SURVEY', 'localisationLongitude_SURVEY',
        'transport_SURVEY',
        # 'lineNumber_SURVEY',
        'peopleInCar_SURVEY', 'additionalStops_SURVEY',
        # 'endingTime_SURVEY',
        'endingTime_SURVEY_SinceMidnight',
        'parkingTime_SURVEY', 'nextTravel_SURVEY',
        # 'travelId',
        'travelAggregation',
        # 'date_OTP',
        'time_OTP_SinceMidnight',
        # 'time_OTP',
        'endTime_OTP_SinceMidnight',
        # 'endTime_OTP',
        # 'fromPlaceGeocoding_OTP',
        'fromPlaceLongitude_OTP', 'fromPlaceLatitude_OTP',
        # 'toPlaceGeocoding_OTP',
        'toPlaceLongitude_OTP', 'toPlaceLatitude_OTP', 'distance_OTP', 'Starting_ZONE',
        'Ending_ZONE', 'PlanType_WALK', 'Routes_WALK',
        # 'cnTransportModes_WALK',
        # 'cnPath_WALK',
        'Distance_WALK',
        'Duration_WALK', 'Speed_WALK', 'ElevationLost_WALK', 'ElevationGained_WALK', 'DistanceGrowth_WALK',
        'PlanType_TRANSIT', 'Routes_TRANSIT',
        # 'cnTransportModes_TRANSIT',
        # 'cnPath_TRANSIT',
        # 'FirstStopName_TRANSIT',
        'FirstStopGeocoding_TRANSIT', 'FirstStopLatitude_TRANSIT', 'FirstStopLongitude_TRANSIT', 'minStops_TRANSIT',
        'avgStops_TRANSIT', 'maxStops_TRANSIT', 'minDistance_TRANSIT', 'avgDistance_TRANSIT', 'maxDistance_TRANSIT',
        'minWalkDistance_TRANSIT', 'avgWalkDistance_TRANSIT', 'maxWalkDistance_TRANSIT', 'minWalkDuration_TRANSIT',
        'avgWalkDuration_TRANSIT', 'maxWalkDuration_TRANSIT', 'minWaitingTime_TRANSIT', 'avgWaitingTime_TRANSIT',
        'maxWaitingTime_TRANSIT', 'minDuration_TRANSIT', 'avgDuration_TRANSIT', 'maxDuration_TRANSIT',
        'minSpeed_TRANSIT', 'avgSpeed_TRANSIT', 'maxSpeed_TRANSIT', 'minTransfersNumber_TRANSIT',
        'avgTransfersNumber_TRANSIT', 'maxTransfersNumber_TRANSIT', 'minTransitTime_TRANSIT', 'avgTransitTime_TRANSIT',
        'maxTransitTime_TRANSIT', 'minPeriod_TRANSIT', 'avgPeriod_TRANSIT', 'maxPeriod_TRANSIT', 'minCost_TRANSIT',
        'avgCost_TRANSIT', 'maxCost_TRANSIT', 'minBusTime_TRANSIT', 'avgBusTime_TRANSIT', 'maxBusTime_TRANSIT',
        'minBusShare_TRANSIT', 'avgBusShare_TRANSIT', 'maxBusShare_TRANSIT', 'minTramTime_TRANSIT',
        'avgTramTime_TRANSIT', 'maxTramTime_TRANSIT', 'minTramShare_TRANSIT', 'avgTramShare_TRANSIT',
        'maxTramShare_TRANSIT', 'minRailTime_TRANSIT', 'avgRailTime_TRANSIT', 'maxRailTime_TRANSIT',
        'minRailShare_TRANSIT', 'avgRailShare_TRANSIT', 'maxRailShare_TRANSIT', 'minSubwayTime_TRANSIT',
        'avgSubwayTime_TRANSIT', 'maxSubwayTime_TRANSIT', 'minSubwayShare_TRANSIT', 'avgSubwayShare_TRANSIT',
        'maxSubwayShare_TRANSIT',
        # 'minDistanceGrowth_TRANSIT', cos popsute wartości
        'avgDistanceGrowth_TRANSIT',
        'maxDistanceGrowth_TRANSIT', 'PlanType_TRANSIT_REAL', 'Routes_TRANSIT_REAL',
        # 'cnTransportModes_TRANSIT_REAL',
        # 'cnPath_TRANSIT_REAL',
        # 'FirstStopName_TRANSIT_REAL',
        'FirstStopGeocoding_TRANSIT_REAL',
        'FirstStopLatitude_TRANSIT_REAL', 'FirstStopLongitude_TRANSIT_REAL', 'minStops_TRANSIT_REAL',
        'avgStops_TRANSIT_REAL', 'maxStops_TRANSIT_REAL', 'minDistance_TRANSIT_REAL', 'avgDistance_TRANSIT_REAL',
        'maxDistance_TRANSIT_REAL', 'minWalkDistance_TRANSIT_REAL', 'avgWalkDistance_TRANSIT_REAL',
        'maxWalkDistance_TRANSIT_REAL', 'minWalkDuration_TRANSIT_REAL', 'avgWalkDuration_TRANSIT_REAL',
        'maxWalkDuration_TRANSIT_REAL', 'minWaitingTime_TRANSIT_REAL', 'avgWaitingTime_TRANSIT_REAL',
        'maxWaitingTime_TRANSIT_REAL', 'minDuration_TRANSIT_REAL', 'avgDuration_TRANSIT_REAL',
        'maxDuration_TRANSIT_REAL', 'minSpeed_TRANSIT_REAL', 'avgSpeed_TRANSIT_REAL', 'maxSpeed_TRANSIT_REAL',
        'minTransfersNumber_TRANSIT_REAL', 'avgTransfersNumber_TRANSIT_REAL', 'maxTransfersNumber_TRANSIT_REAL',
        'minTransitTime_TRANSIT_REAL', 'avgTransitTime_TRANSIT_REAL', 'maxTransitTime_TRANSIT_REAL',
        'minPeriod_TRANSIT_REAL', 'avgPeriod_TRANSIT_REAL', 'maxPeriod_TRANSIT_REAL', 'minCost_TRANSIT_REAL',
        'avgCost_TRANSIT_REAL', 'maxCost_TRANSIT_REAL', 'minBusTime_TRANSIT_REAL', 'avgBusTime_TRANSIT_REAL',
        'maxBusTime_TRANSIT_REAL', 'minBusShare_TRANSIT_REAL', 'avgBusShare_TRANSIT_REAL', 'maxBusShare_TRANSIT_REAL',
        'minTramTime_TRANSIT_REAL', 'avgTramTime_TRANSIT_REAL', 'maxTramTime_TRANSIT_REAL',
        'minTramShare_TRANSIT_REAL', 'avgTramShare_TRANSIT_REAL', 'maxTramShare_TRANSIT_REAL',
        'minRailTime_TRANSIT_REAL', 'avgRailTime_TRANSIT_REAL', 'maxRailTime_TRANSIT_REAL',
        'minRailShare_TRANSIT_REAL', 'avgRailShare_TRANSIT_REAL', 'maxRailShare_TRANSIT_REAL',
        'minSubwayTime_TRANSIT_REAL', 'avgSubwayTime_TRANSIT_REAL', 'maxSubwayTime_TRANSIT_REAL',
        'minSubwayShare_TRANSIT_REAL', 'avgSubwayShare_TRANSIT_REAL', 'maxSubwayShare_TRANSIT_REAL',
        # 'minDistanceGrowth_TRANSIT_REAL', Tutaj są dziwne wartości #?NAME
        'avgDistanceGrowth_TRANSIT_REAL', 'maxDistanceGrowth_TRANSIT_REAL',
        'PlanType_BICYCLE', 'Routes_BICYCLE',
        # 'cnTransportModes_BICYCLE',
        # 'cnPath_BICYCLE',
        'Distance_BICYCLE',
        'Duration_BICYCLE', 'Speed_BICYCLE', 'WalkDistance_BICYCLE', 'WalkDuration_BICYCLE', 'ElevationLost_BICYCLE',
        'ElevationGained_BICYCLE', 'DistanceGrowth_BICYCLE', 'PlanType_CAR', 'Routes_CAR',
        # 'cnTransportModes_CAR',
        # 'cnPath_CAR',
        'Duration_CAR', 'Distance_CAR', 'Speed_CAR', 'WalkDistance_CAR', 'WalkDuration_CAR',
        'DistanceGrowth_CAR', 'DurationInTraffic_CAR', 'SpeedInTraffic_CAR', 'minDurationDifferenceCarToTransit_DIFF',
        'avgDurationDifferenceCarToTransit_DIFF', 'maxDurationDifferenceCarToTransit_DIFF',
        'minDurationRatioCarToTransit_DIFF', 'avgDurationRatioCarToTransit_DIFF', 'maxDurationRatioCarToTransit_DIFF',
        'minSpeedDifferenceCarToTransit_DIFF', 'avgSpeedDifferenceCarToTransit_DIFF',
        'maxSpeedDifferenceCarToTransit_DIFF', 'minSpeedRatioCarToTransit_DIFF', 'avgSpeedRatioCarToTransit_DIFF',
        'maxSpeedRatioCarToTransit_DIFF', 'minDistanceDifferenceCarToTransit_DIFF',
        'avgDistanceDifferenceCarToTransit_DIFF', 'maxDistanceDifferenceCarToTransit_DIFF',
        'minDistanceRatioCarToTransit_DIFF', 'avgDistanceRatioCarToTransit_DIFF', 'maxDistanceRatioCarToTransit_DIFF',
        'minDurationInTrafficDifferenceCarToTransit_DIFF', 'avgDurationInTrafficDifferenceCarToTransit_DIFF',
        'maxDurationInTrafficDifferenceCarToTransit_DIFF', 'minDurationInTrafficRatioCarToTransit_DIFF',
        'avgDurationInTrafficRatioCarToTransit_DIFF', 'maxDurationInTrafficRatioCarToTransit_DIFF',
        'minSpeedInTrafficDifferenceCarToTransit_DIFF', 'avgSpeedInTrafficDifferenceCarToTransit_DIFF',
        'maxSpeedInTrafficDifferenceCarToTransit_DIFF', 'minSpeedInTrafficRatioCarToTransit_DIFF',
        'avgSpeedInTrafficRatioCarToTransit_DIFF', 'maxSpeedInTrafficRatioCarToTransit_DIFF',
        'minDurationDifferenceCarToTransitReal_DIFF', 'avgDurationDifferenceCarToTransitReal_DIFF',
        'maxDurationDifferenceCarToTransitReal_DIFF', 'minDurationRatioCarToTransitReal_DIFF',
        'avgDurationRatioCarToTransitReal_DIFF', 'maxDurationRatioCarToTransitReal_DIFF',
        'minSpeedDifferenceCarToTransitReal_DIFF', 'avgSpeedDifferenceCarToTransitReal_DIFF',
        'maxSpeedDifferenceCarToTransitReal_DIFF', 'minSpeedRatioCarToTransitReal_DIFF',
        'avgSpeedRatioCarToTransitReal_DIFF', 'maxSpeedRatioCarToTransitReal_DIFF',
        'minDistanceDifferenceCarToTransitReal_DIFF', 'avgDistanceDifferenceCarToTransitReal_DIFF',
        'maxDistanceDifferenceCarToTransitReal_DIFF', 'minDistanceRatioCarToTransitReal_DIFF',
        'avgDistanceRatioCarToTransitReal_DIFF', 'maxDistanceRatioCarToTransitReal_DIFF',
        'minDurationInTrafficDifferenceCarToTransitReal_DIFF', 'avgDurationInTrafficDifferenceCarToTransitReal_DIFF',
        'maxDurationInTrafficDifferenceCarToTransitReal_DIFF', 'minDurationInTrafficRatioCarToTransitReal_DIFF',
        'avgDurationInTrafficRatioCarToTransitReal_DIFF', 'maxDurationInTrafficRatioCarToTransitReal_DIFF',
        'minSpeedInTrafficDifferenceCarToTransitReal_DIFF', 'avgSpeedInTrafficDifferenceCarToTransitReal_DIFF',
        'maxSpeedInTrafficDifferenceCarToTransitReal_DIFF', 'minSpeedInTrafficRatioCarToTransitReal_DIFF',
        'avgSpeedInTrafficRatioCarToTransitReal_DIFF', 'maxSpeedInTrafficRatioCarToTransitReal_DIFF',
        'DurationDifferenceCarToBicycle_DIFF', 'DurationRatioCarToBicycle_DIFF', 'SpeedDifferenceCarToBicycle_DIFF',
        'maxDurationDifferenceCarToTransitReal_DIFF', 'minDurationRatioCarToTransitReal_DIFF',
        'avgDurationRatioCarToTransitReal_DIFF', 'maxDurationRatioCarToTransitReal_DIFF',
        'minSpeedDifferenceCarToTransitReal_DIFF', 'avgSpeedDifferenceCarToTransitReal_DIFF',
        'maxSpeedDifferenceCarToTransitReal_DIFF', 'minSpeedRatioCarToTransitReal_DIFF',
        'avgSpeedRatioCarToTransitReal_DIFF', 'maxSpeedRatioCarToTransitReal_DIFF',
        'minDistanceDifferenceCarToTransitReal_DIFF', 'avgDistanceDifferenceCarToTransitReal_DIFF',
        'maxDistanceDifferenceCarToTransitReal_DIFF', 'minDistanceRatioCarToTransitReal_DIFF',
        'avgDistanceRatioCarToTransitReal_DIFF', 'maxDistanceRatioCarToTransitReal_DIFF',
        'minDurationInTrafficDifferenceCarToTransitReal_DIFF', 'avgDurationInTrafficDifferenceCarToTransitReal_DIFF',
        'maxDurationInTrafficDifferenceCarToTransitReal_DIFF', 'minDurationInTrafficRatioCarToTransitReal_DIFF',
        'avgDurationInTrafficRatioCarToTransitReal_DIFF', 'maxDurationInTrafficRatioCarToTransitReal_DIFF',
        'minSpeedInTrafficDifferenceCarToTransitReal_DIFF', 'avgSpeedInTrafficDifferenceCarToTransitReal_DIFF',
        'maxSpeedInTrafficDifferenceCarToTransitReal_DIFF', 'minSpeedInTrafficRatioCarToTransitReal_DIFF',
        'avgSpeedInTrafficRatioCarToTransitReal_DIFF', 'maxSpeedInTrafficRatioCarToTransitReal_DIFF',
        'DurationDifferenceCarToBicycle_DIFF', 'DurationRatioCarToBicycle_DIFF', 'SpeedDifferenceCarToBicycle_DIFF',
        'maxDurationDifferenceCarToTransitReal_DIFF', 'minDurationRatioCarToTransitReal_DIFF',
        'avgDurationRatioCarToTransitReal_DIFF', 'maxDurationRatioCarToTransitReal_DIFF',
        'minSpeedDifferenceCarToTransitReal_DIFF', 'avgSpeedDifferenceCarToTransitReal_DIFF',
        'maxSpeedDifferenceCarToTransitReal_DIFF', 'minSpeedRatioCarToTransitReal_DIFF',
        'avgSpeedRatioCarToTransitReal_DIFF', 'maxSpeedRatioCarToTransitReal_DIFF',
        'minDistanceDifferenceCarToTransitReal_DIFF', 'avgDistanceDifferenceCarToTransitReal_DIFF',
        'maxDistanceDifferenceCarToTransitReal_DIFF', 'minDistanceRatioCarToTransitReal_DIFF',
        'avgDistanceRatioCarToTransitReal_DIFF', 'maxDistanceRatioCarToTransitReal_DIFF',
        'minDurationInTrafficDifferenceCarToTransitReal_DIFF', 'avgDurationInTrafficDifferenceCarToTransitReal_DIFF',
        'maxDurationInTrafficDifferenceCarToTransitReal_DIFF', 'minDurationInTrafficRatioCarToTransitReal_DIFF',
        'avgDurationInTrafficRatioCarToTransitReal_DIFF', 'maxDurationInTrafficRatioCarToTransitReal_DIFF',
        'minSpeedInTrafficDifferenceCarToTransitReal_DIFF', 'avgSpeedInTrafficDifferenceCarToTransitReal_DIFF',
        'maxSpeedInTrafficDifferenceCarToTransitReal_DIFF', 'minSpeedInTrafficRatioCarToTransitReal_DIFF',
        'avgSpeedInTrafficRatioCarToTransitReal_DIFF', 'maxSpeedInTrafficRatioCarToTransitReal_DIFF',
        'DurationDifferenceCarToBicycle_DIFF', 'DurationRatioCarToBicycle_DIFF', 'SpeedDifferenceCarToBicycle_DIFF',
        'maxDurationDifferenceCarToTransitReal_DIFF', 'minDurationRatioCarToTransitReal_DIFF',
        'avgDurationRatioCarToTransitReal_DIFF', 'maxDurationRatioCarToTransitReal_DIFF',
        'minSpeedDifferenceCarToTransitReal_DIFF', 'avgSpeedDifferenceCarToTransitReal_DIFF',
        'maxSpeedDifferenceCarToTransitReal_DIFF', 'minSpeedRatioCarToTransitReal_DIFF',
        'avgSpeedRatioCarToTransitReal_DIFF', 'maxSpeedRatioCarToTransitReal_DIFF',
        'minDistanceDifferenceCarToTransitReal_DIFF', 'avgDistanceDifferenceCarToTransitReal_DIFF',
        'maxDistanceDifferenceCarToTransitReal_DIFF', 'minDistanceRatioCarToTransitReal_DIFF',
        'avgDistanceRatioCarToTransitReal_DIFF', 'maxDistanceRatioCarToTransitReal_DIFF',
        'minDurationInTrafficDifferenceCarToTransitReal_DIFF', 'avgDurationInTrafficDifferenceCarToTransitReal_DIFF',
        'maxDurationInTrafficDifferenceCarToTransitReal_DIFF', 'minDurationInTrafficRatioCarToTransitReal_DIFF',
        'avgDurationInTrafficRatioCarToTransitReal_DIFF', 'maxDurationInTrafficRatioCarToTransitReal_DIFF',
        'minSpeedInTrafficDifferenceCarToTransitReal_DIFF', 'avgSpeedInTrafficDifferenceCarToTransitReal_DIFF',
        'maxSpeedInTrafficDifferenceCarToTransitReal_DIFF', 'minSpeedInTrafficRatioCarToTransitReal_DIFF',
        'avgSpeedInTrafficRatioCarToTransitReal_DIFF', 'maxSpeedInTrafficRatioCarToTransitReal_DIFF',
        'DurationDifferenceCarToBicycle_DIFF', 'DurationRatioCarToBicycle_DIFF', 'SpeedDifferenceCarToBicycle_DIFF',
        'SpeedRatioCarToBicycle_DIFF', 'DistanceDifferenceCarToBicycle_DIFF', 'DistanceRatioCarToBicycle_DIFF',
        'DurationInTrafficDifferenceCarToBicycle_DIFF', 'DurationInTrafficRatioCarToBicycle_DIFF',
        'SpeedInTrafficDifferenceCarToBicycle_DIFF', 'SpeedInTrafficRatioCarToBicycle_DIFF',
        'DurationDifferenceCarToWalk_DIFF', 'DurationRatioCarToWalk_DIFF', 'SpeedDifferenceCarToWalk_DIFF',
        'SpeedRatioCarToWalk_DIFF', 'DistanceDifferenceCarToWalk_DIFF', 'DistanceRatioCarToWalk_DIFF',
        'DurationInTrafficDifferenceCarToWalk_DIFF', 'DurationInTrafficRatioCarToWalk_DIFF',
        'SpeedInTrafficDifferenceCarToWalk_DIFF', 'SpeedInTrafficRatioCarToWalk_DIFF', 'hourSlot_SURVEY',
        'minuteSlot_SURVEY', 'minutesSinceMidnight_SURVEY', 'series_INTERNAL']
    citizens_w1_w2 = citizens_w1_w2[include]
    save_w1_w2_5_1_1_as_arff(citizens_w1_w2)
