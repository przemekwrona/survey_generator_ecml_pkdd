import arff
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

from river import drift
from datetime import datetime, timedelta


def init():
    if not os.path.exists('generated'):
        os.mkdir('generated')

    if not os.path.exists('generated/drift'):
        os.mkdir('generated/drift')


def load_arff():
    arff_path = 'generated/semisupervised/citizens_W1_W2/{}'.format('warsaw_survey_0_init_classificator_0_generated.arff')

    # Load ARFF file
    with open(arff_path, 'r') as f:
        dataset = arff.load(f)

    # Convert to DataFrame
    surveys = pd.DataFrame(dataset['data'], columns=[attr[0] for attr in dataset['attributes']])

    return surveys


def adwin_detector(stream, attribute):
    drift_detector = drift.ADWIN()
    drifts = []

    for index, row in stream.iterrows():
        val = row[attribute]
        drift_detector.update(val)  # Data is processed one sample at a time
        if drift_detector.drift_detected:
            # The drift detector indicates after each sample if there is a drift in the data
            # print(f'Change detected at index {index}')
            drifts.append(row['time_OTP_SinceMidnight'])
            drift_detector = drift.ADWIN()  # As a best practice, we reset the detector

    return drifts


def run():
    init()
    surveys = load_arff()

    base_date = datetime(2024, 1, 1)

    # Convert to datetime.time
    surveys['time'] = surveys['time_OTP_SinceMidnight'].apply(lambda x: base_date + timedelta(minutes=x))

    m_to_km = 1000
    s_to_min = 60
    s_to_h = 1 * 60 * 60
    m_per_s_to_km_per_h = 1 / 3.6

    attributes = [
        ('Distance_WALK', 'Distance by walk [km]', 30, m_to_km),
        ('Distance_CAR', 'Distance by car [km]', 30, m_to_km),
        ('Distance_BICYCLE', 'Distance by bike [km]', 30, m_to_km),
        ('avgDistance_TRANSIT', 'Distance by public transport [km]', 30, m_to_km),

        ('Duration_WALK', 'Duration by walk [min]', 300, s_to_min),
        ('Duration_CAR', 'Duration by car [min]', 300, s_to_min),
        ('Duration_BICYCLE', 'Duration by bike [min]', 300, s_to_min),
        ('avgDuration_TRANSIT', 'Duration by public transport [min]', 300, s_to_min),

        ('Speed_WALK', 'Speed by walk [km/h]', 100, m_per_s_to_km_per_h),
        ('Speed_CAR', 'Speed by car [km/h]', 100, m_per_s_to_km_per_h),
        ('Speed_BICYCLE', 'Speed by bike [km/h]', 100, m_per_s_to_km_per_h),
        ('minSpeed_TRANSIT', 'Speed by public transport [km/h]', 100, m_per_s_to_km_per_h),

        ('avgStops_TRANSIT', 'Number of stops', 60, None),
        ('avgTransfersNumber_TRANSIT', 'Number of transfers', 5, None)
    ]

    for attribute, y_label, max_OY, series_divider in attributes:
        print("{} {}".format(attribute, y_label))
        drifts = adwin_detector(surveys, attribute)
        drift_times = pd.Series(drifts).apply(lambda x: base_date + timedelta(minutes=x))

        if series_divider is not None:
            surveys[attribute] = surveys[attribute] / series_divider

        x = np.arange(len(surveys))

        # Seaborn lineplot
        ax = sns.lineplot(x=x, y=attribute, data=surveys)

        # Format x-axis as HH:MM
        # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        for index, drift_detection in enumerate(drifts):
            plt.axvline(x=drift_detection, color='red', linestyle='--')

        plt.axvline(x=2961, color='black', linestyle='-.', label="End of dataset CW1")

        # Optional: Add legend and title
        custom_line = Line2D([0], [0], color='red', linestyle='--', label='Drift detection')
        custom_line_2 = Line2D([0], [0], color='black', linestyle='-.', label='End of dataset CW1')

        plt.legend(handles=[custom_line, custom_line_2], loc="upper left")

        # # Add titles and labels
        plt.xlabel('Instances')
        plt.ylabel(y_label)
        plt.grid()
        ax.set_xlim(xmin=0, xmax=len(surveys))

        if max_OY is not None:
            ax.set_ylim(ymin=0, ymax=max_OY)
        else:
            ax.set_ylim(ymin=0, y_max=6000)

        sns.set_theme(style="ticks")
        plt.savefig("generated/drift/drift_detection_{}.pdf".format(attribute))  # Save as PNG

        # Show plot
        plt.clf()
