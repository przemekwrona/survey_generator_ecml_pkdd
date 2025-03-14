import pandas as pd


def drift_stream(data_frame, drift_data_frame):
    if data_frame.empty:
        return drift_data_frame

    if drift_data_frame.empty:
        return data_frame

    drift_data_frame = pd.concat([data_frame, drift_data_frame])
    drift_data_frame.index.name = 'survey_id'

    return drift_data_frame.sort_values(by=['minutesSinceMidnight', 'survey_id'])
