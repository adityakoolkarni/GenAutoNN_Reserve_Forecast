import requests
import os
import pandas as pd
import shutil
import time
import datetime


def call_api(url, write_dir, filename, chunk_size=128):
    assert isinstance(url, str)
    assert isinstance(write_dir, str)
    assert isinstance(filename, str)
    assert os.path.exists(write_dir), \
           f'The save path {write_dir} does not exist'

    save_path = os.path.join(write_dir, filename)

    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content():
            fd.write(chunk)

    shutil.unpack_archive(filename=save_path,
                          extract_dir=write_dir,
                          format='zip')


def make_tf(year0, month0, n_months):
    start_date = datetime.date(year=year0, month=month0, day=1)
    tf = pd.DataFrame(columns=['start_date', 'end_date'],
                      index=range(n_months))
    tf.start_date = (
        pd.date_range(start_date, periods=n_months, freq='MS')
        .strftime('%Y%m%d')
    )
    tf.end_date = (
        pd.date_range(start_date, periods=n_months, freq='M')
        .strftime('%Y%m%d')
    )
    # have to actually request less than a month of data to get reply
    tf.end_date = tf.end_date.apply(lambda i: str(int(i) - 1))

    return tf


def clean_dir(save_dir):
    files = os.listdir(save_dir)
    for file in files:
        if 'zip' in file:
            os.remove(os.path.join(save_dir, file))

    return 0


def main():
    # define start date for timeframe (tf)
    year0 = 2018
    month0 = 1
    n_months = 3
    # choose data to query
    query_type_dict = {
        'load': 'SLD_FCST',
        'solar_wind': 'SLD_REN_FCST'
    }
    query_name = query_type_dict['solar_wind']
    # pick market or process... not that ACTUAL is also an option
    market_run_id = 'DAM'
    # definte directory in which to store data
    write_dir = './data/'

    tf = make_tf(year0, month0, n_months)
    for i in range(n_months):
        start_date_str, end_date_str = tf.start_date[i], tf.end_date[i]
        filename = f'start_date_{start_date_str}.zip'
        url = (
            f'http://oasis.caiso.com/oasisapi/SingleZip'
            f'?queryname={query_name}'
            f'&market_run_id={market_run_id}'
            f'&resultformat=6'
            f'&startdatetime={start_date_str}T00:00-0000'
            f'&enddatetime={end_date_str}T23:00-0000&version=1'
        )
        call_api(url, write_dir, filename)
        time.sleep(5)

    clean_dir(write_dir)


if __name__ == '__main__':
    main()
