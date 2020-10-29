import requests
import os 
import pandas as pd
import shutil
import time

def download_url(url, save_dir, file_name, chunk_size=128): 
    '''
    Get the zip file from the url and save it to save_path
    params: url -> str
            save_path -> str
            
    '''
    assert isinstance(url,str) and isinstance(save_dir,str) and isinstance(file_name,str)
    assert os.path.exists(save_dir),'The save path given is %s does not exist'%(save_dir)
    save_path = os.path.join(save_dir,file_name)
    print("Saving File ",save_path)
    print("url fetching is ",url)
    r = requests.get(url, stream=True) 
    with open(save_path, 'wb') as fd: 
        for chunk in r.iter_content(): 
            fd.write(chunk)
    shutil.unpack_archive(save_path,save_path[:-4],'zip')
    csv_files = os.listdir(save_path[:-4])
    for c_file in csv_files:
        shutil.move(os.path.join(save_dir,save_path[:-4],c_file),os.path.join(save_dir,c_file))
    
    shutil.rmtree(os.path.join(save_dir,save_path[:-4]))
    os.remove(os.path.join(save_dir,save_path))

def get_demand_forecast(url,save_dir,tf):
    '''
    '''
    N = len(tf)
    tf.start_date = tf.start_date.dt.strftime('%Y%m%d')
    tf.end_date   = tf.end_date.dt.strftime('%Y%m%d')
    tf.end_date   = tf.end_date.apply(lambda i: str(int(i)-1))
    print(tf.head())
    for n in range(N):
        if n == 0:
            print("Url is ",url)
            print("Start Date ",tf.start_date[n])
            print("End Date ", tf.end_date[n])
            download_url(url,save_dir,'zip_file_%s.zip'%(tf.start_date[n]))
            print()
            print()
        else:
            url_ = url.replace(tf.start_date[0],tf.start_date[n]).replace(tf.end_date[0],tf.end_date[n])
            print("Start Date ",tf.start_date[n])
            print("End Date ", tf.end_date[n])
            print("Url gen is ",url_)
            download_url(url_,save_dir,'zip_file_%s.zip'%(tf.start_date[n]))
            print()
            print()

        time.sleep(5) #Without this the website does not respond!

if __name__ == '__main__':
    months = 3
    print("Getting the files from {} for next {} months".format('20160101',months))

    save_dir = '../data/' 
    url = 'http://oasis.caiso.com/oasisapi/SingleZip?queryname=SLD_FCST&market_run_id=DAM&resultformat=6&startdatetime=20160101T00:00-0000&enddatetime=20160130T23:00-0000&version=1'
    #download_url(url,save_dir,'temp.zip')
    tf = pd.DataFrame(columns=['end_date'],index=range(months))#data=[start_date,end_date])
    tf.end_date = pd.date_range('20160131',periods=months,freq='M') 
    tf['start_date'] = tf['end_date'].values.astype('datetime64[M]') 
    get_demand_forecast(url,save_dir,tf)
