import sys
import os
from urllib.request import build_opener
from pathlib import Path

opener = build_opener()

start_year = int(sys.argv[1])
end_year = int(sys.argv[2])

def year_range(start_year, end_year):
    for year in range(start_year, end_year + 1):
        yield year

Path('nc-files').mkdir(parents=True, exist_ok=True)
os.chdir('./nc-files')
dspath = 'https://downloads.psl.noaa.gov//Datasets/ncep.reanalysis2/pressure/'

year_list = list(year_range(start_year, end_year))

variables = ['air', 'vwnd', 'uwnd']

for year in year_list:
    for variable in variables:
        Path(f'{variable}').mkdir(parents=True, exist_ok=True)
        os.chdir(f'./{variable}')
        file = f'{dspath}{variable}.{year}.nc'
        ofile = os.path.basename(file)
        sys.stdout.write(f'Downloading {ofile} \n')
        sys.stdout.flush()
        infile = opener.open(file)
        outfile = open(ofile, "wb")
        outfile.write(infile.read())
        outfile.close()
        os.chdir(f'./../')
print('All .nc files have been downloaded, please check nc-files folder!')