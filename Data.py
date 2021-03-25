import os
from pathlib import Path
import shutil


def readFile(fileName):
    idFile = open(fileName, "r")  # opens the file in read mode
    id_list = idFile.read().splitlines()  # puts the file into an array
    idFile.close()

    return id_list

def main():

    # read ids.txt as a list
    idstxt_path = ''                         #for example :'val2017/ids.txt'
    ids_list = readFile(idstxt_path)

    # copy the image with id in ids_list to folder selectedData
    dataset = ''
    selectedData = ''

    for filename in os.listdir(dataset):

        #remove .jpg to see if its in ids_list
        name = Path(filename).with_suffix('')

        if str(name) in ids_list:
            path = dataset+str(filename)
            shutil.copy(path, selectedData)
        else:
            continue


main()