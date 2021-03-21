import os
from pathlib import Path
import shutil


def readFile(fileName):
    idFile = open(fileName, "r")  # opens the file in read mode
    id_list = idFile.read().splitlines()  # puts the file into an array
    idFile.close()

    #print(id_list)
    #print(len(id_list))

    return id_list

def main():

    # read ids.txt as a list
    ids_list = readFile('val2017/ids.txt')

    # copy the image with id in ids_list to folder selectedData
    dataset = 'val2017/valdata'
    selectedData = 'val2017/bears'

    #print(ids_list)

    for filename in os.listdir(dataset):

        #remove .jpg to see if its in ids_list
        name = Path(filename).with_suffix('')

        if str(name) in ids_list:
            #print(name)
            #print(filename)
            path = 'val2017/valdata/'+str(filename)
            shutil.copy(path, 'val2017/bears')
        else:
            continue


main()