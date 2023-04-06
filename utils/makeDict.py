import os


def lookForEnvi(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        for file in filenames:
            if file[-4:] != ".hdr":
                continue
            print('\t\'' + file[:-4] + '\',')


if __name__ == "__main__":
    lookForEnvi('E:/BUAA_Spetral_Data/hangzhou/envi/')
    #lookForEnvi('E:/BUAA_Spetral_Data/exp_20210112_car/envi/')