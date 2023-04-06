import os


def mkdir(path):
    path = path.strip()
    path = path.rstrip('\\')
    path = path.rstrip('/')
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path,exist_ok=True)
        print("create dir(" + path + ") successfully")
        return True
    else:
        print("dir(" + path + ") is exist")
        return False

def judegHdrDataType(hdr_dirpath, file):
    with open(hdr_dirpath +"/"+ file + '.hdr', "r") as f:
        data = f.readlines()
    modify_flag = False
    if data[5] != 'data type = 12\n':
        data[5] = 'data type = 12\n'
        modify_flag = True
        # raise HdrDataTypeError("data type = 2, but data type should be 12")
    if modify_flag:
        with open(hdr_dirpath + "/" + file + '.hdr', "w") as f:
            f.writelines(data)
        print("mend the datatype of file : ", file)

if __name__ == "__main__":
    mkdir('./test_mkdir/')
    mkdir('test_mkdir\\')
