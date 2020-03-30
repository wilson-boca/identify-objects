import os

IMAGES_FOLDER = 'images/bees'
NAME = 'image'
EXTENSION = 'jpeg'
LENGTH = 3
CWD_PATH = os.getcwd()
FULL_PATH = os.path.join(CWD_PATH, IMAGES_FOLDER)

if os.path.isdir(FULL_PATH):
    for i, filename in enumerate(os.listdir(FULL_PATH)):
        new_filename = '{}{}.{}'.format(NAME, str(i+1).zfill(LENGTH), EXTENSION)
        if os.path.exists(FULL_PATH + "/" + new_filename):
            print('Skiping {}, the name already exist...'.format(filename))
            continue
        os.rename(FULL_PATH + "/" + filename, '{}/{}'.format(FULL_PATH, new_filename))
