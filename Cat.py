from PIL import Image
from os.path import isfile, join, splitext
from os import listdir
from keras.models import load_model
import numpy as np


class ReduceImages(object):

    def __init__(self):
        self.extensions = ['.jpg', '.jpeg', '.png']
        self.sizex = 250
        self.sizey = 200

    def process_file(self,root,file,fullpath):
        #use libraty PIL just to transform the images
        try:
            with open(fullpath, 'r+b') as f:
                with Image.open(f) as image:
                    resized = image.resize((self.sizex,self.sizey))
                    resized.save(root+"/reduced/"+file)
                return True
        except Exception as e:
            print(e.message)
            return False

    def process_dir(self,path):
        """Processes files in the specified directory matching
        the self.extensions list (case-insensitively)."""

        filecount_ok = 0 # Number of files successfully updated
        filecount_ko = 0  # Number of files not successfully updated
        print('')
        print('Processing directory {}'.format(path))

        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        for file in onlyfiles:
            # Check file extensions against allowed list
            lowercasefile = file.lower()
            matches = False
            if (splitext(file)[1] in self.extensions):
                matches = True

            if matches:
                # File has eligible extension, so process
                fullpath = join(path, file)
                if self.process_file(path, file, fullpath):
                    filecount_ok = filecount_ok + 1
                else:
                    filecount_ko = filecount_ko + 1

        print('Number of images reduced successfully {}'.format(filecount_ok))
        print('Number of images reduced not successfully {}'.format(filecount_ko))
        print('Finished directory {}'.format(path))
        print('')

        return [filecount_ok,filecount_ko]

class DataSet(object):

    def __init__(self,path,sizex,sizey,channels):
        self.path     = path
        self.sizex    = sizex
        self.sizey    = sizey
        self.channels = channels
        self.extensions = ['.jpg', '.jpeg', '.png']


    def load_images_to_dataset(self):
        print('Loading images to datasets x,y...')
        files_names = [f for f in listdir(self.path) if isfile(join(self.path, f)) and splitext(join(self.path, f))[1]in self.extensions]
        number_files = len(files_names)

        aux_dataset_x = np.zeros((number_files, self.sizey, self.sizex, self.channels),dtype='uint8')

        for k in range(number_files):
            aux_dataset_x[k,:, :, :] = np.asarray(Image.open(self.path + "/" + files_names[k]))

        # normalize values
        dataset_x = aux_dataset_x / 255

        print('Finished load images to dataset')
        return dataset_x

class ExecuteModel():

    def __init__(self, image_path, models_path):
        self.image_path = image_path
        self.models_path = models_path

    def process_predit(self):
        # reduce the input image, for that has to be in a folder
        # take care to leave just one image and remove all recursive files from path!!!
        image = ReduceImages()
        image.process_dir(self.image_path)

        # read the image and load to numpy structure
        load_dataset = DataSet(self.image_path + '/reduced', image.sizex, image.sizey, 3)
        dataset = load_dataset.load_images_to_dataset()

        print('')
        print('Predicting cat.......')
        model_inception = load_model(self.models_path + '/cat_model_inception.h5')
        model_dense = load_model(self.models_path + '/cat_model.h5')
        print('Loading/compiling models')
        predict_inception = model_inception.predict(dataset)
        predict_dense = model_dense.predict(predict_inception)

        print('Probability: {}', format(predict_dense))

        print('Predicting cat....... OK')
        return np.argmax(predict_dense)




