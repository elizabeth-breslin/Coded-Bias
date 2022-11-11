import os

path_images = "/Users/sisisnavely/Documents/DS4002/FacialRecognition/Other/"
new_path = "/Users/sisisnavely/Documents/DS4002/FacialRecognition/"

for image in os.listdir(path_images):
    if image.endswith('.jpg'):
        racial_code = image.split("_")[1]  # object_code is something like QRS Eur F699-1-2-2-180
        if not os.path.isdir(racial_code):  # Folder with this object_code does not yet exist, create it
            os.mkdir(racial_code)

        # Move the file to the folder with the object_code name
        os.rename(f"{path_images}/{image}", f"{new_path}/{racial_code}/{image}")

# https://stackoverflow.com/questions/67940644/segregate-files-into-folders-based-on-part-of-filename

