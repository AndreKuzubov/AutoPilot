import imageio


def createGif(imageFileNames, saveFileName='/path/to/movie.gif', frameDuraction=0.2):
    with imageio.get_writer(saveFileName, mode='I', duration=frameDuraction) as writer:
        for filename in imageFileNames:
            image = imageio.imread(filename)
            writer.append_data(image)
