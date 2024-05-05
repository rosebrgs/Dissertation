import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from PIL import Image
from config import dims_dict


def preprocess(array):
    # Normalizes the supplied array and reshapes it into the appropriate format.
    array = array.astype("float64") / 255.0
    array = np.expand_dims(array, axis=3)
    return array


def noise(array, noise_factor=0.4, seed=None, gaussian=False, replacement_val=0):
    # Replace a fraction noise_factor of pixels with replacement_val or gaussian noise
    if seed is not None:
        np.random.seed(seed)
    shape = array.shape
    array = array.flatten()
    indices = np.random.choice(np.arange(array.size), replace=False,
                           size=int(array.size * noise_factor))
    if gaussian is True:
        array[indices] = np.random.normal(loc=0.5, scale=1.0, size=array[indices].shape)
    else:
        array[indices] = replacement_val
    array = array.reshape(shape)
    return np.clip(array, 0.0, 1.0)


def display(array1, array2, seed=None, title='Inputs and outputs of the model', n=10):
    hopfield = False
    
    dim = array1[0].shape[0]
    # Displays ten random images from each one of the supplied arrays.
    if seed is not None:
        np.random.seed(seed)
        
    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]

    fig = plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        if hopfield is True:
            plt.imshow(image1.reshape(dim, dim), cmap='binary', vmin=-1, vmax=1)
        else:
            plt.imshow(image1.reshape(dim, dim))
            plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        if hopfield is True:
            plt.imshow(image2.reshape(dim, dim), cmap='binary', vmin=-1, vmax=1)
        else:
            plt.imshow(image2.reshape(dim, dim))
            plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    fig.suptitle(title)
    plt.show()
    return fig


def load_tfds_dataset(dataset, num=10000, labels=False):
    dim = dims_dict[dataset]
    ds = tfds.load(dataset, split='train', shuffle_files=True, data_dir='./data/')
    ds_info = tfds.builder(dataset).info
    df = tfds.as_dataframe(ds.take(num), ds_info)
    new_train_data = np.empty([num, dim[0], dim[0]])
    test_data = df['image']
    train_data = df['image']
    key_dict = {
        'mnist': 'label',
        'kmnist': 'label',
        'fashion_mnist': 'label',
        'symmetric_solids': 'label_shape',
        'shapes3d': 'label_shape'
    }
    if labels is True:
        key = key_dict[dataset]
        test_labels = df[key].to_numpy()
        train_labels = df[key].to_numpy()

    # black and white datasets
    if dataset == 'mnist' or dataset == 'kmnist' or dataset == 'fashion_mnist':
        for ind, t in enumerate(train_data):
            im = Image.fromarray(t.reshape((28, 28))).resize((dim[0], dim[0]))
            new_train_data[ind] = np.asarray(im)
        new_test_data = np.empty([test_data.shape[0], dim[0], dim[0]])
        for ind, t in enumerate(test_data):
            im = Image.fromarray(t.reshape((28, 28))).resize((dim[0], dim[0]))
            new_test_data[ind] = np.asarray(im)
    # colour datasets
    if dataset == 'symmetric_solids' or dataset == 'shapes3d' or dataset == 'plant_village':
        for ind, t in enumerate(train_data):
            im = Image.fromarray(t).resize((dim[0], dim[0])).convert('L')
            new_train_data[ind] = np.asarray(im)
        new_test_data = np.empty([test_data.shape[0], dim[0], dim[0]])
        for ind, t in enumerate(test_data):
            im = Image.fromarray(t).resize((dim[0], dim[0])).convert('L')
            new_test_data[ind] = np.asarray(im)
    train_data = new_train_data
    test_data = new_test_data

    if labels is True:
        return train_data, test_data, train_labels, test_labels
    else:
        return train_data, test_data
    

def prepare_data(dataset, display=False, noise_factor=0.6, labels=False):

    if labels is True:
        train_data, test_data, train_labels, test_labels = load_tfds_dataset(dataset, labels=True)
    if labels is False:
        train_data, test_data = load_tfds_dataset(dataset, labels=False)

    # Normalize and reshape the data
    train_data = preprocess(train_data)
    test_data = preprocess(test_data)

    # Create a copy of the data with added noise
    noisy_train_data = noise(train_data, noise_factor=noise_factor)
    noisy_test_data = noise(test_data, noise_factor=noise_factor)

    # Display the train data and a version of it with added noise
    if display is True:
        display(train_data, noisy_train_data)

    if labels is True:
        return train_data, test_data, noisy_train_data, noisy_test_data, train_labels, test_labels
    if labels is False:
        return train_data, test_data, noisy_train_data, noisy_test_data
