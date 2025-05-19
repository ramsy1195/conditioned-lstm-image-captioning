#!/usr/bin/env python
# coding: utf-8

# # Conditioned LSTM Language Model for Image Captioning
# Ramya Mohan

# Here are the components of the project:
# 
# * Part I: Creating encoded representations for the images in the flickr dataset using a pretrained image encoder(ResNet)
# * Part II: Preparing the input caption data.
# * Part III: Training an LSTM language model on the caption portion of the data and use it as a generator.
# * Part IV: Modifying the LSTM model to also pass a copy of the input image in each timestep.
# * Part V: Implementing beam search for the image caption generator.
# 
# Note: Access to a GPU is required.

# ### Getting Started
# 
# There are a few required packages.

# In[115]:


import os
import PIL # Python Image Library

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.models import ResNet18_Weights


# In[116]:


if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
    print("You won't be able to train the RNN decoder on a CPU, unfortunately.")
print(DEVICE)


# ### Access to the flickr8k data
# 
# We will use the flickr8k data set, described here in more detail:
# 
# > M. Hodosh, P. Young and J. Hockenmaier (2013) "Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics", Journal of Artificial Intelligence Research, Volume 47, pages 853-899 http://www.jair.org/papers/paper3994.html
# 
# If you are using Colab:
# * The data is available on google drive. You can access the folder here:
# https://drive.google.com/drive/folders/1sXWOLkmhpA1KFjVR0VjxGUtzAImIvU39?usp=sharing
# * Sharing is only enabled for the lionmail domain. 
# 
# N.B.: If you would like to experiment with the dataset beyond this project, I suggest that you submit your own download request here (it's free): https://forms.illinois.edu/sec/1713398
# 
# 

# In[3]:


# OPTIONAL (if not using Colab and the data in Google Drive): Download the data.
get_ipython().system('wget https://storage.googleapis.com/4705_fa24_hw3/hw3data.zip')


# In[4]:


#Then unzip the data
get_ipython().system('unzip hw3data.zip')


# The following variable should point to the location where the data is located.

# In[5]:


#this is where you put the name of your data folder.
#Please make sure it's correct because it'll be used in many places later.
MY_DATA_DIR="hw3data"


# ## Part I: Image Encodings 

# The files Flickr_8k.trainImages.txt Flickr_8k.devImages.txt Flickr_8k.testImages.txt, contain a list of training, development, and test images, respectively. Let's load these lists.

# In[6]:


def load_image_list(filename):
    with open(filename,'r') as image_list_f:
        return [line.strip() for line in image_list_f]


# In[7]:


FLICKR_PATH="hw3data/"


# In[8]:


train_list = load_image_list(os.path.join(FLICKR_PATH, 'Flickr_8k.trainImages.txt'))
dev_list = load_image_list(os.path.join(FLICKR_PATH,'Flickr_8k.devImages.txt'))
test_list = load_image_list(os.path.join(FLICKR_PATH,'Flickr_8k.testImages.txt'))


# Let's see how many images there are

# In[9]:


len(train_list), len(dev_list), len(test_list)


# Each entry is an image filename.

# In[10]:


dev_list[20]


# The images are located in a subdirectory.  

# In[11]:


IMG_PATH = os.path.join(FLICKR_PATH, "Flickr8k_Dataset")


# We can use PIL to open and display the image:

# In[12]:


image = PIL.Image.open(os.path.join(IMG_PATH, dev_list[20]))
image


# ### Preprocessing

# We are going to use an off-the-shelf pre-trained image encoder, the ResNet-18 network. Here is more detail about this model (not required for this project):
# 
# > Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun; Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778
# > https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
# 
# The model was initially trained on an object recognition task over the ImageNet1k data. The task is to predict the correct class label for an image, from a set of 1000 possible classes.
# 
# To feed the flickr images to ResNet, we need to perform the same normalization that was applied to the training images. More details here: https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html

# In[13]:


from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# The resulting images, after preprocessing, are (3,224,244) tensors, where the first dimension represents the three color channels, R,G,B).

# In[14]:


processed_image = preprocess(image)
processed_image.shape


# To the ResNet18 model, the images look like this:

# In[15]:


transforms.ToPILImage()(processed_image)


# ### Image Encoder
# Let's instantiate the ReseNet18 encoder. We are going to use the pretrained weights available in torchvision.

# In[16]:


img_encoder = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)


# In[17]:


img_encoder.eval()


# This is a prediction model,so the output is typically a softmax-activated vector representing 1000 possible object types. Because we are interested in an encoded representation of the image we are just going to use the second-to-last layer as a source of image encodings. Each image will be encoded as a vector of size 512.
# 
# We will use the following hack: remove the last layer, then reinstantiate a Squential model from the remaining layers.

# In[18]:


lastremoved = list(img_encoder.children())[:-1]
img_encoder = torch.nn.Sequential(*lastremoved).to(DEVICE) # also send it to GPU memory
img_encoder.eval()


# Let's try the encoder.

# In[19]:


def get_image(img_name):
    image = PIL.Image.open(os.path.join(IMG_PATH, img_name))
    return preprocess(image)


# In[20]:


preprocessed_image = get_image(train_list[0])
encoded = img_encoder(preprocessed_image.unsqueeze(0).to(DEVICE)) # unsqueeze required to add batch dim (3,224,224) becomes (1,3,224,224)
encoded.shape


# The result isn't quite what we wanted: The final representation is actually a 1x1 "image" (the first dimension is the batch size).
# We can just grab this one pixel:

# In[21]:


encoded = encoded[:,:,0,0] #this is our final image encoded
encoded.shape


# Because we are just using the pretrained encoder, we can simply encode all the images in a preliminary step. We will store them in one big tensor (one for each dataset, train, dev, test). This will save some time when training the conditioned LSTM because we won't have to recompute the image encodings with each training epoch. We can also save the tensors to disk so that we never have to touch the bulky image data again.
# 
# The following function takes a list of image names and returns a tensor of size [n_images, 512] (where each row represents one image).
# 
# For example `encode_imates(train_list)` should return a [6000,512] tensor.

# In[22]:


def encode_images(image_list):
    encoded_images = []

    with torch.no_grad():
        for img_name in image_list:
          preprocessed_image = get_image(img_name)

          encoded = img_encoder(preprocessed_image.unsqueeze(0).to(DEVICE))

          encoded = encoded.squeeze()

          encoded_images.append(encoded)

    encoded_images_tensor = torch.stack(encoded_images)

    return encoded_images_tensor

enc_images_train = encode_images(train_list)
enc_images_train.shape


# We can now save this to disk:

# In[23]:


torch.save(enc_images_train, open('encoded_images_train.pt','wb'))


# It's a good idea to save the resulting matrices, so we do not have to run the encoder again.

# ## Part II Text (Caption) Data Preparation 
# 
# Next, we need to load the image captions and generate training data for the language model. We will train a text-only model first.

# ### Reading image descriptions

# The following function reads the image descriptions from the file `filename` and returns a dictionary in the following format. Take a look at the file `Flickr8k.token.txt` for the format of the input file.
# The keys of the dictionary should be image filenames. Each value should be a list of 5 captions. Each caption should be a list of tokens.  
# 
# The captions in the file are already tokenized, so we can just split them at white spaces. We convert each token to lowercase and then pad each caption with a \<START\> token on the left and an \<END\> token on the right.
# 
# For example, a single caption might look like this:
# ['\<START\>',
#   'a',
#   'child',
#   'in',
#   'a',
#   'pink',
#   'dress',
#   'is',
#   'climbing',
#   'up',
#   'a',
#   'set',
#   'of',
#   'stairs',
#   'in',
#   'an',
#   'entry',
#   'way',
#   '.',
#   '\<EOS\>'],

# In[24]:


def read_image_descriptions(filename):
    image_descriptions = {}

    with open(filename,'r') as in_file:
        for line in in_file:
            line = line.strip()

            image_id, caption_with_index = line.split('#', 1)

            caption = caption_with_index.split('\t', 1)[1]

            tokens = caption.lower().split()

            tokens = ['<START>'] + tokens + ['<EOS>']

            if image_id not in image_descriptions:
                image_descriptions[image_id] = []

            image_descriptions[image_id].append(tokens)

    return image_descriptions


# In[25]:


os.path.join(FLICKR_PATH, "Flickr8k.token.txt")


# In[26]:


descriptions = read_image_descriptions(os.path.join(FLICKR_PATH, "Flickr8k.token.txt"))


# In[27]:


descriptions['1000268201_693b08cb0e.jpg']


# ### Creating Word Indices

# Next, we need to create a lookup table from the **training** data mapping words to integer indices, so we can encode input
# and output sequences using numeric representations.
# 
# Let us create the dictionaries id_to_word and word_to_id, which should map tokens to numeric ids and numeric ids to tokens.  
# We create a set of tokens in the training data first, then convert the set into a list and sort it. This way if the code is run multiple times, we will always get the same dictionaries. 
# 
# We also create word indices for the three special tokens `<PAD>`, `<START>`, and `<EOS>` (end of sentence).

# In[28]:


all_tokens = set()

for image_id, captions in descriptions.items():
    for caption in captions:
        all_tokens.update(caption)

sorted_tokens = sorted(all_tokens)

id_to_word = {}
id_to_word[0] = "<PAD>"
id_to_word[1] = "<START>"
id_to_word[2] = "<EOS>"
word_to_id = {}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<EOS>"] = 2

for idx, token in enumerate(sorted_tokens, start=3):
  if token != "<PAD>" and token != "<START>" and token != "<EOS>":
    id_to_word[idx] = token
    word_to_id[token] = idx


# In[29]:


word_to_id['cat'] # should print an integer


# In[30]:


id_to_word[1] # should print a token


# Note that we do not need an UNK word token because we will only use the model as a generator, once trained.

# ## Part III Basic Decoder Model
# 
# For now, we will just train a model for text generation without conditioning the generator on the image input.

# We will use the LSTM implementation provided by PyTorch. The core idea here is that the recurrent layers (including LSTM) create an "unrolled" RNN. Each time-step is represented as a different position, but the weights for these positions are shared. We are going to use the constant MAX_LEN to refer to the maximum length of a sequence, which turns out to be 40 words in this data set (including START and END).

# In[31]:


MAX_LEN = max(len(description) for image_id in train_list for description in descriptions[image_id])
MAX_LEN


# To train the model, we will convert each description into a set of input output pairs as follows. For example, consider the sequence
# 
# `['<START>', 'a', 'black', 'dog', '<EOS>']`
# 
# We would train the model using the following input/output pairs (note both sequences are padded to the right up to MAX_LEN)
# 
# | i | input                                 | output                              |
# |---|---------------------------------------|-------------------------------------|
# | 0 |[`<START>`,`<PAD>`,`<PAD>`,`<PAD>`,...]| [`a`,`<PAD>`,`<PAD>`,`<PAD>`,...    |  
# | 1 |[`<START>`,`a`,`<PAD>`,`<PAD>`,...]    | [`a`,`black`,`<PAD>`,`<PAD>`,...    |
# | 2 |[`<START>`,`a`,`black`,`<PAD>`,...]    | [`a`,`black`,`dog`,`<PAD>`,...      |
# | 3 |[`<START>`,`a`,`back`,`dog`,...]       | [`a`,`black`,`dog`,`<EOS>`,...      |

# Here is the lange model in pytorch. We will choose input embeddings of dimensionality 512 (for simplicitly, we are not initializing these with pre-trained embeddings here). We will also use 512 for the hidden state vector and the output.

# In[32]:


from torch import nn

vocab_size = len(word_to_id)+1
class GeneratorModel(nn.Module):

    def __init__(self):
        super(GeneratorModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 512)
        self.lstm = nn.LSTM(512, 512, num_layers = 1, bidirectional=False, batch_first=True)
        self.output = nn.Linear(512,vocab_size)

    def forward(self, input_seq):
        hidden = self.lstm(self.embedding(input_seq))
        out = self.output(hidden[0])
        return out


# The input sequence is an integer tensor of size `[batch_size, MAX_LEN]`. Each row is a vector of size MAX_LEN in which each entry is an integer representing a word (according to the `word_to_id` dictionary). If the input sequence is shorter than MAX_LEN, the remaining entries should be padded with '<PAD>'.
# 
# For each input example, the model returns a distribution over possible output words. The model output is a tensor of size `[batch_size, MAX_LEN, vocab_size]`. vocab_size is the number of vocabulary words, i.e. len(word_to_id)

# ### Creating a Dataset for the text training data

# Let us write a Dataset class for the text training data. The __getitem__ method returns an (input_encoding, output_encoding) pair for a single item. Both input_encoding and output_encoding are tensors of size `[MAX_LEN]`, encoding the padded input/output sequence as illustrated above.
# 
# We will first read in all captions in the __init__ method and store them in a list. Above, we used the get_image_descriptions function to load the image descriptions into a dictionary. Let us iterate through the images in img_list, then access the corresponding captions in the `descriptions` dictionary.
# 

# In[33]:


MAX_LEN = 40

class CaptionDataset(Dataset):

    def __init__(self, img_list):

        self.data = []

        for img_id in img_list:
            captions = descriptions[img_id]

            for caption in captions:
                input_seq = caption[:MAX_LEN]
                output_seq = caption[1:MAX_LEN+1]

                input_seq = input_seq + ['<PAD>'] * (MAX_LEN - len(input_seq))
                output_seq = output_seq + ['<PAD>'] * (MAX_LEN - len(output_seq))

                self.data.append((input_seq, output_seq))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,k):

        input_seq, output_seq = self.data[k]

        input_enc = torch.tensor([word_to_id[token] for token in input_seq], dtype=torch.long)
        output_enc = torch.tensor([word_to_id[token] for token in output_seq], dtype=torch.long)
        return input_enc, output_enc


# Let's instantiate the caption dataset and get the first item. We want to see something like this:
# 
# for the input:
# <pre>
# tensor([   1,   74,  805, 2312, 4015, 6488,  170,   74, 8686, 2312, 3922, 7922,
#         7125,   17,    2,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0])
# </pre>
# for the output:
# <pre>
#     tensor([  74,  805, 2312, 4015, 6488,  170,   74, 8686, 2312, 3922, 7922, 7125,
#           17,    2,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0])
# </pre>

# In[34]:


data = CaptionDataset(train_list)


# In[35]:


i, o = data[0]
i


# In[36]:


o


# Let's try the model:

# In[37]:


model = GeneratorModel().to(DEVICE)


# In[38]:


model(i.to(DEVICE)).shape   # should return a [40, vocab_size]  tensor.


# ### Training the Model

# In[39]:


from torch.nn import CrossEntropyLoss
loss_function = CrossEntropyLoss(ignore_index = 0, reduction='mean')

LEARNING_RATE = 1e-03
optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)

loader = DataLoader(data, batch_size = 16, shuffle = True)

def train():
    """
    Train the model for one epoch.
    """
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    total_correct, total_predictions = 0, 0
    tr_preds, tr_labels = [], []
    # put model in training mode
    model.train()

    for idx, batch in enumerate(loader):

        inputs,targets = batch
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        # Run the forward pass of the model
        logits = model(inputs)
        loss = loss_function(logits.transpose(2,1), targets)
        tr_loss += loss.item()
        #print("Batch loss: ", loss.item()) # can comment out if too verbose.
        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        # Calculate accuracy
        predictions = torch.argmax(logits, dim=2)  # Predicted token labels
        not_pads = targets != 0  # Mask for non-PAD tokens
        correct = torch.sum((predictions == targets) & not_pads)
        total_correct += correct.item()
        total_predictions += not_pads.sum().item()

        if idx % 100==0:
            #torch.cuda.empty_cache() # can help if you run into memory issues
            curr_avg_loss = tr_loss/nb_tr_steps
            print(f"Current average loss: {curr_avg_loss}")

        # Run the backward pass to update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy for this batch
        # matching = torch.sum(torch.argmax(logits,dim=2) == targets)
        # predictions = torch.sum(torch.where(targets==-100,0,1))

    epoch_loss = tr_loss / nb_tr_steps
    epoch_accuracy = total_correct / total_predictions if total_predictions != 0 else 0  # Avoid division by zero
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Average accuracy epoch: {epoch_accuracy:.2f}")


# Let us run the training until the accuracy reaches about 0.5 (this would be high for a language model on open-domain text, but the image caption dataset is comparatively small and closed-domain). This will take about 5 epochs.

# In[43]:


train()


# ### Greedy Decoder
# 
# Next, we will write a decoder. The decoder should start with the sequence `["<START>", "<PAD>","<PAD>"...]`, use the model to predict the most likely word in the next position. We append the word to the input sequence and then continue until `"<EOS>"` is predicted or the sequence reaches `MAX_LEN` words.

# In[44]:


def decoder():
    current_sequence = [word_to_id['<START>']] + [word_to_id['<PAD>']] * (MAX_LEN - 1)

    for step in range(MAX_LEN):
        input_tensor = torch.tensor([current_sequence]).to(DEVICE)

        logits = model(input_tensor)

        next_word_id = torch.argmax(logits[0, step, :]).item()

        for i in range(len(current_sequence)):
            if current_sequence[i] == word_to_id['<PAD>']:
                current_sequence[i] = next_word_id
                break

        if next_word_id == word_to_id['<EOS>']:
            break

    decoded_words = [id_to_word[word_id] for word_id in current_sequence]

    return decoded_words


# In[45]:


decoder()


# this will return something like
# ['a',
#  'man',
#  'in',
#  'a',
#  'white',
#  'shirt',
#  'and',
#  'a',
#  'woman',
#  'in',
#  'a',
#  'white',
#  'dress',
#  'walks',
#  'by',
#  'a',
#  'small',
#  'white',
#  'building',
#  '.',
#  '<EOS>']
# 
# This simple decoder will of course always predict the same sequence (and it's not necessarily a good one).
# 
# Let us modify the decoder as follows: instead of choosing the most likely word in each step, sample the next word from the distribution (i.e. the softmax activated output) returned by the model. We apply torch.softmax() to convert the output activations into a distribution.
# 
# To sample fromt he distribution, I recommend you take a look at [np.random.choice](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html), which takes the distribution as a parameter p.

# In[46]:


import numpy as np

def sample_decoder():
    current_sequence = [word_to_id['<START>']] + [word_to_id['<PAD>']] * (MAX_LEN - 1)

    for step in range(MAX_LEN):
        input_tensor = torch.tensor([current_sequence]).to(DEVICE)

        logits = model(input_tensor)

        probs = torch.softmax(logits[0, step, :], dim=0).cpu().detach().numpy()

        next_word_id = np.random.choice(range(vocab_size), p=probs)

        for i in range(len(current_sequence)):
            if current_sequence[i] == word_to_id['<PAD>']:
                current_sequence[i] = next_word_id
                break

        if next_word_id == word_to_id['<EOS>']:
            break

    decoded_words = [id_to_word[word_id] for word_id in current_sequence]

    return decoded_words

for i in range(5):
    print(sample_decoder())


# We are now able to see some interesting output that looks a lot like flickr8k image captions -- only that the captions are generated randomly without any image input.

# ## Part IV - Conditioning on the Image 

# We will now extend the model to condition the next word not only on the partial sequence, but also on the encoded image.
# 
# We will concatenate the 512-dimensional image representation to each 512-dimensional token embedding. The LSTM will therefore see input representations of size 1024.
# 
# Let us write a new Dataset class for the combined image captioning data set. Each call to __getitem__ should return a triple  (image_encoding, input_encoding, output_encoding) for a single item. Both input_encoding and output_encoding should be tensors of size [MAX_LEN], encoding the padded input/output sequence as illustrated above. The image_encoding is the size [512] tensor we pre-computed in part I.
# 
# Note: One tricky issue here is that each image corresponds to 5 captions, so we have to find the correct image for each caption. We can create a mapping from image names to row indices in the image encoding tensor. This way we will be able to find each image by its name.

# In[47]:


MAX_LEN = 40

class CaptionAndImage(Dataset):

    def __init__(self, img_list):

        self.img_data = torch.load(open("encoded_images_train.pt",'rb')) # suggested
        self.img_name_to_id = dict([(i,j) for (j,i) in enumerate(img_list)])

        self.data = []

        for img_name in img_list:
            img_id = self.img_name_to_id[img_name]

            captions = descriptions[img_name]

            for caption in captions:
                input_seq = caption[:MAX_LEN]
                output_seq = caption[1:MAX_LEN+1]

                input_seq = input_seq + ['<PAD>'] * (MAX_LEN - len(input_seq))
                output_seq = output_seq + ['<PAD>'] * (MAX_LEN - len(output_seq))

                self.data.append((img_id, input_seq, output_seq))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,k):

        img_id, input_seq, output_seq = self.data[k]

        img_data = self.img_data[img_id]
        input_enc = torch.tensor([word_to_id[token] for token in input_seq], dtype=torch.long)
        output_enc = torch.tensor([word_to_id[token] for token in output_seq], dtype=torch.long)

        return img_data, input_enc, output_enc


# In[48]:


data = CaptionAndImage(train_list)
img, i, o = data[0]
img.shape # should return torch.Size([512])


# In[49]:


i.shape # should return torch.Size([40])


# In[50]:


o.shape # should return torch.Size([40])


# **Updating the model**
# Let us update the language model code above to include a copy of the image for each position.
# The forward function of the new model takes two inputs:
#     
#    1. a `(batch_size, 2048)` ndarray of image encodings.
#    2. a `(batch_size, MAX_LEN)` ndarray of partial input sequences.
#     
# And one output as before: a `(batch_size, vocab_size)` ndarray of predicted word distributions.   
# 
# The LSTM will take input dimension 1024 instead of 512 (because we are concatenating the 512-dim image encoding).
# 
# In the forward function, we take the image and the embedded input sequence (i.e. AFTER the embedding was applied), and concatenate the image to each input. This requires some tensor manipulation. I recommend taking a look at [torch.Tensor.expand](https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html) and [torch.Tensor.cat](https://pytorch.org/docs/stable/generated/torch.Tensor.cat.html).
# 
# 

# In[51]:


import torch
from torch import nn
vocab_size = len(word_to_id)+1

class CaptionGeneratorModel(nn.Module):

    def __init__(self):
        super(CaptionGeneratorModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, 512)

        self.lstm = nn.LSTM(1024, 512, num_layers=1, bidirectional=False, batch_first=True)

        self.output = nn.Linear(512, vocab_size)

    def forward(self, img, input_seq):

        embedded_input = self.embedding(input_seq)

        img_expanded = img.unsqueeze(1).expand(-1, input_seq.size(1), -1)

        lstm_input = torch.cat((embedded_input, img_expanded), dim=2)

        lstm_out, _ = self.lstm(lstm_input)

        out = self.output(lstm_out)

        return out


# Let's try this new model on one item:

# In[52]:


model = CaptionGeneratorModel().to(DEVICE)


# In[53]:


item = data[0]
img, input_seq, output_seq = item


# In[54]:


logits = model(img.unsqueeze(0).to(DEVICE), input_seq.unsqueeze(0).to(DEVICE))

logits.shape # should return (1,40,8922) = (batch_size, MAX_LEN, vocab_size)


# The training function is, again, mostly unchanged. Keep training until the accuracy exceeds 0.5.

# In[55]:


from torch.nn import CrossEntropyLoss
loss_function = CrossEntropyLoss(ignore_index = 0, reduction='mean')

LEARNING_RATE = 1e-03
optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)

loader = DataLoader(data, batch_size = 16, shuffle = True)

def train():
    """
    Train the model for one epoch.
    """
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    total_correct, total_predictions = 0, 0
    tr_preds, tr_labels = [], []
    # put model in training mode
    model.train()

    for idx, batch in enumerate(loader):

        img, inputs,targets = batch
        img = img.to(DEVICE)
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        # Run the forward pass of the model
        logits = model(img, inputs)
        loss = loss_function(logits.transpose(2,1), targets)
        tr_loss += loss.item()
        #print("Batch loss: ", loss.item()) # can comment out if too verbose.
        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        # Calculate accuracy
        predictions = torch.argmax(logits, dim=2)  # Predicted token labels
        not_pads = targets != 0  # Mask for non-PAD tokens
        correct = torch.sum((predictions == targets) & not_pads)
        total_correct += correct.item()
        total_predictions += not_pads.sum().item()

        if idx % 100==0:
            #torch.cuda.empty_cache() # can help if you run into memory issues
            curr_avg_loss = tr_loss/nb_tr_steps
            print(f"Current average loss: {curr_avg_loss}")

        # Run the backward pass to update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy for this batch
        # matching = torch.sum(torch.argmax(logits,dim=2) == targets)
        # predictions = torch.sum(torch.where(targets==-100,0,1))

    epoch_loss = tr_loss / nb_tr_steps
    epoch_accuracy = total_correct / total_predictions if total_predictions != 0 else 0  # Avoid division by zero
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Average accuracy epoch: {epoch_accuracy:.2f}")


# In[60]:


train()


# **Testing the model**:
# Let us rewrite the greedy decoder from above to take an encoded image representation as input.

# In[97]:


def greedy_decoder(img):
    current_sequence = [word_to_id['<START>']]

    img_tensor = img.unsqueeze(0).to(DEVICE)

    for step in range(1, MAX_LEN):

        input_tensor = torch.tensor([current_sequence]).to(DEVICE)

        logits = model(img_tensor, input_tensor)

        probs = torch.softmax(logits[0, step - 1, :], dim=0).cpu().detach().numpy()

        next_word_id = np.random.choice(range(vocab_size), p=probs)

        current_sequence.append(next_word_id)

        if next_word_id == word_to_id['<EOS>']:
            break

    result = [id_to_word[word_id] for word_id in current_sequence]

    return result


# Now we can load one of the dev images, pass it through the preprocessor and the image encoder, and then into the decoder!

# In[101]:


raw_img = PIL.Image.open(os.path.join(IMG_PATH, dev_list[27]))
preprocessed_img = preprocess(raw_img).to(DEVICE)
encoded_img = img_encoder(preprocessed_img.unsqueeze(0)).reshape((512))
caption = greedy_decoder(encoded_img)
print(caption)
raw_img


# The result looks pretty good for most images, but the model is prone to hallucinations.

# ## Part V - Beam Search Decoder 

# Let us modify the simple greedy decoder for the caption generator to use beam search.
# Instead of always selecting the most probable word, we use a *beam*, which contains the n highest-scoring sequences so far and their total probability (i.e. the product of all word probabilities) using a list of `(probability, sequence)` tuples. After each time-step, we will prune the list to include only the n most probable sequences.
# 
# Then, for each sequence, we compute the n most likely successor words. We append the word to produce n new sequences and compute their score. This way, we create a new list of n*n candidates.
# 
# We prune this list to the best n as before and continue until `MAX_LEN` words have been generated.
# 
# Note that we cannot use the occurence of the `"<EOS>"` tag to terminate generation, because the tag may occur in different positions for different entries in the beam.
# 
# Once `MAX_LEN` has been reached, we return the most likely sequence out of the current n.

# In[87]:


def img_beam_decoder(n, img):

    beam = [(0, [word_to_id['<START>']])]

    img_tensor = img.unsqueeze(0).to(DEVICE)

    for step in range(MAX_LEN):
        candidates = []

        for prob, sequence in beam:
            if sequence[-1] == word_to_id['<EOS>']:
                candidates.append((prob, sequence))
                continue

            input_tensor = torch.tensor([sequence]).to(DEVICE)

            logits = model(img_tensor, input_tensor)

            probs = torch.softmax(logits[0, step, :], dim=0).cpu().detach().numpy()

            top_n_probs, top_n_ids = torch.topk(torch.tensor(probs), n)

            for i in range(n):
                next_word_id = top_n_ids[i].item()
                new_sequence = sequence + [next_word_id]
                new_prob = prob + np.log(top_n_probs[i].item())

                candidates.append((new_prob, new_sequence))

        beam = sorted(candidates, key=lambda x: x[0], reverse=True)[:n]

        if all(seq[-1] == word_to_id['<EOS>'] for _, seq in beam):
            break

    best_sequence = beam[0][1]

    decoded_words = [id_to_word[word_id] for word_id in best_sequence]

    return decoded_words


# Below are 3 development images, each with 1) their greedy output, 2) beam search at n=3 3) beam search at n=5.

# In[89]:


raw_1_img = PIL.Image.open(os.path.join(IMG_PATH, dev_list[165]))
preprocessed_img = preprocess(raw_1_img).to(DEVICE)
encoded_img = img_encoder(preprocessed_img.unsqueeze(0)).reshape((512))
greedy_caption = greedy_decoder(encoded_img)
print(f"greedy caption: {greedy_caption}")
beam_3_caption = img_beam_decoder(3, encoded_img)
print(f"beam n=3 caption: {beam_3_caption}")
beam_5_caption = img_beam_decoder(5, encoded_img)
print(f"beam n=5 caption: {beam_5_caption}")
raw_1_img


# In[102]:


raw_2_img = PIL.Image.open(os.path.join(IMG_PATH, dev_list[85]))
preprocessed_img = preprocess(raw_2_img).to(DEVICE)
encoded_img = img_encoder(preprocessed_img.unsqueeze(0)).reshape((512))
greedy_caption = greedy_decoder(encoded_img)
print(f"greedy caption: {greedy_caption}")
beam_3_caption = img_beam_decoder(3, encoded_img)
print(f"beam n=3 caption: {beam_3_caption}")
beam_5_caption = img_beam_decoder(5, encoded_img)
print(f"beam n=5 caption: {beam_5_caption}")
raw_2_img


# In[114]:


raw_3_img = PIL.Image.open(os.path.join(IMG_PATH, dev_list[66]))
preprocessed_img = preprocess(raw_3_img).to(DEVICE)
encoded_img = img_encoder(preprocessed_img.unsqueeze(0)).reshape((512))
greedy_caption = greedy_decoder(encoded_img)
print(f"greedy caption: {greedy_caption}")
beam_3_caption = img_beam_decoder(3, encoded_img)
print(f"beam n=3 caption: {beam_3_caption}")
beam_5_caption = img_beam_decoder(5, encoded_img)
print(f"beam n=5 caption: {beam_5_caption}")
raw_3_img

