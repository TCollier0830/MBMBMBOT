import os
import httplib2
from bs4 import BeautifulSoup, SoupStrainer
import requests
import re
from urllib.request import urlretrieve
from pathlib import Path
import PyPDF2
from tika import parser
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import numpy as np
import random
import sys
import io
import slate3k as slate
import time
import warnings

warnings.simplefilter("ignore")
tf.get_logger().setLevel('ERROR')

physical_devices = tf.config.experimental.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

Curr = {}
ll=0
class Scrawl:

    def __init__(self, TopLevel, desFolder, NumPages, keyPhrase):
        self.TopLevel = TopLevel
        self.desFolder = desFolder
        self.NumPages = NumPages
        self.keyPhrase = keyPhrase

    def Scrawl(self):

        for p in range(self.NumPages):
        
            http = httplib2.Http()

            if p == 0:
                status, response = http.request(self.TopLevel)
            else:
                status, response = http.request(self.TopLevel + '?_paged=' + str(p))
            
            for ii,link in enumerate(BeautifulSoup(response, 'html.parser',
                                         parseOnlyThese=SoupStrainer('a'))):
                if link.has_attr('href'):
                    if self.keyPhrase in link['href'] and not link['href'] in Curr:
                        Curr.update({link['href']: 1})
                        print(link['href'])
                        name = link['href'].split('/')[-2].replace('transcript-mbmbam-','').replace("-","_")
                        r = requests.get(link['href'])
                        soup = BeautifulSoup(r.text, "html.parser")
                        desFile = os.path.join(self.desFolder, str(name) + '.pdf')
                        f = Path(desFile)
                        for i in soup.find_all('a', href=True):
                            if ".pdf" in i.attrs['href']:
                                ri = requests.get(i['href'])
                                f.write_bytes(ri.content)

        return

def TextIt():
    Brothers = os.path.join(os.getcwd(), "Brothers")
    TextFiles = os.path.join(os.getcwd(), "TextFiles")
    os.makedirs(TextFiles)
    for file in os.listdir(Brothers):
        oFile = open(os.path.join(TextFiles, file.replace("pdf","txt")), "w+", encoding="utf-8")
        #raw = parser.from_file(os.path.join(Brothers,file))
        #raw['content'].replace('\n','')
        pdfFileObj = open(os.path.join(Brothers,file),"rb")
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        for page in range(pdfReader.numPages):
            pageObj = pdfReader.getPage(page)
            body = pageObj.extractText()
            SeparatedBody = [s.replace("\n","") + "\n" for s in body.split(".")]
            for line in SeparatedBody:
                if len(line) > 5:
                    if "..." in line:
                        line = line.replace("...","BIGBONGFGH")
                    oFile.write(line.replace(".",".\n").replace("BIGBONGFGH", "..."))
        oFile.close()
    return

def TextIt2():
    Brothers = os.path.join(os.getcwd(), "Brothers")
    TextFiles = os.path.join(os.getcwd(), "TextFiles")
    os.makedirs(TextFiles)
    for file in os.listdir(Brothers):
        oFile = open(os.path.join(TextFiles, file.replace("pdf","txt")), "w+", encoding="utf-8")
        iFile = open(os.path.join(Brothers, file), "rb")
        doc = slate.PDF(iFile)
        for page in doc:
            page = re.sub(r'\n+ ', '\n', page)
            page = re.sub(r'\n+', '\n', page)
            page = re.sub(r'[^\x00-\x7F]+','', page)
            #page = re.sub(r'[^0-9a-zA-Z]+','', page)
            if len(page)> 6: oFile.write(page[:-1])
        iFile.close()
        oFile.close()
    return

def CombineIt():
    TextFiles = os.path.join(os.getcwd(), "TextFiles")
    oFile = open(os.path.join(TextFiles,"Beeg.txt"), "w+")
    for file in os.listdir(TextFiles)[0:10]:
        if not file == oFile:
            iFile = open(os.path.join(TextFiles,file),"r+")
            lines = iFile.readlines()
            for line in lines:
                oFile.write(line)
                #oFile.write(line.replace("\n","").replace(".","").replace(",","").replace("!","").replace(":",""))
            iFile.close()
    oFile.close()
    return

def RNN():
    TextFiles = os.path.join(os.getcwd(), "TextFiles")
    fileName = os.path.join(TextFiles,"Beeg.txt")
    text = open(fileName, 'rb').read().decode(encoding='utf-8').lower()
    
    vocab = sorted(set(text))

    # Creating a mapping from unique characters to indices
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    text_as_int = np.array([char2idx[c] for c in text])

    # The maximum length sentence we want for a single input in characters
    seq_length = 100
    examples_per_epoch = len(text)//(seq_length+1)

    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target)

    # Batch size
    BATCH_SIZE = 64

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE = 10000

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    # Length of the vocabulary in chars
    vocab_size = len(vocab)

    # The embedding dimension
    embedding_dim = 256

    # Number of RNN units
    rnn_units = 1024
        
    def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
        model = tf.keras.Sequential([
          tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                    batch_input_shape=[batch_size, None]),
          tf.keras.layers.GRU(rnn_units,
                              return_sequences=True,
                              stateful=True,
                              recurrent_initializer='glorot_uniform'),
          tf.keras.layers.Dense(vocab_size)
        ])
        return model

    model = build_model(
        vocab_size = len(vocab),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=BATCH_SIZE)

    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)

    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()

    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    example_batch_loss  = loss(target_example_batch, example_batch_predictions)

    model.compile(optimizer='adam', loss=loss)

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    
    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    EPOCHS=60

    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

    tf.train.latest_checkpoint(checkpoint_dir)

    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    model.build(tf.TensorShape([1, None]))

    def generate_text(model, start_string):
        # Evaluation step (generating text using the learned model)

        # Number of characters to generate
        num_generate = 100

        # Converting our start string to numbers (vectorizing)
        input_eval = [char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        # Empty string to store our results
        text_generated = []

        # Low temperatures results in more predictable text.
        # Higher temperatures results in more surprising text.
        # Experiment to find the best setting.
        temperature = 0.5

        # Here batch size == 1
        model.reset_states()
        ThreeSentences = True
        cc = 0
        num_sent = 10
        while ThreeSentences:
            predictions = model(input_eval)
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 0)

            # using a categorical distribution to predict the character returned by the model
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

            # We pass the predicted character as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)
            if idx2char[predicted_id] in ['.','!','?']:
                cc +=1
            if cc == num_sent:
                ThreeSentences = False
            text_generated.append(idx2char[predicted_id])



        return (start_string + ''.join(text_generated))

    print(generate_text(model, start_string=u"Who's the mayor of flavor town?".lower()))
    return

def Reader():
    TextFiles = os.path.join(os.getcwd(), "TextFiles")
    for file in os.listdir(TextFiles):
        iFile = open(os.path.join(os.path.join(os.getcwd(), "TextFiles"), file),'r')
        lines = iFile.readlines()
        for line in lines:
             print(line)
    return


#Scrawler = Scrawl("https://maximumfun.org/transcripts/my-brother-my-brother-and-me", os.path.join(os.getcwd(), "Brothers"), 54, "/my-brother-my-brother-and-me/transcript")
#Scrawler.Scrawl()

#TextIt2()
CombineIt()
#Reader()
RNN()
