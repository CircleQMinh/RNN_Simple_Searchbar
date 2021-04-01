


from tkinter import *
import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
from utils import *
from datetime import datetime
from rnn_theano import RNNTheano


nltk.download("book")

_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '8000'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '80'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
_NEPOCH = int(os.environ.get('NEPOCH', '100'))
_MODEL_FILE = os.environ.get('MODEL_FILE')

def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=1, evaluate_loss_after=5):
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print ("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5  
                print ("Setting learning rate to %f" % learning_rate)
            sys.stdout.flush()
            save_model_parameters_theano("./data/rnn-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)
        for i in range(len(y_train)):
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

vocabulary_size = _VOCABULARY_SIZE
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"



# Đọc dữ liệu và thêm SENTENCE_START và SENTENCE_END 
print ("Reading CSV file...")
with open('data/reddit-comments-2015-08.csv', encoding="utf8") as f:
    reader = csv.reader(f, skipinitialspace=True)
    next(reader, None)
    # Chia toàn bộ bình luận thành mảng chứa các câu
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
    # Thêm SENTENCE_START vào đầu và SENTENCE_END vào cuối
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
print ("Parsed %d sentences." % (len(sentences)))

# Tách từ các câu trong mảng
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Đếm tần số xuất hiện của các từ
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print ("Found %d unique words tokens." % len(word_freq.items()))

# Tạo mảng từ vựng có các từ thường xuất hiện nhất
vocab = word_freq.most_common(vocabulary_size-1)
# Tạo vector vị trí đối ứng với từ trong mảng từ vựng
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
# Tạo từ điển từ đối ứng với vị trí
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

print ("Using vocabulary size %d." % vocabulary_size)
print ("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

# Thay thế những từ không có trong mảng từ vựng thành 'UNKNOW_TOKEN'
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

## Tạo dữ liệu traning
#X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
#y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

#Tạo model
#model = RNNTheano(vocabulary_size, hidden_dim=_HIDDEN_DIM)
#t1 = time.time()
#model.sgd_step(X_train[10], y_train[10], _LEARNING_RATE)
#t2 = time.time()
#print ("SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.))

model = RNNTheano(vocabulary_size, hidden_dim=50)
# losses = train_with_sgd(model, X_train, y_train, nepoch=50)
# save_model_parameters_theano('./data/trained-model-theano.npz', model)
load_model_parameters_theano('./data/trained-model-theano.npz', model)

print("Load Model Complete...")
#Tạo câu ngẫu nhiên
def generate_sentence(model):
    new_sentence = [word_to_index[sentence_start_token]]
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str

#num_sentences = 10
#senten_min_length = 7

#for i in range(num_sentences):
#    sent = []
#    while len(sent) < senten_min_length:
#        sent = generate_sentence(model)
#    print (" ".join(sent))

#Đoán các từ tiếp theo từ câu cho trước
def guess_sentence(sentence, model):
  sentence = ["%s %s" % (sentence_start_token, sentence)]
  tokenized_sentence = nltk.word_tokenize(sentence[0])
  tokenized_sentence = [w if w in word_to_index else unknown_token for w in tokenized_sentence]
  guess_sent = [word_to_index[w] for w in tokenized_sentence]
  while not guess_sent[-1] == word_to_index[sentence_end_token]:
    next_word_probs = model.forward_propagation(guess_sent)
    sampled_word = word_to_index[unknown_token]
    while sampled_word == word_to_index[unknown_token]:
      samples = np.random.multinomial(1, next_word_probs[-1])
      sampled_word = np.argmax(samples)
    guess_sent.append(sampled_word)
  sentence_str = [index_to_word[x] for x in guess_sent[1:-1]]
  return sentence_str

def returnData(ans):
    num_sentences = 5
    senten_min_length = 9
    strings=[]

    for i in range(num_sentences):
        sent = []
        while len(sent) < senten_min_length:
            sent = guess_sentence(ans, model)
            strings.append(" ".join(sent))
    return strings

root = Tk()
root.title('RNN-Simple Searchbar')
root.geometry("500x300")

# Update the listbox
def update(data):
	# Clear the listbox
	my_list.delete(0, END)
	# Add toppings to listbox
	for item in data:
		my_list.insert(END, item)

# Update entry box with listbox clicked
def fillout(e):
	# Delete whatever is in the entry box
	my_entry.delete(0, END)

	# Add clicked list item to entry box
	my_entry.insert(0, my_list.get(ANCHOR))

# Create function to check entry vs listbox
time_count=0;
def check(e):
    typed = my_entry.get()
    global time_count
    time_count+=1
    if time_count==5:
        if typed == '':
            data = []
        else:
            data = returnData(typed)
        update(data)
        time_count=0



# Create a label
my_label = Label(root, text="Start Typing...",
	font=("Helvetica", 14), fg="grey")

my_label.pack(pady=20)

# Create an entry box
my_entry = Entry(root, font=("Helvetica", 20))
my_entry.pack()

# Create a listbox
my_list = Listbox(root, width=50)
my_list.pack(pady=40)


# Create a binding on the listbox onclick
my_list.bind('<<ListboxSelect>>', fillout)

# Create a binding on the entry box
my_entry.bind("<KeyRelease>", check)

root.mainloop()

