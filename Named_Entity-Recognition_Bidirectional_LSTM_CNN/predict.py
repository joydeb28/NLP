from preprocessing import createBatches,createMatrices,addCharInformatioin,padding
from keras.models import load_model
import pickle


print("Model is loading...")
with open("Benchmark_Models/dict/wd_to_id.txt", "rb") as myFile:
    wd_to_id = pickle.load(myFile)

with open("Benchmark_Models/dict/lb_to_id.txt", "rb") as myFile:
    lb_to_id = pickle.load(myFile)

with open("Benchmark_Models/dict/ch_to_id.txt", "rb") as myFile:
    ch_to_id = pickle.load(myFile)

with open("Benchmark_Models/dict/idx2Label.txt", "rb") as myFile:
    idx2Label = pickle.load(myFile)




#from nn import entity_extract_model_BLSTM
import numpy as np

entity_model = load_model('Benchmark_Models/Benchmark_all_Entity_Model_try_with_book_flight.h5')

print("Completed")


def predict(sentence,model):
    sen_list = [[[i,'O\n'] for i in sentence.split()]]
    test = addCharInformatioin(sen_list)
    
    predLabels = []
    
    test_set = padding(createMatrices(test,wd_to_id,  lb_to_id, ch_to_id))
    
    test_batch,test_batch_len = createBatches(test_set)
    for i,data in enumerate(test_batch):
        tokens, char, labels = data
        tokens = np.asarray([tokens])     
        char = np.asarray([char])
        pred = model.predict([tokens,char], verbose=False)[0] 
        pred = pred.argmax(axis=-1) #Predict the classes            
        predLabels.append(pred)
    entity_labels = []
    j = 0
    words_list = sentence.split()
    for i in predLabels[-1]:
        entity_labels.append((words_list[j],idx2Label[int(i)].replace("\n","")))
        j+=1
    
    return entity_labels

print("Enter/Paste your content. Press double enter to exit.")

contents = []
while True:
    try:
        line = input()
        entity_prediction = predict(line,entity_model)
        for i in entity_prediction:
            print(i)
    except EOFError:
        break
    contents.append(line)

