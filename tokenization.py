#creating custom tokenizer
import re 
with open('/Users/sujalkumar/Desktop/reading /LLM/attention/coding llm from scratch /the-verdict.txt', 'r') as f:
    raw_text=f.read()
print(raw_text[:90])

"""
a vocab is nothing but a directory which have words and its corresponding id for our custom tokenizer we will
consider one token as a word and map it to its corresponding id
"""
#we have seperated words by using space it had between them

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])

all_words=sorted(set(preprocessed))
vocab_size=len(all_words)
print(vocab_size)

vocab={}

for integer,token in enumerate(all_words):
    vocab[token]=integer

print(vocab)

for i,item in enumerate(vocab.items()):
    print(item)
    if i>=50:
        break

#now creating custom tokenizer class

class customtokenizer():
    def __init__(self,vocab):
        self.str_to_int=vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self,text):
        #first we are going to split text into tokens by exploiting the space between them
        preprocessed_txt=re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed= []
        for item in preprocessed_txt:
            if item.strip():
                preprocessed.append(item.strip())

        ids = [
        self.str_to_int[s]
    for s in preprocessed
]
        return ids
    
    def decode(self,id:list):
        text = " ".join([self.int_to_str[i] for i in id])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

sample_text="I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough"
tokenizer=customtokenizer(vocab=vocab)

ids=tokenizer.encode(sample_text)
print(ids) #[53, 44, 149, 1003, 57, 38, 818, 115, 256, 486, 6, 1002, 115, 500, 435, 392]

text=tokenizer.decode(ids)
print(text) #I HAD always thought Jack Gisburn rather a cheap genius -- though a good fellow enough