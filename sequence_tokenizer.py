# Sequence tokenization
import numpy as np
import torch

# english_file = 'drive/MyDrive/translation_en_kn/train.en'
# kannada_file = 'drive/MyDrive/translation_en_kn/train.kn'
english_file = 'english.txt'
kannada_file = 'kannada.txt'

START_TOKEN = '<START>'
PADDING_TOKEN = '<PADDING>'
END_TOKEN = '<END>'

kannada_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
                      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '<', '=', '>', '?', 'ˌ', 
                      'ँ', 'ఆ', 'ఇ', 'ా', 'ి', 'ీ', 'ు', 'ూ', 
                      'ಅ', 'ಆ', 'ಇ', 'ಈ', 'ಉ', 'ಊ', 'ಋ', 'ೠ', 'ಌ', 'ಎ', 'ಏ', 'ಐ', 'ಒ', 'ಓ', 'ಔ', 
                      'ಕ', 'ಖ', 'ಗ', 'ಘ', 'ಙ', 
                      'ಚ', 'ಛ', 'ಜ', 'ಝ', 'ಞ', 
                      'ಟ', 'ಠ', 'ಡ', 'ಢ', 'ಣ', 
                      'ತ', 'ಥ', 'ದ', 'ಧ', 'ನ', 
                      'ಪ', 'ಫ', 'ಬ', 'ಭ', 'ಮ', 
                      'ಯ', 'ರ', 'ಱ', 'ಲ', 'ಳ', 'ವ', 'ಶ', 'ಷ', 'ಸ', 'ಹ', 
                      '಼', 'ಽ', 'ಾ', 'ಿ', 'ೀ', 'ು', 'ೂ', 'ೃ', 'ೄ', 'ೆ', 'ೇ', 'ೈ', 'ೊ', 'ೋ', 'ೌ', '್', 'ೕ', 'ೖ', 'ೞ', 'ೣ', 'ಂ', 'ಃ', 
                      '೦', '೧', '೨', '೩', '೪', '೫', '೬', '೭', '೮', '೯', PADDING_TOKEN, END_TOKEN]

english_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                        ':', '<', '=', '>', '?', '@', 
                        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
                        'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 
                        'Y', 'Z', '[', '\\',
                        ']', '^', '_', '`', 
                        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                        'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
                        'y', 'z', 
                        '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN]

print(f"Size of kannada_vocabulary: {len(kannada_vocabulary)}")
print(f"Size of english_vocabulary: {len(english_vocabulary)}")

index_to_kannada = {k:v for k,v in enumerate(kannada_vocabulary)}
kannada_to_index = {v:k for k,v in enumerate(kannada_vocabulary)}
index_to_english = {k:v for k,v in enumerate(english_vocabulary)}
english_to_index = {v:k for k,v in enumerate(english_vocabulary)}
# print(f"en key ] -> {english_to_index[']']}")

with open(english_file, 'r') as file:
    english_sentences = file.readlines()
with open(kannada_file, 'r') as file:
    kannada_sentences = file.readlines()

# Limit Number of sentences
TOTAL_SENTENCES = 100000
english_sentences = english_sentences[:TOTAL_SENTENCES]
kannada_sentences = kannada_sentences[:TOTAL_SENTENCES]
english_sentences = [sentence.rstrip('\n') for sentence in english_sentences] # Remove newline characters at the end of each line.
kannada_sentences = [sentence.rstrip('\n') for sentence in kannada_sentences]

# Encode every single 'character' into some embedding!
# However, max english sentence has 722 characters, too long!
# Check the 97 percentile: average number of chars in all Kannada/English sentences
PERCENTILE=97
print(f"{PERCENTILE}th percentile # characters in Kannada: {np.percentile([len(str) for str in kannada_sentences], PERCENTILE)}")
print(f"{PERCENTILE}th percentile # characters in english: {np.percentile([len(str) for str in english_sentences], PERCENTILE)}")
# 97% of english sentences have less than or equal to 179 chars.

# Defining a maximum number of sequence length (discard sentences longer than that)
max_sequence_length = 200

def is_valid_tokens(sentence, vocab):
    for token in list(set(sentence)):
        if token not in vocab:
            return False
    return True

def is_valid_length(sentence, max_sequence_length):
    return len(list(sentence)) < (max_sequence_length-1) # need to re-add the end token so leaving 1 space

valid_sentence_indices = []
for index in range(len(kannada_sentences)):
    # Original text files are 1-1 corresponding (input, label) pairs
    kannada_sentence, english_sentence = kannada_sentences[index], english_sentences[index]
    if is_valid_length(kannada_sentence, max_sequence_length) \
        and is_valid_length(english_sentence, max_sequence_length) \
        and is_valid_tokens(kannada_sentence, kannada_vocabulary):
            valid_sentence_indices.append(index)

print(f"After preprocessing, {len(valid_sentence_indices)} valid kannada sequences.")
# Keep only 'valid' kannada/english sentences 
kannada_sentences = [kannada_sentences[i] for i in valid_sentence_indices]
english_sentences = [english_sentences[i] for i in valid_sentence_indices]
# Huge reduction: since both english & kannada sentence have to satisfy the condition

from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset): 
    def __init__(self, english_sentences, kannada_sentences):
        self.english_sentences = english_sentences
        self.kannada_sentences = kannada_sentences

    def __len__(self): # return the number of items in the dataset
        return len(self.english_sentences)
    
    def __getitem__(self, idx): 
        return self.english_sentences[idx], self.kannada_sentences[idx]
    
dataset = TextDataset(english_sentences, kannada_sentences)
print(f"dataset size: {len(dataset)}")

# To speed up training - use "mini-Batch gradient descent"
# This only update the network once it process every 'batch' number of examples
# Also prevent too jaggy update in the loss landscape plot (2D)
batch_size = 3
train_loader = DataLoader(dataset, batch_size)
iterator = iter(train_loader)
# for batch_num, batch in enumerate(iterator):
#     if batch_num > 0:
#         break
#     print(f"batch {batch_num}: {batch}")

# Convert text into numbers
# Input: "Today is sunny."
# Output: [8, 20, 198] -> corresponding indices in english_vocabulary
# For kannada, need to insert <START> and <END> token
def tokenize(sentence, language_to_index, start_token=True, end_token=True)-> torch.Tensor:
    sentence_word_indices = [language_to_index[token] for token in list(sentence)]
    if start_token: # prepend from 0
        sentence_word_indices.insert(0, language_to_index[START_TOKEN])
    if end_token:
        sentence_word_indices.append(language_to_index[START_TOKEN])
    # Fill padding tokens for the rest unfilled values. Ensure all input sentences have 'max_sequence_length' number of words
    for _ in range(len(sentence_word_indices), max_sequence_length):
        sentence_word_indices.append(language_to_index[PADDING_TOKEN])
    
    return torch.tensor(sentence_word_indices)

en_tokenized, kn_tokenized = [],[]
for batch_num, batch in enumerate(iterator):
    
    english_sentences, kannada_sentences = batch
    for (en_str, kn_str) in zip(english_sentences, kannada_sentences):
        en_tensor = tokenize(en_str, english_to_index, start_token=False, end_token=False)
        kn_tensor = tokenize(kn_str, kannada_to_index, start_token=True, end_token=True)
        en_tokenized.append(en_tensor), kn_tokenized.append(kn_str)
        if batch_num==0:
            print(f"{en_str} -> tokenization -> {en_tensor}")
            print(f"{kn_str} -> tokenization -> {kn_tensor}")

print(f"len(en_tokenized) {len(en_tokenized)}")
print(f"len(kn_tokenized) {len(kn_tokenized)}")

# --------- MASKING ----------- #
# Padding mask: do not look at the PADDING token when computing loss function and updating weights. They mean nothing --------- #


NEG_INFTY = -1e9 # Prevent nan during softmax
# Softmax: exponential function. e^-inf = 0 -> if exact 0, for entire row is 0 -> sum = 0, division by 0 -> NaN error

def create_masks(eng_batch, kn_batch):
    num_sentences = len(eng_batch)
    # lookahead mask: only for decoder
    look_ahead_mask = torch.full([max_sequence_length, max_sequence_length], True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal = 1)

    # Padding mask:
    # self-attention: [i][i]
    # cross-attention: [i][j] & [j][i] eng[i] -> kanna[j]
    encoder_padding_mask = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)        
    
    for idx in range(num_sentences):
        eng_sentence_length, kn_sentence_length = len(eng_batch[idx]), len(kn_batch[idx])
        eng_padding_indices = np.arange(eng_sentence_length+1, max_sequence_length)
        kn_padding_indices = np.arange(kn_sentence_length+1, max_sequence_length)

        encoder_padding_mask[idx, :, eng_padding_indices] = True
        encoder_padding_mask[idx, eng_padding_indices, :] = True   

        decoder_padding_mask_self_attention[idx, :, kn_padding_indices] = True
        decoder_padding_mask_self_attention[idx, kn_padding_indices, :] = True   

        decoder_padding_mask_cross_attention[idx, :, eng_padding_indices] = True
        decoder_padding_mask_cross_attention[idx, kn_padding_indices, :] = True   

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0) # Set indices of True to NEG_INFTY, others to 0
    decoder_self_attention_mask = torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)
    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask