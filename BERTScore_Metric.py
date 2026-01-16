import torch
from transformers import BertTokenizer, BertModel


'''
generateEmbeddings(text)
purpose:
returns embeddings for tokens of the input string called text
'''
def generateEmbeddings(text):
    #loading the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #loading BERT
    model = BertModel.from_pretrained('bert-base-uncased')
    #tokenizing
    tokens = tokenizer(text, return_tensor = 'pt')
    #generating embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    #returning a tensor of embeddings from the last layer of the BERT model
    return outputs.last_hidden_states

'''
generateCosineSimilarity(LLM_generated_text, base_truth_text)
purpose:
returns the cosine similarity scores of the embedding of each token in the LLM-generated text with each embedding of the tokens in the base truth text
'''
generateCosineSimilarity(LLM_generated_text, base_truth_text):
    #normalization along the hidden dimension
    LLM_generated_text_normalized = torch.nn.functional.normalize(LLM_generated_text, dim = -1)
    base_truth_text_normalized = torch.nn.functional.normalize(LLM_generated_text, dim = -1)
    #returning cosine similarities using batched matrix multiplication
    return torch.bmm(LLM_generated_text_normalized, base_truth_text_normalized)

'''
computePrecision(cosine_similarity_matrix)
purpose:
returns BERTREcall, which describes the model's accuracy in identifying true positives (candidate tokens that align with reference tokens); quantifies how much of the candidate's content is semantically meaningful relative to the reference
'''
def computePrecision(cosine_similarity_matrix):
    return cosine_similarity_matrix.max(dim = 2)[0].mean()
