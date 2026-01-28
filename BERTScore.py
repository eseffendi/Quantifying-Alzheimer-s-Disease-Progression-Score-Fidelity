import torch
from transformers import BertTokenizer, BertModel

MODEL_NAME = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME, device_map="auto")

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.last_hidden_state

def cosine_similarity(generated_embeddings, reference_embeddings):
    generated_embeddings = torch.nn.functional.normalize(generated_embeddings, dim=-1)
    reference_embeddings = torch.nn.functional.normalize(reference_embeddings, dim=-1)
    return torch.bmm(generated_embeddings, reference_embeddings.transpose(1, 2))

def get_BERT(candidate, reference):
    candidate_embeddings = get_embeddings(candidate)
    reference_embeddings = get_embeddings(reference)
    similarity_matrix = cosine_similarity(candidate_embeddings, reference_embeddings)
    precision = similarity_matrix.max(dim=2)[0].mean()
    recall = similarity_matrix.max(dim=1)[0].mean()
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item() if isinstance(f1, torch.Tensor) else float(f1)
    }