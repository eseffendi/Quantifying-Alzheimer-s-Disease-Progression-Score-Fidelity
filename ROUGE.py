import evaluate 

rouge = rouge = evaluate.load('rouge') 

def get_ROUGE(candidate, reference):  
    rouge_results = rouge.compute(
        predictions=[candidate],
        references=[reference],
        use_aggregator=True
    )
    return rouge_results

