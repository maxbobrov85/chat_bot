import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel
from transformers.optimization import get_linear_schedule_with_warmup
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

MAX_LENGTH=128
DEVICE='cpu'

final_dataset = pd.read_csv('final_dataset_homer.csv')

base_answers = final_dataset['A']
all_answers = list(set(base_answers)) # Список всех ответов из базы

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = AutoModel.from_pretrained("distilbert-base-uncased")
bert_model.from_pretrained("https://drive.google.com/drive/folders/1p7A6jtlYpeau3nnJ9JYiyMjjfplPj004?usp=sharing")

class CrossEncoderBert(torch.nn.Module):
    def __init__(self, max_length: int = MAX_LENGTH):
        super().__init__()
        self.max_length = max_length
        self.bert_model = AutoModel.from_pretrained('distilbert-base-uncased')
        self.bert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.linear = torch.nn.Linear(self.bert_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use the CLS token's output
        return self.linear(pooled_output)
    
model = CrossEncoderBert().to(DEVICE)
model.bert_model.from_pretrained("https://drive.google.com/drive/folders/1-1f05ApQ4vGbqlVQlQA7VvJoQqcEivDO?usp=sharing")

def get_best_rand_reply(
    tokenizer: AutoTokenizer,
    finetuned_ce: CrossEncoderBert,
    base_bert: AutoModel,
    query: str,
    context: str,
    corpus: list[str],
    size_patch = 150,
    qty_rand_choose = 5,
    max_out_context = 200
) -> None:

    dic_answ = dict()
    dic_answ["score"] = []
    dic_answ["answer"] = []

    conext_memory= query+"[SEP]"+context

    if len(corpus) < qty_rand_choose*max_out_context:
        qty_rand_choose = int(len(corpus))

    # так как база большая
    for i in range(qty_rand_choose):
        rand_patch_corpus = list(np.random.choice(corpus, size_patch))
        #print(len(rand_patch_corpus))

        queries = [conext_memory]* len(rand_patch_corpus)
        #print(len(queries))
        tokenized_texts = tokenizer(
            queries,
            rand_patch_corpus,
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(DEVICE)

        # Finetuned CrossEncoder model scoring
        with torch.no_grad():
            ce_scores = finetuned_ce(tokenized_texts['input_ids'],
                                     tokenized_texts['attention_mask']).squeeze(-1)
            ce_scores = torch.sigmoid(ce_scores)  # Apply sigmoid if needed

        # Process scores for finetuned model
        scores = ce_scores.cpu().numpy()
        scores_ix = np.argsort(scores)[::-1][0]
        dic_answ["score"].append(scores[scores_ix])
        dic_answ["answer"].append(rand_patch_corpus[scores_ix])

    id = np.argsort(dic_answ["score"])[::-1][0]# np.array(dic_answ["score"]).argmax()
    answer = dic_answ["answer"][id]
    conext_memory = answer+"[SEP]"+conext_memory
  #  flush_memory()
    return answer, conext_memory[:max_out_context], dic_answ["score"][id]

def answer(question, context):
    answer,_,_ = get_best_rand_reply(
                tokenizer, model, bert_model.to(DEVICE),
                query = question,
                context = context,
                corpus = all_answers)
    return answer
# question = "I'm glad I'm not crying because"  #
# print(f"Реплика: {question}")
# best_answer, conext_memory,  best_score = get_best_rand_reply(
#     tokenizer, model, bert_model.to(DEVICE),
#     query = question,
#     context = "Marg is angry",
#     corpus = all_answers)

# print(f"Лучший ответ: {best_answer}\nscore {best_score}") 
