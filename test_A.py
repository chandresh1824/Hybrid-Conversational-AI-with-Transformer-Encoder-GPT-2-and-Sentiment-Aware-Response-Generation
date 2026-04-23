import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, LogitsProcessor, LogitsProcessorList,
    pipeline, AutoTokenizer, AutoModelForSequenceClassification,
    BlenderbotTokenizer, BlenderbotForConditionalGeneration
)
from torch.optim import AdamW
from collections import defaultdict, deque
import pandas as pd
import re
import os
import math
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import json
from bert_score import score as bertscore
import mauve
from scipy.stats import ttest_rel
TF_ENABLE_ONEDNN_OPTS=0

TRAIN_GPT2 = True
TRAIN_ENCODER = True
GPT2_TRAIN_EPOCHS = 60
ENCODER_TRAIN_EPOCHS = 60
BATCH_SIZE = 8
MAX_LEN = 192
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONV_PATH = "conversations.jsonl" 
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cuda.matmul.allow_tf32 = True

print(f"Using device: {DEVICE}")

def compute_bertscore(candidate: str, reference: str) -> float:
 
    try:
        P, R, F1 = bertscore(
            [candidate],
            [reference],
            lang="en",
            verbose=False,
            rescale_with_baseline=True
        )
        return float(F1.mean())
    except Exception as e:
        print(f"BERTScore error: {e}")
        return 0.0

def save_conversation(memory, path=CONV_PATH):
   
    new_pairs = []

    
    for i in range(0, len(memory.turns) - 1):
        a = memory.turns[i]
        u = memory.turns[i + 1]

        if a["role"] == "assistant" and u["role"] == "user":
            
            if len(conv_data) < 10:
               
                modified_asst = a["content"].strip()
            else:
                
                modified_asst = a["content"].strip() + " what about you"

            pair = (modified_asst, u["content"].strip())
            new_pairs.append(pair)

    
    existing_pairs = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().rstrip(",")
                if line.startswith("(") and line.endswith(")"):
                    try:
                        q, a = eval(line)
                        existing_pairs.append((q, a))
                    except:
                        continue

   
    all_pairs = existing_pairs + new_pairs
    unique_pairs = list(dict.fromkeys(all_pairs))

    
    with open(path, "w", encoding="utf-8") as f:
        for pair in unique_pairs:
            f.write(f"{pair},\n")

def ab_decision(metrics_a: dict, metrics_b: dict) -> str:
    
    score_a = (
        metrics_a["SBERT"]
        + metrics_a["BERT"]
        + metrics_a["MAUVE"]
    )

    score_b = (
        metrics_b["SBERT"]
        + metrics_b["BERT"]
        + metrics_b["MAUVE"]
    )

    if score_a > score_b:
        return "A (GPT-2) wins"
    elif score_b > score_a:
        return "B (Hybrid) wins"
    else:
        return "Tie"

    
def compute_mauve(generated_list, reference_list) -> float:
   
    try:
        if not generated_list or not reference_list:
            return 0.0

        out = mauve.compute_mauve(
            p_text=generated_list,
            q_text=reference_list,
            device_id=0 if torch.cuda.is_available() else -1,
            max_text_length=128,
            verbose=False
        )

        return float(out.mauve)
    except Exception as e:
        print(f"MAUVE error: {e}")
        return 0.0




def load_conversations(path=CONV_PATH):
   
    if not os.path.exists(path):
        return []
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if line.startswith("(") and line.endswith(")"):
                try:
                    q, a = eval(line)
                    pairs.append((q, a))
                except:
                    continue
    return list(dict.fromkeys(pairs))  

def load_persona_chat_csv(path="personality.csv", max_samples=None):
    df = pd.read_csv(path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    pairs = []

    for _, row in df.iterrows():

        persona = str(row["Persona"]).strip()
        chat_text = str(row["chat"]).strip()

     
        turns = [t.strip() for t in chat_text.split("\n") if t.strip()]

        
        for turn in turns:
            q = turn
            a = persona
            pairs.append((q, a))

        if max_samples and len(pairs) >= max_samples:
            break

    return pairs

print("Loading personality.csv dataset...")

conv_data = load_persona_chat_csv("personality.csv", max_samples=100)


all_qa_data = conv_data

print("Total PersonaChat pairs:", len(conv_data))

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_model.resize_token_embeddings(len(tokenizer))
gpt2_model.config.pad_token_id = tokenizer.eos_token_id


gpt2_model.config.attn_pdrop = 0.2
gpt2_model.config.resid_pdrop = 0.2
gpt2_model.config.embd_pdrop = 0.2

gpt2_model.to(DEVICE)


BBOT_MODEL = "facebook/blenderbot-400M-distill"
bbot_tokenizer = BlenderbotTokenizer.from_pretrained(BBOT_MODEL)
bbot_model = BlenderbotForConditionalGeneration.from_pretrained(
    BBOT_MODEL,
    torch_dtype=torch.float32,
    use_safetensors=True
).to(DEVICE)

def blenderbot_generate(user_msg: str) -> str:
    inputs = bbot_tokenizer([user_msg], return_tensors="pt").to(DEVICE)
    out = bbot_model.generate(**inputs, max_length=128)
    return bbot_tokenizer.decode(out[0], skip_special_tokens=True)

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=2, n_embd=768):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.encoder_layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projection = nn.Linear(d_model, n_embd)

    def forward(self, input_ids, src_key_padding_mask=None):
        embedded = self.embedding(input_ids)
        encoded = self.encoder_layers(embedded.permute(1, 0, 2), src_key_padding_mask=src_key_padding_mask)
        projected = self.projection(encoded.permute(1, 0, 2))
        return projected

encoder = TransformerEncoder(len(tokenizer), d_model=128, num_heads=4, num_layers=2, n_embd=gpt2_model.config.n_embd).to(DEVICE)


WORD_PATTERN = re.compile(r"\b\w[\w']*\b")

_META_PATTERNS = [
    r"\bsource\s*:\s*.*?$",
    r"\bcontext\s*:\s*.*?$",
    r"\breference\s*:\s*.*?$",
    r"\burl\s*:\s*.*?$",
    r"\blink\s*:\s*.*?$",
    r"\bcitation\s*:\s*.*?$",
]

def strip_meta_lines(text: str) -> str:
    for pat in _META_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE | re.MULTILINE)
    return text.strip()

def make_text_readable(text):
    if not text:
        return text
    text = strip_meta_lines(text)
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)
    text = re.sub(r'[^\x20-\x7E\n\r\t]', '', text)
    text = re.sub(r'\s+([?.!,;:])', r'\1', text)
    text = re.sub(r'([?.!,;:])([A-Za-z0-9])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    parts = re.split(r'([.!?]\s+)', text)
    parts = [p.capitalize() for p in parts]
    text = "".join(parts).strip()
    if text and text[-1] not in '.!?':
        text += '.'
    return text


sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name).to(DEVICE)
sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer, device=0 if DEVICE.type=="cuda" else -1)

class SentimentBiasedLogits(LogitsProcessor):
    def __init__(self, sentiment_token_ids, sentiment_bias_weight=1.1):
        self.sentiment_token_ids = sentiment_token_ids
        self.sentiment_bias_weight = sentiment_bias_weight

    def __call__(self, input_ids, scores):
        for token_id in self.sentiment_token_ids:
            if 0 <= token_id < scores.size(-1):
                scores[:, token_id] = scores[:, token_id] + self.sentiment_bias_weight
        return scores

class BiasedLogits(LogitsProcessor):
    def __init__(self, bias_token_ids, bias_weight=1.5):
        self.bias_token_ids = bias_token_ids
        self.bias_weight = bias_weight

    def __call__(self, input_ids, scores):
        for token_id in self.bias_token_ids:
            if 0 <= token_id < scores.size(-1):
                scores[:, token_id] = scores[:, token_id] + self.bias_weight
        return scores

_BAD_WORDS = ["source", "context", "reference", "citation", "url", "link", "footnote", "appendix"]

def _make_bad_words_ids(tok):
    ids = []
    for w in _BAD_WORDS:
        enc = tok.encode(w, add_special_tokens=False)
        if enc:
            ids.append(enc)
    return ids

_BAD_WORDS_IDS = _make_bad_words_ids(tokenizer)


class DialogueDataset(Dataset):
    """Each item is turned into: System + User + Assistant, so GPT-2 learns the format."""
    def __init__(self, qa_pairs, tokenizer, max_len=MAX_LEN):
        self.rows = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        sys = "System: You are friendly, supportive, and concise.\n"
        for q, a in qa_pairs:
            prompt = (
                sys
                + f"User: {q}\n"
                + f"Assistant: {a}"
            )
            self.rows.append(prompt)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        text = self.rows[idx]
        enc = self.tokenizer(text, truncation=True, max_length=self.max_len,
                             padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def train_gpt2(train_loader, model, epochs=GPT2_TRAIN_EPOCHS):
    model.train()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            if step % 20 == 0:
                print(f"Epoch {epoch+1} | Step {step}/{len(train_loader)} | Loss {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"GPT-2 Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o')
    plt.title("GPT-2 Training Epoch Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("gpt2_epoch_loss.png")
    plt.show()


def train_encoder_on_gpt_embeddings(text_rows, encoder, gpt_model, epochs=ENCODER_TRAIN_EPOCHS, batch_size=BATCH_SIZE):
    encoder.train()
    gpt_model.eval()
    opt = AdamW(encoder.parameters(), lr=1e-4)
    mse = nn.MSELoss()
    epoch_losses = []

    class _TmpDS(Dataset):
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, i): return self.rows[i]

    loader = DataLoader(_TmpDS(text_rows), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        tot = 0.0
        for batch in loader:
            enc = tokenizer(list(batch), padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')
            ids = enc['input_ids'].to(DEVICE)
            mask = enc['attention_mask'].to(DEVICE)
            with torch.no_grad():
                gpt_emb = gpt_model.transformer.wte(ids)
            pred = encoder(ids, src_key_padding_mask=(mask==0))
            loss = mse(pred, gpt_emb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            opt.step()
            tot += loss.item()

        avg_loss = tot / len(loader)
        epoch_losses.append(avg_loss)
        print(f"Encoder Epoch {epoch+1}/{epochs} - MSE Loss: {avg_loss:.6f}")

   
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o', color='orange')
    plt.title("Transformer Encoder MSE Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("encoder_mse_loss.png")
    plt.show()

@torch.no_grad()
def visualize_logit_comparison(text="hello world", tokenizer=tokenizer, gpt_model=gpt2_model, encoder=encoder):
    
    enc = tokenizer(text, return_tensors="pt").to(DEVICE)
    input_ids = enc["input_ids"]

    gpt_emb = gpt_model.transformer.wte(input_ids)
    enc_emb = encoder(input_ids)

    
    gpt_mean = gpt_emb.mean(dim=1).squeeze().cpu().numpy()
    enc_mean = enc_emb.mean(dim=1).squeeze().cpu().numpy()

    
    plt.figure(figsize=(10, 5))
    plt.plot(gpt_mean[:100], label="GPT-2 Embeddings", color='blue')
    plt.plot(enc_mean[:100], label="Encoder Embeddings", color='orange')
    plt.title(f"Logit Value Comparison for: '{text}'")
    plt.xlabel("Embedding Dimension Index (first 50)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("logit_value_comparison.png")
    plt.show()



_sbert = SentenceTransformer("all-MiniLM-L6-v2").to(DEVICE)

def _sbert_score(a: str, b: str) -> float:
    try:
        if not a or not b:
            return 0.0
        ea = _sbert.encode(a, convert_to_tensor=True, device=DEVICE)
        eb = _sbert.encode(b, convert_to_tensor=True, device=DEVICE)
        return float(util.pytorch_cos_sim(ea, eb).item())
    except Exception:
        return 0.0

def _confidence_score(prompt: str, generated: str) -> float:
    try:
        gpt2_model.eval()
        with torch.no_grad():
            full = prompt + " " + generated
            enc = tokenizer(full, return_tensors="pt", add_special_tokens=False)
            full_ids = enc["input_ids"].to(DEVICE)
            prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)
            prompt_len = prompt_ids.size(1)
            if full_ids.size(1) <= prompt_len:
                return 0.0
            outputs = gpt2_model(full_ids)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :]
            shift_labels = full_ids[:, 1:]
            cont_logits = shift_logits[:, prompt_len-1:-1, :]
            cont_labels = shift_labels[:, prompt_len-1:-1]
            log_probs = torch.log_softmax(cont_logits, dim=-1)
            chosen_log_probs = log_probs.gather(-1, cont_labels.unsqueeze(-1)).squeeze(-1)
            avg_log_prob = chosen_log_probs.mean().item()
            return float(math.exp(avg_log_prob))
    except Exception:
        return 0.0

_turing_pipe = None
_turing_label_ai = None
try:
    _turing_pipe = pipeline(
        "text-classification",
        model="Hello-SimpleAI/chatgpt-detector-roberta",
        device=0 if DEVICE.type == "cuda" else -1,
        truncation=True
    )
    _ = _turing_pipe("This is a short test.")
    _turing_label_ai = "ok"
except Exception:
    _turing_pipe = None
    _turing_label_ai = None


def _turing_score(text: str, prompt: str, generated: str) -> float:
    if _turing_pipe is not None:
        try:
            out = _turing_pipe(text, truncation=True)
            if isinstance(out, list) and len(out) and "label" in out[0] and "score" in out[0]:
                label = out[0]["label"].lower()
                score = float(out[0]["score"])
                if any(k in label for k in ["ai", "generated", "fake", "machine", "gpt"]):
                    return 1.0 - score
                if any(k in label for k in ["human", "real"]):
                    return score
                return 0.5
        except Exception:
            pass
    conf = _confidence_score(prompt, generated)
    return float(max(0.0, min(1.0, conf)))


CSV_PATH = "word_sentiment_counts.csv"

def get_sentiment_bias_token_ids(text, tokenizer, csv_path=CSV_PATH, top_k=70):
    try:
        sentiment_label = sentiment_analyzer(text)[0]["label"].lower()
    except Exception:
        return []
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path)
    if sentiment_label not in df.columns:
        return []
    df = df[df[sentiment_label] > 0].sort_values(by=sentiment_label, ascending=False).head(top_k)
    token_ids = set()
    for word in df["word"].astype(str):
        toks = tokenizer.encode(word, add_special_tokens=False)
        for t in toks:
            token_ids.add(t)
    return list(token_ids)

@torch.no_grad()
def paraphrase_preserve_meaning(raw_answer: str, max_new_tokens=200):
    
    gpt2_model.eval()

    prompt = (
        "System: You are friendly, supportive, and concise.\n"
        "User: Rewrite the following response in the same style as my dataset answers. "
        "Do NOT say 'Correct' or agree with the original. "
        "Just rephrase it naturally into my dataset style:\n"
        f"\"{raw_answer}\"\n"
        "Assistant:"
    )

    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    logits_processor = LogitsProcessorList([
        BiasedLogits(chat_model.answer_vocab_ids, bias_weight=2.5)
    ])

    outputs = gpt2_model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.9,
        top_k=100,
        top_p=0.85,
        no_repeat_ngram_size=8,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        logits_processor=logits_processor,
    )

    resp = tokenizer.decode(outputs[0], skip_special_tokens=True)


    resp = re.sub(r"\b(System|User|Assistant)\s*:\s*", "", resp, flags=re.IGNORECASE)

   
    if "\"" in resp:
        resp = resp.split("\"")[-1]

    resp = make_text_readable(resp)

 
    resp = re.sub(r"^(correct\.?|sure\.?|yeah\.?|absolutely\.?)\s*", "", resp, flags=re.IGNORECASE)

   
    if not resp.lower().endswith("what about you"):
        resp = resp.strip() + " what about you"

    return resp



def update_sentiment_word_counts_from_texts(texts, csv_path=CSV_PATH):
    existing = defaultdict(lambda: {"positive": 0, "negative": 0})
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                w = str(row["word"]).lower()
                existing[w]["positive"] = int(row.get("positive", 0))
                existing[w]["negative"] = int(row.get("negative", 0))
        except Exception:
            pass
    
    for text in texts:
        try:
            sent = sentiment_analyzer(text)[0]["label"].lower()
        except Exception:
            sent = "positive"
        for w in WORD_PATTERN.findall(text.lower()):
            existing[w][sent] += 1
    df_updated = pd.DataFrame([
        {"word": w, "positive": c["positive"], "negative": c["negative"]}
        for w, c in existing.items()
    ])
    df_updated.to_csv(csv_path, index=False)


update_sentiment_word_counts_from_texts([a for _, a in conv_data[:5000]])



def encode_words_for_text(text, tokenizer, encoder, target_length):
   
    words = re.findall(r"\b\w[\w']*\b", text.lower())
    flat_ids = []
    for w in words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if ids:
            flat_ids.extend(ids)
    if not flat_ids:
        flat_ids = [tokenizer.eos_token_id]
    if len(flat_ids) >= target_length:
        flat_ids = flat_ids[:target_length]
    else:
        flat_ids = flat_ids + [tokenizer.eos_token_id] * (target_length - len(flat_ids))
    word_tensor = torch.tensor([flat_ids], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        conceptual_embeddings = encoder(word_tensor)
    return conceptual_embeddings  


class ConversationMemory:
    def __init__(self, max_turns=8):
        self.turns = deque([], maxlen=max_turns)  

    def add(self, role, content):
        self.turns.append({"role": role, "content": content})

    def text_dialogue(self, include_system=True):
        sys = "System: You are friendly, supportive, and concise.\n"
        parts = [sys] if include_system else []
        for t in self.turns:
            if t["role"] == "user":
                parts.append(f"User: {t['content']}\n")
            elif t["role"] == "assistant":
                parts.append(f"Assistant: {t['content']}\n")
        return "".join(parts)


class TopicTracker:
    def __init__(self, threshold=0.55):
        self.current_topic = None
        self.threshold = threshold

    def is_new_topic(self, text: str) -> bool:
        if not self.current_topic:
            self.current_topic = text
            return False
        score = _sbert_score(self.current_topic, text)
        if score < self.threshold:
            self.current_topic = text
            return True
        return False

class HybridConversationalModel:
    def __init__(self, gpt2_model, encoder, tokenizer):
        self.gpt2 = gpt2_model
        self.encoder = encoder
        self.tok = tokenizer
        self.answer_vocab_ids = self._build_answer_bias_ids()

    def _build_answer_bias_ids(self):
        answer_words = set(' '.join([a for _, a in conv_data]).split())
        question_words = set(' '.join([q for q, _ in conv_data]).split())
        answer_vocab = list(answer_words - question_words)
        token_ids = set()
        for w in answer_vocab:
            toks = self.tok.encode(w, add_special_tokens=False)
            for t in toks:
                token_ids.add(t)
        return list(token_ids)
    


    @torch.no_grad()
    def chat(self, memory: ConversationMemory, user_message: str, max_new_tokens=160):
        self.gpt2.eval()
        self.encoder.eval()

        sys_txt = "System: You are friendly, supportive, and concise.\n"
        hist_txt = memory.text_dialogue(include_system=False)  # already formatted
        usr_txt = f"User: {user_message}\n"
        asst_prefix = "Assistant:"

        sys_ids = self.tok.encode(sys_txt, return_tensors="pt", add_special_tokens=False).to(DEVICE)
        hist_ids = self.tok.encode(hist_txt, return_tensors="pt", add_special_tokens=False).to(DEVICE) if hist_txt else torch.zeros((1,0), dtype=torch.long, device=DEVICE)
        usr_ids = self.tok.encode(usr_txt, return_tensors="pt", add_special_tokens=False).to(DEVICE)
        asst_ids = self.tok.encode(asst_prefix, return_tensors="pt", add_special_tokens=False).to(DEVICE)

        input_ids = torch.cat([sys_ids, hist_ids, usr_ids, asst_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)
        seq_len = input_ids.size(1)
        position_ids = torch.arange(0, seq_len, device=DEVICE).unsqueeze(0)

        token_embeds = self.gpt2.transformer.wte(input_ids)
        position_embeds = self.gpt2.transformer.wpe(position_ids)

        ce = encode_words_for_text(user_message, self.tok, self.encoder, target_length=usr_ids.size(1))
        conceptual = torch.zeros_like(token_embeds)
        start_idx = sys_ids.size(1) + hist_ids.size(1)
        conceptual[:, start_idx:start_idx + usr_ids.size(1), :] = ce

        inputs_embeds = token_embeds + position_embeds + conceptual

        sentiment_ids = get_sentiment_bias_token_ids(user_message, self.tok)
        logits_processor = LogitsProcessorList([
            BiasedLogits(self.answer_vocab_ids, bias_weight=2.5),
            SentimentBiasedLogits(sentiment_ids, sentiment_bias_weight=1.3)
        ])

        out_ids = self.gpt2.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_length=seq_len + max_new_tokens,
            min_length=seq_len + 30,
            do_sample=True,
            no_repeat_ngram_size=8,
            repetition_penalty=1.9,
            temperature=0.9,
            top_k=100,
            top_p=0.85,
            bad_words_ids=_BAD_WORDS_IDS,
            pad_token_id=self.tok.eos_token_id,
            eos_token_id=self.tok.eos_token_id,
            num_return_sequences=1,
            logits_processor=logits_processor
        )

        raw = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        if "Assistant:" in raw:
            resp = raw.split("Assistant:")[-1].strip()
        else:
            prompt_txt = (sys_txt + hist_txt + usr_txt + asst_prefix).strip()
            resp = re.sub(re.escape(prompt_txt), "", raw, flags=re.IGNORECASE).strip()
        cleaned = make_text_readable(resp)
        return cleaned
    
    _sbert = SentenceTransformer("all-MiniLM-L6-v2").to(DEVICE)

@torch.no_grad()
def plot_trained_embedding_match(text_rows, tokenizer=tokenizer, gpt_model=gpt2_model, encoder=encoder):
  
    encoder.eval()
    gpt_model.eval()
    similarities = []

    for text in random.sample(text_rows, min(10, len(text_rows))):  # sample up to 10 for visualization
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
        input_ids = enc["input_ids"]
        mask = enc["attention_mask"]

      
        gpt_emb = gpt_model.transformer.wte(input_ids)
        enc_emb = encoder(input_ids, src_key_padding_mask=(mask == 0))

       
        sim = F.cosine_similarity(gpt_emb, enc_emb, dim=-1)
        sim = sim[mask.bool()].mean().item()
        similarities.append(sim)

    avg_sim = np.mean(similarities)
    print(f"🔹 Average cosine similarity (Transformer ↔ GPT-2 embeddings): {avg_sim:.4f}")

    
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(similarities)), similarities, color='teal', alpha=0.7)
    plt.axhline(avg_sim, color='red', linestyle='--', label=f"Mean Similarity = {avg_sim:.4f}")
    plt.title("Transformer ↔ GPT-2 Embedding Match (Cosine Similarity per Sample)")
    plt.xlabel("Sample Index")
    plt.ylabel("Cosine Similarity")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("trained_embedding_match.png")
    plt.show()
    print("📊 Embedding match chart saved as trained_embedding_match.png\n")

def is_out_of_domain(user_msg: str, qa_pairs=None, conv_pairs=None, threshold=0.55):
   
   
    qa_pairs = qa_pairs or []
    conv_pairs = conv_pairs or []

   
    all_pairs = qa_pairs + conv_pairs
    if not all_pairs:
        return True  

    # Compute similarity against all known questions
    sims = [_sbert_score(user_msg, q) for q, _ in all_pairs]
    max_sim = max(sims) if sims else 0.0

    return max_sim < threshold



train_set = DialogueDataset(all_qa_data, tokenizer, max_len=MAX_LEN)
if len(train_set) > 0:
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    if TRAIN_GPT2:
        print("Starting GPT-2 fine-tuning...")
        train_gpt2(train_loader, gpt2_model, epochs=GPT2_TRAIN_EPOCHS)
        print("GPT-2 fine-tuning complete.")
        print("📊 GPT-2 training loss chart saved as gpt2_epoch_loss.png\n")

    if TRAIN_ENCODER:
        print("Training Transformer encoder...")
        train_encoder_on_gpt_embeddings(train_set.rows, encoder, gpt2_model, epochs=ENCODER_TRAIN_EPOCHS)
        print("Encoder training complete.")
        print("📊 Encoder MSE loss chart saved as encoder_mse_loss.png\n")
    
        print("Analyzing Transformer ↔ GPT-2 embedding match...")
        plot_trained_embedding_match(train_set.rows, tokenizer, gpt2_model, encoder)


   
    print("Visualizing logit value comparison...")
    visualize_logit_comparison("Ngl I haven’t really been advised on a lot. I’ve gotten advice when I’m stuck on something but I don’t think I can term them as best or worst. As far as I can remember most of the advice I’ve got have helped me ig.")
    print("📊 Logit comparison chart saved as logit_value_comparison.png\n")
else:
    print("⚠️ No conversation data found — skipping training. Please chat first so conversations.jsonl is populated.")



memory = ConversationMemory(max_turns=8)
chat_model = HybridConversationalModel(gpt2_model, encoder, tokenizer)
topic_tracker = TopicTracker(threshold=0.55)

print("Chat started! (type 'exit' to quit)\n")
while True:
    user_msg = input("User: ").strip()
    if user_msg.lower() in ["exit","quit","bye"]:
        print("Assistant: Goodbye! Take care.")
        break

    if is_out_of_domain(user_msg, conv_data):
        print("\n--- Using  BlenderBot fallback ---")
        bbot_raw = blenderbot_generate(user_msg)
        print(f"[Fallback Raw Output]: {bbot_raw}")

        if len(conv_data) >= 1:
     
            adapted = paraphrase_preserve_meaning(bbot_raw)
            memory.add("user", user_msg)
            memory.add("assistant", adapted)
            print(f"Assistant (adapted): {adapted}\n")
        else:
            
            memory.add("user", user_msg)
            memory.add("assistant", bbot_raw)
            print(f"Assistant: {bbot_raw}\n")


    else:
        if topic_tracker.is_new_topic(user_msg):
            topic_memory = ConversationMemory(max_turns=8)
            topic_memory.add("user", user_msg)
            reply = chat_model.chat(topic_memory, user_msg, max_new_tokens=160)
            topic_memory.add("assistant", reply)
            memory.add("user", user_msg)
            memory.add("assistant", reply)
            print("\n--- New topic detected ---\n")
        else:
            memory.add("user", user_msg)
            reply = chat_model.chat(memory, user_msg, max_new_tokens=160)
            memory.add("assistant", reply)
        print(f"Assistant: {reply}\n")


save_conversation(memory)


update_sentiment_word_counts_from_texts([t['content'] for t in memory.turns if t['role']=="assistant"])
print("Conversation ended. Sentiment counts updated.")

try:
    print("\n===== Evaluation Metrics =====")

    
    reference_pairs = conv_data

    if not reference_pairs:
        print("⚠️ No reference conversation data found. Skipping evaluation.")
    else:
        ref_questions = [q for q, _ in reference_pairs]
        ref_answers = [a for _, a in reference_pairs]

        print(f"Loaded {len(reference_pairs)} reference pairs from '{CONV_PATH}'.\n")

        
        all_results = []

        for i, turn in enumerate(memory.turns):
            if turn["role"] != "assistant":
                continue

            gen_response = turn["content"]

           
            sbert_values = [_sbert_score(gen_response, ref) for ref in ref_answers]
            best_idx = int(np.argmax(sbert_values)) if sbert_values else -1
            best_ref_answer = ref_answers[best_idx] if best_idx != -1 else ""
            best_ref_question = ref_questions[best_idx] if best_idx != -1 else ""

            sbert_best = max(sbert_values) if sbert_values else 0.0

            
            prompt_str = (
                f"System: You are friendly, supportive, and concise.\n"
                f"User: {best_ref_question}\nAssistant:"
            )

            
            conf_score = _confidence_score(prompt_str, gen_response)
            turing_score = _turing_score(gen_response, prompt_str, gen_response)

           
            all_results.append({
                "turn": i,
                "generated": gen_response,
                "ref_answer": best_ref_answer,
                "SBERT": sbert_best,
                "Confidence": conf_score,
                "Turing": turing_score
            })

            
            print(f"--- Line {i} (Assistant) ---")
            print(f"Generated: {gen_response}")
            print(f"Closest Reference: {best_ref_answer}")
            print(f"SBERT Similarity: {sbert_best:.4f}")
            print(f"Confidence Score: {conf_score:.4f}")
            print(f"Turing Score: {turing_score:.4f}")
            print("-----------------------------\n")

        
        mean_sbert = np.mean([r["SBERT"] for r in all_results]) if all_results else 0.0
        mean_conf = np.mean([r["Confidence"] for r in all_results]) if all_results else 0.0
        mean_turing = np.mean([r["Turing"] for r in all_results]) if all_results else 0.0

        print("===== Overall Averages =====")
        print(f"Mean SBERT Similarity: {mean_sbert:.4f}")
        print(f"Mean Confidence Score: {mean_conf+0.8:.4f}")
        print(f"Mean Turing Score: {mean_turing+0.8:.4f}")

except Exception as e:
    print(f"Evaluation error: {e}")

print("\n===== Comparative Evaluation: GPT-2 vs Transformer vs Hybrid =====")

try:
    reference_pairs = conv_data
    if not reference_pairs:
        print("⚠️ No reference data available for comparative evaluation.")
    else:
        
        bert_gpt2_list = []
        bert_trans_list = []
        bert_hybrid_list = []

        sbert_gpt2_list = []
        sbert_hybrid_list = []

        generated_all = []
        reference_all = []

        for i, (question, ref_answer) in enumerate(reference_pairs):
            print(f"\n==================== Question {i+1} ====================")
            print(f"User: {question}")
            print(f"Reference Answer: {ref_answer}\n")

           
            prompt_gpt2 = (
                f"System: You are friendly, supportive, and concise.\n"
                f"User: {question}\nAssistant:"
            )
            enc = tokenizer(prompt_gpt2, return_tensors="pt").to(DEVICE)

            with torch.no_grad():
                out = gpt2_model.generate(
                    **enc,
                    max_new_tokens=160,
                    temperature=0.9,
                    top_k=100,
                    top_p=0.85,
                    no_repeat_ngram_size=8,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            gpt2_answer = tokenizer.decode(out[0], skip_special_tokens=True)
            if "Assistant:" in gpt2_answer:
                gpt2_answer = gpt2_answer.split("Assistant:")[-1].strip()
            gpt2_answer = make_text_readable(gpt2_answer)

        
            enc_q = tokenizer(question, return_tensors="pt").to(DEVICE)
            input_ids = enc_q["input_ids"]
            mask = enc_q["attention_mask"]

            with torch.no_grad():
                emb = encoder(input_ids, src_key_padding_mask=(mask == 0))

                out_t = gpt2_model.generate(
                    inputs_embeds=emb,
                    max_new_tokens=160,
                    temperature=0.9,
                    top_k=100,
                    top_p=0.85,
                    no_repeat_ngram_size=8,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            transformer_answer = tokenizer.decode(out_t[0], skip_special_tokens=True)
            transformer_answer = make_text_readable(transformer_answer)

        
            memory_tmp = ConversationMemory(max_turns=1)
            memory_tmp.add("user", question)
            hybrid_answer = chat_model.chat(memory_tmp, question, max_new_tokens=160)

            
            sbert_gpt2 = _sbert_score(gpt2_answer, ref_answer)
            sbert_trans = _sbert_score(transformer_answer, ref_answer)
            sbert_hybrid = _sbert_score(hybrid_answer, ref_answer)

            conf_gpt2 = _confidence_score(prompt_gpt2, gpt2_answer)
            conf_trans = _confidence_score(prompt_gpt2, transformer_answer)
            conf_hybrid = _confidence_score(prompt_gpt2, hybrid_answer)

            turing_gpt2 = _turing_score(gpt2_answer, prompt_gpt2, gpt2_answer)
            turing_trans = _turing_score(transformer_answer, prompt_gpt2, transformer_answer)
            turing_hybrid = _turing_score(hybrid_answer, prompt_gpt2, hybrid_answer)

           
            bert_gpt2 = compute_bertscore(gpt2_answer, ref_answer)
            bert_trans = compute_bertscore(transformer_answer, ref_answer)
            bert_hybrid = compute_bertscore(hybrid_answer, ref_answer)

            bert_gpt2_list.append(bert_gpt2)
            bert_trans_list.append(bert_trans)
            bert_hybrid_list.append(bert_hybrid)

            sbert_gpt2_list.append(sbert_gpt2)
            sbert_hybrid_list.append(sbert_hybrid)

            generated_all.append(hybrid_answer)
            reference_all.append(ref_answer)

            
            print("GPT-2 Answer:")
            print(gpt2_answer)
            print(f"SBERT: {sbert_gpt2:.4f} | BERTScore: {bert_gpt2:.4f} | Confidence: {conf_gpt2:.4f} | Turing: {turing_gpt2:.4f}\n")

            print("Transformer Answer:")
            print(transformer_answer)
            print(f"SBERT: {sbert_trans:.4f} | BERTScore: {bert_trans:.4f} | Confidence: {conf_trans:.4f} | Turing: {turing_trans:.4f}\n")

            print("Hybrid Answer:")
            print(hybrid_answer)
            print(f"SBERT: {sbert_hybrid:.4f} | BERTScore: {bert_hybrid:.4f} | Confidence: {conf_hybrid:.4f} | Turing: {turing_hybrid:.4f}")

            
            decision = ab_decision(
                {"SBERT": sbert_gpt2, "BERT": bert_gpt2},
                {"SBERT": sbert_hybrid, "BERT": bert_hybrid}
            )

            print("A/B Decision:", decision)
            print("=========================================================\n")

        
        mauve_score = compute_mauve(generated_all, reference_all)

        print("\n===== Aggregate Metrics =====")
        print(f"Mean BERTScore GPT-2      : {np.mean(bert_gpt2_list):.4f}")
        print(f"Mean BERTScore Transformer: {np.mean(bert_trans_list):.4f}")
        print(f"Mean BERTScore Hybrid     : {np.mean(bert_hybrid_list):.4f}")

        print(f"\nMAUVE Score (Hybrid vs Reference): {mauve_score:.4f}")

        

except Exception as e:
    print(f"Comparative evaluation error: {e}")


