import json
from collections import Counter
from tqdm.notebook import tqdm

SPECIAL_TOKENS = {
    "<pad>": 0,  
    "<sos>": 1,  
    "<eos>": 2, 
    "<unk>": 3  
}


def build_vocab(data, min_freq_ratio=0):
    token_document_count = Counter()
    total_articles = len(data)

    print(f"Total articles: {total_articles}")

    for article in tqdm(data, desc='processing articles...'):
        text_tokens = set(article['text'].split())
        title_tokens = set(article['title'].split())
        unique_tokens = text_tokens | title_tokens

        token_document_count.update(unique_tokens)

    min_document_count = max(1, int(min_freq_ratio * total_articles))
    print(f"Minimum document count: {min_document_count} (appears in {min_freq_ratio*100:.1f}% of articles)")


    vocab = {
        word: i + len(SPECIAL_TOKENS)
        for i, (word, count) in tqdm(
            enumerate(token_document_count.items()),
            desc='creating vocabulary'
        )
        if count >= min_document_count
    }

    vocab = {**SPECIAL_TOKENS, **vocab}
    print(f"Final vocabulary size: {len(vocab)}")

    return vocab



file_path = '/kaggle/input/data-json-punc/data_with_punc.json'
with open(file_path, "r") as f:
    data = json.load(f)

training_data = data['training_data']

vocab_src = build_vocab(training_data, min_freq_ratio=0.01)

def re_index_vocab(vocab):
    new_vocab = {}
    for i, token in enumerate(vocab.keys()):
        new_vocab[token] = i
    return new_vocab

vocab_src = re_index_vocab(vocab_src)

vocab_tgt = vocab_src.copy()

with open("vocab_src.json", "w") as f:
    json.dump(vocab_src, f, indent=4)

with open("vocab_tgt.json", "w") as f:
    json.dump(vocab_tgt, f, indent=4)

print("Vocabularies created and saved successfully!")


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
import random
from rouge_score import rouge_scorer
import numpy as np
from tqdm.notebook import tqdm
import os
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np


SPECIAL_TOKENS = {
    "<pad>": 0,  # Padding
    "<sos>": 1,  # Start of sequence
    "<eos>": 2,  # End of sequence
    "<unk>": 3   # Unknown word
}

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
EMB_DIM = 300  # Matches GloVe dimensions
HID_DIM = 300
BATCH_SIZE = 64
LEARNING_RATE = 0.001
N_EPOCHS = 50
MAX_LEN = 10
MAX_LEN_SRC = 500
TEACHER_FORCING_RATIO = 0.7

# Dataset
class HeadlineDataset(Dataset):
    def __init__(self, data, vocab_src):
        self.data = data
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_src

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        title = self.data[idx]['title']

        # Source sequence
        src = [self.vocab_src['<sos>']]
        src += [self.vocab_src.get(word, self.vocab_src['<unk>']) for word in text.split()[:MAX_LEN_SRC-2]]
        src.append(self.vocab_src['<eos>'])

        # Target sequence
        tgt = [self.vocab_tgt['<sos>']]
        tgt += [self.vocab_tgt.get(word, self.vocab_tgt['<unk>']) for word in title.split()[:MAX_LEN-2]]
        tgt.append(self.vocab_tgt['<eos>'])

        vocab_size = len(self.vocab_src)
        # Ensure all tokens are within valid range
        src = [tok if tok < vocab_size else self.vocab_src['<unk>'] for tok in src]
        tgt = [tok if tok < vocab_size else self.vocab_tgt['<unk>'] for tok in tgt]

        return torch.tensor(src), torch.tensor(tgt)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, padding_value=SPECIAL_TOKENS['<pad>'], batch_first=True)
    tgt_padded = pad_sequence(tgt_batch, padding_value=SPECIAL_TOKENS['<pad>'], batch_first=True)
    return src_padded, tgt_padded

def check_dataset_indices(dataset, vocab_size):
    for i in range(min(100, len(dataset))):  # Check a subset for efficiency
        src, tgt = dataset[i]
        if src.max() >= vocab_size or tgt.max() >= vocab_size:
            print(f"Invalid indices in sample {i}")
            return False
    return True

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class HierEncoderRNN(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout_rate=0.5):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Word-level GRU to process individual tokens
        self.word_gru = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        
        # Sentence-level GRU to process sentence embeddings
        self.sent_gru = nn.GRU(hid_dim * 2, hid_dim, batch_first=True, bidirectional=True)
        
        # Linear layer to combine bidirectional outputs for the decoder
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        
        # Sentence ending punctuation marks (mapped to token IDs)
        self.sent_end_tokens = set(['.', '!', '?'])

    def load_embeddings(self, glove_path, word2idx, freeze=True):
        """
        Load pre-trained GloVe embeddings into the embedding layer
        """
        # Initialize embeddings matrix with random values
        embeddings_matrix = torch.randn(len(word2idx), self.embedding.embedding_dim, device=device)

        # Set special tokens embeddings to zeros
        for token in ["<pad>", "<sos>", "<eos>", "<unk>"]:
            if token in word2idx:
                embeddings_matrix[word2idx[token]] = torch.zeros(self.embedding.embedding_dim, device=device)

        # Load GloVe embeddings
        print(f"Loading GloVe embeddings from {glove_path}")
        num_loaded = 0

        try:
            with open(glove_path, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    if word in word2idx:
                        vector = torch.FloatTensor([float(val) for val in values[1:]])
                        embeddings_matrix[word2idx[word]] = vector
                        num_loaded += 1

            print(f"Loaded embeddings for {num_loaded}/{len(word2idx)} words ({num_loaded/len(word2idx)*100:.2f}%)")

            # Load embeddings into the embedding layer
            self.embedding.weight = nn.Parameter(embeddings_matrix)

            # Optionally freeze embeddings
            if freeze:
                self.embedding.weight.requires_grad = False
                print("Embeddings frozen")

        except Exception as e:
            print(f"Error loading GloVe embeddings: {e}")
            print("Using random embeddings instead")

    def _detect_sentence_boundaries(self, tokens, idx2word):
        """
        Detect sentence boundaries based on punctuation
        Returns list of indices where sentences end
        """
        sentence_ends = []
        
        # Convert tokens to words for better sentence detection
        for i, token_idx in enumerate(tokens):
            # Skip padding and special tokens
            if token_idx == 0:  # <pad>
                continue
                
            token = idx2word.get(token_idx.item(), "")
            
            # Check if token is a sentence-ending punctuation
            if token in self.sent_end_tokens:
                sentence_ends.append(i)
                
        # If no sentence boundaries found, treat the whole text as one sentence
        if not sentence_ends:
            # Find the last non-padding token
            for i in range(len(tokens)-1, -1, -1):
                if tokens[i] != 0:  # Not padding
                    sentence_ends.append(i)
                    break
        
        return sentence_ends

    def forward(self, src, idx2word=None):
        """
        Forward pass for hierarchical encoder
        
        Args:
            src: Input token indices [batch_size, seq_len]
            idx2word: Dictionary mapping indices to words (for sentence detection)
        """
        batch_size = src.shape[0]
        device = src.device
        
        # Create a default idx2word dictionary if none provided
        if idx2word is None:
            idx2word = {i: str(i) for i in range(src.max().item() + 1)}
        
        # Embed the tokens
        embedded = self.dropout(self.embedding(src))  # [batch_size, seq_len, emb_dim]
        
        # Word-level encoding with bidirectional GRU
        word_outputs, _ = self.word_gru(embedded)  # [batch_size, seq_len, hid_dim*2]
        
        # Process each sequence in the batch to get sentence-level representations
        sentence_embeddings_list = []
        
        for batch_idx in range(batch_size):
            # Get the current sequence
            seq = src[batch_idx]
            
            # Find sentence boundary positions
            sentence_ends = self._detect_sentence_boundaries(seq, idx2word)
            
            # Extract sentence embeddings by averaging word-level outputs for each sentence
            sentence_embeddings = []
            start_idx = 0
            
            for end_idx in sentence_ends:
                # Skip empty sentences
                if end_idx < start_idx:
                    continue
                    
                # Average word-level hidden states for this sentence
                sent_emb = word_outputs[batch_idx, start_idx:end_idx+1].mean(dim=0, keepdim=True)
                sentence_embeddings.append(sent_emb)
                
                # Update start index for next sentence
                start_idx = end_idx + 1
            
            # If no valid sentences were found, use the average of all word embeddings
            if not sentence_embeddings:
                # Use mask to handle padding
                mask = (seq != 0).float().unsqueeze(-1)
                masked_outputs = word_outputs[batch_idx] * mask
                sent_emb = masked_outputs.sum(dim=0, keepdim=True) / mask.sum(dim=0, keepdim=True).clamp(min=1)
                sentence_embeddings.append(sent_emb)
            
            # Stack all sentence embeddings for this batch item
            batch_sent_emb = torch.cat(sentence_embeddings, dim=0)
            sentence_embeddings_list.append(batch_sent_emb)
        
        # Pad sentence embeddings to the same length in batch
        max_sentences = max(emb.size(0) for emb in sentence_embeddings_list)
        padded_sent_embeddings = []
        
        for emb in sentence_embeddings_list:
            # Pad if needed
            if emb.size(0) < max_sentences:
                padding = torch.zeros(max_sentences - emb.size(0), emb.size(1), device=device)
                padded_emb = torch.cat([emb, padding], dim=0)
            else:
                padded_emb = emb
            padded_sent_embeddings.append(padded_emb.unsqueeze(0))
        
        # Stack to get [batch_size, max_sentences, hid_dim*2]
        sentence_embeddings = torch.cat(padded_sent_embeddings, dim=0)
        
        # Apply sentence-level GRU to get sentence context
        _, sent_hidden = self.sent_gru(sentence_embeddings)
        
        # Process bidirectional hidden states
        hidden_forward = sent_hidden[0, :, :]  # Forward direction
        hidden_backward = sent_hidden[1, :, :]  # Backward direction
        hidden_combined = torch.cat((hidden_forward, hidden_backward), dim=1)
        hidden_transformed = torch.tanh(self.fc(hidden_combined))
        
        # Reshape to match what the decoder expects: [1, batch_size, hid_dim]
        hidden_for_decoder = hidden_transformed.unsqueeze(0)
        
        return word_outputs, hidden_for_decoder

class EncoderRNN(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout_rate=0.5):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        
        # Linear layer to combine bidirectional outputs
        self.fc = nn.Linear(hid_dim * 2, hid_dim)

    def load_embeddings(self, glove_path, word2idx, freeze=True):
        """
        Load pre-trained GloVe embeddings into the embedding layer
        """
        # Initialize embeddings matrix with random values
        embeddings_matrix = torch.FloatTensor(torch.randn(len(word2idx), self.embedding.embedding_dim))

        # Set special tokens embeddings to zeros
        for token in ["<pad>", "<sos>", "<eos>", "<unk>"]:
            if token in word2idx:
                embeddings_matrix[word2idx[token]] = torch.zeros(self.embedding.embedding_dim)

        # Load GloVe embeddings
        print(f"Loading GloVe embeddings from {glove_path}")
        num_loaded = 0

        try:
            with open(glove_path, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    if word in word2idx:
                        vector = torch.FloatTensor([float(val) for val in values[1:]])
                        embeddings_matrix[word2idx[word]] = vector
                        num_loaded += 1

            print(f"Loaded embeddings for {num_loaded}/{len(word2idx)} words ({num_loaded/len(word2idx)*100:.2f}%)")
            self.embedding.weight = nn.Parameter(embeddings_matrix)
            
            if freeze:
                self.embedding.weight.requires_grad = False
                print("Embeddings frozen")

        except Exception as e:
            print(f"Error loading GloVe embeddings: {e}")
            print("Using random embeddings instead")

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)

        # Process bidirectional hidden states
        hidden_forward = hidden[0, :, :]  # First direction
        hidden_backward = hidden[1, :, :]  # Second direction
        hidden_combined = torch.cat((hidden_forward, hidden_backward), dim=1)
        hidden_transformed = torch.tanh(self.fc(hidden_combined))
        hidden_for_decoder = hidden_transformed.unsqueeze(0)  # [1, batch_size, hidden_size]

        return outputs, hidden_for_decoder

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout_rate=0.5):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input_token, hidden):
        # input_token shape: [batch_size, 1]
        embedded = self.dropout(self.embedding(input_token))  # [batch_size, 1, emb_dim]

        # RNN processing
        output, hidden = self.rnn(embedded, hidden)  # output: [batch_size, 1, hid_dim]

        # Final projection to vocabulary
        prediction = self.fc_out(output.squeeze(1))  # [batch_size, output_dim]

        return prediction, hidden

class Decoder2RNN(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout_rate=0.5):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # First GRU
        self.gru1 = nn.GRU(emb_dim, hid_dim, batch_first=True)

        # Second GRU
        self.gru2 = nn.GRU(hid_dim, hid_dim, batch_first=True)

        # Fully connected layer to project GRU2 outputs to vocabulary size
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input_token, hidden):
        # input_token shape: [batch_size, 1]
        embedded = self.dropout(self.embedding(input_token))  # [batch_size, 1, emb_dim]

        # First GRU processing
        output1, hidden1 = self.gru1(embedded, hidden)  # output1: [batch_size, 1, hid_dim]

        # Second GRU processing - use the same initial hidden state from encoder
        output2, hidden2 = self.gru2(output1, hidden)  # output2: [batch_size, 1, hid_dim]

        # Final projection to vocabulary
        prediction = self.fc_out(output2.squeeze(1))  # [batch_size, output_dim]

        return prediction, hidden2

class Seq2seqRNN(nn.Module):
    def __init__(self, encoder_type, vocab_size, emb_dim, hid_dim, device, vocab_src, 
                 max_len=50, use_glove=True, glove_path=None, dropout_rate=0.5, decoder_type='single'):
        super().__init__()
        
        # Create encoder based on specified type
        if encoder_type == 'hierarchical':
            print("using hier encoder")
            self.encoder = HierEncoderRNN(vocab_size, emb_dim, hid_dim, dropout_rate)
        else:  # Default to standard encoder
            print("standard encoder being used!!")
            self.encoder = EncoderRNN(vocab_size, emb_dim, hid_dim, dropout_rate)
            
        # Create decoder based on specified type
        if decoder_type == 'double':
            self.decoder = Decoder2RNN(vocab_size, emb_dim, hid_dim, dropout_rate)
            print("decoder2rnn being used!!")
        else:  # Default to single GRU decoder
            print("standard decoder being used!!")
            self.decoder = Decoder(vocab_size, emb_dim, hid_dim, dropout_rate)
        
        # Store other parameters
        self.device = device
        self.max_len = max_len
        self.vocabulary = vocab_src
        self.encoder_type = encoder_type
        self.idx2word = {v: k for k, v in vocab_src.items()}

        # Load GloVe embeddings if specified
        if use_glove and glove_path:
            if os.path.exists(glove_path):
                self.encoder.load_embeddings(glove_path, vocab_src)
            else:
                print(f"Warning: GloVe embeddings path {glove_path} not found. Using random embeddings.")

    def forward(self, src, tgt=None, teacher_forcing_ratio=1.0, beam_width=1):
        if beam_width > 1 and tgt is None:
            return self.beam_search_decode(src, beam_width)
        return self.greedy_decode(src, tgt, teacher_forcing_ratio)

    def greedy_decode(self, src, tgt, teacher_forcing_ratio):
        batch_size = src.shape[0]

        # Define target length - use tgt length during training, max_len during inference
        tgt_len = tgt.shape[1] if tgt is not None else self.max_len

        # Define vocabulary size
        vocab_size = self.decoder.output_dim

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)

        # Set the first position to <sos> token
        outputs[:, 0, self.vocabulary["<sos>"]] = 1.0

        # Encode the source sequence
        if self.encoder_type == 'hierarchical':
            encoder_outputs, hidden = self.encoder(src, self.idx2word)
        else:
            encoder_outputs, hidden = self.encoder(src)

        # First decoder input is the <sos> token
        input_token = torch.tensor([[self.vocabulary["<sos>"]] * batch_size], device=self.device).T

        # Start from index 1 since index 0 is <sos>
        for t in range(1, tgt_len):
            # Pass through decoder
            prediction, hidden = self.decoder(input_token, hidden)

            # Store prediction
            outputs[:, t, :] = prediction

            # Teacher forcing: decide whether to use real target tokens
            use_teacher_forcing = random.random() < teacher_forcing_ratio

            if use_teacher_forcing and tgt is not None:
                # Use actual next token as next input
                input_token = tgt[:, t].unsqueeze(1)
            else:
                # Use best predicted token
                top1 = prediction.argmax(1).unsqueeze(1)
                input_token = top1

            # Stop if all sequences in batch have generated EOS (only during inference)
            if tgt is None and (input_token == self.vocabulary["<eos>"]).all():
                break

        return outputs

    def beam_search_decode(self, src, beam_width=3):
        batch_size = src.size(0)
        start_token = self.vocabulary["<sos>"]
        end_token = self.vocabulary["<eos>"]
        
        # Encode source sequence
        if self.encoder_type == 'hierarchical':
            _, hidden = self.encoder(src, self.idx2word)
        else:
            _, hidden = self.encoder(src)
        
        # Initialize beams (log_prob, sequence, hidden)
        beams = [([start_token], 0.0, hidden)]
        finished_beams = []

        for _ in range(self.max_len):
            candidates = []
            for seq, score, hidden_state in beams:
                if seq[-1] == end_token:
                    candidates.append((seq, score, hidden_state))
                    continue
                
                # Prepare decoder input
                input_token = torch.tensor([[seq[-1]]], device=self.device)
                
                # Get next token probabilities
                with torch.no_grad():
                    logits, new_hidden = self.decoder(input_token, hidden_state)
                    log_probs = torch.log_softmax(logits, dim=-1)

                # Get top k candidates
                top_scores, top_tokens = log_probs.topk(beam_width)
                for i in range(beam_width):
                    token = top_tokens[0, i].item()
                    new_score = score + top_scores[0, i].item()
                    new_seq = seq + [token]
                    candidates.append((new_seq, new_score, new_hidden))

            # Filter and sort candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = []
            for candidate in candidates[:beam_width]:
                seq, score, hidden = candidate
                if seq[-1] == end_token:
                    finished_beams.append(candidate)
                else:
                    beams.append(candidate)
            
            # Early stopping if all beams finished
            if not beams:
                break

        # Combine finished and unfinished beams
        final_candidates = finished_beams + beams
        final_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Get best sequence
        best_sequence = final_candidates[0][0]
        
        # Convert to output tensor
        outputs = torch.zeros(batch_size, len(best_sequence), len(self.vocabulary)).to(self.device)
        for t, token in enumerate(best_sequence):
            outputs[0, t, token] = 1.0
            
        return outputs

def train_model(data, vocab_src, encoder_type='standard', decoder_type='single', 
               use_glove=False, glove_path=None, beam_width=3):
    # ... (existing setup code remains the same) ...

    print(f"Vocab size: {len(vocab_src)}")

    # Create datasets
    train_data = HeadlineDataset(data['training_data'], vocab_src)
    val_data = HeadlineDataset(data['validation_data'], vocab_src)
    test_data = HeadlineDataset(data['test_data'], vocab_src)

    # Check dataset indices
    assert check_dataset_indices(train_data, len(vocab_src)), "Invalid indices found in training data"

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                            shuffle=True, collate_fn=collate_fn, pin_memory=True)
    
    model = Seq2seqRNN(
        encoder_type=encoder_type,  # 'standard' or 'hierarchical'
        vocab_size=len(vocab_src),
        emb_dim=EMB_DIM,
        hid_dim=HID_DIM,
        device=device,
        vocab_src=vocab_src,
        max_len=MAX_LEN,
        use_glove=use_glove,
        glove_path=glove_path,
        dropout_rate=0.5,
        decoder_type=decoder_type  # 'single' or 'double'
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab_src["<pad>"])

    # Fix any out-of-bounds indices in batches
    vocab_size = len(vocab_src)
    for batch_idx, (src, tgt) in enumerate(train_loader):
        if (src >= vocab_size).any():
            print(f"Found out-of-bounds indices in batch {batch_idx}")
            print(f"Max index: {src.max().item()}, Vocab size: {vocab_size}")
            # Fix the indices by capping them
            src[src >= vocab_size] = vocab_src["<unk>"]

        # Same check for target
        if (tgt >= vocab_size).any():
            tgt[tgt >= vocab_size] = vocab_src["<unk>"]
    
    # Training loop
    best_val_score = -1
    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0

        # Training phase (unchanged)
        for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()

            # Pass tgt for teacher forcing
            output = model(src, tgt, teacher_forcing_ratio=TEACHER_FORCING_RATIO)

            # Reshape output and target for loss calculation
            # output: [batch_size, tgt_len, vocab_size]
            # Target should exclude <sos> token (first token)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)

            # Calculate loss
            loss = criterion(output, tgt)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss/len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # Validation phase with beam search
        if (epoch+1) % 5 == 0 or epoch == N_EPOCHS-1:
            print("\nValidation Evaluation:")
            val_scores = evaluate_rouge(
                model, val_data, vocab_src, vocab_src,
                use_beam_search=True,  # Enable beam search for evaluation
                beam_width=beam_width
            )
            
            # Save best model based on ROUGE-L
            if val_scores['rougeL'] > best_val_score:
                best_val_score = val_scores['rougeL']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'rouge_score': best_val_score,
                    'vocab': vocab_src,
                    'config': {
                        'emb_dim': EMB_DIM,
                        'hid_dim': HID_DIM,
                        'max_len': MAX_LEN,
                        'encoder_type': encoder_type,
                        'decoder_type': decoder_type,
                        'use_glove': use_glove,
                        'beam_width': beam_width
                    }
                }, 'best_headline_generator.pth')
                print(f"New best model saved with ROUGE-L: {best_val_score:.3f}")


    # Final test evaluation with beam search
    print("\nTest Evaluation:")
    test_scores = evaluate_rouge(
        model, test_data, vocab_src, vocab_src,
        use_beam_search=True,
        beam_width=beam_width
    )
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab_src,
        'rouge_score': test_scores['rougeL'],
        'config': {
            'emb_dim': EMB_DIM,
            'hid_dim': HID_DIM,
            'max_len': MAX_LEN,
            'encoder_type': encoder_type,
            'decoder_type': decoder_type,
            'use_glove': use_glove,
            'beam_width': beam_width
        }
    }, 'final_headline_generator.pth')

    return model, val_scores, test_scores

def evaluate_rouge(model, dataset, vocab_src, vocab_tgt, use_beam_search=False, beam_width=3, num_examples=8):
    eval_loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    idx2word_tgt = {v: k for k, v in vocab_tgt.items()}
    
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    examples = []
    example_count = 0
    
    # Special tokens to filter
    special_tokens = {
        vocab_tgt["<pad>"], 
        vocab_tgt["<sos>"], 
        vocab_tgt["<eos>"], 
        vocab_tgt["<unk>"]
    }

    with torch.no_grad():
        for batch_idx, (src, tgt) in enumerate(tqdm(eval_loader, desc="Evaluating")):
            src = src.to(device)
            
            # Generate output with beam search or greedy decoding
            if use_beam_search:
                output = model(src, beam_width=beam_width)
            else:
                output = model(src)
            
            # Convert output to tokens
            predicted_tokens = output.argmax(dim=2)
            
            # Process batch
            for i in range(src.size(0)):
                # Get prediction and reference
                pred_seq = predicted_tokens[i].cpu().numpy()
                ref_seq = tgt[i].cpu().numpy()
                
                # Convert tokens to words
                pred_words = []
                for idx in pred_seq:
                    if idx in idx2word_tgt and idx not in special_tokens:
                        pred_words.append(idx2word_tgt[idx])
                prediction = ' '.join(pred_words)
                
                ref_words = []
                for idx in ref_seq:
                    idx = idx.item()
                    if idx in idx2word_tgt and idx not in special_tokens:
                        ref_words.append(idx2word_tgt[idx])
                reference = ' '.join(ref_words)
                
                # Skip empty predictions or references
                if not prediction.strip() or not reference.strip():
                    continue
                
                # Calculate ROUGE scores
                try:
                    rouge_scores = scorer.score(reference, prediction)
                    for key in scores:
                        scores[key].append(rouge_scores[key].fmeasure)
                    
                    # Store examples
                    if example_count < num_examples:
                        examples.append((
                            prediction,
                            reference,
                            {k: v.fmeasure for k, v in rouge_scores.items()}
                        ))
                        example_count += 1
                        
                    # Debug print first 3 examples
                    if batch_idx == 0 and i < 3:
                        print(f"\nBatch 0, Example {i}:")
                        print(f"Predicted: {prediction}")
                        print(f"Reference: {reference}")
                        print(f"ROUGE Scores: {rouge_scores}")
                        
                except Exception as e:
                    print(f"Error processing example: {e}")

    # Print collected examples
    print("\n\nEvaluation Examples:")
    for i, (pred, ref, rouge) in enumerate(examples):
        print(f"\nExample {i+1}:")
        print(f"Predicted: {pred}")
        print(f"Reference: {ref}")
        print(f"ROUGE-1: {rouge['rouge1']:.3f}")
        print(f"ROUGE-2: {rouge['rouge2']:.3f}")
        print(f"ROUGE-L: {rouge['rougeL']:.3f}")

    # Calculate average scores
    avg_scores = {}
    for metric in scores:
        if scores[metric]:
            avg_scores[metric] = np.mean(scores[metric])
        else:
            avg_scores[metric] = 0.0
            print(f"Warning: No valid scores for {metric}")

    print("\nFinal Average ROUGE Scores:")
    print(f"ROUGE-1: {avg_scores['rouge1']:.3f}")
    print(f"ROUGE-2: {avg_scores['rouge2']:.3f}")
    print(f"ROUGE-L: {avg_scores['rougeL']:.3f}")
    
    return avg_scores


# Updated main function
if __name__ == "__main__":
    # Configuration
    config = {
        'data': data,  # Your dataset loading function
        'vocab_src': vocab_src,  # Your vocab building function
        'encoder_type': 'hierarchical',  # 'standard' or 'hierarchical'
        'decoder_type': 'double',     # 'sxingle' or 'double'
        'use_glove': True,
        'glove_path': "/kaggle/input/glove-dataset/glove.6B.300d.txt",
        'beam_width': 5  # Set beam search width
    }

    # Train model with beam search evaluation
    model, val_scores, test_scores = train_model(**config)

    print("\nTesting Model on Test Dataset:")
    
    # Evaluate model on the entire test dataset using beam search
    test_data = config['data']['test_data']
    rouge_scores = evaluate_rouge(
        model=model,
        dataset=test_data,
        vocab_src=config['vocab_src'],
        vocab_tgt=config['vocab_src'],  # Assuming source and target vocab are the same
        use_beam_search=True,
        beam_width=config['beam_width']
    )

    print("\nFinal ROUGE Scores on Test Dataset:")
    print(f"ROUGE-1: {rouge_scores['rouge1']:.3f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.3f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.3f}")

    # Example inference with beam search
    print("\nExample Inference:")
    test_words = ["<sos>", "breaking", "news", "about", "ai", "advances", "<eos>"]
    test_input = torch.tensor([
        [config['vocab_src'].get(word, config['vocab_src']["<unk>"]) for word in test_words]
    ]).to(model.device)

    with torch.no_grad():
        # Use beam search explicitly
        beam_output = model(test_input, beam_width=config['beam_width'])
        
        # Convert output to tokens (shape: [batch_size, seq_len])
        predicted_tokens = beam_output.argmax(dim=-1)[0]  # Get first batch
        
        # Convert tokens to words
        headline_words = []
        for idx in predicted_tokens.cpu().numpy():
            word = model.idx2word.get(idx, "<unk>")
            if word == "<eos>":
                break
            if word not in ["<sos>", "<pad>", "<unk>"]:
                headline_words.append(word)
                
        print(f"\nGenerated Headline: {' '.join(headline_words)}")
        
        
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
import random
from rouge_score import rouge_scorer
import numpy as np
from tqdm.notebook import tqdm
import os

SPECIAL_TOKENS = {
    "<pad>": 0,  # Padding
    "<sos>": 1,  # Start of sequence
    "<eos>": 2,  # End of sequence
    "<unk>": 3   # Unknown word
}

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
EMB_DIM = 300  # Matches GloVe dimensions
HID_DIM = 300
BATCH_SIZE = 64
LEARNING_RATE = 0.001
N_EPOCHS = 50
MAX_LEN = 10
MAX_LEN_SRC = 500
TEACHER_FORCING_RATIO = 0.7

# Dataset
class HeadlineDataset(Dataset):
    def __init__(self, data, vocab_src):
        self.data = data
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_src

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        title = self.data[idx]['title']

        # Source sequence
        src = [self.vocab_src['<sos>']]
        src += [self.vocab_src.get(word, self.vocab_src['<unk>']) for word in text.split()[:MAX_LEN_SRC-2]]
        src.append(self.vocab_src['<eos>'])

        # Target sequence
        tgt = [self.vocab_tgt['<sos>']]
        tgt += [self.vocab_tgt.get(word, self.vocab_tgt['<unk>']) for word in title.split()[:MAX_LEN-2]]
        tgt.append(self.vocab_tgt['<eos>'])

        vocab_size = len(self.vocab_src)
        # Ensure all tokens are within valid range
        src = [tok if tok < vocab_size else self.vocab_src['<unk>'] for tok in src]
        tgt = [tok if tok < vocab_size else self.vocab_tgt['<unk>'] for tok in tgt]

        return torch.tensor(src), torch.tensor(tgt)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, padding_value=SPECIAL_TOKENS['<pad>'], batch_first=True)
    tgt_padded = pad_sequence(tgt_batch, padding_value=SPECIAL_TOKENS['<pad>'], batch_first=True)
    return src_padded, tgt_padded

def check_dataset_indices(dataset, vocab_size):
    for i in range(min(100, len(dataset))):  # Check a subset for efficiency
        src, tgt = dataset[i]
        if src.max() >= vocab_size or tgt.max() >= vocab_size:
            print(f"Invalid indices in sample {i}")
            return False
    return True

# Modified Encoder with GloVe embedding loading capability
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout_rate=0.5):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)

        # Linear layer to combine bidirectional outputs for the decoder
        self.fc = nn.Linear(hid_dim * 2, hid_dim)

    def load_embeddings(self, glove_path, word2idx, freeze=True):
        """
        Load pre-trained GloVe embeddings into the embedding layer

        Args:
            glove_path (str): Path to the GloVe embeddings file
            word2idx (dict): Vocabulary mapping words to indices
            freeze (bool): Whether to freeze embeddings during training
        """
        # Initialize embeddings matrix with random values
        embeddings_matrix = torch.FloatTensor(torch.randn(len(word2idx), self.embedding.embedding_dim))

        # Set special tokens embeddings to zeros
        for token in SPECIAL_TOKENS:
            if token in word2idx:
                embeddings_matrix[word2idx[token]] = torch.zeros(self.embedding.embedding_dim)

        # Load GloVe embeddings
        print(f"Loading GloVe embeddings from {glove_path}")
        num_loaded = 0

        try:
            with open(glove_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Loading GloVe"):
                    values = line.split()
                    word = values[0]

                    if word in word2idx:
                        vector = torch.FloatTensor([float(val) for val in values[1:]])
                        embeddings_matrix[word2idx[word]] = vector
                        num_loaded += 1

            print(f"Loaded embeddings for {num_loaded}/{len(word2idx)} words ({num_loaded/len(word2idx)*100:.2f}%)")

            # Load embeddings into the embedding layer
            self.embedding.weight = nn.Parameter(embeddings_matrix)

            # Optionally freeze embeddings
            if freeze:
                self.embedding.weight.requires_grad = False
                print("Embeddings frozen")

        except Exception as e:
            print(f"Error loading GloVe embeddings: {e}")
            print("Using random embeddings instead")

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)

        # Process bidirectional hidden states
        hidden_forward = hidden[0, :, :]  # First direction: [1, batch_size, hidden_size]
        hidden_backward = hidden[1, :, :]  # Second direction: [1, batch_size, hidden_size]
        hidden_combined = torch.cat((hidden_forward, hidden_backward), dim=1)
        hidden_transformed = torch.tanh(self.fc(hidden_combined))
        hidden_for_decoder = hidden_transformed.unsqueeze(0)  # [1, batch_size, hidden_size]

        return outputs, hidden_for_decoder

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout_rate=0.5):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input_token, hidden):
        # input_token shape: [batch_size, 1]
        embedded = self.dropout(self.embedding(input_token))  # [batch_size, 1, emb_dim]

        # RNN processing
        output, hidden = self.rnn(embedded, hidden)  # output: [batch_size, 1, hid_dim]

        # Final projection to vocabulary
        prediction = self.fc_out(output.squeeze(1))  # [batch_size, output_dim]

        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, vocab_src, max_len=50, use_glove=True, glove_path=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.max_len = max_len
        self.vocabulary = vocab_src

        # Load GloVe embeddings if specified
        if use_glove and glove_path:
            if os.path.exists(glove_path):
                self.encoder.load_embeddings(glove_path, vocab_src)
            else:
                print(f"Warning: GloVe embeddings path {glove_path} not found. Using random embeddings.")

    def forward(self, src, tgt=None, teacher_forcing_ratio=1.0):
        batch_size = src.shape[0]

        # Define target length - use tgt length during training, max_len during inference
        tgt_len = tgt.shape[1] if tgt is not None else self.max_len

        # Define vocabulary size
        vocab_size = self.decoder.output_dim

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)

        # First token is always <sos>
        outputs[:, 0, self.vocabulary["<sos>"]] = 1.0

        # Encode the source sequence
        encoder_outputs, hidden = self.encoder(src)

        # First decoder input is the <sos> token
        input_token = torch.tensor([[self.vocabulary["<sos>"]] * batch_size], device=self.device).T

        # Start from index 1 since index 0 is <sos>
        for t in range(1, tgt_len):
            # Pass through decoder
            prediction, hidden = self.decoder(input_token, hidden)

            # Store prediction
            outputs[:, t, :] = prediction

            # Teacher forcing: decide whether to use real target tokens
            use_teacher_forcing = random.random() < teacher_forcing_ratio

            if use_teacher_forcing and tgt is not None:
                # Use actual next token as next input
                input_token = tgt[:, t].unsqueeze(1)
            else:
                # Use best predicted token
                top1 = prediction.argmax(1).unsqueeze(1)
                input_token = top1

            # Stop if all sequences in batch have generated EOS (only during inference)
            if tgt is None and (input_token == self.vocabulary["<eos>"]).all():
                break

        return outputs

def evaluate_rouge(model, dataset, vocab_src, vocab_tgt, num_examples=8):
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    idx2word_tgt = {v: k for k, v in vocab_tgt.items()}

    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    examples = []

    # Get special token IDs
    special_tokens = {vocab_tgt["<pad>"], vocab_tgt["<sos>"], vocab_tgt["<eos>"], vocab_tgt["<unk>"]}

    with torch.no_grad():
        for i in range(min(100, len(dataset))):  # Evaluate on max 100 examples
            src, tgt = dataset[i]
            src = src.unsqueeze(0).to(model.device)  # Add batch dimension

            # Generate output with the model (no target provided for inference)
            outputs = model(src, tgt=None, teacher_forcing_ratio=0)

            # Get the predicted token indices
            output_tokens = outputs[0].argmax(dim=1).cpu().numpy()

            # Convert tokens to words, filtering out special tokens
            pred_words = []
            for idx in output_tokens:
                if idx in idx2word_tgt and idx not in special_tokens:
                    pred_words.append(idx2word_tgt[idx])
            pred = ' '.join(pred_words)

            # Process target tokens
            true_words = []
            for idx in tgt:
                idx_item = idx.item()
                if idx_item in idx2word_tgt and idx_item not in special_tokens:
                    true_words.append(idx2word_tgt[idx_item])
            true = ' '.join(true_words)

            # Debug information
            if i < 5:  # Debug first few examples
                print(f"\nDebug Example {i+1}:")
                print(f"Raw prediction tokens: {output_tokens}")
                print(f"Predicted words: {pred_words}")
                print(f"Predicted: '{pred}'")
                print(f"True: '{true}'")

            # Skip empty predictions or targets
            if not pred.strip() or not true.strip():
                print(f"Skipping example {i+1} due to empty prediction or target")
                continue

            # Calculate ROUGE scores
            try:
                rouge_scores = scorer.score(true, pred)
                for key in scores:
                    scores[key].append(rouge_scores[key].fmeasure)

                # Save examples
                if len(examples) < num_examples:
                    examples.append((pred, true, rouge_scores))
            except Exception as e:
                print(f"Error calculating ROUGE for example {i+1}: {e}")

    # Print examples
    print("\nEvaluation Examples:")
    for i, (pred, true, rouge) in enumerate(examples):
        print(f"\nExample {i+1}:")
        print(f"Predicted: {pred}")
        print(f"True: {true}")
        print(f"ROUGE-1: {rouge['rouge1'].fmeasure:.3f}")
        print(f"ROUGE-2: {rouge['rouge2'].fmeasure:.3f}")
        print(f"ROUGE-L: {rouge['rougeL'].fmeasure:.3f}")

    # Calculate average scores
    if not any(scores.values()):
        print("Warning: No valid ROUGE scores calculated!")
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    avg_scores = {k: np.mean(v) if v else 0.0 for k, v in scores.items()}
    print("\nAverage ROUGE Scores:")
    print(f"ROUGE-1: {avg_scores['rouge1']:.3f}")
    print(f"ROUGE-2: {avg_scores['rouge2']:.3f}")
    print(f"ROUGE-L: {avg_scores['rougeL']:.3f}")

    return avg_scores

def load_glove_embeddings(glove_path, vocab_src, emb_dim=300):
    """
    Utility function to load GloVe embeddings separately
    """
    print(f"Loading GloVe embeddings from {glove_path}")
    word_vectors = {}

    try:
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading GloVe file"):
                values = line.split()
                word = values[0]
                vectors = torch.FloatTensor([float(val) for val in values[1:]])
                word_vectors[word] = vectors

        print(f"Loaded {len(word_vectors)} word vectors from GloVe")

        # Create embedding matrix
        embeddings_matrix = torch.randn(len(vocab_src), emb_dim)

        # Set special tokens to zeros
        for token in SPECIAL_TOKENS:
            if token in vocab_src:
                embeddings_matrix[vocab_src[token]] = torch.zeros(emb_dim)

        # Populate with GloVe vectors
        num_loaded = 0
        for word, idx in vocab_src.items():
            if word in word_vectors:
                embeddings_matrix[idx] = word_vectors[word]
                num_loaded += 1

        print(f"Initialized {num_loaded}/{len(vocab_src)} words with GloVe vectors ({num_loaded/len(vocab_src)*100:.2f}%)")
        return embeddings_matrix

    except Exception as e:
        print(f"Error loading GloVe embeddings: {e}")
        return None

def train_model(data, vocab_src, use_glove=False, glove_path=None):
    # Build vocabularies
    print(f"Vocab size: {len(vocab_src)}")

    # Create datasets
    train_data = HeadlineDataset(data['training_data'], vocab_src)
    val_data = HeadlineDataset(data['validation_data'], vocab_src)
    test_data = HeadlineDataset(data['test_data'], vocab_src)

    # Check dataset indices
    assert check_dataset_indices(train_data, len(vocab_src)), "Invalid indices found in training data"

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                            shuffle=True, collate_fn=collate_fn, pin_memory=True)

    # Initialize model components
    encoder = Encoder(len(vocab_src), EMB_DIM, HID_DIM)
    decoder = Decoder(len(vocab_src), EMB_DIM, HID_DIM)

    # Pre-load GloVe embeddings if requested
    if use_glove and glove_path and os.path.exists(glove_path):
        print("Preloading GloVe embeddings...")
        glove_matrix = load_glove_embeddings(glove_path, vocab_src, EMB_DIM)
        if glove_matrix is not None:
            # Set encoder embeddings
            encoder.embedding.weight = nn.Parameter(glove_matrix)
            # Optionally freeze embeddings
            encoder.embedding.weight.requires_grad = False
            print("Encoder embeddings initialized with GloVe and frozen")

    # Create Seq2Seq model
    model = Seq2Seq(encoder, decoder, device, vocab_src,
                   max_len=MAX_LEN,
                   use_glove=False,  # We loaded embeddings manually above
                   glove_path=None).to(device)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=SPECIAL_TOKENS['<pad>'])

    # Fix any out-of-bounds indices in batches
    vocab_size = len(vocab_src)
    for batch_idx, (src, tgt) in enumerate(train_loader):
        if (src >= vocab_size).any():
            print(f"Found out-of-bounds indices in batch {batch_idx}")
            print(f"Max index: {src.max().item()}, Vocab size: {vocab_size}")
            # Fix the indices by capping them
            src[src >= vocab_size] = vocab_src["<unk>"]

        # Same check for target
        if (tgt >= vocab_size).any():
            tgt[tgt >= vocab_size] = vocab_src["<unk>"]

    # Training loop
    best_val_score = -1  # Initialize with negative value to ensure first model is saved
    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0

        for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()

            # Pass tgt for teacher forcing
            output = model(src, tgt, teacher_forcing_ratio=TEACHER_FORCING_RATIO)

            # Reshape output and target for loss calculation
            # output: [batch_size, tgt_len, vocab_size]
            # Target should exclude <sos> token (first token)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)

            # Calculate loss
            loss = criterion(output, tgt)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss/len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # Evaluate every 5 epochs or at the end
        if (epoch+1) % 5 == 0 or epoch == N_EPOCHS-1:
            print("\nValidation Evaluation:")
            val_scores = evaluate_rouge(model, val_data, vocab_src, vocab_src)

            # Save the best model
            rouge_l_score = val_scores['rougeL']
            if rouge_l_score > best_val_score:
                best_val_score = rouge_l_score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'rouge_score': rouge_l_score,
                    'vocab': vocab_src,
                    'config': {
                        'emb_dim': EMB_DIM,
                        'hid_dim': HID_DIM,
                        'max_len': MAX_LEN,
                        'max_len_src': MAX_LEN_SRC,
                        'use_glove': use_glove
                    }
                }, 'best_headline_generator.pth')
                print(f"New best model saved with ROUGE-L: {best_val_score:.3f}")

    # Final evaluation
    print("\nTest Evaluation:")
    test_scores = evaluate_rouge(model, test_data, vocab_src, vocab_src)

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab_src,
        'rouge_score': test_scores['rougeL'],
        'config': {
            'emb_dim': EMB_DIM,
            'hid_dim': HID_DIM,
            'max_len': MAX_LEN,
            'max_len_src': MAX_LEN_SRC,
            'use_glove': use_glove
        }
    }, 'final_headline_generator.pth')

    return model, val_scores, test_scores

# Example usage
if __name__ == "__main__":

    # Path to your pre-downloaded GloVe embeddings
    glove_path = "/kaggle/input/glove-dataset/glove.6B.300d.txt"

    # Train with GloVe embeddings
    model, val_scores, test_scores = train_model(
        data,
        vocab_src,  # Assuming this is already defined
        use_glove=False,
        glove_path=glove_path
    )
    
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
import random
from rouge_score import rouge_scorer
import numpy as np
from tqdm.notebook import tqdm
import os

SPECIAL_TOKENS = {
    "<pad>": 0,  # Padding
    "<sos>": 1,  # Start of sequence
    "<eos>": 2,  # End of sequence
    "<unk>": 3   # Unknown word
}

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
EMB_DIM = 300  # Matches GloVe dimensions
HID_DIM = 300
BATCH_SIZE = 64
LEARNING_RATE = 0.001
N_EPOCHS = 50
MAX_LEN = 10
MAX_LEN_SRC = 500
TEACHER_FORCING_RATIO = 0.7

# Dataset
class HeadlineDataset(Dataset):
    def __init__(self, data, vocab_src):
        self.data = data
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_src

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        title = self.data[idx]['title']

        # Source sequence
        src = [self.vocab_src['<sos>']]
        src += [self.vocab_src.get(word, self.vocab_src['<unk>']) for word in text.split()[:MAX_LEN_SRC-2]]
        src.append(self.vocab_src['<eos>'])

        # Target sequence
        tgt = [self.vocab_tgt['<sos>']]
        tgt += [self.vocab_tgt.get(word, self.vocab_tgt['<unk>']) for word in title.split()[:MAX_LEN-2]]
        tgt.append(self.vocab_tgt['<eos>'])

        vocab_size = len(self.vocab_src)
        # Ensure all tokens are within valid range
        src = [tok if tok < vocab_size else self.vocab_src['<unk>'] for tok in src]
        tgt = [tok if tok < vocab_size else self.vocab_tgt['<unk>'] for tok in tgt]

        return torch.tensor(src), torch.tensor(tgt)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, padding_value=SPECIAL_TOKENS['<pad>'], batch_first=True)
    tgt_padded = pad_sequence(tgt_batch, padding_value=SPECIAL_TOKENS['<pad>'], batch_first=True)
    return src_padded, tgt_padded

def check_dataset_indices(dataset, vocab_size):
    for i in range(min(100, len(dataset))):  # Check a subset for efficiency
        src, tgt = dataset[i]
        if src.max() >= vocab_size or tgt.max() >= vocab_size:
            print(f"Invalid indices in sample {i}")
            return False
    return True

# Modified Encoder with GloVe embedding loading capability
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout_rate=0.5):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)

        # Linear layer to combine bidirectional outputs for the decoder
        self.fc = nn.Linear(hid_dim * 2, hid_dim)

    def load_embeddings(self, glove_path, word2idx, freeze=True):
        """
        Load pre-trained GloVe embeddings into the embedding layer

        Args:
            glove_path (str): Path to the GloVe embeddings file
            word2idx (dict): Vocabulary mapping words to indices
            freeze (bool): Whether to freeze embeddings during training
        """
        # Initialize embeddings matrix with random values
        embeddings_matrix = torch.FloatTensor(torch.randn(len(word2idx), self.embedding.embedding_dim, device=device))

        # Set special tokens embeddings to zeros
        for token in SPECIAL_TOKENS:
            if token in word2idx:
                embeddings_matrix[word2idx[token]] = torch.zeros(self.embedding.embedding_dim, device=device)
    
        # Load GloVe embeddings
        print(f"Loading GloVe embeddings from {glove_path}")
        num_loaded = 0

        

        try:
            new_vocab_src = {}
            
            with open(glove_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Loading GloVe"):
                    values = line.split()
                    word = values[0]

                    if word in word2idx:
                        vector = torch.FloatTensor([float(val) for val in values[1:]])
                        embeddings_matrix[word2idx[word]] = vector
                        num_loaded += 1

            print(f"Loaded embeddings for {num_loaded}/{len(word2idx)} words ({num_loaded/len(word2idx)*100:.2f}%)")
        
            # Load embeddings into the embedding layer
            self.embedding.weight = nn.Parameter(embeddings_matrix)

            # Optionally freeze embeddings
            if freeze:
                self.embedding.weight.requires_grad = False
                print("Embeddings frozen")

        except Exception as e:
            print(f"Error loading GloVe embeddings: {e}")
            print("Using random embeddings instead")

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)

        # Process bidirectional hidden states
        hidden_forward = hidden[0, :, :]  # First direction: [1, batch_size, hidden_size]
        hidden_backward = hidden[1, :, :]  # Second direction: [1, batch_size, hidden_size]
        hidden_combined = torch.cat((hidden_forward, hidden_backward), dim=1)
        hidden_transformed = torch.tanh(self.fc(hidden_combined))
        hidden_for_decoder = hidden_transformed.unsqueeze(0)  # [1, batch_size, hidden_size]

        return outputs, hidden_for_decoder

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout_rate=0.5):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input_token, hidden):
        # input_token shape: [batch_size, 1]
        embedded = self.dropout(self.embedding(input_token))  # [batch_size, 1, emb_dim]

        # RNN processing
        output, hidden = self.rnn(embedded, hidden)  # output: [batch_size, 1, hid_dim]

        # Final projection to vocabulary
        prediction = self.fc_out(output.squeeze(1))  # [batch_size, output_dim]

        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, vocab_src, max_len=50, use_glove=True, glove_path=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.max_len = max_len
        self.vocabulary = vocab_src

        # Load GloVe embeddings if specified
        if use_glove and glove_path:
            if os.path.exists(glove_path):
                self.encoder.load_embeddings(glove_path, vocab_src)
            else:
                print(f"Warning: GloVe embeddings path {glove_path} not found. Using random embeddings.")

    def forward(self, src, tgt=None, teacher_forcing_ratio=1.0):
        batch_size = src.shape[0]

        # Define target length - use tgt length during training, max_len during inference
        tgt_len = tgt.shape[1] if tgt is not None else self.max_len

        # Define vocabulary size
        vocab_size = self.decoder.output_dim

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)

        # First token is always <sos>
        outputs[:, 0, self.vocabulary["<sos>"]] = 1.0

        # Encode the source sequence
        encoder_outputs, hidden = self.encoder(src)

        # First decoder input is the <sos> token
        input_token = torch.tensor([[self.vocabulary["<sos>"]] * batch_size], device=self.device).T

        # Start from index 1 since index 0 is <sos>
        for t in range(1, tgt_len):
            # Pass through decoder
            prediction, hidden = self.decoder(input_token, hidden)

            # Store prediction
            outputs[:, t, :] = prediction

            # Teacher forcing: decide whether to use real target tokens
            use_teacher_forcing = random.random() < teacher_forcing_ratio

            if use_teacher_forcing and tgt is not None:
                # Use actual next token as next input
                input_token = tgt[:, t].unsqueeze(1)
            else:
                # Use best predicted token
                top1 = prediction.argmax(1).unsqueeze(1)
                input_token = top1

            # Stop if all sequences in batch have generated EOS (only during inference)
            if tgt is None and (input_token == self.vocabulary["<eos>"]).all():
                break

        return outputs

def evaluate_rouge(model, dataset, vocab_src, vocab_tgt, num_examples=8):
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    idx2word_tgt = {v: k for k, v in vocab_tgt.items()}

    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    examples = []

    # Get special token IDs
    special_tokens = {vocab_tgt["<pad>"], vocab_tgt["<sos>"], vocab_tgt["<eos>"], vocab_tgt["<unk>"]}

    with torch.no_grad():
        for i in range(min(100, len(dataset))):  # Evaluate on max 100 examples
            src, tgt = dataset[i]
            src = src.unsqueeze(0).to(model.device)  # Add batch dimension

            # Generate output with the model (no target provided for inference)
            outputs = model(src, tgt=None, teacher_forcing_ratio=0)

            # Get the predicted token indices
            output_tokens = outputs[0].argmax(dim=1).cpu().numpy()

            # Convert tokens to words, filtering out special tokens
            pred_words = []
            for idx in output_tokens:
                if idx in idx2word_tgt and idx not in special_tokens:
                    pred_words.append(idx2word_tgt[idx])
            pred = ' '.join(pred_words)

            # Process target tokens
            true_words = []
            for idx in tgt:
                idx_item = idx.item()
                if idx_item in idx2word_tgt and idx_item not in special_tokens:
                    true_words.append(idx2word_tgt[idx_item])
            true = ' '.join(true_words)

            # Debug information
            if i < 5:  # Debug first few examples
                print(f"\nDebug Example {i+1}:")
                print(f"Raw prediction tokens: {output_tokens}")
                print(f"Predicted words: {pred_words}")
                print(f"Predicted: '{pred}'")
                print(f"True: '{true}'")

            # Skip empty predictions or targets
            if not pred.strip() or not true.strip():
                print(f"Skipping example {i+1} due to empty prediction or target")
                continue

            # Calculate ROUGE scores
            try:
                rouge_scores = scorer.score(true, pred)
                for key in scores:
                    scores[key].append(rouge_scores[key].fmeasure)

                # Save examples
                if len(examples) < num_examples:
                    examples.append((pred, true, rouge_scores))
            except Exception as e:
                print(f"Error calculating ROUGE for example {i+1}: {e}")

    # Print examples
    print("\nEvaluation Examples:")
    for i, (pred, true, rouge) in enumerate(examples):
        print(f"\nExample {i+1}:")
        print(f"Predicted: {pred}")
        print(f"True: {true}")
        print(f"ROUGE-1: {rouge['rouge1'].fmeasure:.3f}")
        print(f"ROUGE-2: {rouge['rouge2'].fmeasure:.3f}")
        print(f"ROUGE-L: {rouge['rougeL'].fmeasure:.3f}")

    # Calculate average scores
    if not any(scores.values()):
        print("Warning: No valid ROUGE scores calculated!")
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    avg_scores = {k: np.mean(v) if v else 0.0 for k, v in scores.items()}
    print("\nAverage ROUGE Scores:")
    print(f"ROUGE-1: {avg_scores['rouge1']:.3f}")
    print(f"ROUGE-2: {avg_scores['rouge2']:.3f}")
    print(f"ROUGE-L: {avg_scores['rougeL']:.3f}")

    return avg_scores

def load_glove_embeddings(glove_path, vocab_src, emb_dim=300):
    """
    Utility function to load GloVe embeddings separately
    """
    print(f"Loading GloVe embeddings from {glove_path}")
    word_vectors = {}

    try:
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading GloVe file"):
                values = line.split()
                word = values[0]
                vectors = torch.FloatTensor([float(val) for val in values[1:]])
                word_vectors[word] = vectors

        print(f"Loaded {len(word_vectors)} word vectors from GloVe")

        # Create embedding matrix
        embeddings_matrix = torch.randn(len(vocab_src), emb_dim).to(device)

        # Set special tokens to zeros
        for token in SPECIAL_TOKENS:
            if token in vocab_src:
                embeddings_matrix[vocab_src[token]] = torch.zeros(emb_dim)

        # Populate with GloVe vectors
        num_loaded = 0
        list_of_loaded_words = []
        new_vocab_src = {}
        
        for word, idx in vocab_src.items():
            if word in word_vectors:
                embeddings_matrix[idx] = word_vectors[word]
                num_loaded += 1
                list_of_loaded_words.append(word)
                
        for key, val in vocab_src.items():
            if key  in list_of_loaded_words:
                new_vocab_src[key] = val

        vocab_src = new_vocab_src

        print(f'length of new_vocab is {len(new_vocab_src)}\n')
        print(f'length of vocab is {len(vocab_src)}\n')

        print(f"Initialized {num_loaded}/{len(vocab_src)} words with GloVe vectors ({num_loaded/len(vocab_src)*100:.2f}%)")
        return embeddings_matrix

    except Exception as e:
        print(f"Error loading GloVe embeddings: {e}")
        return None

def train_model(data, vocab_src, use_glove=False, glove_path=None):
    # Build vocabularies
    print(f"Vocab size: {len(vocab_src)}")

    # Create datasets
    train_data = HeadlineDataset(data['training_data'], vocab_src)
    val_data = HeadlineDataset(data['validation_data'], vocab_src)
    test_data = HeadlineDataset(data['test_data'], vocab_src)

    # Check dataset indices
    assert check_dataset_indices(train_data, len(vocab_src)), "Invalid indices found in training data"

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                            shuffle=True, collate_fn=collate_fn, pin_memory=True)

    # Initialize model components
    encoder = Encoder(len(vocab_src), EMB_DIM, HID_DIM)
    decoder = Decoder(len(vocab_src), EMB_DIM, HID_DIM)

    # Pre-load GloVe embeddings if requested
    if use_glove and glove_path and os.path.exists(glove_path):
        print("Preloading GloVe embeddings...")
        glove_matrix = load_glove_embeddings(glove_path, vocab_src, EMB_DIM)
        if glove_matrix is not None:
            # Set encoder embeddings
            encoder.embedding.weight = nn.Parameter(glove_matrix)
            # Optionally freeze embeddings
            encoder.embedding.weight.requires_grad = False
            print("Encoder embeddings initialized with GloVe and frozen")

    # Create Seq2Seq model
    model = Seq2Seq(encoder, decoder, device, vocab_src,
                   max_len=MAX_LEN,
                   use_glove=False,  # We loaded embeddings manually above
                   glove_path=None).to(device)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=SPECIAL_TOKENS['<pad>'])

    # Fix any out-of-bounds indices in batches
    vocab_size = len(vocab_src)
    for batch_idx, (src, tgt) in enumerate(train_loader):
        if (src >= vocab_size).any():
            print(f"Found out-of-bounds indices in batch {batch_idx}")
            print(f"Max index: {src.max().item()}, Vocab size: {vocab_size}")
            # Fix the indices by capping them
            src[src >= vocab_size] = vocab_src["<unk>"]

        # Same check for target
        if (tgt >= vocab_size).any():
            tgt[tgt >= vocab_size] = vocab_src["<unk>"]

    # Training loop
    best_val_score = -1  # Initialize with negative value to ensure first model is saved
    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0

        for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()

            # Pass tgt for teacher forcing
            output = model(src, tgt, teacher_forcing_ratio=TEACHER_FORCING_RATIO)

            # Reshape output and target for loss calculation
            # output: [batch_size, tgt_len, vocab_size]
            # Target should exclude <sos> token (first token)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)

            # Calculate loss
            loss = criterion(output, tgt)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss/len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # Evaluate every 5 epochs or at the end
        if (epoch+1) % 5 == 0 or epoch == N_EPOCHS-1:
            print("\nValidation Evaluation:")
            val_scores = evaluate_rouge(model, val_data, vocab_src, vocab_src)

            # Save the best model
            rouge_l_score = val_scores['rougeL']
            if rouge_l_score > best_val_score:
                best_val_score = rouge_l_score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'rouge_score': rouge_l_score,
                    'vocab': vocab_src,
                    'config': {
                        'emb_dim': EMB_DIM,
                        'hid_dim': HID_DIM,
                        'max_len': MAX_LEN,
                        'max_len_src': MAX_LEN_SRC,
                        'use_glove': use_glove
                    }
                }, 'best_headline_generator.pth')
                print(f"New best model saved with ROUGE-L: {best_val_score:.3f}")

    # Final evaluation
    print("\nTest Evaluation:")
    test_scores = evaluate_rouge(model, test_data, vocab_src, vocab_src)

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab_src,
        'rouge_score': test_scores['rougeL'],
        'config': {
            'emb_dim': EMB_DIM,
            'hid_dim': HID_DIM,
            'max_len': MAX_LEN,
            'max_len_src': MAX_LEN_SRC,
            'use_glove': use_glove
        }
    }, 'final_headline_generator.pth')

    return model, val_scores, test_scores

# Example usage
if __name__ == "__main__":

    # Path to your pre-downloaded GloVe embeddings
    glove_path = "/kaggle/input/glove-dataset/glove.6B.300d.txt"

    # Train with GloVe embeddings
    model, val_scores, test_scores = train_model(
        data,
        vocab_src,  # Assuming this is already defined
        use_glove=True,
        glove_path=glove_path
    )
    
# anant heier RNN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np

EMB_DIM = 300  # Matches GloVe dimensions
HID_DIM = 300
BATCH_SIZE = 64
LEARNING_RATE = 0.001
N_EPOCHS = 30
MAX_LEN = 10
MAX_LEN_SRC = 300
TEACHER_FORCING_RATIO = 0.7


# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class HierEncoderRNN(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout_rate=0.5):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Word-level GRU to process individual tokens
        self.word_gru = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        
        # Sentence-level GRU to process sentence embeddings
        self.sent_gru = nn.GRU(hid_dim * 2, hid_dim, batch_first=True, bidirectional=True)
        
        # Linear layer to combine bidirectional outputs for the decoder
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        
        # Sentence ending punctuation marks (mapped to token IDs)
        self.sent_end_tokens = set(['.', '!', '?'])

    def load_embeddings(self, glove_path, word2idx, freeze=True):
        """
        Load pre-trained GloVe embeddings into the embedding layer
        """
        # Initialize embeddings matrix with random values
        embeddings_matrix = torch.randn(len(word2idx), self.embedding.embedding_dim, device=device)

        # Set special tokens embeddings to zeros
        for token in ["<pad>", "<sos>", "<eos>", "<unk>"]:
            if token in word2idx:
                embeddings_matrix[word2idx[token]] = torch.zeros(self.embedding.embedding_dim, device=device)

        # Load GloVe embeddings
        print(f"Loading GloVe embeddings from {glove_path}")
        num_loaded = 0

        try:
            with open(glove_path, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    if word in word2idx:
                        vector = torch.FloatTensor([float(val) for val in values[1:]])
                        embeddings_matrix[word2idx[word]] = vector
                        num_loaded += 1

            print(f"Loaded embeddings for {num_loaded}/{len(word2idx)} words ({num_loaded/len(word2idx)*100:.2f}%)")

            # Load embeddings into the embedding layer
            self.embedding.weight = nn.Parameter(embeddings_matrix)

            # Optionally freeze embeddings
            if freeze:
                self.embedding.weight.requires_grad = False
                print("Embeddings frozen")

        except Exception as e:
            print(f"Error loading GloVe embeddings: {e}")
            print("Using random embeddings instead")

    def _detect_sentence_boundaries(self, tokens, idx2word):
        """
        Detect sentence boundaries based on punctuation
        Returns list of indices where sentences end
        """
        sentence_ends = []
        
        # Convert tokens to words for better sentence detection
        for i, token_idx in enumerate(tokens):
            # Skip padding and special tokens
            if token_idx == 0:  # <pad>
                continue
                
            token = idx2word.get(token_idx.item(), "")
            
            # Check if token is a sentence-ending punctuation
            if token in self.sent_end_tokens:
                sentence_ends.append(i)
                
        # If no sentence boundaries found, treat the whole text as one sentence
        if not sentence_ends:
            # Find the last non-padding token
            for i in range(len(tokens)-1, -1, -1):
                if tokens[i] != 0:  # Not padding
                    sentence_ends.append(i)
                    break
        
        return sentence_ends

    def forward(self, src, idx2word=None):
        """
        Forward pass for hierarchical encoder
        
        Args:
            src: Input token indices [batch_size, seq_len]
            idx2word: Dictionary mapping indices to words (for sentence detection)
        """
        batch_size = src.shape[0]
        device = src.device
        
        # Create a default idx2word dictionary if none provided
        if idx2word is None:
            idx2word = {i: str(i) for i in range(src.max().item() + 1)}
        
        # Embed the tokens
        embedded = self.dropout(self.embedding(src))  # [batch_size, seq_len, emb_dim]
        
        # Word-level encoding with bidirectional GRU
        word_outputs, _ = self.word_gru(embedded)  # [batch_size, seq_len, hid_dim*2]
        
        # Process each sequence in the batch to get sentence-level representations
        sentence_embeddings_list = []
        
        for batch_idx in range(batch_size):
            # Get the current sequence
            seq = src[batch_idx]
            
            # Find sentence boundary positions
            sentence_ends = self._detect_sentence_boundaries(seq, idx2word)
            
            # Extract sentence embeddings by averaging word-level outputs for each sentence
            sentence_embeddings = []
            start_idx = 0
            
            for end_idx in sentence_ends:
                # Skip empty sentences
                if end_idx < start_idx:
                    continue
                    
                # Average word-level hidden states for this sentence
                sent_emb = word_outputs[batch_idx, start_idx:end_idx+1].mean(dim=0, keepdim=True)
                sentence_embeddings.append(sent_emb)
                
                # Update start index for next sentence
                start_idx = end_idx + 1
            
            # If no valid sentences were found, use the average of all word embeddings
            if not sentence_embeddings:
                # Use mask to handle padding
                mask = (seq != 0).float().unsqueeze(-1)
                masked_outputs = word_outputs[batch_idx] * mask
                sent_emb = masked_outputs.sum(dim=0, keepdim=True) / mask.sum(dim=0, keepdim=True).clamp(min=1)
                sentence_embeddings.append(sent_emb)
            
            # Stack all sentence embeddings for this batch item
            batch_sent_emb = torch.cat(sentence_embeddings, dim=0)
            sentence_embeddings_list.append(batch_sent_emb)
        
        # Pad sentence embeddings to the same length in batch
        max_sentences = max(emb.size(0) for emb in sentence_embeddings_list)
        padded_sent_embeddings = []
        
        for emb in sentence_embeddings_list:
            # Pad if needed
            if emb.size(0) < max_sentences:
                padding = torch.zeros(max_sentences - emb.size(0), emb.size(1), device=device)
                padded_emb = torch.cat([emb, padding], dim=0)
            else:
                padded_emb = emb
            padded_sent_embeddings.append(padded_emb.unsqueeze(0))
        
        # Stack to get [batch_size, max_sentences, hid_dim*2]
        sentence_embeddings = torch.cat(padded_sent_embeddings, dim=0)
        
        # Apply sentence-level GRU to get sentence context
        _, sent_hidden = self.sent_gru(sentence_embeddings)
        
        # Process bidirectional hidden states
        hidden_forward = sent_hidden[0, :, :]  # Forward direction
        hidden_backward = sent_hidden[1, :, :]  # Backward direction
        hidden_combined = torch.cat((hidden_forward, hidden_backward), dim=1)
        hidden_transformed = torch.tanh(self.fc(hidden_combined))
        
        # Reshape to match what the decoder expects: [1, batch_size, hid_dim]
        hidden_for_decoder = hidden_transformed.unsqueeze(0)
        
        return word_outputs, hidden_for_decoder

class EncoderRNN(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout_rate=0.5):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        
        # Linear layer to combine bidirectional outputs
        self.fc = nn.Linear(hid_dim * 2, hid_dim)

    def load_embeddings(self, glove_path, word2idx, freeze=True):
        """
        Load pre-trained GloVe embeddings into the embedding layer
        """
        # Initialize embeddings matrix with random values
        embeddings_matrix = torch.FloatTensor(torch.randn(len(word2idx), self.embedding.embedding_dim))

        # Set special tokens embeddings to zeros
        for token in ["<pad>", "<sos>", "<eos>", "<unk>"]:
            if token in word2idx:
                embeddings_matrix[word2idx[token]] = torch.zeros(self.embedding.embedding_dim)

        # Load GloVe embeddings
        print(f"Loading GloVe embeddings from {glove_path}")
        num_loaded = 0

        try:
            with open(glove_path, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    if word in word2idx:
                        vector = torch.FloatTensor([float(val) for val in values[1:]])
                        embeddings_matrix[word2idx[word]] = vector
                        num_loaded += 1

            print(f"Loaded embeddings for {num_loaded}/{len(word2idx)} words ({num_loaded/len(word2idx)*100:.2f}%)")
            self.embedding.weight = nn.Parameter(embeddings_matrix)
            
            if freeze:
                self.embedding.weight.requires_grad = False
                print("Embeddings frozen")

        except Exception as e:
            print(f"Error loading GloVe embeddings: {e}")
            print("Using random embeddings instead")

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)

        # Process bidirectional hidden states
        hidden_forward = hidden[0, :, :]  # First direction
        hidden_backward = hidden[1, :, :]  # Second direction
        hidden_combined = torch.cat((hidden_forward, hidden_backward), dim=1)
        hidden_transformed = torch.tanh(self.fc(hidden_combined))
        hidden_for_decoder = hidden_transformed.unsqueeze(0)  # [1, batch_size, hidden_size]

        return outputs, hidden_for_decoder

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout_rate=0.5):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input_token, hidden):
        # input_token shape: [batch_size, 1]
        embedded = self.dropout(self.embedding(input_token))  # [batch_size, 1, emb_dim]

        # RNN processing
        output, hidden = self.rnn(embedded, hidden)  # output: [batch_size, 1, hid_dim]

        # Final projection to vocabulary
        prediction = self.fc_out(output.squeeze(1))  # [batch_size, output_dim]

        return prediction, hidden

class Seq2seqRNN(nn.Module):
    def __init__(self, encoder_type, vocab_size, emb_dim, hid_dim, device, vocab_src, 
                 max_len=50, use_glove=True, glove_path=None, dropout_rate=0.5):
        super().__init__()
        
        # Create encoder based on specified type
        if encoder_type == 'hier':
            self.encoder = HierEncoderRNN(vocab_size, emb_dim, hid_dim, dropout_rate)
            print(encoder_type)
        else:  # Default to standard encoder
            print("not heir")
            self.encoder = EncoderRNN(vocab_size, emb_dim, hid_dim, dropout_rate)
            
        # Create decoder
        self.decoder = Decoder(vocab_size, emb_dim, hid_dim, dropout_rate)
        
        # Store other parameters
        self.device = device
        self.max_len = max_len
        self.vocabulary = vocab_src
        self.encoder_type = encoder_type
        self.idx2word = {v: k for k, v in vocab_src.items()}

        # Load GloVe embeddings if specified
        if use_glove and glove_path:
            if os.path.exists(glove_path):
                self.encoder.load_embeddings(glove_path, vocab_src)
            else:
                print(f"Warning: GloVe embeddings path {glove_path} not found. Using random embeddings.")

    def forward(self, src, tgt=None, teacher_forcing_ratio=1.0):
        batch_size = src.shape[0]

        # Define target length - use tgt length during training, max_len during inference
        tgt_len = tgt.shape[1] if tgt is not None else self.max_len

        # Define vocabulary size
        vocab_size = self.decoder.output_dim

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)

        # Set the first position to <sos> token
        outputs[:, 0, self.vocabulary["<sos>"]] = 1.0

        # Encode the source sequence
        if self.encoder_type == 'hierarchical':
            encoder_outputs, hidden = self.encoder(src, self.idx2word)
        else:
            encoder_outputs, hidden = self.encoder(src)

        # First decoder input is the <sos> token
        input_token = torch.tensor([[self.vocabulary["<sos>"]] * batch_size], device=self.device).T

        # Start from index 1 since index 0 is <sos>
        for t in range(1, tgt_len):
            # Pass through decoder
            prediction, hidden = self.decoder(input_token, hidden)

            # Store prediction
            outputs[:, t, :] = prediction

            # Teacher forcing: decide whether to use real target tokens
            use_teacher_forcing = random.random() < teacher_forcing_ratio

            if use_teacher_forcing and tgt is not None:
                # Use actual next token as next input
                input_token = tgt[:, t].unsqueeze(1)
            else:
                # Use best predicted token
                top1 = prediction.argmax(1).unsqueeze(1)
                input_token = top1

            # Stop if all sequences in batch have generated EOS (only during inference)
            if tgt is None and (input_token == self.vocabulary["<eos>"]).all():
                break

        return outputs
def evaluate_rouge(model, dataset, vocab_src, vocab_tgt, num_examples=8):
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    idx2word_tgt = {v: k for k, v in vocab_tgt.items()}

    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    examples = []

    # Get special token IDs
    special_tokens = {vocab_tgt["<pad>"], vocab_tgt["<sos>"], vocab_tgt["<eos>"], vocab_tgt["<unk>"]}

    with torch.no_grad():
        for i in range(min(100, len(dataset))):  # Evaluate on max 100 examples
            src, tgt = dataset[i]
            src = src.unsqueeze(0).to(model.device)  # Add batch dimension

            # Generate output with the model (no target provided for inference)
            outputs = model(src, tgt=None, teacher_forcing_ratio=0)

            # Get the predicted token indices
            output_tokens = outputs[0].argmax(dim=1).cpu().numpy()

            # Convert tokens to words, filtering out special tokens
            pred_words = []
            for idx in output_tokens:
                if idx in idx2word_tgt and idx not in special_tokens:
                    pred_words.append(idx2word_tgt[idx])
            pred = ' '.join(pred_words)

            # Process target tokens
            true_words = []
            for idx in tgt:
                idx_item = idx.item()
                if idx_item in idx2word_tgt and idx_item not in special_tokens:
                    true_words.append(idx2word_tgt[idx_item])
            true = ' '.join(true_words)

            # Debug information
            if i < 5:  # Debug first few examples
                print(f"\nDebug Example {i+1}:")
                print(f"Raw prediction tokens: {output_tokens}")
                print(f"Predicted words: {pred_words}")
                print(f"Predicted: '{pred}'")
                print(f"True: '{true}'")

            # Skip empty predictions or targets
            if not pred.strip() or not true.strip():
                print(f"Skipping example {i+1} due to empty prediction or target")
                continue

            # Calculate ROUGE scores
            try:
                rouge_scores = scorer.score(true, pred)
                for key in scores:
                    scores[key].append(rouge_scores[key].fmeasure)

                # Save examples
                if len(examples) < num_examples:
                    examples.append((pred, true, rouge_scores))
            except Exception as e:
                print(f"Error calculating ROUGE for example {i+1}: {e}")

    # Print examples
    print("\nEvaluation Examples:")
    for i, (pred, true, rouge) in enumerate(examples):
        print(f"\nExample {i+1}:")
        print(f"Predicted: {pred}")
        print(f"True: {true}")
        print(f"ROUGE-1: {rouge['rouge1'].fmeasure:.3f}")
        print(f"ROUGE-2: {rouge['rouge2'].fmeasure:.3f}")
        print(f"ROUGE-L: {rouge['rougeL'].fmeasure:.3f}")

    # Calculate average scores
    if not any(scores.values()):
        print("Warning: No valid ROUGE scores calculated!")
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    avg_scores = {k: np.mean(v) if v else 0.0 for k, v in scores.items()}
    print("\nAverage ROUGE Scores:")
    print(f"ROUGE-1: {avg_scores['rouge1']:.3f}")
    print(f"ROUGE-2: {avg_scores['rouge2']:.3f}")
    print(f"ROUGE-L: {avg_scores['rougeL']:.3f}")

    return avg_scores

def load_glove_embeddings(glove_path, vocab_src, emb_dim=300):
    """
    Utility function to load GloVe embeddings separately
    """
    print(f"Loading GloVe embeddings from {glove_path}")
    word_vectors = {}

    try:
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading GloVe file"):
                values = line.split()
                word = values[0]
                vectors = torch.FloatTensor([float(val) for val in values[1:]])
                word_vectors[word] = vectors

        print(f"Loaded {len(word_vectors)} word vectors from GloVe")

        # Create embedding matrix
        # Create embedding matrix on the desired device
        embeddings_matrix = torch.randn(len(vocab_src), emb_dim, device=device)


        # Set special tokens to zeros
        for token in SPECIAL_TOKENS:
            if token in vocab_src:
                embeddings_matrix[vocab_src[token]] = torch.zeros(emb_dim)

        # Populate with GloVe vectors
        num_loaded = 0
        list_of_loaded_words = []
        new_vocab_src = {}
        
        for word, idx in vocab_src.items():
            if word in word_vectors:
                embeddings_matrix[idx] = word_vectors[word]
                num_loaded += 1
                list_of_loaded_words.append(word)
                
        for key, val in vocab_src.items():
            if key  in list_of_loaded_words:
                new_vocab_src[key] = val

        vocab_src = new_vocab_src

        print(f'length of new_vocab is {len(new_vocab_src)}\n')
        print(f'length of vocab is {len(vocab_src)}\n')

        print(f"Initialized {num_loaded}/{len(vocab_src)} words with GloVe vectors ({num_loaded/len(vocab_src)*100:.2f}%)")
        return embeddings_matrix

    except Exception as e:
        print(f"Error loading GloVe embeddings: {e}")
        return None

def train_model(data, vocab_src, use_glove=False, glove_path=None):
    # Build vocabularies
    print(f"Vocab size: {len(vocab_src)}")

    # Create datasets
    train_data = HeadlineDataset(data['training_data'], vocab_src)
    val_data = HeadlineDataset(data['validation_data'], vocab_src)
    test_data = HeadlineDataset(data['test_data'], vocab_src)

    # Check dataset indices
    assert check_dataset_indices(train_data, len(vocab_src)), "Invalid indices found in training data"

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                            shuffle=True, collate_fn=collate_fn, pin_memory=True)

    model = Seq2seqRNN("hier", len(vocab_src), 
                    EMB_DIM, HID_DIM, device, 
                    vocab_src, 50, True, glove_path, 0.5).to(device)

    # # Initialize model components
    # encoder = Encoder(len(vocab_src), EMB_DIM, HID_DIM)
    # decoder = Decoder(len(vocab_src), EMB_DIM, HID_DIM)

    # Pre-load GloVe embeddings if requested
    # if use_glove and glove_path and os.path.exists(glove_path):
    #     print("Preloading GloVe embeddings...")
    #     glove_matrix = load_glove_embeddings(glove_path, vocab_src, EMB_DIM)
    #     if glove_matrix is not None:
    #         # Set encoder embeddings
    #         encoder.embedding.weight = nn.Parameter(glove_matrix)
    #         # Optionally freeze embeddings
    #         encoder.embedding.weight.requires_grad = False
    #         print("Encoder embeddings initialized with GloVe and frozen")

    # Create Seq2Seq model
    # model = Seq2Seq(encoder, decoder, device, vocab_src,
    #                max_len=MAX_LEN,
    #                use_glove=False,  # We loaded embeddings manually above
    #                glove_path=None).to(device)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=SPECIAL_TOKENS['<pad>'])

    # Fix any out-of-bounds indices in batches
    vocab_size = len(vocab_src)
    for batch_idx, (src, tgt) in enumerate(train_loader):
        if (src >= vocab_size).any():
            print(f"Found out-of-bounds indices in batch {batch_idx}")
            print(f"Max index: {src.max().item()}, Vocab size: {vocab_size}")
            # Fix the indices by capping them
            src[src >= vocab_size] = vocab_src["<unk>"]

        # Same check for target
        if (tgt >= vocab_size).any():
            tgt[tgt >= vocab_size] = vocab_src["<unk>"]

    # Training loop
    best_val_score = -1  # Initialize with negative value to ensure first model is saved
    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0

        for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()

            # Pass tgt for teacher forcing
            output = model(src, tgt, teacher_forcing_ratio=TEACHER_FORCING_RATIO)

            # Reshape output and target for loss calculation
            # output: [batch_size, tgt_len, vocab_size]
            # Target should exclude <sos> token (first token)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)

            # Calculate loss
            loss = criterion(output, tgt)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss/len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # Evaluate every 5 epochs or at the end
        if (epoch+1) % 5 == 0 or epoch == N_EPOCHS-1:
            print("\nValidation Evaluation:")
            val_scores = evaluate_rouge(model, val_data, vocab_src, vocab_src)

            # Save the best model
            rouge_l_score = val_scores['rougeL']
            if rouge_l_score > best_val_score:
                best_val_score = rouge_l_score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'rouge_score': rouge_l_score,
                    'vocab': vocab_src,
                    'config': {
                        'emb_dim': EMB_DIM,
                        'hid_dim': HID_DIM,
                        'max_len': MAX_LEN,
                        'max_len_src': MAX_LEN_SRC,
                        'use_glove': use_glove
                    }
                }, 'best_headline_generator.pth')
                print(f"New best model saved with ROUGE-L: {best_val_score:.3f}")

    # Final evaluation
    print("\nTest Evaluation:")
    test_scores = evaluate_rouge(model, test_data, vocab_src, vocab_src)

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab_src,
        'rouge_score': test_scores['rougeL'],
        'config': {
            'emb_dim': EMB_DIM,
            'hid_dim': HID_DIM,
            'max_len': MAX_LEN,
            'max_len_src': MAX_LEN_SRC,
            'use_glove': use_glove
        }
    }, 'final_headline_generator.pth')

    return model, val_scores, test_scores

# Example usage
if __name__ == "__main__":

    # Path to your pre-downloaded GloVe embeddings
    glove_path = "/kaggle/input/glove-dataset/glove.6B.300d.txt"

    # Train with GloVe embeddings
    model, val_scores, test_scores = train_model(
        data,
        vocab_src,  # Assuming this is already defined
        use_glove=True,
        glove_path=glove_path
    )
    
    
import torch
import torch.nn as nn
import random

EMB_DIM = 300  # Matches GloVe dimensions
HID_DIM = 300
BATCH_SIZE = 16
LEARNING_RATE = 0.001
N_EPOCHS = 30
MAX_LEN = 10
MAX_LEN_SRC = 300
TEACHER_FORCING_RATIO = 0.7


class Decoder2RNN(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout_rate=0.5):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # First GRU
        self.gru1 = nn.GRU(emb_dim, hid_dim, batch_first=True)

        # Second GRU
        self.gru2 = nn.GRU(hid_dim, hid_dim, batch_first=True)

        # Fully connected layer to project GRU2 outputs to vocabulary size
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input_token, hidden):
        # input_token shape: [batch_size, 1]
        embedded = self.dropout(self.embedding(input_token))  # [batch_size, 1, emb_dim]

        # First GRU processing
        output1, hidden1 = self.gru1(embedded, hidden)  # output1: [batch_size, 1, hid_dim]

        # Second GRU processing - use the same initial hidden state from encoder
        output2, hidden2 = self.gru2(output1, hidden)  # output2: [batch_size, 1, hid_dim]

        # Final projection to vocabulary
        prediction = self.fc_out(output2.squeeze(1))  # [batch_size, output_dim]

        return prediction, hidden2

class Seq2seqRNN(nn.Module):
    def __init__(self, encoder_type, vocab_size, emb_dim, hid_dim, device, vocab_src, 
                 max_len=50, use_glove=True, glove_path=None, dropout_rate=0.5, decoder_type='single'):
        super().__init__()
        
        # Create encoder based on specified type
        if encoder_type == 'hierarchical':
            print("using heir encoder")
            self.encoder = HierEncoderRNN(vocab_size, emb_dim, hid_dim, dropout_rate)
        else:  # Default to standard encoder
            print("using standard encoder")
            self.encoder = EncoderRNN(vocab_size, emb_dim, hid_dim, dropout_rate)
            
        # Create decoder based on specified type
        if decoder_type == 'double':
            self.decoder = Decoder2RNN(vocab_size, emb_dim, hid_dim, dropout_rate)
            print("decoder2rnn being used!!")
        else:  # Default to single GRU decoder
            self.decoder = Decoder(vocab_size, emb_dim, hid_dim, dropout_rate)
        
        # Store other parameters
        self.device = device
        self.max_len = max_len
        self.vocabulary = vocab_src
        self.encoder_type = encoder_type
        self.idx2word = {v: k for k, v in vocab_src.items()}

        # Load GloVe embeddings if specified
        if use_glove and glove_path:
            if os.path.exists(glove_path):
                self.encoder.load_embeddings(glove_path, vocab_src)
            else:
                print(f"Warning: GloVe embeddings path {glove_path} not found. Using random embeddings.")

    def forward(self, src, tgt=None, teacher_forcing_ratio=1.0):
        batch_size = src.shape[0]

        # Define target length - use tgt length during training, max_len during inference
        tgt_len = tgt.shape[1] if tgt is not None else self.max_len

        # Define vocabulary size
        vocab_size = self.decoder.output_dim

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)

        # Set the first position to <sos> token
        outputs[:, 0, self.vocabulary["<sos>"]] = 1.0

        # Encode the source sequence
        if self.encoder_type == 'hierarchical':
            encoder_outputs, hidden = self.encoder(src, self.idx2word)
        else:
            encoder_outputs, hidden = self.encoder(src)

        # First decoder input is the <sos> token
        input_token = torch.tensor([[self.vocabulary["<sos>"]] * batch_size], device=self.device).T

        # Start from index 1 since index 0 is <sos>
        for t in range(1, tgt_len):
            # Pass through decoder
            prediction, hidden = self.decoder(input_token, hidden)

            # Store prediction
            outputs[:, t, :] = prediction

            # Teacher forcing: decide whether to use real target tokens
            use_teacher_forcing = random.random() < teacher_forcing_ratio

            if use_teacher_forcing and tgt is not None:
                # Use actual next token as next input
                input_token = tgt[:, t].unsqueeze(1)
            else:
                # Use best predicted token
                top1 = prediction.argmax(1).unsqueeze(1)
                input_token = top1

            # Stop if all sequences in batch have generated EOS (only during inference)
            if tgt is None and (input_token == self.vocabulary["<eos>"]).all():
                break

        return outputs

def train_model(data, vocab_src, encoder_type='standard', decoder_type='single', use_glove=False, glove_path=None):
    # Build vocabularies
    print(f"Vocab size: {len(vocab_src)}")

    # Create datasets
    train_data = HeadlineDataset(data['training_data'], vocab_src)
    val_data = HeadlineDataset(data['validation_data'], vocab_src)
    test_data = HeadlineDataset(data['test_data'], vocab_src)

    # Check dataset indices
    assert check_dataset_indices(train_data, len(vocab_src)), "Invalid indices found in training data"

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                            shuffle=True, collate_fn=collate_fn, pin_memory=True)

    # Initialize the Seq2seqRNN model with the specified encoder and decoder types
    model = Seq2seqRNN(
        encoder_type=encoder_type,  # 'standard' or 'hierarchical'
        vocab_size=len(vocab_src),
        emb_dim=EMB_DIM,
        hid_dim=HID_DIM,
        device=device,
        vocab_src=vocab_src,
        max_len=MAX_LEN,
        use_glove=use_glove,
        glove_path=glove_path,
        dropout_rate=0.5,
        decoder_type=decoder_type  # 'single' or 'double'
    ).to(device)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab_src["<pad>"])

    # Fix any out-of-bounds indices in batches
    vocab_size = len(vocab_src)
    for batch_idx, (src, tgt) in enumerate(train_loader):
        if (src >= vocab_size).any():
            print(f"Found out-of-bounds indices in batch {batch_idx}")
            print(f"Max index: {src.max().item()}, Vocab size: {vocab_size}")
            # Fix the indices by capping them
            src[src >= vocab_size] = vocab_src["<unk>"]

        # Same check for target
        if (tgt >= vocab_size).any():
            tgt[tgt >= vocab_size] = vocab_src["<unk>"]

    # Training loop
    best_val_score = -1  # Initialize with negative value to ensure first model is saved
    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0

        for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()

            # Pass tgt for teacher forcing
            output = model(src, tgt, teacher_forcing_ratio=TEACHER_FORCING_RATIO)

            # Reshape output and target for loss calculation
            # output: [batch_size, tgt_len, vocab_size]
            # Target should exclude <sos> token (first token)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)

            # Calculate loss
            loss = criterion(output, tgt)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss/len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # Evaluate every 5 epochs or at the end
        if (epoch+1) % 5 == 0 or epoch == N_EPOCHS-1:
            print("\nValidation Evaluation:")
            val_scores = evaluate_rouge(model, val_data, vocab_src, vocab_src)

            # Save the best model
            rouge_l_score = val_scores['rougeL']
            if rouge_l_score > best_val_score:
                best_val_score = rouge_l_score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'rouge_score': rouge_l_score,
                    'vocab': vocab_src,
                    'config': {
                        'emb_dim': EMB_DIM,
                        'hid_dim': HID_DIM,
                        'max_len': MAX_LEN,
                        'encoder_type': encoder_type,
                        'decoder_type': decoder_type,
                        'use_glove': use_glove
                    }
                }, 'best_headline_generator.pth')
                print(f"New best model saved with ROUGE-L: {best_val_score:.3f}")

    # Final evaluation
    print("\nTest Evaluation:")
    test_scores = evaluate_rouge(model, test_data, vocab_src, vocab_src)

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab_src,
        'rouge_score': test_scores['rougeL'],
        'config': {
            'emb_dim': EMB_DIM,
            'hid_dim': HID_DIM,
            'max_len': MAX_LEN,
            'encoder_type': encoder_type,
            'decoder_type': decoder_type,
            'use_glove': use_glove
        }
    }, 'final_headline_generator.pth')

    return model, val_scores, test_scores

# Example usage
if __name__ == "__main__":
    # Path to your pre-downloaded GloVe embeddings
    glove_path = "/kaggle/input/glove-dataset/glove.6B.300d.txt"

    # Train with GloVe embeddings and double GRU decoder
    model, val_scores, test_scores = train_model(
        data,
        vocab_src,  # Assuming this is already defined
        encoder_type='hierarchical',  # or 'hierarchical'
        decoder_type='double',    # Use the new Decoder2RNN
        use_glove=True,
        glove_path=glove_path
    )


EMB_DIM = 300  # Matches GloVe dimensions
HID_DIM = 300
BATCH_SIZE = 16
LEARNING_RATE = 0.001
N_EPOCHS = 30
MAX_LEN = 10
MAX_LEN_SRC = 300
TEACHER_FORCING_RATIO = 0.7

class Seq2seqRNN(nn.Module):
    def __init__(self, encoder_type, vocab_size, emb_dim, hid_dim, device, vocab_src, 
                 max_len=50, use_glove=True, glove_path=None, dropout_rate=0.5, decoder_type='single'):
        super().__init__()
        
        # Create encoder based on specified type
        if encoder_type == 'hierarchical':
            print("using hier encoder")
            self.encoder = HierEncoderRNN(vocab_size, emb_dim, hid_dim, dropout_rate)
        else:  # Default to standard encoder
            print("standard encoder being used!!")
            self.encoder = EncoderRNN(vocab_size, emb_dim, hid_dim, dropout_rate)
            
        # Create decoder based on specified type
        if decoder_type == 'double':
            self.decoder = Decoder2RNN(vocab_size, emb_dim, hid_dim, dropout_rate)
            print("decoder2rnn being used!!")
        else:  # Default to single GRU decoder
            print("standard decoder being used!!")
            self.decoder = Decoder(vocab_size, emb_dim, hid_dim, dropout_rate)
        
        # Store other parameters
        self.device = device
        self.max_len = max_len
        self.vocabulary = vocab_src
        self.encoder_type = encoder_type
        self.idx2word = {v: k for k, v in vocab_src.items()}

        # Load GloVe embeddings if specified
        if use_glove and glove_path:
            if os.path.exists(glove_path):
                self.encoder.load_embeddings(glove_path, vocab_src)
            else:
                print(f"Warning: GloVe embeddings path {glove_path} not found. Using random embeddings.")

    def forward(self, src, tgt=None, teacher_forcing_ratio=1.0, beam_width=1):
        if beam_width > 1 and tgt is None:
            return self.beam_search_decode(src, beam_width)
        return self.greedy_decode(src, tgt, teacher_forcing_ratio)

    def greedy_decode(self, src, tgt, teacher_forcing_ratio):
        batch_size = src.shape[0]

        # Define target length - use tgt length during training, max_len during inference
        tgt_len = tgt.shape[1] if tgt is not None else self.max_len

        # Define vocabulary size
        vocab_size = self.decoder.output_dim

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)

        # Set the first position to <sos> token
        outputs[:, 0, self.vocabulary["<sos>"]] = 1.0

        # Encode the source sequence
        if self.encoder_type == 'hierarchical':
            encoder_outputs, hidden = self.encoder(src, self.idx2word)
        else:
            encoder_outputs, hidden = self.encoder(src)

        # First decoder input is the <sos> token
        input_token = torch.tensor([[self.vocabulary["<sos>"]] * batch_size], device=self.device).T

        # Start from index 1 since index 0 is <sos>
        for t in range(1, tgt_len):
            # Pass through decoder
            prediction, hidden = self.decoder(input_token, hidden)

            # Store prediction
            outputs[:, t, :] = prediction

            # Teacher forcing: decide whether to use real target tokens
            use_teacher_forcing = random.random() < teacher_forcing_ratio

            if use_teacher_forcing and tgt is not None:
                # Use actual next token as next input
                input_token = tgt[:, t].unsqueeze(1)
            else:
                # Use best predicted token
                top1 = prediction.argmax(1).unsqueeze(1)
                input_token = top1

            # Stop if all sequences in batch have generated EOS (only during inference)
            if tgt is None and (input_token == self.vocabulary["<eos>"]).all():
                break

        return outputs

    def beam_search_decode(self, src, beam_width=3):
        batch_size = src.size(0)
        start_token = self.vocabulary["<sos>"]
        end_token = self.vocabulary["<eos>"]
        
        # Encode source sequence
        if self.encoder_type == 'hierarchical':
            _, hidden = self.encoder(src, self.idx2word)
        else:
            _, hidden = self.encoder(src)
        
        # Initialize beams (log_prob, sequence, hidden)
        beams = [([start_token], 0.0, hidden)]
        finished_beams = []

        for _ in range(self.max_len):
            candidates = []
            for seq, score, hidden_state in beams:
                if seq[-1] == end_token:
                    candidates.append((seq, score, hidden_state))
                    continue
                
                # Prepare decoder input
                input_token = torch.tensor([[seq[-1]]], device=self.device)
                
                # Get next token probabilities
                with torch.no_grad():
                    logits, new_hidden = self.decoder(input_token, hidden_state)
                    log_probs = torch.log_softmax(logits, dim=-1)

                # Get top k candidates
                top_scores, top_tokens = log_probs.topk(beam_width)
                for i in range(beam_width):
                    token = top_tokens[0, i].item()
                    new_score = score + top_scores[0, i].item()
                    new_seq = seq + [token]
                    candidates.append((new_seq, new_score, new_hidden))

            # Filter and sort candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = []
            for candidate in candidates[:beam_width]:
                seq, score, hidden = candidate
                if seq[-1] == end_token:
                    finished_beams.append(candidate)
                else:
                    beams.append(candidate)
            
            # Early stopping if all beams finished
            if not beams:
                break

        # Combine finished and unfinished beams
        final_candidates = finished_beams + beams
        final_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Get best sequence
        best_sequence = final_candidates[0][0]
        
        # Convert to output tensor
        outputs = torch.zeros(batch_size, len(best_sequence), len(self.vocabulary)).to(self.device)
        for t, token in enumerate(best_sequence):
            outputs[0, t, token] = 1.0
            
        return outputs


def train_model(data, vocab_src, encoder_type='standard', decoder_type='single', 
               use_glove=False, glove_path=None, beam_width=3):
    # ... (existing setup code remains the same) ...

    print(f"Vocab size: {len(vocab_src)}")

    # Create datasets
    train_data = HeadlineDataset(data['training_data'], vocab_src)
    val_data = HeadlineDataset(data['validation_data'], vocab_src)
    test_data = HeadlineDataset(data['test_data'], vocab_src)

    # Check dataset indices
    assert check_dataset_indices(train_data, len(vocab_src)), "Invalid indices found in training data"

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                            shuffle=True, collate_fn=collate_fn, pin_memory=True)
    
    model = Seq2seqRNN(
        encoder_type=encoder_type,  # 'standard' or 'hierarchical'
        vocab_size=len(vocab_src),
        emb_dim=EMB_DIM,
        hid_dim=HID_DIM,
        device=device,
        vocab_src=vocab_src,
        max_len=MAX_LEN,
        use_glove=use_glove,
        glove_path=glove_path,
        dropout_rate=0.5,
        decoder_type=decoder_type  # 'single' or 'double'
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab_src["<pad>"])

    # Fix any out-of-bounds indices in batches
    vocab_size = len(vocab_src)
    for batch_idx, (src, tgt) in enumerate(train_loader):
        if (src >= vocab_size).any():
            print(f"Found out-of-bounds indices in batch {batch_idx}")
            print(f"Max index: {src.max().item()}, Vocab size: {vocab_size}")
            # Fix the indices by capping them
            src[src >= vocab_size] = vocab_src["<unk>"]

        # Same check for target
        if (tgt >= vocab_size).any():
            tgt[tgt >= vocab_size] = vocab_src["<unk>"]
    
    # Training loop
    best_val_score = -1
    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0

        # Training phase (unchanged)
        for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()

            # Pass tgt for teacher forcing
            output = model(src, tgt, teacher_forcing_ratio=TEACHER_FORCING_RATIO)

            # Reshape output and target for loss calculation
            # output: [batch_size, tgt_len, vocab_size]
            # Target should exclude <sos> token (first token)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)

            # Calculate loss
            loss = criterion(output, tgt)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss/len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # Validation phase with beam search
        if (epoch+1) % 5 == 0 or epoch == N_EPOCHS-1:
            print("\nValidation Evaluation:")
            val_scores = evaluate_rouge(
                model, val_data, vocab_src, vocab_src,
                use_beam_search=True,  # Enable beam search for evaluation
                beam_width=beam_width
            )
            
            # Save best model based on ROUGE-L
            if val_scores['rougeL'] > best_val_score:
                best_val_score = val_scores['rougeL']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'rouge_score': best_val_score,
                    'vocab': vocab_src,
                    'config': {
                        'emb_dim': EMB_DIM,
                        'hid_dim': HID_DIM,
                        'max_len': MAX_LEN,
                        'encoder_type': encoder_type,
                        'decoder_type': decoder_type,
                        'use_glove': use_glove,
                        'beam_width': beam_width
                    }
                }, 'best_headline_generator.pth')
                print(f"New best model saved with ROUGE-L: {best_val_score:.3f}")


    # Final test evaluation with beam search
    print("\nTest Evaluation:")
    test_scores = evaluate_rouge(
        model, test_data, vocab_src, vocab_src,
        use_beam_search=True,
        beam_width=beam_width
    )
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab_src,
        'rouge_score': test_scores['rougeL'],
        'config': {
            'emb_dim': EMB_DIM,
            'hid_dim': HID_DIM,
            'max_len': MAX_LEN,
            'encoder_type': encoder_type,
            'decoder_type': decoder_type,
            'use_glove': use_glove,
            'beam_width': beam_width
        }
    }, 'final_headline_generator.pth')

    return model, val_scores, test_scores

def evaluate_rouge(model, dataset, vocab_src, vocab_tgt, use_beam_search=False, beam_width=3, num_examples=8):
    eval_loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    idx2word_tgt = {v: k for k, v in vocab_tgt.items()}
    
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    examples = []
    example_count = 0
    
    # Special tokens to filter
    special_tokens = {
        vocab_tgt["<pad>"], 
        vocab_tgt["<sos>"], 
        vocab_tgt["<eos>"], 
        vocab_tgt["<unk>"]
    }

    with torch.no_grad():
        for batch_idx, (src, tgt) in enumerate(tqdm(eval_loader, desc="Evaluating")):
            src = src.to(device)
            
            # Generate output with beam search or greedy decoding
            if use_beam_search:
                output = model(src, beam_width=beam_width)
            else:
                output = model(src)
            
            # Convert output to tokens
            predicted_tokens = output.argmax(dim=2)
            
            # Process batch
            for i in range(src.size(0)):
                # Get prediction and reference
                pred_seq = predicted_tokens[i].cpu().numpy()
                ref_seq = tgt[i].cpu().numpy()
                
                # Convert tokens to words
                pred_words = []
                for idx in pred_seq:
                    if idx in idx2word_tgt and idx not in special_tokens:
                        pred_words.append(idx2word_tgt[idx])
                prediction = ' '.join(pred_words)
                
                ref_words = []
                for idx in ref_seq:
                    idx = idx.item()
                    if idx in idx2word_tgt and idx not in special_tokens:
                        ref_words.append(idx2word_tgt[idx])
                reference = ' '.join(ref_words)
                
                # Skip empty predictions or references
                if not prediction.strip() or not reference.strip():
                    continue
                
                # Calculate ROUGE scores
                try:
                    rouge_scores = scorer.score(reference, prediction)
                    for key in scores:
                        scores[key].append(rouge_scores[key].fmeasure)
                    
                    # Store examples
                    if example_count < num_examples:
                        examples.append((
                            prediction,
                            reference,
                            {k: v.fmeasure for k, v in rouge_scores.items()}
                        ))
                        example_count += 1
                        
                    # Debug print first 3 examples
                    if batch_idx == 0 and i < 3:
                        print(f"\nBatch 0, Example {i}:")
                        print(f"Predicted: {prediction}")
                        print(f"Reference: {reference}")
                        print(f"ROUGE Scores: {rouge_scores}")
                        
                except Exception as e:
                    print(f"Error processing example: {e}")

    # Print collected examples
    print("\n\nEvaluation Examples:")
    for i, (pred, ref, rouge) in enumerate(examples):
        print(f"\nExample {i+1}:")
        print(f"Predicted: {pred}")
        print(f"Reference: {ref}")
        print(f"ROUGE-1: {rouge['rouge1']:.3f}")
        print(f"ROUGE-2: {rouge['rouge2']:.3f}")
        print(f"ROUGE-L: {rouge['rougeL']:.3f}")

    # Calculate average scores
    avg_scores = {}
    for metric in scores:
        if scores[metric]:
            avg_scores[metric] = np.mean(scores[metric])
        else:
            avg_scores[metric] = 0.0
            print(f"Warning: No valid scores for {metric}")

    print("\nFinal Average ROUGE Scores:")
    print(f"ROUGE-1: {avg_scores['rouge1']:.3f}")
    print(f"ROUGE-2: {avg_scores['rouge2']:.3f}")
    print(f"ROUGE-L: {avg_scores['rougeL']:.3f}")
    
    return avg_scores


# Updated main function
if __name__ == "__main__":
    # Configuration
    config = {
        'data': data,  # Your dataset loading function
        'vocab_src': vocab_src,  # Your vocab building function
        'encoder_type': 'hierarchical',  # 'standard' or 'hierarchical'
        'decoder_type': 'double',     # 'sxingle' or 'double'
        'use_glove': True,
        'glove_path': "/kaggle/input/glove-dataset/glove.6B.300d.txt",
        'beam_width': 5  # Set beam search width
    }

    # Train model with beam search evaluation
    model, val_scores, test_scores = train_model(**config)

    print("\nTesting Model on Test Dataset:")
    
    # Evaluate model on the entire test dataset using beam search
    test_data = config['data']['test_data']
    rouge_scores = evaluate_rouge(
        model=model,
        dataset=test_data,
        vocab_src=config['vocab_src'],
        vocab_tgt=config['vocab_src'],  # Assuming source and target vocab are the same
        use_beam_search=True,
        beam_width=config['beam_width']
    )

    print("\nFinal ROUGE Scores on Test Dataset:")
    print(f"ROUGE-1: {rouge_scores['rouge1']:.3f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.3f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.3f}")

    # Example inference with beam search
    print("\nExample Inference:")
    test_words = ["<sos>", "breaking", "news", "about", "ai", "advances", "<eos>"]
    test_input = torch.tensor([
        [config['vocab_src'].get(word, config['vocab_src']["<unk>"]) for word in test_words]
    ]).to(model.device)

    with torch.no_grad():
        # Use beam search explicitly
        beam_output = model(test_input, beam_width=config['beam_width'])
        
        # Convert output to tokens (shape: [batch_size, seq_len])
        predicted_tokens = beam_output.argmax(dim=-1)[0]  # Get first batch
        
        # Convert tokens to words
        headline_words = []
        for idx in predicted_tokens.cpu().numpy():
            word = model.idx2word.get(idx, "<unk>")
            if word == "<eos>":
                break
            if word not in ["<sos>", "<pad>", "<unk>"]:
                headline_words.append(word)
                
        print(f"\nGenerated Headline: {' '.join(headline_words)}")
