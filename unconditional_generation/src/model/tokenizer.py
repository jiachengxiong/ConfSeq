import os
import json
from transformers import PreTrainedTokenizer
from typing import Dict


class WhitespaceTokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        """
        Initializes the tokenizer with a custom vocabulary.
        
        :param kwargs: Additional arguments to be passed to the parent constructor.
        """
        # Build the vocabulary using the _build_vocab method
        self.encoder: Dict[str, int] = self._build_vocab()  # Token to ID mapping
        # Create a reverse mapping from ID to token
        self.decoder: Dict[int, str] = {idx: token for token, idx in self.encoder.items()}
        
        # Call the parent constructor with necessary special tokens
        super().__init__(
            pad_token='<PAD>',  # Padding token
            bos_token='<BOS>',  # Beginning of sentence token
            eos_token='<EOS>',  # End of sentence token
            unk_token='<UNK>',  # Unknown token
            mask_token='<MASK>',  # Mask token
            **kwargs  # Pass any other additional arguments to the parent constructor
        )

    def _build_vocab(self) -> Dict[str, int]:
        """
        Builds the vocabulary, which includes:
        1. Predefined special tokens (`<BOS>`, `<EOS>`, `<PAD>`, `<UNK>`, `<MASK>`)
        2. All visible ASCII characters (ASCII 33-126)
        3. Custom tokens in the format `<i>` where i is in the range [-180, 180].
        
        :return: A dictionary mapping tokens to unique integer IDs.
        """
        # Initialize the vocabulary with special tokens
        vocab = ['<BOS>', '<EOS>', '<PAD>', '<UNK>', '<MASK>']
        
        # Add all visible ASCII characters (characters with ASCII values 33-126)
        vocab.extend(map(chr, range(33, 127)))
        
        # Add custom tokens in the format <i>, where i ranges from -180 to 180
        vocab.extend(f'<{i}>' for i in range(-180, 181))

        # Add special tokens for prefix
        vocab.extend(['<std>', '<aug>', '<unk>'])

        # Return a dictionary mapping each token to a unique ID
        return {token: idx for idx, token in enumerate(vocab)}
    
    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenizes the input text by splitting it on whitespace.
        
        :param text: The input text to be tokenized.
        :return: A list of tokens (words) from the input text.
        """
        return text.strip().split()

    def _convert_token_to_id(self, token: str) -> int:
        """
        Converts a token into its corresponding ID. If the token is not found in the vocabulary,
        returns the ID of the unknown token (`<UNK>`).
        
        :param token: The token to be converted to an ID.
        :return: The ID corresponding to the token.
        """
        return self.encoder.get(token, self.encoder[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        """
        Converts an ID back to its corresponding token. If the ID is not found in the vocabulary,
        returns the unknown token (`<UNK>`).
        
        :param index: The ID to be converted to a token.
        :return: The token corresponding to the given ID.
        """
        return self.decoder.get(index, self.unk_token)

    def get_vocab(self) -> Dict[str, int]:
        """
        Returns the entire vocabulary (token to ID mapping).
        
        :return: The vocabulary as a dictionary mapping tokens to their corresponding IDs.
        """
        return self.encoder
    
    @property
    def vocab_size(self) -> int:
        """
        Returns the size of the vocabulary, i.e., the number of tokens in the vocabulary.
        
        :return: The size of the vocabulary.
        """
        return len(self.encoder)
    
    def save_pretrained(self, save_dir: str) -> None:
        """
        Saves the vocabulary to a specified directory as a JSON file.
        If the directory does not exist, it will be created.
        
        :param save_dir: The directory where the vocabulary should be saved.
        """
        os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
        # Save the vocabulary to a JSON file
        with open(os.path.join(save_dir, 'tokenizer.json'), 'w', encoding='utf-8') as f:
            json.dump(self.encoder, f, ensure_ascii=False, indent=2)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Adds the <BOS> token at the beginning and <EOS> token at the end of the sequence.
        """
        bos = [self.bos_token_id]
        eos = [self.eos_token_id]
        return bos + token_ids_0 + eos
