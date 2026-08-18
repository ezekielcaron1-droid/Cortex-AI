"""
cortex/tokenizer.py
Tokenizer CamemBERT pour CORTEX.

Remplace l'ancien tokenizer byte-level (260 classes) par le tokenizer
SentencePiece de CamemBERT (camembert-base) - specifiquement entraine sur
du francais (contrairement a LLaMA, entraine majoritairement sur de
l'anglais), avec un vrai vocabulaire de 32 005 tokens qui correspond
exactement a config.vocab_size (au lieu de 260 classes reelles noyees
dans 32 000 - voir l'analyse du plateau de loss).

IMPORTANT : ce changement rend les checkpoints entraines avec l'ancien
ByteTokenizer INCOMPATIBLES (semantique des ids totalement differente -
byte-level vs sous-mots). Un nouvel entrainement complet est necessaire.
"""

from typing import List, Optional
import torch
from transformers import AutoTokenizer


class CamembertTokenizerWrapper:
    """Wrapper autour du tokenizer CamemBERT de HuggingFace, avec la meme
    interface que l'ancien ByteTokenizer (encode/decode/encode_batch,
    PAD_ID/BOS_ID/EOS_ID/UNK_ID, propriete vocab_size) pour ne rien casser
    ailleurs dans le pipeline."""

    def __init__(self, max_seq_len: int = 1024, vocab_size: Optional[int] = None):
        self.max_seq_len = max_seq_len

        self.tokenizer = AutoTokenizer.from_pretrained("camembert-base")

        self.PAD_ID = self.tokenizer.pad_token_id
        self.BOS_ID = self.tokenizer.bos_token_id
        self.EOS_ID = self.tokenizer.eos_token_id
        self.UNK_ID = self.tokenizer.unk_token_id
        self._vocab_size = self.tokenizer.vocab_size

        # Tous les tokens speciaux/reserves du tokenizer CamemBERT, PAS
        # seulement PAD/BOS/EOS/UNK. CamemBERT herite de RoBERTa/XLM et
        # contient des tokens "placeholder" avec des IDs totalement
        # separes des vrais tokens fonctionnels : <s>NOTUSED (id=0),
        # </s>NOTUSED (id=2), <unk>NOTUSED (id=32005), en plus de <mask>
        # (id=32004). Sans les inclure ici, un de ces placeholders choisi
        # par argmax pendant la generation fuitait tel quel dans le texte
        # affiche (observe en pratique : "<s>NOTUSED?..." dans une vraie
        # reponse). tokenizer.all_special_ids couvre TOUS ces cas d'un
        # coup, de maniere robuste (pas de liste codee en dur a maintenir).
        self._speciaux_ids = set(self.tokenizer.all_special_ids)

        print(f"[INFO] Tokenizer CamemBERT charge - vocab_size = {self._vocab_size}")

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    # ------------------------------------------------------------------ #
    # Encodage : texte -> liste d'ids
    # ------------------------------------------------------------------ #
    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        truncate: bool = True,
    ) -> List[int]:
        if text is None:
            text = ""

        ids = self.tokenizer.encode(text, add_special_tokens=False)

        final_ids = []
        if add_bos:
            final_ids.append(self.BOS_ID)
        final_ids.extend(ids)
        if add_eos:
            final_ids.append(self.EOS_ID)

        if truncate and len(final_ids) > self.max_seq_len:
            if add_eos:
                final_ids = final_ids[: self.max_seq_len - 1] + [self.EOS_ID]
            else:
                final_ids = final_ids[: self.max_seq_len]

        return final_ids

    # ------------------------------------------------------------------ #
    # Décodage : liste d'ids -> texte
    # ------------------------------------------------------------------ #
    def decode(self, ids, skip_special: bool = True) -> str:
        if torch.is_tensor(ids):
            ids = ids.tolist()

        if skip_special:
            ids = [i for i in ids if i not in self._speciaux_ids]

        return self.tokenizer.decode(ids)

    # ------------------------------------------------------------------ #
    # Batch : plusieurs textes -> tenseur (B, T) aligné avec padding
    # ------------------------------------------------------------------ #
    def encode_batch(
        self,
        texts: List[str],
        device=None,
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        encoded = [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]
        max_len = max((len(e) for e in encoded), default=1)
        max_len = max(max_len, 1)

        input_ids = torch.full((len(encoded), max_len), self.PAD_ID, dtype=torch.long)
        attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long)

        for row, e in enumerate(encoded):
            input_ids[row, : len(e)] = torch.tensor(e, dtype=torch.long)
            attention_mask[row, : len(e)] = 1

        if device is not None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

        return input_ids, attention_mask


# Alias pour compatibilite avec le reste du code (bridge.py, train.py, etc.
# importent tous ByteTokenizer - inutile de renommer partout)
ByteTokenizer = CamembertTokenizerWrapper


if __name__ == "__main__":
    tok = ByteTokenizer(max_seq_len=64)

    essais = [
        "Bonjour !",
        "Ça va ? Édouard était énervé.",
        "café, garçon, hôtel, forêt",
        "",
    ]

    tous_ok = True
    for texte in essais:
        ids = tok.encode(texte)
        retour = tok.decode(ids)
        ok = (retour.strip() == texte.strip()) or (len(texte) > tok.max_seq_len)
        tous_ok = tous_ok and ok
        print(f"[{'OK' if ok else 'ECHEC'}] {len(texte)} caracteres -> {len(ids)} ids -> match={retour.strip() == texte.strip()}")

    print(f"\nTOUS OK: {tous_ok}")
