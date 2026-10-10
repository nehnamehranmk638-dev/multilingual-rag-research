
import json
import re

class ClaimExtractor:
    def __init__(self, generator, max_claims=12):
        self.generator = generator
        self.max_claims = max_claims

    def _split_sentence(self, sentence):
        prompt = (
            "Extract the atomic factual claims from the sentence below.\n\n"
            "Rules:\n"
            "1. Keep only factual information stated in the sentence.\n"
            "2. Remove introductory or conversational phrases such as "
            "'Based on the provided evidence'.\n"
            "3. Do not add information.\n"
            "4. Do not infer anything.\n"
            "5. Do not use outside knowledge.\n"
            "6. If the sentence contains one factual claim, return exactly "
            "one claim.\n"
            "7. If the sentence contains multiple independent factual claims, "
            "return each separately.\n"
            "8. Preserve the original factual meaning.\n"
            "9. Return ONLY a JSON array of strings.\n\n"
            f"Sentence:\n{sentence}"
        )
        messages = [
            {
                "role": "system",
                "content": "You extract atomic factual claims without adding information."
            },
            {"role": "user", "content": prompt}
        ]
        tok = self.generator.tokenizer
        text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tok(
            text,
            return_tensors="pt"
        ).to(self.generator.model.device)

        import torch
        with torch.no_grad():
            out = self.generator.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False
            )

        decoded = tok.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        try:
            claims = json.loads(decoded)
        except Exception:
            match = re.search(r"\[.*\]", decoded, flags=re.S)
            if match:
                try:
                    claims = json.loads(match.group(0))
                except Exception:
                    claims = []
            else:
                claims = []

        if not isinstance(claims, list):
            return [sentence]

        claims = [str(x).strip() for x in claims if str(x).strip()]
        return claims if claims else [sentence]

    def extract(self, answer):
        sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
        claims = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_claims = self._split_sentence(sentence)
            claims.extend(sentence_claims)

            if len(claims) >= self.max_claims:
                break

        return claims[:self.max_claims]
