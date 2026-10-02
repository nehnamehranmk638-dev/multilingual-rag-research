import json
import re

class ClaimExtractor:
    def __init__(self, generator, max_claims=12):
        self.generator = generator
        self.max_claims = max_claims

    def extract(self, answer):
        prompt = (
            "Break the following answer into atomic factual claims. "
            "Each item must contain one independently verifiable fact. "
            "Return ONLY a JSON array of strings.\n\n"
            f"Answer:\n{answer}"
        )
        messages = [
            {"role": "system", "content": "You extract atomic factual claims."},
            {"role": "user", "content": prompt}
        ]
        tok = self.generator.tokenizer
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(self.generator.model.device)
        with __import__("torch").no_grad():
            out = self.generator.model.generate(
                **inputs, max_new_tokens=256, do_sample=False
            )
        decoded = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

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
            return []
        return [str(x).strip() for x in claims if str(x).strip()][:self.max_claims]
