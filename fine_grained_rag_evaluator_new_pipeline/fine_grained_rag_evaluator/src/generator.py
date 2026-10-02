import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class QwenGenerator:
    def __init__(self, model_name, max_new_tokens=256, temperature=0.0, do_sample=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

    def generate(self, question, evidence):
        context = "\n\n".join(
            f"[{i+1}] {x['text']}" for i, x in enumerate(evidence)
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the question using only the supplied evidence. "
                    "Do not invent facts. If the evidence is insufficient, say so."
                )
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nEvidence:\n{context}"
            }
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample
        }
        if self.do_sample:
            kwargs["temperature"] = self.temperature
        with torch.no_grad():
            output = self.model.generate(**inputs, **kwargs)
        generated = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
