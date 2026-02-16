import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLM:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        """
        Initialize the model and tokenizer for local GPU inference.
        Automatically splits model across available GPUs.
        """
        print(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"Loading model for {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={0: "7GB", 1: "7GB", 2: "7GB", 3:"7GB"},
        )
        self.model.eval()

    def generate(self, messages, temperature=0.7, max_new_tokens=10):
        """
        Generate text based on a list of messages.
        `messages` should be a list of dicts with a 'role' and 'content' key (like OpenAI format).
        """
        prompt = "\n".join([m["content"] for m in messages])

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_tokens = outputs[0][input_len:]

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
