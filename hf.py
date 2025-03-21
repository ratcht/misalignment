from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, PreTrainedTokenizer, PreTrainedModel
import torch
from typing import Dict, Tuple, Any
from threading import Thread
from peft import PeftModel




class HF:
  DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


  @classmethod
  def load_model(cls, model_name: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
      model_name,
      torch_dtype="auto",
      device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

  @classmethod
  def raw_query(cls, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, query: str, stream=False, skip_special_tokens=False, **kwargs) -> Dict[str, Any]:
    print(f"Running on device: {model.device}")
    model_inputs = tokenizer([query], return_tensors="pt").to(model.device)
    
    if not stream:
      with torch.no_grad():
        output = model.generate(
          **model_inputs,
          max_new_tokens=512,
          do_sample=True,
          temperature=0.7,
          top_p=0.9,
          **kwargs
        )
      if isinstance(output, list) or isinstance(output, tuple):
        response = tokenizer.decode(output[0][0], skip_special_tokens=skip_special_tokens)
      else:
        response = tokenizer.decode(output[0], skip_special_tokens=skip_special_tokens)
      return {"query": query, "response": response, "output": output}
    else:
      # Create a streamer for token-by-token generation
      streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
      
      # Set up generation parameters
      generation_kwargs = dict(
        **model_inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        streamer=streamer
      )
      
      # Start generation in a separate thread
      thread = Thread(target=model.generate, kwargs=generation_kwargs)
      thread.start()
      
      # Return the streamer object for the caller to iterate through
      return {
        "query": query, 
        "streamer": streamer, 
        "thread": thread, 
      }

  @classmethod
  def query(cls, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, query: str, system_prompt: str = None, stream: bool = False, **kwargs) -> Dict[str, Any]:    
    
    if system_prompt:
      messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
      ]
    else:
      messages = [
        {"role": "user", "content": query}
      ] 

    text = tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True
    )

    return HF.raw_query(
      model, tokenizer, text, stream, **kwargs
    )
    
    
    
  @classmethod
  def load_base_model_with_adapter(self, base_model_name, adapter_path):
    # Load the base model and tokenizer
    base_model, tokenizer = HF.load_model(base_model_name)
    
    # Load the adapter onto the model
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    return model, tokenizer
  
