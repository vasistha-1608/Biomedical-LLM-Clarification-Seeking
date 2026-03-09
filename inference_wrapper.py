# %%
import torch

def predict_missing_pico(abstract_text, model, tokenizer, device, max_length=512):
    
    
    inputs = tokenizer(
        abstract_text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    
    model.eval()
    with torch.no_grad():
       
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
    
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    
    
    label_map = {0: "Complete", 1: "Missing P", 2: "Missing I", 3: "Missing O"}
    
    return label_map[predicted_class_id]



# %%
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = "./saved_biobert_pico_model/"  
print(f"Loading fine-tuned model from {model_path}...")


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model successfully loaded and moved to {device}!")

sample_abstract = (
    "We conducted a randomized controlled trial to evaluate the efficacy of a new "
    "cognitive behavioral therapy approach. The primary outcome measured was the "
    "reduction in severe anxiety symptoms after six months."
)


prediction = predict_missing_pico(sample_abstract, model, tokenizer, device)

print(f"Abstract: {sample_abstract}")
print(f"BioBERT Prediction: {prediction}")

# %%



