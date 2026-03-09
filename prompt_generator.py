def build_pico_prompt(abstract_text, prediction):
    
    # 1. Base system instruction establishing the LLM's persona
    system_prompt = (
        "You are an expert clinical data extractor and reviewer. "
        "Your task is to analyze a medical abstract based on findings from an upstream classification model.\n\n"
    )
    
    # 2. Handle the "Complete" scenario
    if prediction == "Complete":
        task_instruction = (
            "The classification model found all required PICO elements (Population, Intervention, Outcome). "
            "Please output a short, structured summary of the Patient, Intervention, and Outcome "
            "based strictly on the text provided. Do not hallucinate external information."
        )
        
    # 3. Handle the "Missing" scenarios
    else:
        # Extract just the letter (P, I, or O) from the prediction string
        missing_letter = prediction.split(" ")[1] 
        element_map = {"P": "Patient/Population", "I": "Intervention", "O": "Outcome"}
        full_name = element_map.get(missing_letter, "PICO element")
        
        task_instruction = (
            f"The classification model has flagged this abstract as MISSING the {full_name} element. "
            f"Generate a single, professional clarification question directed at the authors "
            f"requesting the missing {full_name} details. Do not answer the question yourself, "
            f"and do not include information not present in the abstract."
        )
        
    # 4. Assemble the final prompt structure
    final_prompt = (
        f"{system_prompt}"
        f"--- TASK ---\n{task_instruction}\n\n"
        f"--- ABSTRACT ---\n{abstract_text}\n\n"
        f"--- RESPONSE ---\n"
    )
    
    return final_prompt


sample_abstract = (
    "We conducted a randomized controlled trial to evaluate the efficacy of a new "
    "cognitive behavioral therapy approach. The primary outcome measured was the "
    "reduction in severe anxiety symptoms after six months."
)

mock_prediction = "Missing P"

llm_prompt = build_pico_prompt(sample_abstract, mock_prediction)
print(llm_prompt)