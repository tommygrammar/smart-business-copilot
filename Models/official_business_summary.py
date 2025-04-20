from Models.wave_driver_analysis import generate_business_narrative
from Models.strengths_weakness_assessment_model import find_strengths, find_weaknesses
import re

def create_business_summary():
    # Get the narrative overview for sales and revenue.
    narrative_data = generate_business_narrative("sales", "revenue")
    # Extract the narrative string from the returned dict.
    narrative = narrative_data.get("narrative", "")
    wave_data = narrative_data.get("wave_graph_data", "")
    
    # Get the strength and weakness analysis based on revenue data.
    strengths = find_strengths("revenue")
    weaknesses = find_weaknesses("revenue")
    
    # Convert strengths and weaknesses to strings if they aren't already.
    if isinstance(strengths, list):
        strengths_text = "\n".join(str(item) for item in strengths)
    else:
        strengths_text = str(strengths)
    
    if isinstance(weaknesses, list):
        weaknesses_text = "\n".join(str(item) for item in weaknesses)
    else:
        weaknesses_text = str(weaknesses)
    
    # --- Extract key performance details from the narrative ---
    # Look for lines that mention sales and revenue performance.
    sales_perf = ""
    revenue_perf = ""
    sales_driver_line = ""
    
    for line in narrative.splitlines():
        line_strip = line.strip()
        if "Sales is forecasted" in line_strip:
            sales_perf = line_strip
        if "Revenue is projected" in line_strip:
            revenue_perf = line_strip
        # Look for a line where sales is a major or not a major driver
        if re.search(r'\bsales\b.*(major|driven)', line_strip, re.IGNORECASE):
            sales_driver_line = line_strip
        if "revenue is not solely driven by sales" in line_strip:
            extra = "Sales conversion yako to revenue haiko efficient vile inafaa. Angalia inventory na expenses na kama promotions zako ziko poa. Inaweza kuwa hao indio kuna shida. Ukikosa kuifix unaweza endelea kulose do"
        elif  "revenue is not solely driven by sales" not in line_strip:
            extra = "sales conversion yako to revenue iko poa sai. Tuko sure ukiuza revenue inagrow directly so sai focus on increasing sales etc"
            

    performance_summary = f"{sales_perf} \n\n {revenue_perf}\n\n".strip()
    # Append the sales driver analysis if found.
    if sales_driver_line:
        performance_summary += f"\n\n {sales_driver_line}\n\n"
    
    # --- Extract key drivers from the strengths and weaknesses texts ---
    # For strengths, look for bullet points starting with "• " followed by the driver name and a colon.
    strong_drivers = re.findall(r"•\s*\*+([\w_]+)\*+\s*:", strengths_text)
    # For weaknesses, look for bullet points that include markdown bold formatting for the driver name.
    weak_drivers = re.findall(r"•\s*\*+([\w_]+)\*+\s*:", weaknesses_text)
    
    # Create a concise summary from the key points.
    narrative = (
        "-----------------------------\n\n"
        "Business Summary:\n"
        "=================\n\n"
        "**Performance Overview:**\n\n"
        f"{performance_summary}\n\n"
        "**Key Revenue Drivers:**\n\n"
        f"  **Strongest:** {', '.join(strong_drivers) if strong_drivers else 'No strengths are present. This is bad.'}\n\n"
        f"  **Weakest:** {', '.join(weak_drivers) if weak_drivers else 'No weaknesses found. Bloody brilliant. This means you need to focus on optimizing.'}\n\n"
        f"  **To note:**  {extra if extra else 'N/A'}\n\n"
    )

    result = {
        "narrative": narrative
        
    }
    
    return result
