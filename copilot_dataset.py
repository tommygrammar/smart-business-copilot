from probabilistic_intent_language_model import model

##impact analysis
model.train("am i making losses", "invoke: loss")
model.train("kuna mahali napoteza pesa", "invoke: loss")
model.train("what inefficiencies do i have", "invoke: loss")
##impact analysis
model.train("was the promotion effective", "invoke: impact_analysis")
model.train("promotion ilifanya kazi", "invoke: impact_analysis")
model.train("promotion inaendelea aje", "invoke: impact_analysis")
model.train("nafaa kufanya promotion tena", "invoke: impact_analysis")
model.train("how did the promotion perform", "invoke: impact_analysis")
model.train("did sales improve after the promotion", "invoke: impact_analysis")
model.train("how is the promotion doing", "invoke: impact_analysis")
model.train("did the promotion work", "invoke: impact_analysis")
model.train("competitor analysis", "invoke: competitor_analysis")
model.train("kuna shop ingine imeingia", "invoke: competitor_analysis")
model.train("sales zangu zitareduce juu ya iyo shop mpya", "invoke: competitor_analysis")
model.train("how is the competitor doing", "invoke: competitor_analysis")

#HOW MUCH
model.train("how much sales did i make", "invoke:how_many_sales")
model.train("nitauza vitu ngapi this week", "invoke:how_many_sales")
model.train("how much inventory do i have", "invoke:how_much_inventory")
model.train("nimebaki na inventory ya ngapi", "invoke:how_much_inventory")
model.train("inventory imebaki?", "invoke:how_much_inventory")
model.train("how much revenue did i make", "invoke:how_much_revenue")
model.train("nilimake revenue ya how much", "invoke:how_much_revenue")

#cashflow analysis
model.train("how many sales am i likely to make this week?", "invoke:cashflow_analysis")
model.train("nitauza vitu ngapi this week?", "invoke:cashflow_analysis")
model.train("naweza uza vitu ngapi this week?", "invoke:cashflow_analysis")
model.train("this week niko likely kumake sales ngapi?", "invoke:cashflow_analysis")
model.train("is my cashflow healthy for this week?", "invoke:cashflow_analysis")
model.train("cashflow yangu iko aje?", "invoke:cashflow_analysis")
model.train("are my sales likely to be low this week?", "invoke:cashflow_analysis")
model.train("sales zangu zitakuwa low this week", "invoke:cashflow_analysis")
model.train("are my sales likely to be high this week?", "invoke:cashflow_analysis")
model.train("sales zangu zitakuwa high this week?", "invoke:cashflow_analysis")
model.train("will my cashflow be a negative?", "invoke:cashflow_analysis")

model.train("is my supply chain gonna fail", "invoke:supply_chain")
model.train("how likely is it that my  supply chain is gonna fail", "invoke:supply_chain")
model.train("is my supply chain fragile", "invoke:supply_chain")

model.train("how likely is it that i will run out of stock", "invoke:stockout")
model.train("are my stockout chances high", "invoke:stockout")
model.train("how much stock do i need", "invoke:stockout")
model.train("niko karibu kuishiwa na stock", "invoke:stockout")
model.train("nahitaji stock ya how much", "invoke:stockout")


model.train("when do i make lots of sales", "invoke:sales_time")
model.train("ni saa ngapi huwa sales ziko juu", "invoke:sales_time")
model.train("what is my demand looking like this week", "invoke:demand")
model.train("unaona demand itakuwa aje this week", "invoke:demand")
model.train("demand itakuwa juu this weekk", "invoke:demand")
model.train("what are my strongest products", "invoke:segment")
model.train("ni product gani huenda sana", "invoke:segment")
model.train("what are my weakest products", "invoke:segment")
model.train("ni product gani haileti profit sana", "invoke:segment")

# Basic revenue analysis phrases
model.train("how is my business doing", "invoke: show_summary")
model.train("biashara iko aje", "invoke: show_summary")
model.train("naendelea vizuri?", "invoke: show_summary")
model.train("executive summary", "invoke: show_summary")
model.train("business summary", "invoke: show_summary")
model.train("how is my business performing", "invoke: show_summary")
model.train("what is my business performance summary", "invoke: show_summary")
model.train("give me my business overview", "invoke: show_summary")
model.train("summary of my business results", "invoke: show_summary")
model.train("show my company's performance summary", "invoke: show_summary")
model.train("provide executive summary of my business", "invoke: show_summary")
model.train("what's the update on my business performance", "invoke: show_summary")
model.train("summarize my business performance", "invoke: show_summary")
model.train("provide a summary of my business health", "invoke: show_summary")
model.train("business performance report", "invoke: show_summary")
model.train("company performance summary", "invoke: show_summary")

# Greetings
model.train("Hi", "Hi, how may i help you today?")
model.train("Niaje", "Niko fiti, nikusaidie aje kwa bizna yako")
model.train("Good Morning", "Good Morning, how may i help you today?")
model.train("hello", "Hello, how may i help you today?")
model.train("hey there", "Hey, how may i help you today?")

# Expanded Training Pairs for Additional Factors
additional_factors = [
    "sales", "revenue", "marketing", "customer_sat", "website_traffic", "employee_productivity",
    "operational_efficiency", "competitive_advantage", "inventory", "customer_loyalty",
    "brand_awareness", "supply_chain_reliability", "innovation", "employee_satisfaction",
    "market_share", "digital_engagement", "social_media_presence", "product_quality",
    "operational_costs", "financial_health"
]

# Generate multiple training pair variations for each additional factor.
for factor in additional_factors:
    query_factor = factor.replace("_", " ")
    # Variations for DECREASE intent:
    model.train(f"do a standard analysis of {query_factor} and {query_factor} for <days> days ", "invoke: deep_analysis")
    model.train(f"analyze {query_factor} and {query_factor} for <days> days ", "invoke: deep_analysis")
    model.train(f"how {query_factor} is {query_factor} this <days> days ", "invoke: deep_analysis")
    model.train(f"how has {query_factor} and {query_factor} been for the past <days> days", "invoke: deep_analysis")
    model.train(f"how has {query_factor} and {query_factor} been for the last <days> days", "invoke: deep_analysis")

# Closing and courtesy pairs
model.train("thank you", "You are welcome. Feel free to approach me with anything to do with your business that i can help you with.")
model.train("thanks", "You are welcome. Feel free to ask me anything else about your business.")