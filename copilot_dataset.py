from probabilistic_intent_language_model import model
from itertools import combinations


##impact analysis
model.train("am i making losses", "invoke: loss")
##impact analysis
model.train("was the promotion effective", "invoke: impact_analysis")
model.train("how did the promotion perform", "invoke: impact_analysis")
model.train("did sales improve after the promotion", "invoke: impact_analysis")
model.train("how is the promotion doing", "invoke: impact_analysis")
model.train("did the promotion work", "invoke: impact_analysis")
model.train("competitor analysis", "invoke: competitor_analysis")

##risk-modeling analysis
model.train("do a risk analysis", "invoke: risk_modeling")
model.train("assess my risk", "invoke: risk_modeling")
model.train("are my risks bad", "invoke: risk_modeling")
model.train("how good is my risk", "invoke: risk_modeling")


##impact analysis
model.train("calculate an event probability", "invoke: event_analysis")
model.train("am i likely to reach my revenue target?", "invoke: event_analysis")

#cashflow analysis
model.train("how many sales am i likely to make this week?", "invoke:cashflow_analysis")
model.train("is my cashflow healthy for this week?", "invoke:cashflow_analysis")
model.train("are my sales likely to be low this week?", "invoke:cashflow_analysis")
model.train("are my sales likely to be high this week?", "invoke:cashflow_analysis")
model.train("will my cashflow be a negative?", "invoke:cashflow_analysis")

model.train("is my supply chain gonna fail", "invoke:supply_chain")
model.train("how likely is it that my  supply chain is gonna fail", "invoke:supply_chain")
model.train("is my supply chain fragile", "invoke:supply_chain")

model.train("how likely is it that i will run out of stock", "invoke:stockout")
model.train("are my stockout chances high", "invoke:stockout")
model.train("how much stock do i need", "invoke:stockout")


model.train("when do i make lots of sales", "invoke:sales_time")

model.train("what is my demand looking like this week", "invoke:demand")

model.train("what are my strongest products", "invoke:segment")
model.train("what are my weakest products", "invoke:segment")

# Basic revenue analysis phrases

model.train("how is my business doing", "invoke: show_summary")
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


# Existing training pairs for dynamic optimization queries using a revenue placeholder.
model.train("how can i keep marketing budget constant and scale my sales in order to reach <revenue> revenue", 
            "invoke: optimize")
model.train("how can i keep my sales constant and optimize my marketing to reach the target of <revenue>", 
            "invoke: optimize")
model.train("how can i optimize both my scales and marketing to reach <revenue>", 
            "invoke: optimize")

# Additional variations
model.train("tell me about marketing", "invoke: generate_marketing_summary")
model.train("give me a marketing report", "invoke: generate_marketing_summary")
model.train("how effective is our marketing", "invoke: generate_marketing_summary")
model.train("marketing update", "invoke: generate_marketing_summary")
model.train("explain marketing performance", "invoke: generate_marketing_summary")

# Cost Efficiency Analysis Training Pairs
model.train("analyze cost efficiency", "invoke: generate_cost_summary")
model.train("analyze costs", "invoke: generate_cost_summary")
model.train("analyze expenses", "invoke: generate_cost_summary")
model.train("cost efficiency analysis", "invoke: generate_cost_summary")
model.train("what are our cost trends", "invoke: generate_cost_summary")
model.train("how efficient are our costs", "invoke: generate_cost_summary")
model.train("analyze our expenses", "invoke: generate_cost_summary")
model.train("what is the cost trend", "invoke: generate_cost_summary")

# Additional variations
model.train("give me a cost report", "invoke: generate_cost_summary")
model.train("update me on our expenses", "invoke: generate_cost_summary")
model.train("cost update", "invoke: generate_cost_summary")
model.train("explain cost efficiency", "invoke: generate_cost_summary")
model.train("tell me about our expenses", "invoke: generate_cost_summary")

# Greetings
model.train("Hi", "Hi Tom, how may i help you today?")
model.train("Good Morning", "Good Morning Tom, how may i help you today?")
model.train("hello", "Hi Tom, how may i help you today?")
model.train("hey there", "Hi Tom, how may i help you today?")

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

# Generate multiple training pair variations for each additional factor.
for factor in additional_factors:
    query_factor = factor.replace("_", " ")
    # Variations for DECREASE intent:
    model.train(f"decrease {query_factor} by <rate> and optimize {query_factor} for <revenue> revenue", "invoke: optimize")
    model.train(f"reduce {query_factor} by <rate> and adjust {query_factor} to achieve <revenue> revenue", "invoke: optimize")
    model.train(f"cut {query_factor} by <rate> so that {query_factor} meets the <revenue> revenue target", "invoke: optimize")
    model.train(f"lower {query_factor} by <rate> and calibrate {query_factor} for <revenue> revenue", "invoke: optimize")
    model.train(f"trim {query_factor} by <rate> to optimize performance for <revenue> revenue", "invoke: optimize")
    model.train(f"minimize {query_factor} by <rate> while striving for <revenue> revenue", "invoke: optimize")
    model.train(f"curtail {query_factor} by <rate> and improve {query_factor} to hit <revenue> revenue", "invoke: optimize")
    model.train(f"reduce costs: lower {query_factor} by <rate> and adjust to achieve <revenue> revenue", "invoke: optimize")
    model.train(f"cut back on {query_factor} by <rate> and streamline it for <revenue> revenue", "invoke: optimize")
    model.train(f"decline {query_factor} by <rate> and reconfigure {query_factor} for <revenue> revenue", "invoke: optimize")
    model.train(f"scale down {query_factor} by <rate> to meet the target of <revenue> revenue", "invoke: optimize")
    model.train(f"reduce the value of {query_factor} by <rate> and set it for <revenue> revenue", "invoke: optimize")
    model.train(f"shrink {query_factor} by <rate> so that it aligns with <revenue> revenue goals", "invoke: optimize")
    model.train(f"cut {query_factor} by <rate> and fine-tune it to reach <revenue> revenue", "invoke: optimize")
    model.train(f"trim down {query_factor} by <rate> to ensure <revenue> revenue is met", "invoke: optimize")

    # Variations for INCREASE intent:
    model.train(f"increase {query_factor} by <rate> and optimize {query_factor} for <revenue> revenue", "invoke: optimize")
    model.train(f"boost {query_factor} by <rate> and adjust {query_factor} to achieve <revenue> revenue", "invoke: optimize")
    model.train(f"raise {query_factor} by <rate> so that {query_factor} drives <revenue> revenue", "invoke: optimize")
    model.train(f"amplify {query_factor} by <rate> and fine-tune it for <revenue> revenue", "invoke: optimize")
    model.train(f"expand {query_factor} by <rate> to meet the target of <revenue> revenue", "invoke: optimize")
    model.train(f"enhance {query_factor} by <rate> and recalibrate for <revenue> revenue", "invoke: optimize")
    model.train(f"upgrade {query_factor} by <rate> and set it to hit <revenue> revenue", "invoke: optimize")
    model.train(f"improve {query_factor} by <rate> and adjust it for <revenue> revenue", "invoke: optimize")
    model.train(f"elevate {query_factor} by <rate> so that it achieves <revenue> revenue", "invoke: optimize")
    model.train(f"raise the level of {query_factor} by <rate> to ensure <revenue> revenue", "invoke: optimize")
    model.train(f"ramp up {query_factor} by <rate> and optimize for a <revenue> revenue target", "invoke: optimize")
    model.train(f"grow {query_factor} by <rate> and align it to generate <revenue> revenue", "invoke: optimize")
    model.train(f"augment {query_factor} by <rate> and recalibrate for <revenue> revenue", "invoke: optimize")
    model.train(f"upgrade our {query_factor} by <rate> so that it meets <revenue> revenue goals", "invoke: optimize")
    model.train(f"increase our {query_factor} by <rate> to drive <revenue> revenue", "invoke: optimize")
    model.train(f"how much {query_factor} do i need  while my {query_factor} is constant to get to a <revenue> revenue", "invoke: optimize")

# Generate multiple training pair variations for each additional factor.
for factor in additional_factors:
    query_factor = factor.replace("_", " ")
    # Keeping marketing constant and adjusting sales
    model.train(f"I want to keep my {query_factor} unchanged while increasing {query_factor} to achieve a revenue goal of <revenue>", 
                "invoke: optimize")
    model.train(f"Maintain my {query_factor} spend and adjust my {query_factor} to reach <revenue> revenue", 
                "invoke: optimize")
    model.train(f"How do I keep {query_factor} costs fixed and scale my {query_factor} to hit <revenue> in revenue?", 
                "invoke: optimize")
    model.train(f"Keep my {query_factor} constant and let my {query_factor} drive revenue to <revenue>", 
                "invoke: optimize")
    model.train(f"optimize {query_factor} keep {query_factor} for <revenue> revenue", 
                "invoke: optimize")

    

    # Keeping {query_factor} constant and adjusting {query_factor}
    model.train(f"I need to keep my {query_factor} stable and improve my {query_factor} strategy to target <revenue> revenue", 
                "invoke: optimize")
    model.train(f"Keep {query_factor} constant and optimize {query_factor} so that revenue reaches <revenue>", 
                "invoke: optimize")
    model.train(f"If I hold my {query_factor} steady, what {query_factor} changes are needed to hit <revenue> revenue?", 
                "invoke: optimize")
    model.train(f"I want to keep my {query_factor} numbers unchanged, but adjust {query_factor} to achieve <revenue> revenue", 
                "invoke: optimize")

    # Optimizing both {query_factor} and {query_factor}
    model.train(f"How can I optimize both {query_factor} and {query_factor} to reach a revenue target of <revenue>?", 
                "invoke: optimize")
    model.train(f"What adjustments should I make to both {query_factor} and {query_factor} to hit <revenue> revenue?", 
                "invoke: optimize")
    model.train(f"I need to optimize both my {query_factor} and {query_factor} efforts to achieve <revenue> revenue", 
                "invoke: optimize")

    # Additional variations with different phrasing
    model.train(f"I want to keep my {query_factor} budget constant and modify {query_factor} to reach <revenue> revenue", 
                "invoke: optimize")
    model.train(f"I want to keep my {query_factor} constant and modify {query_factor} to reach <revenue> revenue", 
                "invoke: optimize")
    model.train(f"How can I maintain constant {query_factor} and boost my {query_factor} to achieve <revenue> revenue?", 
                "invoke: optimize")
    model.train(f"If I hold my {query_factor} steady, what {query_factor} adjustments are needed to hit <revenue> revenue?", 
                "invoke: optimize")
    model.train(f"How can I adjust my {query_factor} while keeping {query_factor} fixed to get <revenue> revenue?", 
                "invoke: optimize")
    model.train(f"Keep my {query_factor} unchanged and tell me what {query_factor} adjustment is needed to hit <revenue> revenue", 
                "invoke: optimize")
    model.train(f"How can I reach a revenue of <revenue> by only changing my {query_factor} strategy and keeping {query_factor} constant?", 
                "invoke: optimize")
    model.train(f"optimize {query_factor} , keep {query_factor} constant for a target revenue of <revenue>", 
                "invoke: optimize")
    model.train(f"optimize {query_factor} , keep {query_factor} constant for a target revenue of <revenue>", 
                "invoke: optimize")

# Generate multiple training pair variations for each additional factor.
for factor in additional_factors:
    query_factor = factor.replace("_", " ")
    model.train(f"what would happen if i increased {query_factor} by <rate>", "invoke: business_twin")
    model.train(f"how would our {query_factor} change if we boosted it by <rate>", "invoke: business_twin")
    model.train(f"if we raised {query_factor} by <rate>, what impact would it have on our performance", "invoke: business_twin")
    model.train(f"what is the effect of increasing {query_factor} by <rate> on our overall business", "invoke: business_twin")
    model.train(f"simulate a scenario where {query_factor} is increased by <rate>", "invoke: business_twin")
    model.train(f"what would happen if i increase {query_factor} by <rate> to my business", "invoke: business_twin")
    model.train(f"analyze the impact if we increased {query_factor} by <rate>", "invoke: business_twin")
    model.train(f"what would be the outcome if {query_factor} was increased by <rate>", "invoke: business_twin")
    model.train(f"how would a <rate> increase in {query_factor} affect our business", "invoke: business_twin")
    model.train(f"can you simulate an increase of <rate> in our {query_factor} and show the effect", "invoke: business_twin")
    model.train(f"project the effect of a <rate> boost in {query_factor}", "invoke: business_twin")
    model.train(f"if we were to increase {query_factor} by <rate>, how would that change our business outcomes", "invoke: business_twin")
    model.train(f"what results do we see when {query_factor} is increased by <rate>?", "invoke: business_twin")
    model.train(f"explain the impact on our business if {query_factor} grows by <rate>", "invoke: business_twin")
    model.train(f"forecast the effects of boosting {query_factor} by <rate>", "invoke: business_twin")
    model.train(f"simulate a future scenario where {query_factor} is raised by <rate>", "invoke: business_twin")
    model.train(f"how would increasing {query_factor} by <rate> affect my business", "invoke: business_twin")

for factor in additional_factors:
    query_factor = factor.replace("_", " ")
    # --- Weakness Training Pairs (Expanded) ---
    model.train(f"why is my {query_factor} performing poorly", f"invoke: find_weaknesses({factor})")
    model.train(f"what are my {query_factor} weakpoints", f"invoke: find_weaknesses({factor})")
    model.train(f"explain why {query_factor} is down", f"invoke: find_weaknesses({factor})")
    model.train(f"what is dragging {query_factor} down", f"invoke: find_weaknesses({factor})")
    model.train(f"what is causing poor {query_factor}", f"invoke: find_weaknesses({factor})")
    model.train(f"what is driving {query_factor} down", f"invoke: find_weaknesses({factor})")
    model.train(f"why has {query_factor} declined", f"invoke: find_weaknesses({factor})")
    model.train(f"why is my {query_factor} lagging", f"invoke: find_weaknesses({factor})")
    model.train(f"what are the shortcomings of my {query_factor}", f"invoke: find_weaknesses({factor})")
    model.train(f"what factors are hindering my {query_factor}", f"invoke: find_weaknesses({factor})")
    model.train(f"explain why my {query_factor} is underperforming", f"invoke: find_weaknesses({factor})")
    model.train(f"why is my {query_factor} not meeting expectations", f"invoke: find_weaknesses({factor})")
    model.train(f"what issues are affecting my {query_factor}", f"invoke: find_weaknesses({factor})")
    model.train(f"what problems are dragging {query_factor} down", f"invoke: find_weaknesses({factor})")
    model.train(f"identify the weaknesses in my {query_factor}", f"invoke: find_weaknesses({factor})")
    model.train(f"where is my {query_factor} falling short", f"invoke: find_weaknesses({factor})")
    model.train(f"what are the weaknesses in {query_factor}", f"invoke: find_weaknesses({factor})")
    
    # --- Strength Training Pairs (Expanded) ---
    model.train(f"why is my {query_factor} strong", f"invoke: find_strengths({factor})")
    model.train(f"what are my {query_factor} strongpoints", f"invoke: find_strengths({factor})")
    model.train(f"explain why {query_factor} is up", f"invoke: find_strengths({factor})")
    model.train(f"what is driving {query_factor} up", f"invoke: find_strengths({factor})")
    model.train(f"what is causing high {query_factor}", f"invoke: find_strengths({factor})")
    model.train(f"why has {query_factor} improved", f"invoke: find_strengths({factor})")
    model.train(f"why is my {query_factor} excelling", f"invoke: find_strengths({factor})")
    model.train(f"what factors are boosting my {query_factor}", f"invoke: find_strengths({factor})")
    model.train(f"explain why my {query_factor} is performing well", f"invoke: find_strengths({factor})")
    model.train(f"what are the advantages of my {query_factor}", f"invoke: find_strengths({factor})")
    model.train(f"what is contributing to my {query_factor} success", f"invoke: find_strengths({factor})")
    model.train(f"why is my {query_factor} thriving", f"invoke: find_strengths({factor})")
    model.train(f"identify the strengths in my {query_factor}", f"invoke: find_strengths({factor})")
    model.train(f"where is my {query_factor} excelling", f"invoke: find_strengths({factor})")
    model.train(f"find the strengths in our {query_factor} performance", f"invoke: find_strengths({factor})")
    model.train(f"what improvements are seen in my {query_factor}", f"invoke: find_strengths({factor})")

# Closing and courtesy pairs
model.train("so good things are coming", "yes sir, based on our data, for the next 6 months, your business should perform significantly well")
model.train("thank you", "You are welcome. Feel free to approach me with anything to do with your business that i can help you with.")
model.train("thanks", "You are welcome. Feel free to ask me anything else about your business.")


# Generate training pairs for analyze_drivers (comparison of two different factors)
for factor1, factor2 in combinations(additional_factors, 2):
    qf1 = factor1.replace("_", " ")
    qf2 = factor2.replace("_", " ")
    model.train(f"analyze {qf1} and {qf2}", "invoke: analyze_drivers")
    model.train(f"{qf1} and {qf2} analysis", "invoke: analyze_drivers")
    model.train(f"compare {qf1} and {qf2}", "invoke: analyze_drivers")
    model.train(f"what is the relationship between {qf1} and {qf2}", "invoke: analyze_drivers")
    model.train(f"explain the correlation between {qf1} and {qf2}", "invoke: analyze_drivers")
    model.train(f"give me a {qf1} and {qf2} report", "invoke: analyze_drivers")
    model.train(f"i need a {qf1} and {qf2} report", "invoke: analyze_drivers")
    model.train(f"provide a summary of {qf1} and {qf2}", "invoke: analyze_drivers")
    model.train(f"how are {qf1} and {qf2} performing together", "invoke: analyze_drivers")
    model.train(f"{qf1} versus {qf2} analysis", "invoke: analyze_drivers")
    model.train(f"analyze the interplay between {qf1} and {qf2}", "invoke: analyze_drivers")
    model.train(f"explain how {qf1} and {qf2} interact", "invoke: analyze_drivers")
    model.train(f"what are the trends in {qf1} and {qf2}", "invoke: analyze_drivers")
    model.train(f"give me an overview of {qf1} and {qf2} trends", "invoke: analyze_drivers")
    model.train(f"business analysis of {qf1} and {qf2}", "invoke: analyze_drivers")
    model.train(f"narrative for {qf1} and {qf2} performance", "invoke: analyze_drivers")
    model.train(f"summarize {qf1} and {qf2} dynamics", "invoke: analyze_drivers")
    model.train(f"how do {qf1} and {qf2} compare", "invoke: analyze_drivers")
    model.train(f"provide a comparison of {qf1} versus {qf2}", "invoke: analyze_drivers")
    model.train(f"what is driving {qf1} and {qf2}", "invoke: analyze_drivers")
    model.train(f"explain future trends for {qf1} and {qf2}", "invoke: analyze_drivers")
    model.train(f"analyze the combined behavior of {qf1} and {qf2}", "invoke: analyze_drivers")
    model.train(f"{qf1} and {qf2} forecast summary", "invoke: analyze_drivers")
    model.train(f"what factors influence both {qf1} and {qf2}", "invoke: analyze_drivers")
    model.train(f"business narrative for {qf1} and {qf2}", "invoke: analyze_drivers")
    model.train(f"how are our {qf1} and {qf2} related", "invoke: analyze_drivers")
    model.train(f"compare our {qf1} and {qf2} cycles", "invoke: analyze_drivers")
    model.train(f"what is the overlap between {qf1} and {qf2} trends", "invoke: analyze_drivers")
    model.train(f"explain the dynamic between {qf1} and {qf2}", "invoke: analyze_drivers")
    model.train(f"analyze {qf1} and {qf2} together", "invoke: analyze_drivers")
    model.train(f"how is {qf1} and {qf2} together", "invoke: analyze_drivers")
    model.train(f"how is {qf1} and {qf2}", "invoke: analyze_drivers")
    model.train(f"how is {qf1} and {qf2} doing", "invoke: analyze_drivers")

