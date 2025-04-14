import networkx as nx
import re
import math
from Models.wave_driver_analysis import generate_business_narrative
from Models.deep_wave_driver_analysis_period_support import trend_generate_business_narrative
from Data.business_data import historical_data
from Models.strengths_weakness_assessment_model import find_weaknesses, find_strengths
from Models.gradient_ascent_sensitivity_allocation_model import optimize
from Models.business_twin_sensitivity_model import business_twin
from Models.official_business_summary import create_business_summary
import pickle
from Models.impact_analysis_model import bayesian_impact as impact_analysis
from Models.risk_model import run_risk_analysis
from Models.event_probability_models import calculate_event_probability
from Models.competitor_analysis import bayesian_competitor_analysis
# Use combinations of distinct factors for analysis comparisons.
from itertools import combinations
from collections import defaultdict

# Helper function for defaultdict initialization that is pickle-friendly.
def default_inner_dict():
    return defaultdict(int)


def default_int_dict():
    return defaultdict(int)

# ---------------------
# Probabilistic Language Model for Business Domain with Function Invocation
# ---------------------
class ProbabilisticLanguageModel:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.query_count = defaultdict(int)
        # Replace the lambda with a top-level function
        self.query_response_count = defaultdict(default_inner_dict)
        self.training_queries = set()
        self.context = {}
        self.entities = {}
        self.stopwords = {"this", "is", "very", "and", "the", "a", "an", "of", "to", "with", "in", "it", "for", "on", "so", "exactly"}
        self.domain_factors = ["revenue", "sales", "marketing", "cost", "profit",
                               "customer_sat", "website_traffic", "employee_productivity",
                               "operational_efficiency", "competitive_advantage", "inventory",
                               "customer_loyalty", "brand_awareness", "supply_chain_reliability",
                               "innovation", "employee_satisfaction", "market_share",
                               "digital_engagement", "social_media_presence", "product_quality",
                               "operational_costs", "financial_health"]

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s\+\-\*/\.,]', '', text)
        words = text.split()
        filtered = [w for w in words if w not in self.stopwords]
        return " ".join(filtered).strip()
    
    def standardize_query(self, text):
        text = re.sub(r'(\d[\d,\.]*)\s*(?=revenue)', '<revenue>', text)
        text = re.sub(r'\d[\d,\.]*', '<specific>', text)
        return text

    def update_conversation_context(self, text):
        for factor in self.domain_factors:
            if factor in text:
                self.context["last_topic"] = factor
                break

    def resolve_pronouns(self, text):
        words = text.split()
        resolved = []
        for word in words:
            lw = word.lower()
            if lw in {"it", "this"} and "last_topic" in self.context:
                resolved.append(self.context["last_topic"])
            else:
                resolved.append(word)
        return " ".join(resolved)

    def train(self, query, response):
        clean_query = self.clean_text(query)
        clean_response = response.strip()
        self.update_conversation_context(clean_query)
        self.training_queries.add(clean_query)
        self.query_count[clean_query] += 1
        self.query_response_count[clean_query][clean_response] += 1

        for q_node in clean_query.split():
            for r_node in clean_response.split():
                if not self.graph.has_edge(q_node, r_node):
                    self.graph.add_edge(q_node, r_node, weight=0)
                self.graph[q_node][r_node]['weight'] += 1

    # --- Similarity Methods ---
    def letter_ngrams(self, text, n=3):
        text = text.replace(" ", "")
        if len(text) < n:
            return {text}
        return {text[i:i+n] for i in range(len(text)-n+1)}

    def jaccard_similarity(self, set1, set2):
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union) if union else 0

    def word_f1_similarity(self, input_words, training_words):
        overlap = input_words.intersection(training_words)
        if not input_words or not training_words:
            return 0
        return (2 * len(overlap)) / (len(input_words) + len(training_words))

    def letter_ngram_similarity(self, input_text, training_text, n=3):
        ngrams1 = self.letter_ngrams(input_text, n)
        ngrams2 = self.letter_ngrams(training_text, n)
        return self.jaccard_similarity(ngrams1, ngrams2)

    def length_penalty(self, input_text, training_text):
        len_ratio = len(input_text.split()) / len(training_text.split())
        return min(len_ratio, 1/len_ratio)

    def combined_similarity(self, input_query, training_query):
        input_words = set(input_query.split())
        training_words = set(training_query.split())
        word_sim = self.word_f1_similarity(input_words, training_words)
        letter_sim = self.letter_ngram_similarity(input_query, training_query)
        len_pen = self.length_penalty(input_query, training_query)
        weight_word = 0.5
        weight_letter = 0.3
        return (weight_word * word_sim + weight_letter * letter_sim) * len_pen

    # --- Bayesian Scoring Method with Directional Bonus ---
    def bayesian_score(self, input_query, training_query, p_match=0.9, p_nomatch=0.1):
        input_tokens = self.clean_text(input_query).split()
        training_tokens = set(self.clean_text(training_query).split())
        score = 0.0
        directional = {
            "up": {"up", "strong", "increasing"},
            "down": {"down", "poorly", "weak", "decreasing", "underperforming"}
        }
        for token in input_tokens:
            if token in training_tokens:
                token_score = math.log(p_match)
            else:
                token_score = math.log(p_nomatch)
            if token in directional:
                if any(d in training_query for d in directional[token]):
                    token_score += 0.5
                else:
                    token_score -= 0.5
            score += token_score
        total = sum(self.query_count.values())
        epsilon = 1e-9
        ratio = self.query_count[training_query] / total if total > 0 else epsilon
        prior = math.log(max(ratio, epsilon))
        return score + prior

    def is_arithmetic_query(self, query):
        return bool(re.search(r'\d+\s*[\+\-\*/]\s*\d+', query))

    def compute_arithmetic(self, query):
        expr_match = re.findall(r'\d+\s*[\+\-\*/]\s*\d+', query)
        if expr_match:
            try:
                expr = expr_match[0]
                result = eval(expr)
                return f"The answer is {result}."
            except Exception:
                return "Error computing the arithmetic expression."
        return None

    def split_queries(self, query):
        segments = re.split(r'[;.]\s*', query)
        segments = [seg.strip() for seg in segments if seg.strip()]
        if len(segments) == 1 and " and " in segments[0]:
            segments = [seg.strip() for seg in segments[0].split(" and ") if seg.strip()]
        return segments

    def process_query(self, query):
        # Check for arithmetic queries first.
        if self.is_arithmetic_query(query):
            result = self.compute_arithmetic(query)
            if result:
                return result

        self.update_conversation_context(query)
        query = self.resolve_pronouns(query)
        clean_query = self.clean_text(query)
        standardized_query = self.standardize_query(clean_query)
        
        # Compute training query similarity for all training queries.
        training_scores = []
        for trained_query in self.training_queries:
            standardized_training = self.standardize_query(trained_query)
            score = 0.5 * self.bayesian_score(standardized_query, standardized_training) \
                    + 0.5 * self.combined_similarity(standardized_query, standardized_training)
            training_scores.append((trained_query, score))

        if training_scores:
            top_five = sorted(training_scores, key=lambda x: x[1], reverse=True)[:5]
            best_training_query, best_score = top_five[0]
            print(f"[DEBUG] Selected best training query: '{best_training_query}' with score {best_score}")
        else:
            best_training_query = None

        # Use the best training query if available.
        if best_training_query is None:
            resp = "intent not recognized. Kindly stick to the point and make it more direct'."
        else:
            responses_dict = self.query_response_count[best_training_query]
            best_response = max(responses_dict, key=lambda r: responses_dict[r])
            if best_response.startswith("invoke:"):
                function_name = best_response.split("invoke:")[1].strip()
                print(f"[DEBUG] Invoking function: '{function_name}' based on best training query")
                # If the best training query is of type business_twin, use extraction.
                if function_name.startswith("business_twin"):
                    lower_query = query.lower()
                    twin_pattern = r'\b((?:increased|increasin(?:g))|boost(?:ed|ing)?|raise(?:d|ing)?|(?:decrease(?:d)?|decreasin(?:g))|drop(?:ped|ing)?)\s+([A-Za-z\s]+)\s+(?:by|of)\s+(\d+(?:\.\d+)?)(\%?)\b'
                    twin_match = re.search(twin_pattern, lower_query, re.IGNORECASE)
                    if twin_match:
                        action = twin_match.group(1).strip()
                        factor_str = twin_match.group(2).strip()
                        value_str = twin_match.group(3).strip()
                        percent_sign = twin_match.group(4).strip()
                        try:
                            rate_value = float(value_str)
                        except ValueError:
                            resp = "Could not extract numeric parameters for business twin from best training query."
                        else:
                            if action.startswith("decrease") or action.startswith("decreasin") or action.startswith("drop"):
                                rate_value = -abs(rate_value)
                            else:
                                rate_value = abs(rate_value)
                            resp = business_twin(factor_str, rate_value)
                    else:
                        resp = "Could not extract parameters for business twin from best training query."

                elif function_name == "analyze_drivers":
                    lower_query = query.lower()
                    factors_found = []
                    for factor in self.domain_factors:
                        display_factor = factor.replace("_", " ")
                        pos = lower_query.find(display_factor)
                        if pos == -1:
                            pos = lower_query.find(factor)
                        if pos != -1:
                            factors_found.append((pos, factor))
                    factors_found.sort(key=lambda x: x[0])
                    if len(factors_found) >= 2:
                        factor1, factor2 = factors_found[0][1], factors_found[1][1]
                    else:
                        factor1, factor2 = "sales", "revenue"
                    resp = generate_business_narrative(factor1, factor2)
                elif function_name.startswith("optimize"):
                    lower_query = query.lower()
                    fixed_m = None
                    fixed_s = None
                    m_mark = re.search(r'(reduce|decrease|increase|boost).*?marketing.*?(?:by|at|to)\s*(\-?\d[\d,\.]*)', lower_query)
                    if m_mark:
                        action = m_mark.group(1)
                        value = float(m_mark.group(2).replace(',', ''))
                        fixed_m = -abs(value) if action in ["reduce", "decrease"] else abs(value)
                        constant_mode = "marketing_constant"
                    m_sales = re.search(r'(reduce|decrease|increase|boost).*?sales.*?(?:by|at|to)\s*(\-?\d[\d,\.]*)', lower_query)
                    if m_sales:
                        action = m_sales.group(1)
                        value = float(m_sales.group(2).replace(',', ''))
                        fixed_s = -abs(value) if action in ["reduce", "decrease"] else abs(value)
                        constant_mode = "sales_constant"
                    if fixed_m is None and fixed_s is None:
                        if "keep my sales constant" in lower_query or re.search(r'\bsales\b.*\bconstant\b', lower_query):
                            constant_mode = "sales_constant"
                        elif "keep my marketing constant" in lower_query or re.search(r'\bmarketing\b.*\bconstant\b', lower_query):
                            constant_mode = "marketing_constant"
                        else:
                            constant_mode = "none"
                    match = re.search(r'(\d[\d,\.]*)\s*(?:revenue)', lower_query)
                    if match:
                        target = float(match.group(1).replace(',', ''))
                    else:
                        numbers = re.findall(r'\d[\d,\.]*', lower_query)
                        target = float(numbers[-1].replace(',', '')) if numbers else 0.0
                    if constant_mode == "sales_constant" and fixed_s is None:
                        fixed_s = 0.0
                    if constant_mode == "marketing_constant" and fixed_m is None:
                        fixed_m = 0.0
                    resp = optimize(constant_mode, target, fixed_m, fixed_s)
                elif function_name.startswith("find_weaknesses"):
                    match = re.match(r"find_weaknesses\((\w+)\)", function_name)
                    if match:
                        factor = match.group(1)
                        resp = find_weaknesses(factor)
                    else:
                        resp = "Function not recognized (weakness)."
                elif function_name.startswith("find_strengths"):
                    match = re.match(r"find_strengths\((\w+)\)", function_name)
                    if match:
                        factor = match.group(1)
                        resp = find_strengths(factor)
                    else:
                        resp = "Function not recognized (strengths)."
                
                elif function_name.startswith("show_summary"):
                    resp = create_business_summary()
                elif function_name.startswith("deep_analysis"):
                    lower_query = query.lower()
                    factors_found = []
                    for factor in self.domain_factors:
                        display_factor = factor.replace("_", " ")
                        pos = lower_query.find(display_factor)
                        if pos == -1:
                            pos = lower_query.find(factor)
                        if pos != -1:
                            factors_found.append((pos, factor))
                    factors_found.sort(key=lambda x: x[0])
                    if len(factors_found) >= 2:
                        factor1, factor2 = factors_found[0][1], factors_found[1][1]
                    
                    # Extract any number from the query to use as the period (days)
                    days_found = re.findall(r'\d+', lower_query)
                    days = int(days_found[0]) if days_found else 365

                    resp = trend_generate_business_narrative(factor1, factor2, days)
                elif function_name.startswith("impact_analysis"):
                    resp = impact_analysis()
                elif function_name.startswith("risk_modeling"):
                    resp = run_risk_analysis("revenue")
                elif function_name.startswith("event_analysis"):
                    resp = calculate_event_probability("revenue", 651.4, 98, use_gbm=False, use_rolling_window=True, plot_residuals=False)
                elif function_name.startswith("competitor_analysis"):
                    resp = bayesian_competitor_analysis()





                else:
                    resp = "Function not recognized."
            else:
                resp = best_response

        self.train(query, resp)
        return resp

# ---------------------
# Expanded Training Pairs Mapping Query Data to Function Invocation

# Revenue Analysis Training Pairs
model = ProbabilisticLanguageModel()


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


##impact analysis
model.train("calculate an event probability", "invoke: event_analysis")


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


#response_obj = model.process_query("hi")
#print(response_obj)