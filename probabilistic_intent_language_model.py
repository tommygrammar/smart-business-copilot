import networkx as nx
import re
import math
#Model Fuctions Imports
from Models.wave_driver_analysis import generate_business_narrative
from Models.deep_wave_driver_analysis_period_support import trend_generate_business_narrative
from Data.business_data import historical_data
from Models.strengths_weakness_assessment_model import losses
from Models.official_business_summary import create_business_summary
from Models.impact_analysis_model import bayesian_impact as impact_analysis
from Models.competitor_analysis import bayesian_competitor_analysis
from Models.cash_flow import weekly_cashflow_sales_analysis
from Models.supply_chain_fragility import supply_chain_fragility
from Models.stockout import stockout
from Models.sales_time import time_analysis
from Models.demand_analysis import demand_analysis
from Models.product_segmentation import type_shit
from Models.metrics import how_much_inventory, how_much_revenue, how_much_sales
# Use combinations of distinct factors for analysis compar
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
        self.stopwords = {"this", "is", "very", "and", "the", "a", "an", "of", "to", "with", "in", "it", "for", "on", "so", "exactly", "my"}
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
            #print(f"[DEBUG] Selected best training query: '{best_training_query}' with score {best_score}")
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
                #print(f"[DEBUG] Invoking function: '{function_name}' based on best training query")
                # If the best training query is of type business_twin, use extraction.

                if function_name == "analyze_drivers":
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
                elif function_name.startswith("competitor_analysis"):
                    resp = bayesian_competitor_analysis()
                elif function_name.startswith("cashflow_analysis"):
                    resp = weekly_cashflow_sales_analysis()
                elif function_name.startswith("supply_chain"):
                    resp =  supply_chain_fragility()
                elif function_name.startswith("stockout"):
                    resp = stockout()
                elif function_name.startswith("sales_time"):
                    resp = time_analysis()
                elif function_name.startswith("demand"):
                    resp = demand_analysis()
                elif function_name.startswith("segment"):
                    resp = type_shit()
                elif function_name.startswith("loss"):
                    resp = losses("revenue")
                elif function_name.startswith("how_many_sales"):
                    resp = how_much_sales(1)
                elif function_name.startswith("how_much_revenue"):
                    resp = f" {how_much_revenue(1)}"
                elif function_name.startswith("how_much_inventory"):
                    resp = how_much_inventory(1)






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


#response_obj = model.process_query("hi")
#print(response_obj)