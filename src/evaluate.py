import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pipeline import build_rag_pipeline
from retriever import build_retriever_by_type
import pandas as pd



proxy_hosts = {"localhost", "127.0.0.1", "::1"}
proxy_hosts.update(
    host.strip()
    for host in os.environ.get("NO_PROXY", os.environ.get("no_proxy", "")).split(",")
    if host.strip()
)
os.environ["NO_PROXY"] = ",".join(sorted(proxy_hosts))
os.environ["no_proxy"] = os.environ["NO_PROXY"]




# Define root directory and append to sys.path if not present
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


# File paths for datasets and evaluation output directories
TEST_DATA_PATH = ROOT_DIR.parent / "dataset" / "Q&A.json"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REPORT_JSON_PATH = RESULTS_DIR / "comparative_evaluation_report.json"
OVERALL_CSV_PATH = RESULTS_DIR / "overall_results.csv"
CATEGORY_CSV_PATH = RESULTS_DIR / "category_matrix.csv"
DETAILED_QUESTIONS_CSV_PATH = RESULTS_DIR / "detailed_questions_results.csv"

# Load environment variables
load_dotenv()

# Initialize the Groq Chat model used as LLM Judge
judge_llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0.0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)


def evaluate_response_with_judge(
    question: str, context: str, answer: str, ground_truth: str, is_answerable: bool = True
) -> dict:
    
    prompt_template = ChatPromptTemplate.from_template("""
You are a strict, objective, and critical evaluator for a Retrieval-Augmented Generation (RAG) system.

Your task: evaluate a single RAG response by comparing the Generated Answer against the Ground Truth, using the Retrieved Context as the ONLY allowed knowledge source.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Question: {question}
Ground Truth Answer: {ground_truth}
Retrieved Context: {context}
Generated Answer: {answer}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVALUATION RUBRIC (1 to 5):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1] FAITHFULNESS (1-5)
Does the Generated Answer rely ONLY on the Retrieved Context?

  5 = Every claim is directly supported by the context.
  4 = Minor paraphrasing, no new facts introduced.
  3 = One small claim is unsupported but is common knowledge.
  2 = One or more significant claims NOT in context (minor hallucination).
  1 = Answer invents facts, contradicts context, or is fabricated.

  Special case: If the answer refuses ("I don't know") while the context DOES contain the answer → score 2.
  Special case: If the answer refuses while the context does NOT contain the answer → score 5.

[2] ANSWER RELEVANCE (1-5)
Does the Generated Answer fully and accurately answer the Question, compared to the Ground Truth?

  5 = Fully correct, semantically matches Ground Truth.
  4 = Correct but missing a minor detail present in Ground Truth.
  3 = Partially correct: main idea captured but key component missing or minor error.
  2 = Mostly incorrect: wrong core answer with tangential relevance.
  1 = Completely wrong, irrelevant, or contradicts Ground Truth.

  Special case: For unanswerable questions, correct refusal = 5, any attempted answer = 1.

[3] CONTEXT RELEVANCE (1-5)
Does the Retrieved Context contain the EXACT information needed, without excessive noise or missing crucial facts?

  5 = Contains all needed facts; no irrelevant content.
  4 = Contains needed facts + minor irrelevant content.
  3 = Contains needed facts but mixed with significant noise, OR missing one secondary fact.
  2 = Missing a crucial fact required to answer correctly.
  1 = Entirely irrelevant to the question.

  Special case: For questions requiring information from multiple sections, score 5 ONLY if context covers ALL required sections. If one is missing → max score 3.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Be STRICT. Do NOT give 5 by default. Reserve 5 for exceptional cases.
2. Ignore writing style, grammar, or length. Judge CONTENT only.
3. Semantic equivalence counts as correct (different wording is OK).
4. If the answer is correct but context is missing facts → penalize CONTEXT_RELEVANCE, not FAITHFULNESS.
5. If the answer adds correct external knowledge not in context → still penalize FAITHFULNESS.
6. Do NOT be lenient. A "4" means there is a real, identifiable flaw.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (STRICT JSON ONLY, no extra text):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{{
    "reasoning": {{
    "faithfulness_analysis": "Which claims are/are not supported by context.",
    "answer_relevance_analysis": "Compare answer to Ground Truth. What is correct, missing, wrong.",
    "context_relevance_analysis": "Which facts were needed, missing, or noise."
    }},
    "scores": {{
    "faithfulness": <1-5>,
    "answer_relevance": <1-5>,
    "context_relevance": <1-5>
    }},
  "verdict": "<correct|partially_correct|incorrect|refused>"
}}
""")
    try:
        response = judge_llm.invoke(
            prompt_template.format(
                question=question,
                is_answerable=is_answerable,
                ground_truth=ground_truth,
                context=context if context.strip() else "NO CONTEXT RETRIEVED",
                answer=answer,
            )
        )
        json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
        if json_match:
            parsed_result = json.loads(json_match.group())
            scores = parsed_result.get("scores", parsed_result)
            return {
                "reasoning": parsed_result.get("reasoning", ""),
                "faithfulness": scores.get("faithfulness", 1),
                "answer_relevance": scores.get("answer_relevance", 1),
                "context_relevance": scores.get("context_relevance", 1),
            }
    except Exception as e:
        print(f"Evaluation error: {e}")

    return {"reasoning": "Failed to parse judge output", "faithfulness": 1, "answer_relevance": 1, "context_relevance": 1}


def load_overall_json() -> dict:
    """Loads existing evaluation JSON report from disk to preserve previously evaluated models."""
    if REPORT_JSON_PATH.exists():
        try:
            with open(REPORT_JSON_PATH, "r", encoding="utf-8") as f:
                stored_results = json.load(f)
                return {
                    model_name: result
                    for model_name, result in stored_results.items()
                    if model_name in {"faiss", "proposed"}
                }
        except Exception as e:
            print(f"⚠️ Warning loading existing report: {e}")
    return {}


def update_and_save_all_reports(comparative_results: dict):
    """
    Persists updated evaluation results instantly to JSON and CSV files on disk 
    without overwriting data from other models.
    """
    # 1. Save full JSON report
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(comparative_results, f, ensure_ascii=False, indent=2)

    # 2. Save overall summary CSV
    overall_data = []
    for model_name, res in comparative_results.items():
        if "overall" in res:
            row = {"Model": model_name.upper()}
            row.update(res["overall"])
            overall_data.append(row)
    if overall_data:
        pd.DataFrame(overall_data).to_csv(OVERALL_CSV_PATH, index=False, encoding="utf-8-sig")

    # 3. Save Category x Difficulty breakdown matrix CSV
    all_cat_subcats = set()
    for res in comparative_results.values():
        if "by_category" in res:
            all_cat_subcats.update(res["by_category"].keys())

    if all_cat_subcats:
        matrix_rows = []
        # Sort category combinations alphabetically
        for key in sorted(all_cat_subcats):
            cat_name, difficulty = key.split(" | ") if " | " in key else (key, "General")
            row = {"Category": cat_name, "Difficulty": difficulty}
            
            # Aggregate metrics across models present in report
            for model in ["faiss", "proposed"]:
                if model in comparative_results and "by_category" in comparative_results[model]:
                    cat_data = comparative_results[model]["by_category"].get(key, {})
                    row[f"{model.upper()}_Faithfulness"] = cat_data.get("avg_faithfulness", 0)
                    row[f"{model.upper()}_Answer_Rel"] = cat_data.get("avg_answer_relevance", 0)
                    row[f"{model.upper()}_Context_Rel"] = cat_data.get("avg_context_relevance", 0)
            matrix_rows.append(row)
            
        pd.DataFrame(matrix_rows).to_csv(CATEGORY_CSV_PATH, index=False, encoding="utf-8-sig")

    # 4. Save detailed aggregated results for all questions across models
    detailed_rows = []
    for model_name, res in comparative_results.items():
        for q_item in res.get("detailed_questions", []):
            row = {"Model": model_name.upper()}
            row.update(q_item)
            detailed_rows.append(row)
    if detailed_rows:
        pd.DataFrame(detailed_rows).to_csv(DETAILED_QUESTIONS_CSV_PATH, index=False, encoding="utf-8-sig")


def run_benchmark_for_model(model_name: str, md_path: str, test_cases: list, comparative_results: dict) -> dict:
    # Runs end-to-end evaluation benchmark for a specific model variant.
    print(f"\n==========================================")
    print(f"🚀 Evaluating Model Variant: [{model_name.upper()}]")
    print(f"==========================================")

    model_specific_csv = RESULTS_DIR / f"detailed_questions_{model_name.lower()}.csv"

    # Resume checkpoint -> check for existing evaluated questions under this model
    existing_model_data = comparative_results.get(model_name, {})
    detailed_case_results = [
        item
        for item in existing_model_data.get("detailed_questions", [])
        if item.get("generated_answer") != "Error generating response"
    ]
    processed_q_ids = {q["question_id"] for q in detailed_case_results}

    if processed_q_ids:
        print(f"🔄 Checkpoint found! Skipping {len(processed_q_ids)} already evaluated questions for [{model_name.upper()}].")

    # Initialize retriever and RAG pipeline for the selected model variant
    retriever = build_retriever_by_type(md_path, retriever_type=model_name)
    rag_chain = build_rag_pipeline(retriever, include_context=True)

    num_cases = len(test_cases)
    model_start_time = time.time()

    for idx, item in enumerate(test_cases, 1):
        # Retrieve unique question identifier
        q_id = item.get("id", idx)
        
        # Skip question if already evaluated
        if q_id in processed_q_ids or idx in processed_q_ids:
            continue

        question = item["question"]
        ground_truth = item["answer"]
        is_answerable = item.get("category") != "trap"
        
        # Extract Category and Difficulty fields
        cat = item.get("category", "General")
        difficulty = item.get("difficulty", "General")
        chapter = item.get("chapter", "General")

        print(f"\n--- [{model_name.upper()}] Q{idx}/{num_cases} ({q_id}) ---")
        print(f"❓ Question: {question}")
        print(f"🏷️ Category: {cat} | 📂 Difficulty: {difficulty}")

        q_start_time = time.time()

        # Execute RAG chain with retry logic for network or API errors
        print("internal API")
        max_retries = 3
        rag_result = {}
        for attempt in range(max_retries):
            try:
                rag_result = rag_chain.invoke(question)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"RAG attempt {attempt + 1} failed for question {q_id}: {e}")
                    time.sleep(2)
                else:
                    raise RuntimeError(
                        f"RAG generation failed for question {q_id} after {max_retries} attempts"
                    ) from e


        
        q_execution_time = round(time.time() - q_start_time, 2)
        context_str = rag_result.get("context", "")
        answer_str = rag_result.get("answer", "")

        print("external API")
        # Evaluate output using LLM Judge
        eval_res = evaluate_response_with_judge(
            question, context_str, answer_str, ground_truth, is_answerable
        )

        f_s, a_s, c_s = eval_res["faithfulness"], eval_res["answer_relevance"], eval_res["context_relevance"]
        reasoning = eval_res.get("reasoning", "")

        q_item_data = {
            "question_id": q_id,
            "category": cat,
            "chapter": chapter,
            "difficulty": difficulty,
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": answer_str,
            "retrieved_context": context_str,
            "faithfulness": f_s,
            "answer_relevance": a_s,
            "context_relevance": c_s,
            "latency_seconds": q_execution_time,
            "reasoning": reasoning,
        }
        detailed_case_results.append(q_item_data)

        # Calculate updated overall averages
        tot_f = sum(x["faithfulness"] for x in detailed_case_results)
        tot_a = sum(x["answer_relevance"] for x in detailed_case_results)
        tot_c = sum(x["context_relevance"] for x in detailed_case_results)
        tot_time = sum(x["latency_seconds"] for x in detailed_case_results)
        processed_count = len(detailed_case_results)

        # Calculate category and difficulty metrics
        cat_scores = defaultdict(lambda: {"count": 0, "f": 0, "a": 0, "c": 0})
        for x in detailed_case_results:
            key = f"{x['category']} | {x['difficulty']}"
            cat_scores[key]["count"] += 1
            cat_scores[key]["f"] += x["faithfulness"]
            cat_scores[key]["a"] += x["answer_relevance"]
            cat_scores[key]["c"] += x["context_relevance"]

        by_category = {}
        for c_key, val in cat_scores.items():
            cnt = val["count"]
            by_category[c_key] = {
                "count": cnt,
                "avg_faithfulness": round(val["f"] / cnt, 2),
                "avg_answer_relevance": round(val["a"] / cnt, 2),
                "avg_context_relevance": round(val["c"] / cnt, 2),
            }

        comparative_results[model_name] = {
            "overall": {
                "avg_faithfulness": round(tot_f / processed_count, 2),
                "avg_answer_relevance": round(tot_a / processed_count, 2),
                "avg_context_relevance": round(tot_c / processed_count, 2),
                "total_execution_time_sec": round(tot_time, 2),
                "avg_latency_per_question_sec": round(tot_time / processed_count, 2),
            },
            "by_category": by_category,
            "detailed_questions": detailed_case_results,
        }

        # Instantly persist updated JSON and CSV reports to disk
        update_and_save_all_reports(comparative_results)
        
        # Save model-specific detailed CSV
        model_rows = [{"Model": model_name.upper(), **q} for q in detailed_case_results]
        pd.DataFrame(model_rows).to_csv(model_specific_csv, index=False, encoding="utf-8-sig")

        print(f"📊 Scores -> F: {f_s} | A: {a_s} | C: {c_s} | ⏱️ Time: {q_execution_time}s [JSON & CSVs Updated 💾]")

    total_model_time = round(time.time() - model_start_time, 2)
    print(f"\n==========================================")
    print(f"✅ Completed Model [{model_name.upper()}] in {total_model_time} sec")
    print(f"==========================================")

    return comparative_results[model_name]


def main():
    if not TEST_DATA_PATH.exists():
        print(f"Error: Dataset not found at {TEST_DATA_PATH}")
        return

    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)["qa_pairs"]

    md_path = str(ROOT_DIR.parent / "dataset" / "md-book.md")
    
    # Specify model variants to run
    models_to_test = [ "proposed"]

    # Load existing benchmark reports to append results safely
    comparative_results = load_overall_json()

    for model in models_to_test:
        run_benchmark_for_model(model, md_path, test_cases, comparative_results)


if __name__ == "__main__":
    main()
