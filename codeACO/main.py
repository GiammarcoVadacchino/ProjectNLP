from PromptDAG import PromptDAG
from PromptACO import PromptACO
import random
from transformers import T5Tokenizer, T5ForConditionalGeneration
from Dataset import Dataset
from PromptEvaluator import PromptEvaluator
from StrategyACO import StrategyACO
import time


DEVICE = "mps"
MODEL_NAME = "t5-base"
HUMAN_PROMPT = False
SEED = 40

model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

"""
PART OF A PROMPT:

RoleInstruction,
TaskInstruction,
FidelityConstraint,
SalienceGuidance,
StyleConstraint,
Input,
LengthConstraint,
OutputFormat
"""


def build_dataset(
    csv_path,
    input_col,
    output_col,
    n_samples=20,
    target_tokens=350,
    tolerance=20,
    train_fraction=0.8,
    seed=SEED
):
    dataset = Dataset(
        csv_path=csv_path,
        input_col=input_col,
        output_col=output_col,
        model_name=MODEL_NAME
    )

    dataset.load_with_token_limit(
        number_of_samples=n_samples,
        target_tokens=target_tokens,
        tolerance=tolerance
    )

    dataset.split_train_test(
        train_fraction=train_fraction,
        seed=seed,
        shuffle=True
    )

    return dataset


def build_prompt_dag(
    levels,
    base_texts,
    model,
    tokenizer,
    nodes_per_level,
    variants_csv="prompt_variants.csv"
):
    assert len(levels) == len(base_texts)
    assert len(levels) == len(nodes_per_level)

    dag = PromptDAG(levels)
    dag.initialize_levels(base_texts)

    dag.prepare_variants(
        load_variants=True,
        path_csv=variants_csv,
        nodes_per_level=list(nodes_per_level),
        model=model,
        tokenizer=tokenizer
    )

    dag.build_graph()
    return dag


def run_experiment(
    levels,
    base_texts,
    nodes_per_level,
    model,
    tokenizer,
    dataset,              
    strategyACO,
    strategy_name,
    n_ants=4,
    iterations=4,
    alpha=1,
    evaporation_rate=0.2
):
    random.seed(SEED)

    dag = build_prompt_dag(
        levels=levels,
        base_texts=base_texts,
        nodes_per_level=nodes_per_level,
        model=model,
        tokenizer=tokenizer
    )

    evaluator = PromptEvaluator(
        model,
        tokenizer,
        dataset.train_pairs
    )

    aco = PromptACO(
        dag=dag,
        evaluator=evaluator,
        strategyACO=strategyACO,
        strategy_name=strategy_name,
        n_ants=n_ants,
        iterations=iterations,
        alpha=alpha,
        evaporation_rate=evaporation_rate
    )

    best_prompt, best_score = aco.run()
    aco.save_history_to_csv(file_path="../results/statsACO.csv") 

    return best_prompt, best_score, aco.history


EXPERIMENTS = [
    {
        "levels": ["RoleInstruction","TaskInstruction","FidelityConstraint","SalienceGuidance","StyleConstrain","Input","LengthConstraint"],
        "base_texts": [
            "Pretend you are an expert that summarize text",
            "Summarize this text.",
            "Write a clear and curt summary",
            "Focus on key points",
            "Use a clear, concise, and neutral style",
            "the text:",
            "Limit the summary to a few sentences"
        ],
        "nodes_per_level": (15,15, 15,15,15, 15,15),
        "strategy": StrategyACO(),
        "strategy_name": "Basic ACO"
    },
    {
        "levels": ["RoleInstruction","TaskInstruction","FidelityConstraint","SalienceGuidance","StyleConstrain","Input","LengthConstraint"],
        "base_texts": [
            "Pretend you are an expert that summarize text",
            "Summarize this text.",
            "Write a clear and curt summary",
            "Focus on key points",
            "Use a clear, concise, and neutral style",
            "the text:",
            "Limit the summary to a few sentences"
        ],
        "nodes_per_level": (15,15, 15,15,15, 15,15),
        "strategy": StrategyACO(),
        "strategy_name": "EAS"
    },
    {
        "levels": ["RoleInstruction","TaskInstruction","FidelityConstraint","SalienceGuidance","StyleConstrain","Input","LengthConstraint"],
        "base_texts": [
            "Pretend you are an expert that summarize text",
            "Summarize this text.",
            "Write a clear and curt summary",
            "Focus on key points",
            "Use a clear, concise, and neutral style",
            "the text:",
            "Limit the summary to a few sentences"
        ],
        "nodes_per_level": (15,15, 15,15,15, 15,15),
        "strategy": StrategyACO(),
        "strategy_name": "Rank"
    },
    {
        "levels": ["RoleInstruction","TaskInstruction","FidelityConstraint","SalienceGuidance","StyleConstrain","Input","LengthConstraint"],
        "base_texts": [
            "Pretend you are an expert that summarize text",
            "Summarize this text.",
            "Write a clear and curt summary",
            "Focus on key points",
            "Use a clear, concise, and neutral style",
            "the text:",
            "Limit the summary to a few sentences"
        ],
        "nodes_per_level": (15,15, 15,15,15, 15,15),
        "strategy": StrategyACO(),
        "strategy_name": "MMAS"
    },
    {
        "levels": ["RoleInstruction","TaskInstruction","FidelityConstraint","SalienceGuidance","StyleConstrain","Input","LengthConstraint"],
        "base_texts": [
            "Pretend you are an expert that summarize text",
            "Summarize this text.",
            "Write a clear and curt summary",
            "Focus on key points",
            "Use a clear, concise, and neutral style",
            "the text:",
            "Limit the summary to a few sentences"
        ],
        "nodes_per_level": (15,15, 15,15,15, 15,15),
        "strategy": StrategyACO(),
        "strategy_name": "BWAS"
    }
]


def main():
    random.seed(SEED)
    dataset = build_dataset(
        csv_path="../data/test.csv",
        input_col="article",
        output_col="highlights",
        seed=SEED
    )

    if not HUMAN_PROMPT:
        for idx, exp in enumerate(EXPERIMENTS):

            begin = time.time()

            print(f"\nRunning experiment {idx + 1}/{len(EXPERIMENTS)}")
            print("Levels:", exp["levels"])
            print("Nodes per level:", exp["nodes_per_level"])
            print("Strategy:", exp["strategy_name"])

            run_experiment(
                levels=exp["levels"],
                base_texts=exp["base_texts"],
                nodes_per_level=exp["nodes_per_level"],
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,                 
                strategyACO=exp["strategy"],
                strategy_name=exp["strategy_name"]
            )

            print(f"\nTOTAL TIME TAKEN: {time.time() - begin} secs ==========================================================")

    else:
        prompt = "Pretend you are an expert that summarize text\nSummarize this text\nWrite a clear and curt summary\nFocus on key points\nUse a clear, concise, and neutral style\nthe text: {INPUT}\nLimit the summary to a few sentences."
        promptEvaluator = PromptEvaluator(model=model, tokenizer=tokenizer, dataset=dataset.train_pairs)
        promptEvaluator.evaluatePrompt(
            prompt=prompt,
            humanPrompt=HUMAN_PROMPT,
            humanPromptcsvPath="../results/statsHumanPrompt.csv"
        )


if __name__ == "__main__":
    main()
